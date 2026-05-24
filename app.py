"""
app.py — Flask application for FoodRoute AI Dashboard
Enhanced with: Auth, Roles, NGO Requests, Homeless Clusters, Messaging,
               Route Map, Food Listings, Homeless Survey, Notifications,
               Volunteer Delivery Tracking, Registration
"""

import io
import os
import csv
import base64
import random
import threading
import functools
from typing import List, Optional
from datetime import datetime

import torch  # type: ignore[import]
import numpy as np  # type: ignore[import]
import matplotlib  # type: ignore[import]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import]

from flask import (
    Flask, jsonify, render_template, request,
    session, redirect, url_for, flash
)

from environment import FoodRedistributionEnv
from rl_model import train_agent, run_greedy_baseline

app = Flask(__name__)
app.secret_key = "foodroute-ai-secret-2026"

# ---------------------------------------------------------------------------
# In-Memory User Store
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# In-Memory User Store (Extended)
# ---------------------------------------------------------------------------
USERS = {
    "restaurant1": {"password": "pass123", "role": "Restaurant", "display": "Green Plate Bistro", "score": 1500, "donation_mode": False},
    "restaurant2": {"password": "pass123", "role": "Restaurant", "display": "Spice Garden",       "score": 1200, "donation_mode": False},
    "restaurant3": {"password": "pass123", "role": "Restaurant", "display": "Urban Kitchen",      "score": 850,  "donation_mode": False},
    "ngo1":        {"password": "pass123", "role": "NGO",        "display": "Feeding India",      "score": 0},
    "ngo2":        {"password": "pass123", "role": "NGO",        "display": "Robin Hood Army",    "score": 0},
    "volunteer1":  {"password": "pass123", "role": "Volunteer",  "display": "Arjun (NCC)",        "score": 210,  "depot": None},
    "volunteer2":  {"password": "pass123", "role": "Volunteer",  "display": "Priya (NSS)",        "score": 180,  "depot": None},
}

DEPOTS = ["T Nagar Depot", "Guindy Depot", "Anna Nagar Depot", "Tambaram Depot", "Central Depot"]

donations = []
messages = []

# ---------------------------------------------------------------------------
# In-Memory Application State
# ---------------------------------------------------------------------------
_simulation_data: dict = {}
_results: dict        = {}
_training_lock        = threading.Lock()

# NGO Requests
_ngo_requests: list = []
# Messages: keyed by request_id -> list of message dicts
_messages: dict = {}
# Food Listings submitted by restaurants
_food_listings: list = []
# Volunteer delivery tracking
_delivery_status: dict = {}   # username -> {"status": "idle"|"en_route"|"delivered", "started_at": ...}
# Notifications: username -> list of notification dicts
_notifications: dict = {}

_request_id_counter = 1

FOOD_TYPES   = ["Cooked Rice", "Bread Loaves", "Dal & Curry", "Fruits", "Biryani", "Rotis", "Soup", "Snack Boxes"]
DEMAND_TYPES = ["Cooked Meal", "Dry Ration", "Any Food", "Fresh Produce", "Packaged Snacks"]
DENSITY_OPTS = ["High", "Medium", "Low"]


# ---------------------------------------------------------------------------
# Helper: Push Notification
# ---------------------------------------------------------------------------
def push_notif(username: str, msg: str, category: str = "info"):
    """Append a notification for a given username."""
    global _notifications
    if username not in _notifications:
        _notifications[username] = []
    _notifications[username].append({
        "msg": msg,
        "category": category,
        "time": datetime.now().strftime("%H:%M"),
        "read": False,
    })


# ---------------------------------------------------------------------------
# Auth Helpers
# ---------------------------------------------------------------------------

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

def restaurant_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session or session["user"].get("role") != "Restaurant":
            return jsonify({"success": False, "error": "Unauthorized: Restaurant only"}), 403
        return f(*args, **kwargs)
    return decorated

def ngo_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session or session["user"].get("role") != "NGO":
            return jsonify({"success": False, "error": "Unauthorized: NGO only"}), 403
        return f(*args, **kwargs)
    return decorated

def volunteer_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session or session["user"].get("role") != "Volunteer":
            return jsonify({"success": False, "error": "Unauthorized: Volunteer only"}), 403
        return f(*args, **kwargs)
    return decorated


def current_user():
    return session.get("user", None)


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET"])
def login_page():
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_post():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    role_attempt = request.form.get("role", "").strip()

    user = USERS.get(username)
    if not user or user["password"] != password:
        flash("Invalid username or password.", "error")
        return redirect(url_for("login_page"))
    if role_attempt and user["role"] != role_attempt:
        flash("Selected role does not match your account.", "error")
        return redirect(url_for("login_page"))

    session["user"] = {
        "username": username,
        "role": user["role"],
        "display": user["display"],
    }
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@app.route("/api/register", methods=["POST"])
def register():
    """Register a new user (in-memory)."""
    data = request.get_json(force=True)
    username    = data.get("username", "").strip()
    password    = data.get("password", "").strip()
    display     = data.get("display", "").strip()
    role        = data.get("role", "NGO")

    if not username or not password or not display:
        return jsonify({"success": False, "error": "All fields required."}), 400
    if username in USERS:
        return jsonify({"success": False, "error": "Username already taken."}), 409
    if role not in ("Restaurant", "NGO", "Volunteer"):
        return jsonify({"success": False, "error": "Invalid role."}), 400

    USERS[username] = {"password": password, "role": role, "display": display, "score": 0}
    if role == "Restaurant":
        USERS[username]["donation_mode"] = False
    if role == "Volunteer":
        USERS[username]["depot"] = None

    return jsonify({"success": True, "message": f"Account created! Login as {username}."})


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    return render_template("index.html", user=current_user())


# ---------------------------------------------------------------------------
# Route Map Page
# ---------------------------------------------------------------------------

@app.route("/route-map")
@login_required
def route_map_page():
    import json
    results_json = json.dumps(_results) if _results else "{}"
    sim_json = json.dumps(_simulation_data) if _simulation_data else "{}"
    return render_template("route_map.html", user=current_user(),
                           results_json=results_json, sim_json=sim_json)


# ---------------------------------------------------------------------------
# Utility: Matplotlib helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _generate_data_internal(seed: Optional[int] = None) -> dict:
    """Generate fresh simulation data including homeless clusters."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    restaurants = []
    for i in range(20):
        surplus = round(random.uniform(10, 50), 2)
        restaurants.append({
            "id": i + 1,
            "name": f"Restaurant {i + 1}",
            "x": round(random.uniform(5, 95), 2),
            "y": round(random.uniform(5, 95), 2),
            "surplus": surplus,
            "food_type": random.choice(FOOD_TYPES),
            "expiry_hours": random.choice([1, 2, 3, 4, 6]),
            "priority": "High" if surplus > 35 else "Medium" if surplus > 20 else "Low",
        })

    ngos = []
    for j in range(10):
        demand = round(random.uniform(5, 40), 2)
        ngos.append({
            "id": j + 1,
            "name": f"NGO {j + 1}",
            "x": round(random.uniform(5, 95), 2),
            "y": round(random.uniform(5, 95), 2),
            "demand": demand,
            "demand_type": random.choice(DEMAND_TYPES),
            "capacity": round(random.uniform(demand, demand + 30), 1),
            "priority": "High" if demand > 30 else "Medium" if demand > 18 else "Low",
        })

    homeless_clusters = []
    for k in range(8):
        size = random.randint(15, 80)
        homeless_clusters.append({
            "id": k + 1,
            "name": f"Cluster {k + 1}",
            "x": round(random.uniform(5, 95), 2),
            "y": round(random.uniform(5, 95), 2),
            "cluster_size": size,
            "demand": round(size * random.uniform(0.4, 0.8), 2),
            "density": random.choice(DENSITY_OPTS),
            "location_desc": random.choice([
                "Near Railway Station", "Under Flyover", "Market Area",
                "Bus Stand", "Park Zone", "Industrial Area", "River Bank", "Old City"
            ]),
        })

    return {
        "restaurants": restaurants,
        "ngos": ngos,
        "homeless": homeless_clusters,
    }


def _build_reward_graph(rewards: List[float], smoothed: List[float]) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    episodes = list(range(1, len(rewards) + 1))
    ax.plot(episodes, rewards, color="#60a5fa", alpha=0.3, linewidth=1, label="Episode Reward")
    ax.plot(episodes, smoothed, color="#34d399", linewidth=2.5, label="10-ep Moving Avg")
    ax.set_xlabel("Episode", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Total Reward", color="#94a3b8", fontsize=11)
    ax.set_title("DQN Reward Convergence", color="#f1f5f9", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=9)
    ax.grid(True, color="#334155", linestyle="--", linewidth=0.5, alpha=0.5)
    if smoothed:
        ax.annotate(f"Final avg: {smoothed[-1]:.1f}",
                    xy=(len(smoothed), smoothed[-1]),
                    xytext=(-60, 15), textcoords="offset points",
                    color="#34d399", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#34d399", lw=1))
    return _fig_to_b64(fig)


def _build_route_map(restaurants, ngos, route, homeless=None) -> str:
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    rx = [r["x"] for r in restaurants]; ry = [r["y"] for r in restaurants]
    ax.scatter(rx, ry, s=120, c="#f97316", zorder=5, label="Restaurants")
    for r in restaurants:
        ax.annotate(f"R{r['id']}", (r["x"], r["y"]),
                    textcoords="offset points", xytext=(6, 4), color="#fdba74", fontsize=7)
    nx = [n["x"] for n in ngos]; ny = [n["y"] for n in ngos]
    ax.scatter(nx, ny, s=120, c="#818cf8", zorder=5, marker="D", label="NGOs")
    for n in ngos:
        ax.annotate(f"N{n['id']}", (n["x"], n["y"]),
                    textcoords="offset points", xytext=(6, 4), color="#a5b4fc", fontsize=7)
    if homeless:
        hx = [h["x"] for h in homeless]; hy = [h["y"] for h in homeless]
        ax.scatter(hx, hy, s=100, c="#34d399", zorder=5, marker="^", label="Homeless Clusters")
        for h in homeless:
            ax.annotate(f"HC{h['id']}", (h["x"], h["y"]),
                        textcoords="offset points", xytext=(6, 4), color="#6ee7b7", fontsize=7)
    ax.scatter([50], [50], s=250, c="#f43f5e", zorder=6, marker="*", label="Depot")
    if route and len(route) > 1:
        xs = [route[0][1]]; ys = [route[0][2]]
        for stop in route[1:]:
            xs.append(stop[1]); ys.append(stop[2])
        for i in range(len(xs) - 1):
            color = "#f97316" if (i < len(route)-1 and len(route[i+1]) > 3
                                  and route[i+1][0] == "restaurant") else "#818cf8"
            ax.annotate("", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("X Coordinate", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Y Coordinate", color="#94a3b8", fontsize=11)
    ax.set_title("Optimized Redistribution Route", color="#f1f5f9", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.grid(True, color="#334155", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", loc="upper left")
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# API Routes — Data
# ---------------------------------------------------------------------------

@app.route("/api/generate", methods=["POST"])
@login_required
@restaurant_required
def generate():
    global _simulation_data, _results
    _simulation_data = _generate_data_internal()
    _results = {}
    return jsonify({"status": "ok", "data": _simulation_data})


@app.route("/api/status", methods=["GET"])
@login_required
def status():
    return jsonify({
        "data_ready": bool(_simulation_data),
        "results_ready": bool(_results),
    })


# ---------------------------------------------------------------------------
# API Routes — Optimization
# ---------------------------------------------------------------------------

@app.route("/run_optimization", methods=["POST"])
@login_required
@restaurant_required
def run_optimization():
    global _results
    try:
        if not _simulation_data:
            return jsonify({"success": False, "error": "Generate simulation data first."}), 400

        n_episodes = int(request.json.get("episodes", 300)) if request.is_json else 300
        n_episodes = max(10, min(n_episodes, 500))

        with _training_lock:
            env = FoodRedistributionEnv()
            env.seed_data(_simulation_data["restaurants"], _simulation_data["ngos"])

            agent, rewards, smoothed = train_agent(env, n_episodes=n_episodes)
            greedy_summary = run_greedy_baseline(env)

            state = env.reset(use_seeded=True)
            dqn_done = False
            while not dqn_done:
                s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = int(agent.policy_net(s_t).argmax(dim=1).item())
                state, _, dqn_done, _ = env.step(action)
            dqn_summary = env.get_summary()

            reward_graph = _build_reward_graph(rewards, smoothed)
            route_map = _build_route_map(
                _simulation_data["restaurants"],
                _simulation_data["ngos"],
                dqn_summary["route"],
                _simulation_data.get("homeless", []),
            )

            # Compute 50/50 distribution allocation
            total_ngo_demand      = sum(n["demand"] for n in _simulation_data["ngos"])
            total_homeless_demand = sum(h["demand"] for h in _simulation_data.get("homeless", []))
            delivered = dqn_summary["total_delivered"]
            half = delivered / 2
            ngo_delivered      = round(min(half, total_ngo_demand), 2)
            homeless_delivered = round(min(half, total_homeless_demand), 2)

            # Fairness score: 1 = perfect 50/50, 0 = totally one-sided
            total_dist = ngo_delivered + homeless_delivered
            fairness_score = round(1 - abs(ngo_delivered - homeless_delivered) / max(total_dist, 1), 3)

            _results = {
                "dqn": dqn_summary,
                "greedy": greedy_summary,
                "rewards": rewards,
                "smoothed": smoothed,
                "reward_graph": reward_graph,
                "route_map": route_map,
                "episodes": n_episodes,
                "distribution": {
                    "ngo": ngo_delivered,
                    "homeless": homeless_delivered,
                    "total": delivered,
                    "fairness_score": fairness_score,
                },
                "volunteer": "Arjun (NCC) + Priya (NSS)",
            }

        return jsonify({
            "success": True,
            "route": dqn_summary["route"],
            "total_food": dqn_summary["total_delivered"],
            "total_distance": dqn_summary["total_distance"],
            "rewards": rewards,
            "dqn": dqn_summary,
            "greedy": greedy_summary,
            "reward_graph": reward_graph,
            "route_map": route_map,
            "rewards_last10": smoothed[-10:] if smoothed else [],
            "episodes": n_episodes,
            "distribution": _results["distribution"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
@app.route("/api/optimize-route", methods=["POST"])
@login_required
@volunteer_required
def volunteer_optimize_route():
    """Identical to /run_optimization, but authorized for Volunteers."""
    global _results
    try:
        if not _simulation_data:
            return jsonify({"success": False, "error": "Generate simulation data first."}), 400

        n_episodes = int(request.json.get("episodes", 300)) if request.is_json else 300
        n_episodes = max(10, min(n_episodes, 500))

        with _training_lock:
            env = FoodRedistributionEnv()
            env.seed_data(_simulation_data["restaurants"], _simulation_data["ngos"])

            agent, rewards, smoothed = train_agent(env, n_episodes=n_episodes)
            greedy_summary = run_greedy_baseline(env)

            state = env.reset(use_seeded=True)
            dqn_done = False
            while not dqn_done:
                s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = int(agent.policy_net(s_t).argmax(dim=1).item())
                state, _, dqn_done, _ = env.step(action)
            dqn_summary = env.get_summary()

            reward_graph = _build_reward_graph(rewards, smoothed)
            route_map = _build_route_map(
                _simulation_data["restaurants"],
                _simulation_data["ngos"],
                dqn_summary["route"],
                _simulation_data.get("homeless", []),
            )

            total_ngo_demand      = sum(n["demand"] for n in _simulation_data["ngos"])
            total_homeless_demand = sum(h["demand"] for h in _simulation_data.get("homeless", []))
            delivered = dqn_summary["total_delivered"]
            half = delivered / 2
            ngo_delivered      = round(min(half, total_ngo_demand), 2)
            homeless_delivered = round(min(half, total_homeless_demand), 2)

            total_dist = ngo_delivered + homeless_delivered
            fairness_score = round(1 - abs(ngo_delivered - homeless_delivered) / max(total_dist, 1), 3)

            _results = {
                "dqn": dqn_summary,
                "greedy": greedy_summary,
                "rewards": rewards,
                "smoothed": smoothed,
                "reward_graph": reward_graph,
                "route_map": route_map,
                "episodes": n_episodes,
                "distribution": {
                    "ngo": ngo_delivered,
                    "homeless": homeless_delivered,
                    "total": delivered,
                    "fairness_score": fairness_score,
                },
                "volunteer": current_user()["display"],
            }

        return jsonify({
            "success": True,
            "route": dqn_summary["route"],
            "total_food": dqn_summary["total_delivered"],
            "total_distance": dqn_summary["total_distance"],
            "rewards": rewards,
            "dqn": dqn_summary,
            "greedy": greedy_summary,
            "reward_graph": reward_graph,
            "route_map": route_map,
            "rewards_last10": smoothed[-10:] if smoothed else [],
            "episodes": n_episodes,
            "distribution": _results["distribution"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# API Routes — Common & New Round 3/4 APIs
# ---------------------------------------------------------------------------
@app.route("/api/ngo/available-donations", methods=["GET"])
@login_required
@ngo_required
def ngo_available_donations():
    """Returns active food listings from global donations list based on allocation rules."""
    user = current_user()
    user_display = user["display"]
    
    available = []
    for d in donations:
        # Rules:
        # 1. Public and Available
        # 2. Assigned to this NGO (regardless of status, though usually available/reserved)
        if d.get("donation_type") == "public" and d.get("status") == "available":
            available.append(d)
        elif d.get("donation_type") == "assigned" and d.get("assigned_ngo") == user_display:
            available.append(d)
            
    return jsonify({"success": True, "listings": available})

@app.route("/api/ngo/reserve-donation", methods=["POST"])
@login_required
@ngo_required
def reserve_donation():
    """FCFS Reservation for Public Donations."""
    data = request.get_json(force=True)
    donation_id = int(data.get("id"))
    user = current_user()
    user_display = user["display"]

    for d in donations:
        if d["id"] == donation_id:
            if d["status"] != "available" or d["donation_type"] != "public":
                return jsonify({"success": False, "error": "Donation is no longer available."}), 400
            
            d["status"] = "reserved"
            d["assigned_ngo"] = user_display
            
            # Notify restaurant
            owner_restaurant = d["restaurant"]
            # Find the username for this display name
            owner_username = None
            for u, udata in USERS.items():
                if udata["display"] == owner_restaurant:
                    owner_username = u
                    break
            
            if owner_username:
                push_notif(owner_username, f"🤝 {user_display} reserved your public donation: {d['food_type']}", "success")
            
            return jsonify({"success": True})
            
    return jsonify({"success": False, "error": "Donation not found."}), 404

@app.route("/api/messages/<peer_display>", methods=["GET"])
@login_required
def get_conversation(peer_display):
    """Returns conversation history between current user and a peer."""
    user = current_user()
    my_username = user["username"]
    
    # Find peer's username
    peer_username = None
    for u, udata in USERS.items():
        if udata["display"] == peer_display:
            peer_username = u
            break
    
    if not peer_username:
        # Fallback if display name is actually a username (e.g. from an old session)
        peer_username = peer_display

    history = []
    for m in messages:
        # Message matches if (me -> peer) OR (peer -> me)
        if (m["sender"] == my_username and m["receiver"] == peer_username) or \
           (m["sender"] == peer_username and m["receiver"] == my_username):
            history.append(m)
            
    return jsonify({"success": True, "messages": history})

@app.route("/api/message/send", methods=["POST"])
@login_required
def send_message():
    """Stores a Direct Message between users."""
    data = request.get_json(force=True)
    sender_user = current_user()
    sender_username = sender_user["username"]
    receiver_display = data.get("receiver", "").strip()
    msg_text = data.get("message", "").strip()

    # Find receiver username by display name
    receiver_username = None
    for u, udata in USERS.items():
        if udata["display"] == receiver_display:
            receiver_username = u
            break
    
    if not receiver_username:
        receiver_username = receiver_display

    new_msg = {
        "sender": sender_username,
        "receiver": receiver_username,
        "sender_display": sender_user["display"],
        "message": msg_text,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }
    messages.append(new_msg)
    
    # Debug Print
    print(f"DEBUG DM: {sender_username} -> {receiver_username}: {msg_text}")
    print(f"CURRENT MESSAGES COUNT: {len(messages)}")

    return jsonify({"success": True, "message": new_msg})
@app.route("/api/leaderboard", methods=["GET"])
@login_required
def get_leaderboard():
    """Returns sorted lists for Top Restaurants and Top Volunteers."""
    rests = [{"username": u, "display": d["display"], "score": d.get("score", 0), "donation_mode": d.get("donation_mode", False)}
             for u, d in USERS.items() if d["role"] == "Restaurant"]
    vols  = [{"username": u, "display": d["display"], "score": d.get("score", 0), "depot": d.get("depot")}
             for u, d in USERS.items() if d["role"] == "Volunteer"]
    rests.sort(key=lambda x: x["score"], reverse=True)
    vols.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"success": True, "restaurants": rests, "volunteers": vols})

@app.route("/api/restaurant/donation-mode", methods=["POST"])
@login_required
@restaurant_required
def toggle_donation_mode():
    data = request.get_json(force=True)
    enabled = bool(data.get("enabled", False))
    user = current_user()
    if user and user["username"] in USERS:
        USERS[user["username"]]["donation_mode"] = enabled
        if enabled:
            for u, d in USERS.items():
                if d["role"] == "NGO":
                    push_notif(u, f"{user['display']} is now a willing active donor!", "purple")
        return jsonify({"success": True, "enabled": enabled})
@app.route("/api/ngos", methods=["GET"])
@login_required
def get_ngos():
    """Returns list of registered NGOs for the assignment dropdown."""
    ngos = [{"username": u, "display": d["display"]} for u, d in USERS.items() if d["role"] == "NGO"]
    return jsonify({"success": True, "ngos": ngos})

# ---------------------------------------------------------------------------
# API Routes — Voluntary Donations (Round 6/7)
# ---------------------------------------------------------------------------
@app.route("/api/restaurant/donate", methods=["POST"])
@restaurant_required
def submit_voluntary_donation():
    """Restaurant explicitly donates food with optional NGO assignment."""
    user = current_user()
    data = request.get_json(force=True)
    qty = float(data.get("qty", 0))
    if qty <= 0:
        return jsonify({"success": False, "error": "Quantity must be > 0."}), 400

    donation_type = data.get("donation_type", "public")
    assigned_ngo = data.get("assigned_ngo") if donation_type == "assigned" else None

    new_donation = {
        "id": len(donations) + 1,
        "food_type": data.get("food_type", "Cooked Meals"),
        "quantity": qty,
        "expiry_hours": int(data.get("expiry", 2)),
        "location": data.get("location", "").strip(),
        "notes": data.get("notes", "").strip(),
        "restaurant": user["display"],
        "donation_type": donation_type,
        "assigned_ngo": assigned_ngo,
        "status": "available",
        "is_volunteering": True,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }
    donations.append(new_donation)
    
    # Notify specific NGO if assigned
    if donation_type == "assigned" and assigned_ngo:
        ngo_username = None
        for u, udata in USERS.items():
            if udata["display"] == assigned_ngo:
                ngo_username = u
                break
        if ngo_username:
            push_notif(ngo_username, f"🎁 {user['display']} assigned a direct donation to you: {new_donation['food_type']}", "food")
    elif donation_type == "public":
        for uname, udata in USERS.items():
            if udata["role"] == "NGO":
                push_notif(uname, f"💚 {user['display']} published a public donation: {new_donation['food_type']}!", "success")
            
    return jsonify({"success": True, "donation": new_donation})

@app.route("/api/restaurant/donations", methods=["GET"])
@login_required
@restaurant_required
def get_restaurant_donations():
    """Returns the logged-in restaurant's explicit donations."""
    user = current_user()
    my_donations = [d for d in donations if d["restaurant"] == user["display"]]
    return jsonify({"success": True, "donations": list(reversed(my_donations))})

# ---------------------------------------------------------------------------
# API Routes — Food Listings (Restaurant)
# ---------------------------------------------------------------------------

@app.route("/api/restaurant/food-listing", methods=["POST"])
@login_required
@restaurant_required
def add_food_listing():
    """Restaurant uploads a standard surplus food listing."""
    user = current_user()
    if user["role"] != "Restaurant":
        return jsonify({"success": False, "error": "Only restaurants can upload food listings."}), 403
    
    data = request.get_json(force=True)
    qty = float(data.get("quantity", 0))
    if qty <= 0:
        return jsonify({"success": False, "error": "Quantity must be > 0."}), 400

    listing = {
        "id": len(_food_listings) + 1,
        "restaurant": user["display"],
        "restaurant_username": user["username"],
        "food_type": data.get("food_type", "Cooked Rice"),
        "quantity": qty,
        "expiry_hours": int(data.get("expiry_hours", 2)),
        "location": data.get("location", "").strip(),
        "timestamp": datetime.now().strftime("%H:%M"),
        "date": datetime.now().strftime("%d %b"),
        "status": "Available",
    }
    _food_listings.append(listing)

    # Notify all NGOs about new surplus listing
    for uname, udata in USERS.items():
        if udata["role"] == "NGO":
            push_notif(uname,
                       f"🍽️ {user['display']} posted {qty} units of standard surplus {listing['food_type']}!",
                       "food")
    return jsonify({"success": True, "listing": listing})

@app.route("/api/restaurant/generate-random-food", methods=["POST"])
@login_required
@restaurant_required
def generate_random_food():
    """Generates dummy food listings exclusively for demo testing."""
    foods = ["Cooked Rice", "Dal & Curry", "Rotis", "Biryani"]
    locations = ["Anna Nagar", "T Nagar", "Velachery", "Adyar"]
    qty = random.randint(20, 100)
    ftype = random.choice(foods)
    loc = random.choice(locations)
    
    user = current_user()
    listing = {
        "id": len(_food_listings) + 1,
        "restaurant": user["display"],
        "restaurant_user": user["username"],
        "food_type": ftype,
        "quantity": qty,
        "expiry_hours": random.choice([2, 4, 6]),
        "location": loc,
        "timestamp": datetime.now().strftime("%H:%M"),
        "date": datetime.now().strftime("%d %b"),
        "status": "Available",
    }
    _food_listings.append(listing)
    
    # Increase score
    username = user["username"]
    USERS[username]["score"] = USERS.get(username, {}).get("score", 0) + int(qty * 0.5)
    
    for u, d in USERS.items():
        if d["role"] == "NGO":
            push_notif(u, f"🍽️ {user['display']} uploaded {qty} units of {ftype} (DEMO)!", "food")

    return jsonify({"success": True, "listing": listing})


@app.route("/api/restaurant/food-listings", methods=["GET"])
@login_required
def get_food_listings():
    """Return all food listings (paginated front N)."""
    return jsonify({"listings": _food_listings[-50:]})  # last 50


# ---------------------------------------------------------------------------
# API Routes — NGO Requests & Messaging
# ---------------------------------------------------------------------------

@app.route("/api/ngo/request", methods=["POST"])
@login_required
@ngo_required
def ngo_request():
    global _request_id_counter
    data = request.get_json(force=True)
    req = {
        "id": _request_id_counter,
        "ngo_name": current_user()["display"],
        "ngo_username": current_user()["username"],
        "quantity": data.get("quantity", 0),
        "location": data.get("location", ""),
        "urgency": data.get("urgency", "Medium"),
        "status": "Pending",
        "timestamp": datetime.now().strftime("%H:%M"),
        "accepted_by": None,
    }
    _ngo_requests.append(req)
    _messages[str(_request_id_counter)] = []
    _request_id_counter += 1
    # Notify all restaurants
    for uname, udata in USERS.items():
        if udata["role"] == "Restaurant":
            push_notif(uname,
                       f"📩 {req['ngo_name']} requested {req['quantity']} units [{req['urgency']} urgency]",
                       "request")
    return jsonify({"success": True, "request": req})


@app.route("/api/ngo/requests", methods=["GET"])
@login_required
def get_ngo_requests():
    user = current_user()
    if user["role"] == "Restaurant":
        filtered = _ngo_requests
    elif user["role"] == "NGO":
        filtered = [r for r in _ngo_requests if r["ngo_username"] == user["username"]]
    else:
        filtered = _ngo_requests  # Volunteers see all
    return jsonify({"requests": filtered})


@app.route("/api/restaurant/accept", methods=["POST"])
@login_required
@restaurant_required
def accept_request():
    data = request.get_json(force=True)
    req_id = int(data.get("request_id", -1))
    for req in _ngo_requests:
        if req["id"] == req_id:
            req["status"] = "Accepted"
            req["accepted_by"] = current_user()["display"]
            # Notify the NGO
            push_notif(req["ngo_username"],
                       f"✅ {current_user()['display']} accepted your food request for {req['quantity']} units!",
                       "accepted")
            return jsonify({"success": True, "request": req})
    return jsonify({"success": False, "error": "Request not found."}), 404


@app.route("/api/restaurant/reject", methods=["POST"])
@login_required
@restaurant_required
def reject_request():
    data = request.get_json(force=True)
    req_id = int(data.get("request_id", -1))
    for req in _ngo_requests:
        if req["id"] == req_id:
            req["status"] = "Rejected"
            push_notif(req["ngo_username"],
                       f"❌ Your request was declined. Try requesting from another restaurant.",
                       "rejected")
            return jsonify({"success": True, "request": req})
    return jsonify({"success": False, "error": "Request not found."}), 404


@app.route("/api/messages", methods=["GET"])
@login_required
def get_messages():
    req_id = request.args.get("request_id", "")
    msgs = _messages.get(req_id, [])
    return jsonify({"messages": msgs})


@app.route("/api/messages", methods=["POST"])
@login_required
def post_message():
    data = request.get_json(force=True)
    req_id = str(data.get("request_id", ""))
    text = data.get("text", "").strip()
    if not text or req_id not in _messages:
        return jsonify({"success": False, "error": "Invalid message or request."}), 400
    msg = {
        "sender": current_user()["display"],
        "role": current_user()["role"],
        "text": text,
        "time": datetime.now().strftime("%H:%M"),
    }
    _messages[req_id].append(msg)
    return jsonify({"success": True, "message": msg})


# ---------------------------------------------------------------------------
# API Routes — Homeless Survey
# ---------------------------------------------------------------------------

@app.route("/api/homeless/survey", methods=["POST"])
@login_required
def homeless_survey():
    """Manually add a homeless cluster from survey data."""
    global _simulation_data
    data = request.get_json(force=True)
    location_desc = data.get("location_desc", "").strip()
    cluster_size  = int(data.get("cluster_size", 20))
    density       = data.get("density", "Medium")
    if not location_desc:
        return jsonify({"success": False, "error": "Location description required."}), 400
def add_survey_cluster():
    """Manual input of homeless cluster."""
    data = request.get_json(force=True)
    cluster = {
        "id": len(_simulation_data.get("homeless", [])) + 1,
        "name": f"Survey Node {len(_simulation_data.get('homeless', [])) + 1}",
        "x": round(random.uniform(5, 95), 1),
        "y": round(random.uniform(5, 95), 1),
        "cluster_size": data.get("cluster_size", 20),
        "demand": data.get("cluster_size", 20),
        "density": data.get("density", "Medium"),
        "location_desc": data.get("location_desc", "Unknown")
    }
    if "homeless" not in _simulation_data:
        _simulation_data["homeless"] = []
    _simulation_data["homeless"].append(cluster)
    return jsonify({"success": True, "cluster": cluster})

@app.route("/api/homeless/survey-data", methods=["GET"])
@login_required
def get_survey_data():
    """Read static/chennai_survey.csv to display on frontend."""
    filepath = "static/chennai_survey.csv"
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "Survey data not found"})
    
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return jsonify({"success": True, "data": records})

@app.route("/api/homeless/download-csv", methods=["GET"])
@login_required
def download_survey_csv():
    """Trigger download for the CSV file."""
    filepath = "static/chennai_survey.csv"
    if not os.path.exists(filepath):
        return "File not found", 404
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    from flask import Response
    return Response(
        content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=chennai_survey.csv"}
    )


# ---------------------------------------------------------------------------
# API Routes — Volunteer
# ---------------------------------------------------------------------------

@app.route("/api/volunteer/set-depot", methods=["POST"])
@login_required
@volunteer_required
def volunteer_set_depot():
    data = request.get_json(force=True)
    depot = data.get("depot")
    if depot not in DEPOTS:
        return jsonify({"success": False, "error": "Invalid depot."}), 400
    username = current_user()["username"]
    USERS[username]["depot"] = depot
    
    push_notif(username, f"Assigned to {depot}. Ready for dispatch.", "green")
    
    teammates = [d["display"] for u, d in USERS.items() if d["role"] == "Volunteer" and d.get("depot") == depot and u != username]
    return jsonify({"success": True, "depot": depot, "teammates": teammates})


@app.route("/api/volunteer/start", methods=["POST"])
@login_required
@volunteer_required
def volunteer_start():
    if not _results:
        return jsonify({"success": False, "error": "No optimization results yet."}), 400
    user = current_user()
    _delivery_status[user["username"]] = {
        "status": "en_route",
        "started_at": datetime.now().strftime("%H:%M"),
        "completed_at": None,
        "volunteer": user["display"],
        "route_stops": len(_results.get("dqn", {}).get("route", [])),
    }
    return jsonify({
        "success": True,
        "message": f"Delivery started by {user['display']}!",
        "route_stops": _delivery_status[user["username"]]["route_stops"],
    })


@app.route("/api/volunteer/complete", methods=["POST"])
@login_required
@volunteer_required
def volunteer_complete():
    """Mark delivery as completed."""
    user = current_user()
    uname = user["username"]
    if uname not in _delivery_status or _delivery_status[uname]["status"] != "en_route":
        return jsonify({"success": False, "error": "No active delivery to complete."}), 400
    _delivery_status[uname]["status"] = "delivered"
    _delivery_status[uname]["completed_at"] = datetime.now().strftime("%H:%M")
    # Notify all NGOs
    for un, ud in USERS.items():
        if ud["role"] == "NGO":
            push_notif(un, f"🚀 {user['display']} has completed the food delivery!", "delivery")
    return jsonify({"success": True, "message": f"Delivery marked complete by {user['display']}!"})


@app.route("/api/volunteer/status", methods=["GET"])
@login_required
def volunteer_delivery_status():
    """Get delivery status for the current volunteer."""
    user = current_user()
    status_key = user["username"]
    info = _delivery_status.get(status_key, {"status": "idle"})
    return jsonify({"delivery": info})


# ---------------------------------------------------------------------------
# API Routes — Notifications
# ---------------------------------------------------------------------------

@app.route("/api/notifications", methods=["GET"])
@login_required
def get_notifications():
    user = current_user()
    notifs = _notifications.get(user["username"], [])
    unread = [n for n in notifs if not n["read"]]
    return jsonify({"notifications": notifs[-20:], "unread_count": len(unread)})


@app.route("/api/notifications/read", methods=["POST"])
@login_required
def mark_notifications_read():
    user = current_user()
    if user["username"] in _notifications:
        for n in _notifications[user["username"]]:
            n["read"] = True
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5001)
