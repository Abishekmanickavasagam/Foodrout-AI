/**
 * script.js — FoodRoute AI Dashboard
 * Enhanced: Themes, Homeless Clusters + Leaflet Map, Google Maps Nav, Strict Roles
 */
"use strict";

let appData    = null;
let appResults = null;
let activeRequestId = null;
let barChart   = null;
let _notifOpen = false;
let leafletMap = null;
let googleMapsOrigin = null;
let googleMapsDest   = null;
let googleMapsWaypoints = null;

// ============================================================
// Theme logic
// ============================================================
function initTheme() {
  const t = localStorage.getItem("fr-theme") || "dark";
  document.documentElement.setAttribute("data-theme", t);
  const btn = document.getElementById("theme-btn");
  if (btn) btn.textContent = t === "dark" ? "🌙" : "☀️";
}

function toggleTheme() {
  const cur  = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("fr-theme", next);
  const btn = document.getElementById("theme-btn");
  if (btn) btn.textContent = next === "dark" ? "🌙" : "☀️";
  
  // Re-render map tiles context if map exists
  if (leafletMap) {
    setTimeout(() => { leafletMap.invalidateSize(); }, 300);
  }
}
initTheme();

// ============================================================
// Navigation
// ============================================================
const pageTitles = {
  dashboard:      ["Dashboard",       "Real-time AI food redistribution optimizer"],
  insights:       ["Homeless Insights", "Survey data mapping & visualization"],
  "food-listing": ["Post Surplus",    "Upload surplus food for redistribution"],
  "ngo-panel":    ["Request Hub",     "Food requests & partner messaging"],
  results:        ["AI Routing",      "DQN agent routes & Google Maps Navigation"],
};

document.querySelectorAll(".nav-item").forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    navigateTo(link.dataset.page);
  });
});

function navigateTo(page) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(l => l.classList.remove("active"));
  
  const pageEl = document.getElementById("page-" + page);
  if (pageEl) pageEl.classList.add("active");
  
  const navEl  = document.getElementById("nav-" + page);
  if (navEl)  navEl.classList.add("active");
  
  const titles = pageTitles[page] || ["Dashboard", ""];
  const titleEl = document.getElementById("page-title");
  if (titleEl) {
    titleEl.textContent = titles[0];
    document.getElementById("page-sub").textContent = titles[1];
  }

  if (page === "ngo-panel") {
      loadRequests();
      loadAvailableDonations();
  }
  if (page === "food-listing") { loadFoodListings(); loadMyDonations(); }
  if (page === "insights")     loadChennaiClusters();
}

// ============================================================
// Data Gen & Optimization
// ============================================================
async function generateData() {
  setBtn("btn-generate", true, "⏳ Generating…");
  showLoader("Generating Simulation Data…", "Creating restaurant nodes & local demands.");
  try {
    const res  = await fetch("/api/generate", { method: "POST" });
    const json = await res.json();
    if (json.status !== "ok") throw new Error(json.message);

    appData = json.data;
    const surplus = appData.restaurants.reduce((s, r) => s + r.surplus, 0);
    animateValue("stat-surplus", 0, Math.round(surplus), 800, 0);
    setBtn("btn-optimize", false);
    showToast("✅ Data generated successfully", "success");
  } catch (err) {
    showToast("❌ " + err.message, "error");
  } finally {
    hideLoader();
    setBtn("btn-generate", false, "⚡ Sim Data");
  }
}

async function runOptimization() {
  if (!appData) return;
  setBtn("btn-optimize", true, "⏳ Computing…");
  showLoader("Running Deep Q-Network…", "Optimizing routes for max fairness and min spoilage.");
  try {
    const res  = await fetch("/run_optimization", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes: 300 }),
    });
    const json = await res.json();
    if (!json.success) throw new Error(json.error);

    appResults = json;
    populateResults(json);
    showToast("🚀 Optimization complete! Review routes.", "success");
    addActivity("AI Optimization complete. Route generated for volunteers.", "blue");
    navigateTo("results");
  } catch (err) {
    showToast("❌ " + err.message, "error");
  } finally {
    hideLoader();
    setBtn("btn-optimize", false, "🤖 Run AI");
  }
}

// ============================================================
// Chennai Leaflet Map & Survey Data
// ============================================================
async function loadChennaiClusters() {
  try {
    const res = await fetch("/api/homeless/survey-data");
    const json = await res.json();
    if (!json.success) throw new Error(json.error);
    
    renderSurveyTable(json.data);
    renderLeafletMap(json.data);
  } catch(err) {
    console.error(err);
    showToast("Could not load survey data.", "error");
  }
}

function renderSurveyTable(data) {
  const tbody = document.getElementById("survey-tbody");
  if(!tbody) return;
  
  const prioritySelect = document.getElementById("start-cluster-priority");
  if(prioritySelect) {
      prioritySelect.innerHTML = '<option value="none">AI Determined</option>';
  }

  if(!data || data.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-row">No survey data found.</td></tr>`;
    return;
  }
  tbody.innerHTML = data.map(row => {
    let densCls = "db-Low";
    let prob = Math.floor(Math.random() * 20) + 15; // 15-34%
    if (row.density === "High") { densCls = "db-High"; prob = Math.floor(Math.random() * 15) + 85; } // 85-99%
    if (row.density === "Medium") { densCls = "db-Medium"; prob = Math.floor(Math.random() * 30) + 50; } // 50-79%
    
    if(prioritySelect) {
        prioritySelect.innerHTML += `<option value="${row.id}">${row.location} (${prob}% Prob)</option>`;
    }

    // Only render probability for volunteer/restaurant via the HTML headers
    return `<tr>
      <td>${row.id}</td>
      <td><strong>${row.location}</strong></td>
      <td>${row.area}</td>
      <td>${row.estimated_people}</td>
      <td>${row.food_needed}</td>
      <td><span class="density-badge ${densCls}">${row.density}</span></td>
      <td><strong>${prob}%</strong></td>
    </tr>`;
  }).join("");
}

function renderLeafletMap(data) {
  if (!document.getElementById("cluster-leaflet-map")) return;
  
  if (!leafletMap) {
    leafletMap = L.map("cluster-leaflet-map").setView([13.0827, 80.2707], 12); // Chennai center
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap',
      subdomains: 'abcd',
      maxZoom: 19
    }).addTo(leafletMap);
  } else {
    // clear markers
    leafletMap.eachLayer((layer) => {
      if(layer instanceof L.Marker || layer instanceof L.CircleMarker) leafletMap.removeLayer(layer);
    });
  }

  // Plot data
  data.forEach(row => {
    const lat = parseFloat(row.lat);
    const lng = parseFloat(row.lng);
    const rad = parseInt(row.estimated_people) || 20;
    
    let color = "#4caf50";
    if (row.density === "High") color = "#ef5350";
    else if (row.density === "Medium") color = "#ff9800";
    
    // Circle marker
    L.circleMarker([lat, lng], {
      radius: Math.min(Math.max(rad / 2, 8), 24),
      fillColor: color,
      color: "#fff",
      weight: 2,
      opacity: 1,
      fillOpacity: 0.7
    }).addTo(leafletMap).bindPopup(`
      <div style="font-family:Inter; font-size:13px">
        <strong>${row.location}</strong><br/>
        People: ${row.estimated_people}<br/>
        Type: ${row.food_needed}
      </div>
    `);
  });
  setTimeout(() => { leafletMap.invalidateSize(); }, 300);
}

function downloadSurveyCSV() {
  window.open("/api/homeless/download-csv", "_blank");
}

// ============================================================
// Google Maps Navigation
// ============================================================
function buildGoogleMapsLink(route) {
  const btn = document.getElementById("gm-btn");
  if (!btn) return;
  
  if (!route || route.length < 2) {
    btn.disabled = true;
    return;
  }
  btn.disabled = false;
  
  // Real coords for Chennai (approx mappings)
  // We use dummy realistic co-ords mapped by scale for the demo
  const mapToChennai = (x, y) => {
    // scale roughly over chennai (lat 12.9 to 13.1, lng 80.1 to 80.3)
    const lat = 12.9 + (y / 100) * 0.2;
    const lng = 80.1 + (x / 100) * 0.2;
    return `${lat.toFixed(5)},${lng.toFixed(5)}`;
  };
  
  // start point
  googleMapsOrigin = mapToChennai(route[0][1], route[0][2]);
  
  // end point (last node)
  const last = route[route.length - 1];
  googleMapsDest   = mapToChennai(last[1], last[2]);
  
  let wp = [];
  for(let i=1; i<route.length-1; i++) {
     wp.push(mapToChennai(route[i][1], route[i][2]));
  }
  googleMapsWaypoints = wp.join("|");
}

function openGoogleMapsNav() {
  if (!googleMapsOrigin || !googleMapsDest) {
    showToast("Route not valid.", "error"); return;
  }
  let url = `https://www.google.com/maps/dir/?api=1&origin=${googleMapsOrigin}&destination=${googleMapsDest}`;
  if (googleMapsWaypoints && googleMapsWaypoints.length > 0) {
      url += `&waypoints=${googleMapsWaypoints}`;
  }
  window.open(url, "_blank");
}

// ============================================================
// Results population
// ============================================================
function populateResults(json) {
  const dqn = json.dqn || {};
  const dist = json.distribution || {};
  
  document.getElementById("result-summary-row").style.display = "grid";
  document.getElementById("charts-row").style.display = "grid";
  document.getElementById("fairness-card").style.display = "block";

  animateValue("stat-delivered", 0, dqn.total_delivered || 0, 900, 1);
  animateValue("stat-spoilage",  0, Math.random() * 50 + 20, 900, 1); // Mock spoilage diff

  if (document.getElementById("reward-graph")) {
    document.getElementById("reward-graph").src = "data:image/png;base64," + json.reward_graph;
  }

  // Render Route Table
  const tbody = document.getElementById("route-tbody");
  if (tbody && dqn.route) {
    tbody.innerHTML = dqn.route.map((step, idx) => {
      const type = step[0];
      const tag  = type === 'depot' ? '<span class="tag tag-depot">DEPOT</span>' : type === 'restaurant' ? '<span class="tag tag-restaurant">REST</span>' : '<span class="tag tag-ngo">DEST</span>';
      return `<tr>
        <td>${idx}</td>
        <td>${tag}</td>
        <td>ID ${step[3] || 0}</td>
        <td>${step[1].toFixed(1)}</td>
        <td>${step[2].toFixed(1)}</td>
        <td>${step[4] ? step[4].toFixed(1) : "—"}</td>
      </tr>`;
    }).join("");
    buildGoogleMapsLink(dqn.route);
  }

  // Bar Chart
  const barCtx = document.getElementById("bar-chart")?.getContext("2d");
  if (barCtx) {
    if (barChart) barChart.destroy();
    barChart = new Chart(barCtx, {
      type: "bar",
      data: {
        labels: ["NGO Allocations", "Homeless Allocations"],
        datasets: [{ label: "Units", data: [dist.ngo, dist.homeless], backgroundColor: ["#ab47bc", "#4caf50"], borderRadius: 6 }]
      },
      options: { responsive: true, plugins: { legend: { display: false } } }
    });
  }

  // Fairness Meter
  const ngoPct = Math.round(((dist.ngo || 1) / ((dist.ngo || 1) + (dist.homeless || 1))) * 100);
  const homePct = 100 - ngoPct;
  document.getElementById("fairness-bar-ngo").style.width = ngoPct + "%";
  document.getElementById("fairness-bar-homeless").style.width = homePct + "%";
  document.getElementById("fairness-ngo-val").textContent = dist.ngo.toFixed(1);
  document.getElementById("fairness-homeless-val").textContent = dist.homeless.toFixed(1);
  
  if (window.USER_ROLE === "Volunteer") {
    document.getElementById("btn-start-delivery").disabled = false;
  }
}

// ============================================================
// Volunteer Delivery
// ============================================================
async function startDelivery() {
  document.getElementById("btn-start-delivery").style.display = "none";
  document.getElementById("btn-complete-delivery").style.display = "inline-flex";
  
  const tracker = document.getElementById("delivery-tracker");
  tracker.style.display = "flex";
  document.getElementById("dts-assigned").classList.add("completed");
  document.getElementById("dts-enroute").classList.add("active");
  addActivity("Volunteer Arjun (NCC) started delivery dispatch.", "green");
}

async function completeDelivery() {
  document.getElementById("btn-complete-delivery").style.display = "none";
  document.getElementById("dts-enroute").classList.add("completed");
  document.getElementById("dts-enroute").classList.remove("active");
  document.getElementById("dts-delivered").classList.add("completed");
  document.getElementById("delivery-status-msg").style.display = "block";
  document.getElementById("delivery-status-msg").textContent = "🎉 Delivery confirmed! Zero waste achieved.";
  showToast("Delivery complete!", "success");
  addActivity("Volunteer Arjun (NCC) successfully completed delivery.", "green");
}

// ============================================================
// NGO Requests API Call
// ============================================================
async function sendNGORequest() {
  const q = document.getElementById("req-qty").value;
  const l = document.getElementById("req-location").value;
  if(!q || !l) return showToast("Enter required fields", "error");
  try {
    const res = await fetch("/api/ngo/request", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({quantity: q, location: l, urgency: "High"})
    });
    const json = await res.json();
    if(json.success) { showToast("Request broadcasted!", "success"); loadRequests(); addActivity(`Food request for ${q} units broadcasted.`, "purple"); }
  } catch(e) { }
}

async function loadRequests() {
  try {
    const res = await fetch("/api/ngo/requests");
    const json = await res.json();
    const list = document.getElementById("requests-list");
    if(!list) return;
    if(json.requests && json.requests.length) {
      list.innerHTML = json.requests.map(r => `
        <div class="request-card">
          <div class="req-info">
            <div class="req-ngo">${r.ngo_name}</div>
            <div class="req-qty">${r.quantity} units</div>
            <div class="req-meta">Status: <span style="font-weight:600;color:var(--text)">${r.status}</span></div>
          </div>
          ${window.USER_ROLE === 'Restaurant' && r.status === 'Pending' ? `<button class="btn btn-accept" style="font-size:0.75rem;padding:4px 8px;" onclick="acceptReq(${r.id})">Accept</button>` : `<button class="btn btn-secondary" style="font-size:0.7rem;padding:3px 8px;">DM</button>`}
        </div>
      `).join("");
      const b = document.getElementById("nav-ngo-badge");
      if(b) { b.textContent = json.requests.length; b.style.display="inline-block"; }
    } else {
      list.innerHTML = `<div class="empty-row">No active network requests.</div>`;
    }
  } catch(e) {}
}

async function acceptReq(id) {
  try {
    const res = await fetch("/api/restaurant/accept", {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({request_id:id})
    });
    showToast("Accepted!", "success");
    loadRequests();
    addActivity("Restaurant accepted food request.", "orange");
  } catch(e) {}
}

// ============================================================
// Food Listings API Call
// ============================================================
async function uploadFoodListing() {
  const t = document.getElementById("food-type").value;
  const q = document.getElementById("food-qty").value;
  const l = document.getElementById("food-location").value;
  if(!q||!l) return showToast("Enter qty and location","error");
  try {
    await fetch("/api/restaurant/food-listing", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({food_type:t, quantity:q, expiry_hours:2, location:l})
    });
    showToast("Listing published!", "success");
    addActivity(`Uploaded ${q} units of ${t} surplus.`, "orange");
    loadFoodListings();
  } catch(e) {}
}

async function loadFoodListings() {
  try {
    const res = await fetch("/api/restaurant/food-listings");
    const json = await res.json();
    const tbody = document.getElementById("listings-tbody");
    if(!tbody) return;
    if(json.listings && json.listings.length) {
      tbody.innerHTML = json.listings.map(l => `
        <tr><td>${l.food_type}</td><td>${l.quantity}</td><td>${l.expiry_hours}h</td><td><span class="pill pill-low">${l.status}</span></td></tr>
      `).join("");
    } else { tbody.innerHTML = `<tr><td colspan="4" class="empty-row">No listings yet.</td></tr>`; }
  } catch(e) {}
}

// ============================================================
// Notification + Activity UI
// ============================================================
function toggleNotifDropdown() {
  const d = document.getElementById("notif-dropdown");
  if(d) {
    _notifOpen = !_notifOpen;
    d.classList.toggle("open", _notifOpen);
    document.getElementById("notif-dot")?.classList.remove("show");
  }
}
async function checkNotifications() {
  try {
    const res = await fetch("/api/notifications");
    const json = await res.json();
    if(json.unread_count > 0 && !_notifOpen) {
      document.getElementById("notif-dot")?.classList.add("show");
    }
    const l = document.getElementById("notif-list");
    if(l && json.notifications && json.notifications.length) {
      l.innerHTML = json.notifications.reverse().map(n => `<div class="notif-item">${n.msg}<div class="notif-time">${n.time}</div></div>`).join("");
    }
  } catch(e) {}
}
setInterval(checkNotifications, 10000);
setTimeout(checkNotifications, 1000);

function addActivity(text, color) {
  const f = document.getElementById("activity-feed");
  if(!f) return;
  const t = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  const html = `<div class="activity-item"><div class="activity-dot ad-${color}"></div><div><div class="activity-text">${text}</div><div class="activity-time">${t}</div></div></div>`;
  f.insertAdjacentHTML("afterbegin", html);
}

// Utils
function animateValue(id, from, to, duration, decimals=0) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = performance.now();
  function update(ts) {
    const prog = Math.min((ts - start) / duration, 1);
    el.textContent = (from + (to - from) * (1 - Math.pow(1 - prog, 3))).toFixed(decimals);
    if (prog < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}
function setBtn(id, dis, text) {
  const btn = document.getElementById(id);
  if(btn) { btn.disabled=dis; if(text) btn.innerHTML=text; }
}
function showLoader(title, sub) {
  document.getElementById("loader-title").textContent=title;
  document.getElementById("loader-sub").textContent=sub;
  document.getElementById("loader").style.display="flex";
}
function hideLoader() { document.getElementById("loader").style.display="none"; }
let toastTimer;
function showToast(msg, type="info") {
  const el = document.getElementById("toast");
  el.textContent = msg; el.className = "toast show " + type;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => {el.className="toast";}, 4000);
}

// ============================================================
// Advanced Features (Round 3)
// ============================================================

async function generateDemoFood() {
  setBtn("btn-generate", true, "⏳ Generating...");
  try {
    const res = await fetch("/api/restaurant/generate-random-food", { method:"POST" });
    const json = await res.json();
    if(json.success) { showToast("Demo data generated!", "success"); loadFoodListings(); }
  } catch(e) {}
  setBtn("btn-generate", false, "🎲 Demo Data");
}

let currentLb = 'Restaurant';
async function switchLeaderboard(role) {
  currentLb = role;
  document.getElementById("btn-lb-res")?.classList.toggle("active", role === 'Restaurant');
  document.getElementById("btn-lb-vol")?.classList.toggle("active", role === 'Volunteer');
  loadLeaderboard();
}

async function loadLeaderboard() {
  try {
    const res = await fetch("/api/leaderboard");
    const json = await res.json();
    const lbd = document.getElementById("leaderboard-list");
    if(!lbd) return;
    
    if (currentLb === 'Restaurant') {
      lbd.innerHTML = json.restaurants.map((r, i) => `
        <div class="activity-item" style="align-items:center;">
          <div style="font-size:1.5rem; width:30px; text-align:center">${i===0?'🥇':i===1?'🥈':i===2?'🥉':'🏅'}</div>
          <div style="flex:1;">
             <div class="activity-text" style="font-weight:600">${r.display}</div>
             <div class="activity-time">${r.donation_mode ? '<span style="color:var(--green)">💚 Active Donor</span>' : 'Standard'}</div>
          </div>
          <div style="font-weight:700; color:var(--text)">${r.score} pt</div>
        </div>
      `).join("");
    } else {
      lbd.innerHTML = json.volunteers.map((v, i) => `
        <div class="activity-item" style="align-items:center;">
           <div style="font-size:1.5rem; width:30px; text-align:center">${i===0?'🥇':i===1?'🥈':i===2?'🥉':'🏅'}</div>
           <div style="flex:1;">
             <div class="activity-text" style="font-weight:600">${v.display}</div>
             <div class="activity-time">${v.depot || 'No Depot'}</div>
           </div>
           <div style="font-weight:700; color:var(--text)">${v.score} pt</div>
        </div>
      `).join("");
    }
  } catch(e) {}
}

let donationModeOn = false;
async function toggleDonationMode() {
  donationModeOn = !donationModeOn;
  try {
    await fetch("/api/restaurant/donation-mode", {
       method: "POST", headers:{"Content-Type":"application/json"},
       body: JSON.stringify({enabled: donationModeOn})
    });
    document.getElementById("donation-icon").textContent = donationModeOn ? '💚' : '⚪';
    document.getElementById("donation-text").innerHTML = donationModeOn ? '<span style="color:var(--green)">Active</span>' : 'Disabled';
    showToast(donationModeOn ? "Donation Mode Enabled" : "Donation Mode Disabled", "success");
    addActivity(`Restaurant set Donation Mode to ${donationModeOn ? "ON" : "OFF"}.`, "green");
  } catch(e) {}
}

async function setVolunteerDepot() {
  const depot = document.getElementById("depot-select").value;
  if(!depot) return showToast("Please select a depot", "error");
  
  try {
    const res = await fetch("/api/volunteer/set-depot", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({depot: depot})
    });
    const json = await res.json();
    if(json.success) {
      document.getElementById("depot-selection-overlay").style.display = "none";
      document.getElementById("current-depot").textContent = json.depot;
      document.getElementById("teammates-count").textContent = json.teammates.length;
      showToast("Checked in at " + json.depot, "success");
    } else {
      showToast(json.error, "error");
    }
  } catch(e) {}
}

// Init leaderboard on load
document.addEventListener("DOMContentLoaded", () => {
    loadLeaderboard();
    // Pre-populate if missing depot
    const depotOv = document.getElementById("depot-selection-overlay");
    if (depotOv) depotOv.style.display = "flex";

    // NGO List for Restaurant Modal
    loadNgoList();

    // Auto-refresh Active Donations on NGO Portal
    setInterval(() => {
        const ngoPanel = document.getElementById("page-ngo-panel");
        if (ngoPanel && ngoPanel.classList.contains("active")) {
            loadAvailableDonations();
        }
    }, 5000);
});

// ============================================================
// Round 4 Features (Advanced Endpoints)
// ============================================================

async function loadAvailableDonations() {
  try {
    const res = await fetch("/api/ngo/available-donations");
    const json = await res.json();
    const tbody = document.getElementById("available-donations-tbody");
    if(!tbody) return;
    
    if(json.listings && json.listings.length) {
      tbody.innerHTML = json.listings.map(l => {
        const isAssigned = l.donation_type === 'assigned';
        const isReserved = l.status === 'reserved';
        const typeBadge = isAssigned ? '<span class="badge" style="background:var(--purple); color:white; font-size:0.6rem; padding:0.1rem 0.3rem; border-radius:4px; margin-right:0.4rem;">DIRECT</span>' : '<span class="badge" style="background:var(--green-light); color:#121212; font-size:0.6rem; padding:0.1rem 0.3rem; border-radius:4px; margin-right:0.4rem;">PUBLIC</span>';
        
        let actionHtml = '';
        if (isAssigned || isReserved) {
            // If assigned or already reserved, show DM directly
            actionHtml = `<button class="btn btn-secondary" style="padding:0.2rem 0.5rem;font-size:0.75rem; border-color:var(--green-light); color:var(--green-light);" onclick="showDmModal('${l.restaurant}')">💬 DM</button>`;
        } else {
            // Public and Available -> Show Reserve
            actionHtml = `<button class="btn btn-primary" style="padding:0.2rem 0.5rem;font-size:0.75rem;" onclick="reserveDonation(${l.id})">Reserve</button>`;
        }

        return `
        <tr>
          <td>${typeBadge} ${l.food_type}</td>
          <td>${l.quantity}</td>
          <td>${l.expiry_hours}h</td>
          <td>${l.location}</td>
          <td><strong>${l.restaurant}</strong></td>
          <td>
            <div style="display:flex; gap:0.25rem;">
                ${actionHtml}
            </div>
          </td>
        </tr>
      `;}).join("");
    } else {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No relevant donations available right now.</td></tr>';
    }
  } catch(e) {}
}

async function reserveDonation(id) {
    try {
        const res = await fetch("/api/ngo/reserve-donation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: id })
        });
        const json = await res.json();
        if (json.success) {
            showToast("Donation Reserved! You can now message the restaurant.", "success");
            loadAvailableDonations();
        } else {
            showToast(json.error || "Reservation failed", "error");
        }
    } catch(e) { showToast("Network error", "error"); }
}

function requestDonation(id) {
  // Logic replaced by reserveDonation for Public/FCFS flow
}

async function runVolunteerOptimization() {
  if (!appData) {
      showToast("Generating simulation environment first...", "info");
      await generateData();
  }
  
  setBtn("btn-vol-opt", true, "⏳ Computing...");
  showLoader("Running AI Route Optimization...", "Deep Q-Network finding the best distribution route.");
  
  try {
    const startCluster = document.getElementById("start-cluster-priority")?.value || "none";
    const res = await fetch("/api/optimize-route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes: 300, start_cluster: startCluster })
    });
    
    const json = await res.json();
    if (!json.success) throw new Error(json.error);
    
    appResults = json;
    populateResults(json);
    showToast("Route Optimized Successfully!", "success");
    addActivity("Volunteer finished AI routing simulation.", "blue");
    
    navigateTo("results");
  } catch (err) {
    showToast("❌ " + err.message, "error");
  } finally {
    hideLoader();
    setBtn("btn-vol-opt", false, "🚀 Run AI Optimization");
  }
}

// ============================================================
// Round 6 Features (Voluntary Donations & DM)
// ============================================================

function showDonationModal() { 
    document.getElementById('donation-modal').style.display = 'flex'; 
    // Reset radio and hide NGO select
    const radios = document.getElementsByName('df-type-radio');
    if(radios[0]) radios[0].checked = true;
    toggleNgoSelect(false);
}
function hideDonationModal() { document.getElementById('donation-modal').style.display = 'none'; }

function toggleNgoSelect(show) {
    document.getElementById('df-ngo-select-wrap').style.display = show ? 'block' : 'none';
}

async function loadNgoList() {
    try {
        const res = await fetch("/api/ngos");
        const json = await res.json();
        const select = document.getElementById("df-assigned-ngo");
        if(!select) return;
        select.innerHTML = '<option value="">-- Choose NGO --</option>';
        json.ngos.forEach(ngo => {
            const opt = document.createElement("option");
            opt.value = ngo.display;
            opt.textContent = ngo.display;
            select.appendChild(opt);
        });
    } catch(e) {}
}

async function submitVoluntaryDonation() {
    const typeRadios = document.getElementsByName('df-type-radio');
    let donationType = 'public';
    for(const r of typeRadios) if(r.checked) donationType = r.value;

    const data = {
        food_type: document.getElementById('df-type').value,
        qty: document.getElementById('df-qty').value,
        expiry: document.getElementById('df-expiry').value,
        location: document.getElementById('df-location').value,
        notes: document.getElementById('df-notes').value,
        donation_type: donationType,
        assigned_ngo: document.getElementById('df-assigned-ngo').value
    };
    
    if (!data.qty || !data.food_type) return showToast("Please fill all required fields", "warning");
    if (donationType === 'assigned' && !data.assigned_ngo) return showToast("Please select a target NGO", "warning");

    try {
        const res = await fetch("/api/restaurant/donate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        const json = await res.json();
        if (json.success) {
            showToast("Voluntary Donation Submitted!", "success");
            hideDonationModal();
            loadMyDonations();
        } else {
            showToast("Error: " + json.error, "error");
        }
    } catch(e) { showToast("Submission failed", "error"); }
}

async function loadMyDonations() {
    try {
        const res = await fetch("/api/restaurant/donations");
        const json = await res.json();
        const tbody = document.getElementById("my-donations-tbody");
        if (!tbody) return;

        if (json.donations && json.donations.length) {
            tbody.innerHTML = json.donations.map(d => {
                const typeLabel = d.donation_type === 'assigned' ? 'Assigned' : 'Public';
                const ngoLabel = d.assigned_ngo || '-';
                let statusColor = 'var(--green)';
                if (d.status === 'reserved') statusColor = 'var(--yellow)';
                if (d.status === 'completed') statusColor = 'var(--blue)';

                return `
                <tr>
                    <td>${d.food_type}</td>
                    <td>${d.quantity}</td>
                    <td>${d.expiry_hours}h</td>
                    <td>${d.location}</td>
                    <td>${typeLabel}</td>
                    <td>${ngoLabel}</td>
                    <td>
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="color:${statusColor}; font-weight:600;">${d.status.toUpperCase()}</span>
                            ${d.assigned_ngo ? `<button class="btn btn-secondary" style="padding:0.15rem 0.4rem; font-size:0.65rem;" onclick="showDmModal('${d.assigned_ngo}')">💬 Chat</button>` : ''}
                        </div>
                    </td>
                </tr>
            `;}).join("");
        } else {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-row">No active voluntary donations.</td></tr>';
        }
    } catch(e) {}
}

let currentChatPeer = null;
let chatRefreshInterval = null;

function showDmModal(peerName) {
    currentChatPeer = peerName;
    document.getElementById('dm-modal-title').innerHTML = `Chat with <span style="color:var(--green-light)">${peerName}</span>`;
    document.getElementById('dm-text').value = "";
    document.getElementById('dm-modal').style.display = 'flex';
    
    // Initial load
    loadChatHistory(peerName);
    
    // Auto refresh every 4 seconds
    if(chatRefreshInterval) clearInterval(chatRefreshInterval);
    chatRefreshInterval = setInterval(() => loadChatHistory(currentChatPeer), 4000);
}

function hideDmModal() {
    document.getElementById('dm-modal').style.display = 'none';
    currentChatPeer = null;
    if(chatRefreshInterval) {
        clearInterval(chatRefreshInterval);
        chatRefreshInterval = null;
    }
}

async function loadChatHistory(peerName) {
    if(!peerName) return;
    try {
        const res = await fetch(`/api/messages/${encodeURIComponent(peerName)}`);
        const json = await res.json();
        const container = document.getElementById("dm-chat-history");
        if(!container) return;

        if(json.messages && json.messages.length) {
            container.innerHTML = json.messages.map(m => {
                const isSent = m.sender_display !== peerName;
                const bubbleClass = isSent ? 'msg-sent' : 'msg-received';
                return `
                    <div class="msg-item ${bubbleClass}">
                        ${m.message}
                        <span class="msg-time">${m.timestamp}</span>
                    </div>
                `;
            }).join("");
            // Scroll to bottom
            container.scrollTop = container.scrollHeight;
        } else {
            container.innerHTML = '<div style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem; margin-top:2rem;">No messages yet. Say hi!</div>';
        }
    } catch(e) {}
}

async function sendDmModalMessage() {
    const msg = document.getElementById('dm-text').value.trim();
    if (!msg || !currentChatPeer) return;

    try {
        const res = await fetch("/api/message/send", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ receiver: currentChatPeer, message: msg })
        });
        const json = await res.json();
        if (json.success) {
            document.getElementById('dm-text').value = "";
            loadChatHistory(currentChatPeer);
        } else {
            showToast("Failed to send message", "error");
        }
    } catch(e) { showToast("Error sending message", "error"); }
}
