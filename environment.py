"""
environment.py
Gym-style RL environment for perishable food redistribution.
Supports 20 restaurants + 10 NGOs with time-decay spoilage.
"""

import math
import numpy as np  # type: ignore[import]


class FoodRedistributionEnv:
    """
    Custom RL Environment for food redistribution in urban environments.

    State space:
        - Food quantities at each restaurant (20,)
        - Demand remaining at each NGO (10,)
        - Vehicle's current position (2,)
        - Time elapsed (1,)
        Total: 33 features

    Action space:
        0-19  -> visit restaurant i
        20-29 -> deliver to NGO (i-20)
        Total: 30 discrete actions

    Reward:
        + food_delivered
        - 0.1 * distance_traveled
        - 0.2 * spoilage_incurred
    """

    N_RESTAURANTS = 20
    N_NGOS = 10
    MAX_STEPS = 200
    SPOILAGE_RATE = 0.05          # lambda for exponential decay
    VEHICLE_CAPACITY = 150        # max units truck can carry at once

    def __init__(self):
        self.n_actions = self.N_RESTAURANTS + self.N_NGOS
        self.n_states = self.N_RESTAURANTS + self.N_NGOS + 2 + 1  # 33

        # Fixed coordinate arrays (re-randomised every reset unless seeded)
        self.restaurant_coords = np.zeros((self.N_RESTAURANTS, 2), dtype=np.float32)
        self.ngo_coords = np.zeros((self.N_NGOS, 2), dtype=np.float32)

        # Baseline food/demand arrays (set via reset or seed_data)
        self._initial_food = np.zeros(self.N_RESTAURANTS, dtype=np.float32)
        self._initial_demand = np.zeros(self.N_NGOS, dtype=np.float32)

        # Runtime state — initialised properly in reset()
        self.food = np.zeros(self.N_RESTAURANTS, dtype=np.float32)
        self.demand = np.zeros(self.N_NGOS, dtype=np.float32)
        self.time_at_pickup = np.zeros(self.N_RESTAURANTS, dtype=np.float32)
        self.vehicle_pos = np.array([50.0, 50.0], dtype=np.float32)
        self.vehicle_load = 0.0
        self.current_step = 0
        self.done = False
        self.total_delivered = 0.0
        self.total_distance = 0.0
        self.total_spoilage = 0.0
        self.visited_restaurants = set()
        self.delivered_ngos = set()
        self.route: list[tuple] = []

        # Generate initial random coords
        self._generate_coordinates()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def seed_data(self, restaurant_data, ngo_data):
        """Load pre-generated restaurant/NGO data dicts into the env."""
        self.restaurant_coords = np.array(
            [[r["x"], r["y"]] for r in restaurant_data], dtype=np.float32
        )
        self.ngo_coords = np.array(
            [[n["x"], n["y"]] for n in ngo_data], dtype=np.float32
        )
        self._initial_food = np.array(
            [r["surplus"] for r in restaurant_data], dtype=np.float32
        )
        self._initial_demand = np.array(
            [n["demand"] for n in ngo_data], dtype=np.float32
        )

    def reset(self, use_seeded=False):
        """Reset environment, returns initial state vector."""
        if not use_seeded or np.all(self.restaurant_coords == 0):
            self._generate_coordinates()
            self._initial_food = np.random.uniform(
                10, 50, self.N_RESTAURANTS
            ).astype(np.float32)
            self._initial_demand = np.random.uniform(
                5, 40, self.N_NGOS
            ).astype(np.float32)

        # Mutable runtime state
        self.food = self._initial_food.copy()
        self.demand = self._initial_demand.copy()
        self.time_at_pickup = np.zeros(self.N_RESTAURANTS, dtype=np.float32)
        self.vehicle_pos = np.array([50.0, 50.0], dtype=np.float32)
        self.vehicle_load = 0.0
        self.current_step = 0
        self.done = False
        self.total_delivered = 0.0
        self.total_distance = 0.0
        self.total_spoilage = 0.0
        self.visited_restaurants = set()
        self.delivered_ngos = set()
        self.route = [("depot", 50.0, 50.0)]

        return self._get_state()

    def step(self, action):
        """Execute action, return (next_state, reward, done, info)."""
        assert not self.done, "Episode is done — call reset()."
        reward = 0.0

        if action < self.N_RESTAURANTS:
            # --- Pickup from restaurant ---
            r_idx = action
            target = self.restaurant_coords[r_idx]
            dist = self._euclidean(self.vehicle_pos, target)
            self.vehicle_pos = target.copy()
            self.total_distance += dist
            reward -= 0.1 * dist

            # Apply spoilage since last visit
            elapsed = self.current_step - self.time_at_pickup[r_idx]
            remaining = float(self.food[r_idx]) * math.exp(
                -self.SPOILAGE_RATE * elapsed
            )
            spoiled = float(self.food[r_idx]) - remaining
            self.total_spoilage += spoiled
            reward -= 0.2 * spoiled
            self.food[r_idx] = remaining

            # Load vehicle
            pickup_amt = min(remaining, self.VEHICLE_CAPACITY - self.vehicle_load)
            self.vehicle_load += pickup_amt
            self.food[r_idx] -= pickup_amt
            self.time_at_pickup[r_idx] = self.current_step
            self.visited_restaurants.add(r_idx)
            self.route.append(
                ("restaurant", float(target[0]), float(target[1]), int(r_idx), float(pickup_amt))
            )

        else:
            # --- Delivery to NGO ---
            n_idx = action - self.N_RESTAURANTS
            target = self.ngo_coords[n_idx]
            dist = self._euclidean(self.vehicle_pos, target)
            self.vehicle_pos = target.copy()
            self.total_distance += dist
            reward -= 0.1 * dist

            deliver_amt = min(self.vehicle_load, self.demand[n_idx])
            self.vehicle_load -= deliver_amt
            self.demand[n_idx] -= deliver_amt
            self.total_delivered += deliver_amt
            reward += deliver_amt
            self.delivered_ngos.add(n_idx)
            self.route.append(
                ("ngo", float(target[0]), float(target[1]), int(n_idx), float(deliver_amt))
            )

        self.current_step += 1

        # Terminal conditions
        all_demand_met = bool(np.all(self.demand <= 0))
        out_of_steps = self.current_step >= self.MAX_STEPS
        self.done = all_demand_met or out_of_steps

        state = self._get_state()
        info = {
            "delivered": self.total_delivered,
            "distance": self.total_distance,
            "spoilage": self.total_spoilage,
            "step": self.current_step,
        }
        return state, reward, self.done, info

    def get_summary(self):
        """Return episode summary dict."""
        return {
            "total_delivered": round(float(self.total_delivered), 2),  # type: ignore[call-overload]
            "total_distance": round(float(self.total_distance), 2),    # type: ignore[call-overload]
            "total_spoilage": round(float(self.total_spoilage), 2),    # type: ignore[call-overload]
            "route": self.route,
            "steps": self.current_step,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_coordinates(self):
        self.restaurant_coords = np.random.uniform(
            0, 100, (self.N_RESTAURANTS, 2)
        ).astype(np.float32)
        self.ngo_coords = np.random.uniform(
            0, 100, (self.N_NGOS, 2)
        ).astype(np.float32)

    def _get_state(self):
        """Normalised state vector of length 33."""
        food_norm = self.food / 50.0
        demand_norm = self.demand / 40.0
        pos_norm = self.vehicle_pos / 100.0
        time_norm = np.array([self.current_step / self.MAX_STEPS], dtype=np.float32)
        return np.concatenate(
            [food_norm, demand_norm, pos_norm, time_norm]
        ).astype(np.float32)

    @staticmethod
    def _euclidean(a, b):
        return float(np.sqrt(np.sum((a - b) ** 2)))
