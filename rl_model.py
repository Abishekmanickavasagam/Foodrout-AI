"""
rl_model.py
Deep Q-Network (DQN) agent for the food redistribution environment.
Architecture: 33-input → 256 → 256 → 30-output
Includes: ReplayBuffer, DQNNetwork, DQNAgent, train_agent()
"""

import math
import random
from collections import deque
import numpy as np  # type: ignore[import] # pylint: disable=import-error

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.optim as optim  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """Dueling DQN architecture for improved value estimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        # Dueling combination: Q = V + (A - mean(A))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN agent with epsilon-greedy exploration and target network."""

    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10     # episodes
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 300             # episodes for decay

    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.steps_done = 0

        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory = ReplayBuffer(capacity=20_000)
        self.episode_count = 0

    # ---- Exploration -------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -self.episode_count / self.EPS_DECAY
        )
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.policy_net(s).argmax(dim=1).item())

    # ---- Learning ----------------------------------------------------------

    def learn(self) -> float:
        if len(self.memory) < self.BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.BATCH_SIZE
        )

        states_t = torch.tensor(states).to(self.device)
        actions_t = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t = torch.tensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q values (Double DQN: action selected by policy, evaluated by target)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.GAMMA * next_q * (1 - dones_t)

        loss = F.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def end_episode(self):
        self.episode_count += 1
        if self.episode_count % self.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def push(self, *args):
        self.memory.push(*args)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_agent(env, n_episodes: int = 300, progress_cb=None):
    """
    Train DQN agent for n_episodes.

    Args:
        env: FoodRedistributionEnv instance
        n_episodes: number of training episodes
        progress_cb: optional callable(episode, reward) called each episode

    Returns:
        agent: trained DQNAgent
        rewards_history: list of per-episode total rewards
        smoothed: list of 10-episode moving-average rewards
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(state_dim=env.n_states, action_dim=env.n_actions, device=device)
    rewards_history = []

    for episode in range(1, n_episodes + 1):
        state = env.reset(use_seeded=True)
        total_reward = 0.0
        ep_loss = 0.0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            loss = agent.learn()
            ep_loss += loss
            steps += 1
            if done:
                break

        agent.end_episode()
        rewards_history.append(float(total_reward))

        if progress_cb:
            progress_cb(episode, total_reward)

    # Compute smoothed rewards (10-episode moving average)
    rewards_arr = np.array(rewards_history, dtype=np.float64)  # type: ignore[call-overload]
    smoothed = []
    window = 10
    for i in range(len(rewards_arr)):
        start = max(0, i - window + 1)
        smoothed.append(float(np.mean(rewards_arr[start: i + 1])))

    return agent, rewards_history, smoothed


# ---------------------------------------------------------------------------
# Greedy (Nearest Neighbour) Baseline for comparison
# ---------------------------------------------------------------------------

def run_greedy_baseline(env):
    """
    Simple nearest-neighbour greedy: always visit the closest unvisited
    restaurant first, then deliver to closest NGO.
    Returns summary dict.
    """
    env.reset(use_seeded=True)
    visited_r = set()
    delivered_n = set()

    def nearest(coords, visited):
        best, best_d = None, float("inf")
        for i, c in enumerate(coords):
            if i not in visited:
                d = env._euclidean(env.vehicle_pos, c)
                if d < best_d:
                    best, best_d = i, d
        return best

    for _ in range(env.MAX_STEPS):
        if env.done:
            break
        if env.vehicle_load < env.VEHICLE_CAPACITY * 0.3:
            action = nearest(env.restaurant_coords, visited_r)
            if action is None:
                action = 0
            visited_r.add(action)
        else:
            action = nearest(env.ngo_coords, delivered_n)
            if action is None:
                action = 0
            action += env.N_RESTAURANTS
            delivered_n.add(action - env.N_RESTAURANTS)

        _, _, done, _ = env.step(action)
        if done:
            break

    return env.get_summary()
