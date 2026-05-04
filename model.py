"""
model.py — DQN & Double DQN Implementation  v3
================================================
KEY CHANGES FROM v2:
────────────────────
1. SMALLER NETWORK: 128→64→32 (was 256→128→64).
   8-dimensional state + 4 actions doesn't need 256-wide layers.
   Smaller net = less overfitting to replay buffer, faster convergence.

2. SOFT TARGET UPDATE (Polyak averaging, τ=0.005):
   Old: hard copy every 300 steps → sudden Q-target shifts → loss spikes
   New: target_params = τ × online + (1-τ) × target every step → smooth

3. SCHEDULER NOT CALLED IN learn():
   Old: scheduler.step() called per gradient update (12×/episode)
   New: train.py calls agent.step_episode() once per episode

4. GAMMA = 0.99 (was 0.95):
   12-step episodes need less discounting. With γ=0.95, step 12 is
   discounted to 0.57. With γ=0.99, step 12 is 0.89 — agent values
   the full customer lifecycle.

Architecture unchanged: Dueling Double DQN (well-suited for the task).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Q-NETWORK  (smaller for 8-dim state)
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Dueling DQN architecture:
        Shared: Linear(8→128) → LN → GELU → Linear(128→64) → LN → GELU → Linear(64→32) → GELU
        Value stream:     Linear(32→16) → GELU → Linear(16→1)
        Advantage stream: Linear(32→16) → GELU → Linear(32→N_ACTIONS)
        Q = V + A − mean(A)
    """

    def __init__(
        self,
        state_dim:   int,
        n_actions:   int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dueling:     bool = True,
    ):
        super().__init__()
        self.dueling = dueling

        layers = []
        in_dim = state_dim
        for h in hidden_dims[:-1]:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, hidden_dims[-1]), nn.GELU()]
        self.feature_net = nn.Sequential(*layers)

        feat_dim = hidden_dims[-1]

        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(feat_dim, 16), nn.GELU(), nn.Linear(16, 1)
            )
            self.adv_stream = nn.Sequential(
                nn.Linear(feat_dim, 16), nn.GELU(), nn.Linear(16, n_actions)
            )
        else:
            self.output = nn.Linear(feat_dim, n_actions)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_net(x)
        if self.dueling:
            V = self.value_stream(feat)
            A = self.adv_stream(feat)
            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            Q = self.output(feat)
        return Q

    def get_action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q = self.forward(state.unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def get_q_values(self, state: torch.Tensor) -> np.ndarray:
        """Return all Q-values as numpy array (for diagnostics)."""
        with torch.no_grad():
            q = self.forward(state.unsqueeze(0)).squeeze(0).cpu().numpy()
        return q


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITIZED EXPERIENCE REPLAY (PER)
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2015).
    Samples transitions proportional to |TD error|^alpha.
    """

    def __init__(
        self,
        capacity:  int,
        state_dim: int,
        device:    torch.device,
        alpha:     float = 0.6,
        beta:      float = 0.4,
        beta_end:  float = 1.0,
        epsilon:   float = 1e-4,
    ):
        self.capacity  = capacity
        self.state_dim = state_dim
        self.device    = device
        self.alpha     = alpha
        self.beta      = beta
        self.beta_end  = beta_end
        self.epsilon   = epsilon
        self.ptr  = 0
        self.size = 0

        # Storage
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)
        self.priorities  = np.zeros(capacity, dtype=np.float32)

        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.dones[self.ptr]       = float(done)
        self.priorities[self.ptr]  = self._max_priority
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        idx = np.random.choice(self.size, size=batch_size, replace=False, p=probs)

        # Importance sampling weights
        weights = (self.size * probs[idx]) ** (-self.beta)
        weights /= weights.max()

        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
            torch.from_numpy(weights.astype(np.float32)).to(self.device),
            idx,
        )

    def update_priorities(self, indices, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.priorities[indices] = priorities
        self._max_priority = max(self._max_priority, priorities.max())

    def anneal_beta(self, current_step, total_steps):
        frac = min(1.0, current_step / max(total_steps, 1))
        self.beta = self.beta + frac * (self.beta_end - self.beta)

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# UNIFORM REPLAY BUFFER  (fallback)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Standard uniform experience replay buffer."""

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity  = capacity
        self.device    = device
        self.state_dim = state_dim
        self.ptr       = 0
        self.size      = 0

        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.dones[self.ptr]       = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        weights = torch.ones(batch_size).to(self.device)
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
            weights,
            idx,
        )

    def update_priorities(self, indices, td_errors):
        pass

    def anneal_beta(self, *args, **kwargs):
        pass

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# DQN AGENT
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN + Dueling + Prioritized Replay + Soft Target Update.

    Key changes from v2:
    ────────────────────
    • Soft target update (τ=0.005) instead of hard copy every 300 steps
    • Scheduler NOT called in learn() — use step_episode() instead
    • γ=0.99 for 12-step episodes
    • Smaller network (128→64→32)
    """

    def __init__(
        self,
        state_dim:     int,
        n_actions:     int,
        gamma:         float = 0.99,
        lr:            float = 1e-3,
        tau:           float = 0.005,         # soft target update rate
        batch_size:    int   = 128,
        buffer_size:   int   = 50_000,
        dueling:       bool  = True,
        double_dqn:    bool  = True,
        use_per:       bool  = True,
        n_train_episodes: int = 1_000,       # for LR scheduler
        device:        Optional[str] = None,
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.tau           = tau
        self.batch_size    = batch_size
        self.double_dqn    = double_dqn
        self.use_per       = use_per
        self.train_steps   = 0

        # Epsilon — managed externally by train.py (linear schedule)
        self.eps = 1.0

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.online_net = QNetwork(state_dim, n_actions, dueling=dueling).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, dueling=dueling).to(self.device)
        self._hard_sync()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=lr, eps=1e-5
        )
        # Cosine annealing: steps once per EPISODE (not per gradient)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(n_train_episodes, 1), eta_min=lr * 0.1
        )

        # Buffer
        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size, state_dim, self.device,
                alpha=0.6, beta=0.4, beta_end=1.0,
            )
        else:
            self.buffer = ReplayBuffer(buffer_size, state_dim, self.device)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """ε-greedy action selection."""
        if not eval_mode and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        s = torch.from_numpy(state).float().to(self.device)
        return self.online_net.get_action(s)

    # ── Learning step ─────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        """One gradient update. Returns loss or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, weights, idx = \
            self.buffer.sample(self.batch_size)

        # Current Q-values
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values

            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # TD errors (for PER)
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # Weighted Huber loss
        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.train_steps += 1

        # Update PER priorities
        self.buffer.update_priorities(idx, td_errors)

        # Soft target update (every gradient step)
        self._soft_update()

        return float(loss.item())

    # ── Episode-level updates ─────────────────────────────────────────────────

    def step_episode(self):
        """Call once per EPISODE (not per gradient step)."""
        self.scheduler.step()
        # Anneal PER beta
        self.buffer.anneal_beta(self.train_steps, 50_000)

    # ── Memory ────────────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def set_epsilon(self, eps: float):
        """Set epsilon externally (linear schedule in train.py)."""
        self.eps = eps

    # ── Target network ────────────────────────────────────────────────────────

    def _soft_update(self):
        """Polyak averaging: θ_target ← τ·θ_online + (1-τ)·θ_target"""
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def _hard_sync(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_q_stats(self, state: np.ndarray) -> dict:
        """Return Q-values and action preferences for diagnostics."""
        s = torch.from_numpy(state).float().to(self.device)
        q = self.online_net.get_q_values(s)
        return {
            "q_values": q,
            "best_action": int(np.argmax(q)),
            "q_gap": float(q.max() - np.sort(q)[-2]),
        }

    def get_current_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "eps":         self.eps,
            "train_steps": self.train_steps,
        }, path)
        print(f"[DQNAgent] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.eps = ckpt["eps"]
        self.train_steps = ckpt["train_steps"]
        print(f"[DQNAgent] Loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from env import N_ACTIONS
    from data import N_FEATURES

    agent = DQNAgent(state_dim=N_FEATURES, n_actions=N_ACTIONS, use_per=True)
    print(f"Device     : {agent.device}")
    print(f"N_ACTIONS  : {N_ACTIONS}  (Increase, Decrease, Maintain, Block)")
    print(f"Network    :\n{agent.online_net}")

    rng = np.random.default_rng(0)
    for _ in range(500):
        s  = rng.random(N_FEATURES).astype(np.float32)
        a  = agent.select_action(s)
        r  = rng.normal(0, 1)  # rewards now in ~[-3, +2]
        s2 = rng.random(N_FEATURES).astype(np.float32)
        d  = rng.random() < 0.1
        agent.store(s, a, r, s2, d)

    loss = agent.learn()
    print(f"First loss : {loss:.6f}")

    sample_state = rng.random(N_FEATURES).astype(np.float32)
    stats = agent.get_q_stats(sample_state)
    from env import ACTIONS as ACT
    print(f"Q-values   : { {ACT[i][0]: f'{q:.3f}' for i, q in enumerate(stats['q_values'])} }")
    print(f"Best action: {ACT[stats['best_action']][0]}")
    print("model.py smoke test passed ✓")
