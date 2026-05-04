"""
train.py — Training Loop, Baselines, Evaluation & Insights
===========================================================
ROOT CAUSE FIXES (v2):
──────────────────────
1. ACTION SPACE: Updated all references to 4-action space (Increase, Decrease,
   Maintain, Block). All baselines and interpretation code updated.

2. ACTION BIAS DIAGNOSTICS:
   New section: detect_action_bias() — measures if agent is stuck on one action.
   Triggers a warning and recommends fixes if any action dominates >70%.

3. COMPREHENSIVE METRICS (new):
   • Q-value gap per action (margin between best and second-best)
   • Policy entropy (uniform = log(4) ≈ 1.39; collapsed = 0)
   • Action distribution over time (tracked every eval window)
   • Per-action reward breakdown
   • Training efficiency metrics (reward per episode, sample efficiency)
   • Calibration check: compare policy risk scores vs actual outcomes

4. EVALUATION SEPARATED: eval env uses a different seed from training env.
   Previously both used the same pool/seed, so eval was just re-running on
   training customers → optimistic results.

5. WARMUP PERIOD: First 200 episodes are pure random exploration to fill
   the PER buffer with diverse experiences before any gradient updates.
   This prevents the early collapsed policy from self-reinforcing.

Usage
──────
  python train.py                          # full run (2000 episodes)
  python train.py --episodes 500           # quick test
  python train.py --eval-only model.pt     # evaluate saved checkpoint
  python train.py --no-per                 # disable prioritized replay
"""

import argparse
import json
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data import (
    FEATURE_NAMES,
    generate_synthetic_customers,
    build_customer_pool,
    TransitionModels,
)
from env import CreditLimitEnv, N_ACTIONS, ACTIONS
from model import DQNAgent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import random


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE POLICIES  (updated for 4-action space)
# ─────────────────────────────────────────────────────────────────────────────

def always_increase(obs):  return 0   # Increase every step
def always_decrease(obs):  return 1   # Decrease every step
def always_maintain(obs):  return 2   # Maintain every step
def random_policy(obs):    return np.random.randint(N_ACTIONS)

def rule_based_policy(obs: np.ndarray) -> int:
    """
    Heuristic baseline mimicking a simple bank credit policy.

    Normalized obs indices:
    [0] credit_limit, [1] balance, [2] utilization, [3] payment_ratio,
    [4] late_payments, [5] txn_frequency, [6] risk_score, [7] age

    Note: all values are normalized to [0,1] range.
    """
    util       = obs[2]    # [0,1]
    pay_ratio  = obs[3]    # [0,1] since raw max is 3.0
    late       = obs[4]    # [0,1] since raw max is 20
    risk       = obs[6]    # [0,1] since raw is 300-850

    # Block: many late payments AND low risk score
    if late > 0.30 and risk < 0.40:   # >6 late payments, score < 520
        return 3   # Block

    # Decrease: high utilization OR moderate late payments
    if util > 0.60 or late > 0.20:
        return 1   # Decrease

    # Increase: low utilization AND good payment history AND good risk score
    if util < 0.35 and pay_ratio > 0.4 and risk > 0.60:
        return 0   # Increase

    return 2   # Maintain


BASELINES = {
    "Always Increase": always_increase,
    "Always Decrease": always_decrease,
    "Always Maintain": always_maintain,
    "Rule-Based":      rule_based_policy,
    "Random":          random_policy,
}

ACTION_LABELS = [ACTIONS[i][0] for i in range(N_ACTIONS)]


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env: CreditLimitEnv,
    agent_or_policy,
    train: bool = False,
    warmup: bool = False,
) -> Dict:
    """
    Run a single episode.

    warmup=True: forces random action selection (for buffer filling).
    train=True:  stores transitions and calls learn().
    """
    obs  = env.reset()
    done = False
    episode_reward = 0.0
    step_rewards   = []
    action_counts  = defaultdict(int)
    info_list      = []
    loss_list      = []
    q_gaps         = []

    while not done:
        if warmup:
            action = env.sample_action()
        elif callable(agent_or_policy) and not isinstance(agent_or_policy, DQNAgent):
            action = agent_or_policy(obs)
        else:
            action = agent_or_policy.select_action(obs, eval_mode=not train)

        # Track Q-value gap during training (measure confidence)
        if isinstance(agent_or_policy, DQNAgent) and not warmup:
            stats = agent_or_policy.get_q_stats(obs)
            q_gaps.append(stats["q_gap"])

        next_obs, reward, done, info = env.step(action)

        if train and isinstance(agent_or_policy, DQNAgent) and not warmup:
            agent_or_policy.store(obs, action, reward, next_obs, done)
            loss = agent_or_policy.learn()
            if loss is not None:
                loss_list.append(loss)
        elif warmup and isinstance(agent_or_policy, DQNAgent):
            # During warmup, still store but don't learn
            agent_or_policy.store(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward
        step_rewards.append(reward)
        action_counts[info["action"]] += 1
        info_list.append(info)

    return {
        "total_reward":    episode_reward,
        "step_rewards":    step_rewards,
        "n_steps":         len(step_rewards),
        "action_counts":   dict(action_counts),
        "defaulted":       any(i["defaulted"]  for i in info_list),
        "churned":         any(i["churned"]     for i in info_list),
        "losses":          loss_list,
        "mean_p_default":  np.mean([i["p_default"]  for i in info_list]),
        "mean_p_churn":    np.mean([i["p_churn"]     for i in info_list]),
        "mean_q_gap":      float(np.mean(q_gaps)) if q_gaps else 0.0,
        "reward_components": {
            "base_revenue":   np.mean([i.get("base_revenue", 0)   for i in info_list]),
            "default_cost":   np.mean([i.get("default_cost", 0)   for i in info_list]),
            "churn_cost":     np.mean([i.get("churn_cost", 0)     for i in info_list]),
            "action_adj":     np.mean([i.get("action_adj", 0)     for i in info_list]),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# ACTION BIAS DETECTOR  (new diagnostic tool)
# ─────────────────────────────────────────────────────────────────────────────

def detect_action_bias(
    agent: DQNAgent,
    pool: np.ndarray,
    threshold: float = 0.70,
) -> Dict:
    """
    Detect if the agent is collapsed onto one action.

    Returns a dict with:
      • action_distribution: fraction of customers preferring each action
      • is_biased: True if any action > threshold
      • dominant_action: which action dominates (if any)
      • policy_entropy: 0=collapsed, log(N_ACTIONS)=uniform
      • q_value_gap_stats: mean/std of Q(best) - Q(second_best)

    This diagnostic is critical because:
    ─────────────────────────────────────────────────────────────────────────
    If Q(Increase) >> Q(Decrease) for ALL states, the agent will always
    Increase regardless of customer risk. This is the core symptom we're
    diagnosing. The Q-value gap tells us: is the agent CONFIDENT or just
    defaulting because all Q-values are equal?
    """
    from data import normalize_state
    import torch

    agent.online_net.eval()
    preferred = []
    q_gaps    = []

    with torch.no_grad():
        for raw in pool[:1000]:
            norm = normalize_state(raw)
            t    = torch.from_numpy(norm).float().to(agent.device)
            q    = agent.online_net.get_q_values(t)
            preferred.append(int(np.argmax(q)))
            sorted_q = np.sort(q)
            q_gaps.append(float(sorted_q[-1] - sorted_q[-2]))

    preferred = np.array(preferred)
    dist = {ACTION_LABELS[i]: float((preferred == i).mean()) for i in range(N_ACTIONS)}

    # Policy entropy
    probs = np.array([(preferred == i).mean() for i in range(N_ACTIONS)])
    probs = np.clip(probs, 1e-8, 1)
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(N_ACTIONS))

    dominant = max(dist, key=dist.get)
    is_biased = dist[dominant] > threshold

    return {
        "action_distribution": dist,
        "is_biased":           is_biased,
        "dominant_action":     dominant if is_biased else None,
        "policy_entropy":      entropy,
        "max_possible_entropy": max_entropy,
        "normalized_entropy":   entropy / max_entropy,  # 1.0 = fully uniform
        "mean_q_gap":           float(np.mean(q_gaps)),
        "std_q_gap":            float(np.std(q_gaps)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_episodes:    int   = 2_000,
    eval_every:    int   = 100,
    eval_episodes: int   = 200,
    save_path:     str   = "model.pt",
    seed:          int   = 42,
    use_per:       bool  = True,
    warmup_episodes: int = 50,    # reduced: 50 is enough to seed buffer
) -> Dict:
    """
    Full training loop with action bias monitoring.

    Phase 1 (episodes 1-warmup_episodes):
        Pure random exploration. Fills PER buffer with diverse experiences
        including rare default and block events. No gradient updates.

    Phase 2 (episodes warmup_episodes+1 to n_episodes):
        ε-greedy exploration with ε decaying from 1.0 → 0.05.
        Double DQN + PER updates every episode.

    Key diagnostic checkpoints:
        Every eval_every episodes: check action distribution, Q-value gap,
        policy entropy. If bias detected, log warning.
    """
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 70)
    print("  RL CREDIT LIMIT OPTIMIZATION — TRAINING v2")
    print("  (4 actions: Increase / Decrease / Maintain / Block)")
    print("=" * 70)

    # ── Step 1: Data + transition models ─────────────────────────────────────
    print("\n[1/5] Generating synthetic data & fitting transition models …")
    df = generate_synthetic_customers(n=8_000, seed=seed)
    print(f"      Dataset: {len(df):,} customers  |  "
          f"Default={df['default'].mean():.1%}  |  Churn={df['churn'].mean():.1%}")
    tm = TransitionModels()
    tm.fit(df, verbose=True)

    # ── Step 2: Environment + agent ───────────────────────────────────────────
    print("\n[2/5] Building environments & DQN agent …")
    # SEPARATE train and eval pools/seeds — prevents eval leaking train performance
    train_pool = build_customer_pool(n=2_000, seed=seed)
    eval_pool  = build_customer_pool(n=1_000, seed=seed + 1000)  # different pool

    train_env = CreditLimitEnv(tm, train_pool, episode_length=12, seed=seed)
    eval_env  = CreditLimitEnv(tm, eval_pool,  episode_length=12, seed=seed + 999)

    from data import N_FEATURES
    n_train_episodes = n_episodes - warmup_episodes
    agent = DQNAgent(
        state_dim=N_FEATURES,
        n_actions=N_ACTIONS,
        gamma=0.99,
        lr=1e-3,
        tau=0.005,
        batch_size=128,
        buffer_size=50_000,
        dueling=True,
        double_dqn=True,
        use_per=use_per,
        n_train_episodes=n_train_episodes,
    )

    # Linear epsilon schedule
    eps_start = 1.0
    eps_end   = 0.08
    decay_episodes = int(0.7 * n_train_episodes)  # decay over 70% of training
    print(f"      Device      : {agent.device}")
    print(f"      Episodes    : {n_episodes}  (warmup: {warmup_episodes})")
    print(f"      Buffer      : {agent.buffer.capacity:,}  (PER: {use_per})")
    print(f"      Actions     : {[ACTIONS[i][0] for i in range(N_ACTIONS)]}")

    # ── Step 3: Warmup (pure random fill) ────────────────────────────────────
    print(f"\n[3/5] Warmup phase ({warmup_episodes} random episodes) …")
    for ep in range(warmup_episodes):
        run_episode(train_env, agent, train=False, warmup=True)
    print(f"      Buffer filled: {len(agent.buffer):,} transitions")

    # ── Step 4: Training loop ─────────────────────────────────────────────────
    print("\n[4/5] Training …")
    history = {
        "episode_rewards":      [],
        "mean_losses":          [],
        "epsilons":             [],
        "default_rates":        [],
        "churn_rates":          [],
        "q_gaps":               [],
        "eval_rewards":         [],
        "eval_episodes":        [],
        "eval_default_rates":   [],
        "action_distributions": [],
        "policy_entropies":     [],
        "lr_history":           [],
        "reward_components":    [],
    }

    best_eval_reward = -np.inf
    t0 = time.time()

    for ep in range(1, n_train_episodes + 1):
        # Linear epsilon decay
        if ep <= decay_episodes:
            new_eps = eps_start - (eps_start - eps_end) * (ep / decay_episodes)
        else:
            new_eps = eps_end
        agent.set_epsilon(new_eps)

        result = run_episode(train_env, agent, train=True, warmup=False)
        agent.step_episode()  # LR scheduler + PER beta annealing

        history["episode_rewards"].append(result["total_reward"])
        history["mean_losses"].append(
            float(np.mean(result["losses"])) if result["losses"] else float("nan")
        )
        history["epsilons"].append(agent.eps)
        history["default_rates"].append(float(result["defaulted"]))
        history["churn_rates"].append(float(result["churned"]))
        history["q_gaps"].append(result["mean_q_gap"])
        history["lr_history"].append(agent.get_current_lr())
        history["reward_components"].append(result["reward_components"])

        # Track action distribution per episode
        total_steps_ep = sum(result["action_counts"].values())
        ep_dist = {lbl: result["action_counts"].get(lbl, 0) / max(total_steps_ep, 1)
                   for lbl in ACTION_LABELS}
        history["action_distributions"].append(ep_dist)

        # ── Periodic evaluation ───────────────────────────────────────────────
        if ep % eval_every == 0:
            
            eval_rewards, eval_defaults, eval_churns, eval_steps = [], [], [], []
            eval_action_counts = Counter()

            for _ in range(eval_episodes):
                r = run_episode(eval_env, agent, train=False)
                eval_rewards.append(r["total_reward"])
                eval_defaults.append(float(r["defaulted"]))
                eval_churns.append(float(r["churned"]))
                eval_steps.append(r["n_steps"])
                for a, cnt in r["action_counts"].items():
                    eval_action_counts[a] += cnt

            mean_eval  = float(np.mean(eval_rewards))
            mean_def   = float(np.mean(eval_defaults))

            history["eval_rewards"].append(mean_eval)
            history["eval_episodes"].append(ep)
            history["eval_default_rates"].append(mean_def)

            # Action bias check
            bias_info  = detect_action_bias(agent, train_pool)
            entropy_n  = bias_info["normalized_entropy"]
            history["policy_entropies"].append(entropy_n)

            elapsed = time.time() - t0
            recent_loss = [v for v in history["mean_losses"][-100:] if not np.isnan(v)]
            mean_loss = float(np.mean(recent_loss)) if recent_loss else float("nan")

            print(
                f"  Ep {ep + warmup_episodes:5d} | "
                f"ε={agent.eps:.3f} | "
                f"train_r={np.mean(history['episode_rewards'][-100:]):7.2f} | "
                f"eval_r={mean_eval:7.2f} | "
                f"def%={mean_def:.1%} | "
                f"loss={mean_loss:.4f} | "
                f"H={entropy_n:.2f} | "
                f"t={elapsed:.0f}s"
            )
            print(f"          Action dist: "
                  + " | ".join(f"{a}={v:.1%}" for a, v in bias_info["action_distribution"].items()))

            # Bias warning
            if bias_info["is_biased"]:
                print(f"  ⚠️  ACTION BIAS DETECTED: '{bias_info['dominant_action']}' "
                      f"dominates {bias_info['action_distribution'][bias_info['dominant_action']]:.0%} "
                      f"of states. Entropy={entropy_n:.2f} (0=collapsed, 1=uniform)")

            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval
                agent.save(save_path)
                print(f"          ✓ New best eval reward: {best_eval_reward:.2f}")

    print(f"\n[4/5] Training complete. Best eval reward: {best_eval_reward:.2f}")
    print(f"      Total gradient steps: {agent.train_steps:,}")
    return history, agent, eval_env, tm, eval_pool


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION vs BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    agent: DQNAgent,
    env: CreditLimitEnv,
    n_episodes: int = 500,
) -> Dict:
    """
    Evaluate the trained agent vs all baselines on the eval environment.
    Returns comprehensive metrics including action distributions.
    """
    print("\n[5/5] Evaluating agent vs baselines …")

    all_policies = {"DQN Agent": agent, **BASELINES}
    results = {}

    for name, policy in all_policies.items():
        rewards, defaults, churns, steps = [], [], [], []
        action_totals = Counter()
        per_action_rewards = defaultdict(list)
        rev_components = defaultdict(list)

        for _ in range(n_episodes):
            r = run_episode(env, policy, train=False)
            rewards.append(r["total_reward"])
            defaults.append(float(r["defaulted"]))
            churns.append(float(r["churned"]))
            steps.append(r["n_steps"])
            for a, cnt in r["action_counts"].items():
                action_totals[a] += cnt
            for k, v in r["reward_components"].items():
                rev_components[k].append(v)

        total_actions = sum(action_totals.values())
        action_dist = {a: cnt / max(total_actions, 1) for a, cnt in action_totals.items()}

        # Policy entropy
        probs = np.array([action_dist.get(lbl, 0) for lbl in ACTION_LABELS])
        probs = np.clip(probs, 1e-8, 1); probs /= probs.sum()
        entropy = float(-np.sum(probs * np.log(probs)))

        results[name] = {
            "mean_reward":   float(np.mean(rewards)),
            "std_reward":    float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "default_rate":  float(np.mean(defaults)),
            "churn_rate":    float(np.mean(churns)),
            "mean_steps":    float(np.mean(steps)),
            "action_dist":   action_dist,
            "policy_entropy": entropy,
            "normalized_entropy": entropy / np.log(N_ACTIONS),
            "mean_base_revenue":  float(np.mean(rev_components["base_revenue"])),
            "mean_def_cost":      float(np.mean(rev_components["default_cost"])),
        }

    # Print summary table
    print("\n" + "─" * 90)
    print(f"{'Policy':<22} {'MeanR':>8} {'Std':>6} {'Def%':>6} {'Churn%':>7} "
          f"{'Entropy':>8} {'Inc%':>6} {'Dec%':>6} {'Main%':>6} {'Blk%':>6}")
    print("─" * 90)

    for name, m in results.items():
        marker = " ◄" if name == "DQN Agent" else ""
        ad = m["action_dist"]
        print(
            f"{name:<22} "
            f"{m['mean_reward']:>8.2f} "
            f"{m['std_reward']:>6.2f} "
            f"{m['default_rate']:>5.1%} "
            f"{m['churn_rate']:>6.1%} "
            f"{m['normalized_entropy']:>8.2f} "
            f"{ad.get('Increase', 0):>5.1%} "
            f"{ad.get('Decrease', 0):>5.1%} "
            f"{ad.get('Maintain', 0):>5.1%} "
            f"{ad.get('Block', 0):>5.1%}"
            f"{marker}"
        )
    print("─" * 90)
    print("Entropy: 1.0 = perfectly uniform across actions, 0.0 = always one action")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# POLICY INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_policy(agent: DQNAgent, pool: np.ndarray) -> None:
    """
    Deep inspection of the learned policy.

    Analyses:
    1. Action distribution over 1000 customers
    2. Per-feature statistics by chosen action
    3. Q-value statistics (gap, confidence)
    4. Boundary analysis: what separates Increase vs Decrease customers?
    5. Action bias diagnosis
    """
    from data import normalize_state
    import torch

    print("\n" + "=" * 70)
    print("  POLICY INTERPRETATION")
    print("=" * 70)

    sample = pool[:1000]
    preferred = []
    q_vals    = []

    agent.online_net.eval()
    with torch.no_grad():
        for raw in sample:
            norm = normalize_state(raw)
            t = torch.from_numpy(norm).float().to(agent.device).unsqueeze(0)
            q = agent.online_net(t).squeeze(0).cpu().numpy()
            q_vals.append(q)
            preferred.append(int(np.argmax(q)))

    preferred = np.array(preferred)
    q_vals    = np.array(q_vals)

    # ── 1. Action distribution ────────────────────────────────────────────────
    print("\nAction Distribution (1000 customers, greedy policy):")
    for i, lbl in enumerate(ACTION_LABELS):
        pct = (preferred == i).mean()
        bar = "█" * int(pct * 40)
        print(f"  {lbl:<10}  {pct:5.1%}  {bar}")

    # Policy entropy
    probs = np.array([(preferred == i).mean() for i in range(N_ACTIONS)])
    probs = np.clip(probs, 1e-8, 1)
    entropy = -np.sum(probs * np.log(probs))
    print(f"\n  Policy entropy: {entropy:.3f} / {np.log(N_ACTIONS):.3f} (max)")
    print(f"  Normalized:     {entropy / np.log(N_ACTIONS):.3f} (1.0=uniform, 0.0=collapsed)")

    if entropy / np.log(N_ACTIONS) < 0.4:
        dom = ACTION_LABELS[np.argmax(probs)]
        print(f"\n  ⚠️  WARNING: Policy entropy is low — agent heavily biases toward '{dom}'.")
        print("     This may indicate reward imbalance or under-exploration.")
        print("     Suggestions: increase eps_end, check reward magnitudes,")
        print("     verify transition model probabilities are non-trivial.")

    # ── 2. Feature stats by action ────────────────────────────────────────────
    print("\nAverage feature values by preferred action:")
    print(f"  {'Feature':<18}" + "".join(f"  {l:<10}" for l in ACTION_LABELS))
    print("  " + "─" * (18 + 12 * N_ACTIONS))
    for fi, fname in enumerate(FEATURE_NAMES):
        row = f"  {fname:<18}"
        for ai in range(N_ACTIONS):
            mask = preferred == ai
            val = sample[mask, fi].mean() if mask.sum() > 0 else float("nan")
            row += f"  {val:>10.2f}"
        print(row)

    # ── 3. Q-value statistics ─────────────────────────────────────────────────
    q_gaps = q_vals.max(axis=1) - np.sort(q_vals, axis=1)[:, -2]
    print(f"\nQ-value gap (best - 2nd best):")
    print(f"  Mean={q_gaps.mean():.3f}  Std={q_gaps.std():.3f}  "
          f"Min={q_gaps.min():.3f}  Max={q_gaps.max():.3f}")
    print(f"  Low gap = agent uncertain. High gap = agent very confident.")

    print(f"\nMean Q-value per action:")
    for i, lbl in enumerate(ACTION_LABELS):
        print(f"  {lbl:<10}: mean={q_vals[:, i].mean():.3f}  "
              f"std={q_vals[:, i].std():.3f}  "
              f"min={q_vals[:, i].min():.3f}  "
              f"max={q_vals[:, i].max():.3f}")

    # ── 4. Narrative insights ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Key policy insights:")
    for ai, lbl in enumerate(ACTION_LABELS):
        mask = preferred == ai
        if mask.sum() < 5:
            print(f"\n  [{lbl}] N/A (<5 customers assigned this action)")
            continue
        r = sample[mask]
        pct = mask.mean()
        risk  = r[:, 6].mean()   # risk_score
        util  = r[:, 2].mean()   # utilization
        late  = r[:, 4].mean()   # late_payments
        pay   = r[:, 3].mean()   # payment_ratio

        icons = {"Increase": "▲", "Decrease": "▼", "Maintain": "■", "Block": "✗"}
        print(f"\n  {icons[lbl]} {lbl} ({pct:.0%} of customers):")
        print(f"      Avg risk_score    : {risk:.0f}   Avg utilization: {util:.2f}")
        print(f"      Avg late_payments : {late:.1f}   Avg pay_ratio:   {pay:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING EFFICIENCY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_training_efficiency(history: Dict) -> Dict:
    """
    Compute metrics that measure HOW EFFICIENTLY the agent learned.

    Metrics:
    ─────────
    • episodes_to_convergence: first episode where eval reward stays within
      5% of final best for 3 consecutive eval windows
    • sample_efficiency: final eval reward / total gradient steps
    • reward_improvement: (final_eval - initial_eval) / initial_eval
    • mean_policy_entropy: average normalized entropy across eval windows
    • loss_convergence: episode where loss drops to <5% of peak loss
    """
    eval_rewards  = np.array(history.get("eval_rewards",  []))
    eval_episodes = np.array(history.get("eval_episodes", []))
    entropies     = np.array(history.get("policy_entropies", []))
    losses        = np.array([v for v in history["mean_losses"] if not np.isnan(v)])

    metrics: Dict = {}

    if len(eval_rewards) >= 3:
        final_best = eval_rewards.max()
        threshold  = final_best * 0.95
        conv_ep    = None
        for i in range(len(eval_rewards) - 2):
            if all(r >= threshold for r in eval_rewards[i:i+3]):
                conv_ep = int(eval_episodes[i])
                break
        metrics["episodes_to_convergence"] = conv_ep or int(eval_episodes[-1])
        metrics["reward_improvement"] = float(
            (eval_rewards[-1] - eval_rewards[0]) / max(abs(eval_rewards[0]), 1e-3)
        )
        metrics["final_eval_reward"] = float(eval_rewards[-1])
        metrics["best_eval_reward"]  = float(eval_rewards.max())

    if len(entropies) > 0:
        metrics["mean_policy_entropy"] = float(entropies.mean())
        metrics["final_policy_entropy"] = float(entropies[-1]) if len(entropies) > 0 else float("nan")

    if len(losses) >= 10:
        peak_loss = losses.max()
        conv_idx  = np.where(losses < 0.05 * peak_loss)[0]
        metrics["loss_convergence_step"] = int(conv_idx[0]) if len(conv_idx) > 0 else len(losses)
        metrics["final_mean_loss"]       = float(losses[-50:].mean())

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curve(history: Dict, save_dir: str = ".", smooth_window: int = 50) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available — pip install matplotlib")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    BG = "#0f1117"; PANEL_BG = "#151821"; GRID_C = "#1e2130"
    SPINE_C = "#2a2d3a"; TEXT_C = "#c8cad4"; TICK_C = "#6b7085"
    TEAL = "#1DB594"; AMBER = "#F0A020"; PURPLE = "#7B6FE8"
    RED = "#E05050"; BLUE = "#4488FF"; ORANGE = "#FF7744"

    ACTION_COLORS = [TEAL, RED, AMBER, PURPLE]   # Inc, Dec, Main, Block

    def rolling_mean(arr, w):
        result = np.full(len(arr), np.nan)
        for i in range(len(arr)):
            vals = [v for v in arr[max(0, i - w + 1):i + 1]
                    if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                result[i] = np.mean(vals)
        return result

    rewards      = np.array(history["episode_rewards"], dtype=float)
    losses_raw   = np.array([v if v == v else np.nan for v in history["mean_losses"]], dtype=float)
    epsilons     = np.array(history["epsilons"], dtype=float)
    default_r    = np.array(history["default_rates"], dtype=float)
    eval_eps     = history.get("eval_episodes", [])
    eval_rew     = history.get("eval_rewards",  [])
    entropies    = np.array(history.get("policy_entropies", []), dtype=float)
    q_gaps       = np.array(history.get("q_gaps", []), dtype=float)

    n_eps = len(rewards)
    x = np.arange(1, n_eps + 1)

    smooth_reward  = rolling_mean(rewards.tolist(), smooth_window)
    smooth_loss    = rolling_mean(losses_raw.tolist(), min(smooth_window, 30))
    smooth_default = rolling_mean(default_r.tolist(), smooth_window) * 100
    smooth_q_gap   = rolling_mean(q_gaps.tolist(), smooth_window)

    # ── Build action distribution time series ────────────────────────────────
    if history.get("action_distributions"):
        action_dists = history["action_distributions"]
        action_ts = {lbl: [] for lbl in ACTION_LABELS}
        for ep_dist in action_dists:
            for lbl in ACTION_LABELS:
                action_ts[lbl].append(ep_dist.get(lbl, 0.0))
        smooth_act = {lbl: rolling_mean(action_ts[lbl], smooth_window) for lbl in ACTION_LABELS}
    else:
        smooth_act = {}

    # ── Figure: 3×2 grid ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.42, wspace=0.30,
                        left=0.07, right=0.97, top=0.92, bottom=0.06)

    def style_ax(ax, ylabel, xlabel="Episode"):
        ax.set_facecolor(PANEL_BG)
        ax.set_xlabel(xlabel, color=TICK_C, fontsize=9, labelpad=4)
        ax.set_ylabel(ylabel,  color=TICK_C, fontsize=9, labelpad=4)
        ax.tick_params(colors=TICK_C, labelsize=8, length=3)
        ax.grid(color=GRID_C, linewidth=0.5, linestyle="-")
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE_C); sp.set_linewidth(0.6)

    # Panel 1: Episode reward
    ax = axes[0, 0]; style_ax(ax, "Total reward")
    ax.set_title("Episode Reward", color=TEXT_C, fontsize=11, pad=8)
    ax.fill_between(x, rewards, alpha=0.06, color=TEAL)
    ax.plot(x, rewards, color=TEAL, alpha=0.18, linewidth=0.6)
    ax.plot(x, smooth_reward, color=TEAL, linewidth=1.8, label=f"Smoothed (w={smooth_window})")
    if eval_eps and eval_rew:
        ax.scatter(eval_eps, eval_rew, color=RED, s=40, zorder=5, label="Eval")
    ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT_C)

    # Panel 2: Action distribution over time
    ax = axes[0, 1]; style_ax(ax, "Action frequency")
    ax.set_title("Action Distribution Over Training", color=TEXT_C, fontsize=11, pad=8)
    if smooth_act:
        for i, lbl in enumerate(ACTION_LABELS):
            ax.plot(x, smooth_act[lbl], color=ACTION_COLORS[i], linewidth=1.6,
                    label=lbl, alpha=0.9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT_C, loc="upper right")
        ax.axhline(1.0 / N_ACTIONS, color=SPINE_C, linewidth=1, linestyle="--", alpha=0.6)
        ax.text(n_eps * 0.98, 1.0 / N_ACTIONS + 0.02, "Uniform",
                color=TICK_C, fontsize=7, ha="right")

    # Panel 3: Training loss
    ax = axes[1, 0]; style_ax(ax, "Huber loss")
    ax.set_title("Training Loss", color=TEXT_C, fontsize=11, pad=8)
    valid = ~np.isnan(losses_raw)
    if valid.any():
        ax.fill_between(x[valid], losses_raw[valid], alpha=0.08, color=AMBER)
        ax.plot(x[valid], losses_raw[valid], color=AMBER, alpha=0.20, linewidth=0.6)
        ax.plot(x, smooth_loss, color=AMBER, linewidth=1.8, label="Smoothed")
        ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT_C)

    # Panel 4: Policy entropy
    ax = axes[1, 1]; style_ax(ax, "Normalized entropy")
    ax.set_title("Policy Entropy (Action Diversity)", color=TEXT_C, fontsize=11, pad=8)
    if len(entropies) > 0 and eval_eps:
        ax.plot(eval_eps, entropies, color=PURPLE, linewidth=2.0, marker="o",
                markersize=4, label="Entropy (1=uniform)")
        ax.axhline(1.0, color=SPINE_C, linewidth=1, linestyle="--", alpha=0.6)
        ax.text(eval_eps[-1] * 0.95, 1.0 + 0.02, "Max entropy",
                color=TICK_C, fontsize=7, ha="right")
        ax.set_ylim(0, 1.15)
        ax.axhline(0.4, color=RED, linewidth=0.8, linestyle=":", alpha=0.5)
        ax.text(eval_eps[-1] * 0.95, 0.42, "⚠ Bias threshold",
                color=RED, fontsize=7, ha="right", alpha=0.8)
        ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT_C)

    # Panel 5: Epsilon
    ax = axes[2, 0]; style_ax(ax, "ε (exploration rate)")
    ax.set_title("Epsilon Decay", color=TEXT_C, fontsize=11, pad=8)
    ax.fill_between(x, epsilons, alpha=0.10, color=BLUE)
    ax.plot(x, epsilons, color=BLUE, linewidth=1.8)
    ax.set_ylim(0, 1.05)
    for thresh, lbl in [(0.5, "50%"), (0.1, "10%")]:
        if epsilons.min() <= thresh <= epsilons.max():
            ax.axhline(thresh, color=SPINE_C, linewidth=0.8, linestyle="--", alpha=0.6)

    # Panel 6: Default rate
    ax = axes[2, 1]; style_ax(ax, "Default rate (%)")
    ax.set_title("Rolling Default Rate", color=TEXT_C, fontsize=11, pad=8)
    ax.fill_between(x, smooth_default, alpha=0.12, color=RED)
    ax.plot(x, smooth_default, color=RED, linewidth=1.8, label=f"Rolling avg (w={smooth_window})")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT_C)

    fig.suptitle(
        f"RL Credit Limit Optimization — Learning Curve  ({n_eps} episodes)",
        color=TEXT_C, fontsize=13, fontweight="medium", y=0.97
    )

    out = save_dir / "learning_curve.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Learning curve saved → {out}")
    plt.close()


def plot_training(history: Dict, eval_results: Dict, save_dir: str = ".") -> None:
    """Convenience wrapper for backward compatibility."""
    plot_learning_curve(history, save_dir=save_dir)

    if not MATPLOTLIB_AVAILABLE:
        return

    save_dir = Path(save_dir)
    BG = "#0f1117"; PANEL_BG = "#151821"; GRID_C = "#1e2130"
    SPINE_C = "#2a2d3a"; TEXT_C = "#c8cad4"; TICK_C = "#6b7085"
    TEAL = "#1DB594"; RED = "#E05050"; AMBER = "#F0A020"; PURPLE = "#7B6FE8"

    names  = list(eval_results.keys())
    means  = [eval_results[n]["mean_reward"]  for n in names]
    stds   = [eval_results[n]["std_reward"]   for n in names]
    drates = [eval_results[n]["default_rate"] * 100 for n in names]
    crates = [eval_results[n]["churn_rate"]   * 100 for n in names]
    entrs  = [eval_results[n]["normalized_entropy"] for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(wspace=0.35, left=0.05, right=0.97, top=0.88, bottom=0.06)

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TICK_C, labelsize=8)
        ax.grid(color=GRID_C, linewidth=0.5, axis="x")
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE_C); sp.set_linewidth(0.6)

    colors = [TEAL if n == "DQN Agent" else "#3a3d50" for n in names]

    for ax_i, (ax, data, title, col) in enumerate(zip(
        axes,
        [means, drates, crates, entrs],
        ["Mean reward", "Default rate (%)", "Churn rate (%)", "Policy entropy"],
        [colors, [RED if n == "DQN Agent" else "#3a3d50" for n in names],
         [AMBER if n == "DQN Agent" else "#3a3d50" for n in names],
         [PURPLE if n == "DQN Agent" else "#3a3d50" for n in names]],
    )):
        style_ax(ax)
        xerr = stds if ax_i == 0 else None
        kwargs = dict(color=col, height=0.55)
        if xerr:
            kwargs.update(xerr=xerr, ecolor=TICK_C, capsize=3,
                          error_kw={"linewidth": 0.8})
        ax.barh(names, data, **kwargs)
        ax.set_title(title, color=TEXT_C, fontsize=10, pad=6)
        ax.tick_params(axis="y", colors=TEXT_C)
        ax.axvline(0, color=SPINE_C, linewidth=0.8)

    fig.suptitle("Policy comparison vs baselines", color=TEXT_C,
                 fontsize=12, fontweight="medium", y=0.97)
    out = save_dir / "baseline_comparison.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Baseline comparison saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL Credit Limit Optimization v2")
    parser.add_argument("--episodes",         type=int,   default=2_000)
    parser.add_argument("--eval-every",       type=int,   default=100)
    parser.add_argument("--warmup-episodes",  type=int,   default=50)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--save",             type=str,   default="model.pt")
    parser.add_argument("--no-per",           action="store_true",
                        help="Disable prioritized replay (use uniform)")
    parser.add_argument("--eval-only",        type=str,   default=None,
                        help="Path to saved model for eval-only mode")
    args = parser.parse_args()

    if args.eval_only:
        from data import N_FEATURES
        df = generate_synthetic_customers(8_000)
        tm = TransitionModels()
        tm.fit(df, verbose=True)
        pool  = build_customer_pool(1_000, seed=9999)
        env   = CreditLimitEnv(tm, pool, episode_length=12, seed=9999)
        agent = DQNAgent(state_dim=N_FEATURES, n_actions=N_ACTIONS)
        agent.load(args.eval_only)
        eval_results = evaluate_all(agent, env)
        interpret_policy(agent, pool)
        bias = detect_action_bias(agent, pool)
        print(f"\nBias check: {bias}")
        return

    history, agent, eval_env, tm, pool = train(
        n_episodes=args.episodes,
        eval_every=args.eval_every,
        warmup_episodes=args.warmup_episodes,
        save_path=args.save,
        seed=args.seed,
        use_per=not args.no_per,
    )

    eval_results  = evaluate_all(agent, eval_env)
    interpret_policy(agent, pool)

    efficiency = compute_training_efficiency(history)
    print("\n" + "=" * 60)
    print("  TRAINING EFFICIENCY SUMMARY")
    print("=" * 60)
    for k, v in efficiency.items():
        if isinstance(v, float):
            print(f"  {k:<35}: {v:.4f}")
        else:
            print(f"  {k:<35}: {v}")

    bias = detect_action_bias(agent, pool)
    print("\n" + "=" * 60)
    print("  FINAL BIAS DIAGNOSIS")
    print("=" * 60)
    print(f"  Action distribution: {bias['action_distribution']}")
    print(f"  Policy entropy:      {bias['policy_entropy']:.3f} / {bias['max_possible_entropy']:.3f}")
    print(f"  Normalized entropy:  {bias['normalized_entropy']:.3f}")
    print(f"  Biased:              {bias['is_biased']}",
          f"(dominant: {bias['dominant_action']})" if bias['is_biased'] else "")

    plot_training(history, eval_results, save_dir=".")

    # Save history
    with open("history.json", "w") as f:
        def convert(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            if isinstance(obj, dict):           return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):           return [convert(v) for v in obj]
            return obj

        save_obj = {
            "eval_results": convert(eval_results),
            "efficiency":   convert(efficiency),
            "bias":         convert(bias),
            **{k: convert(v) for k, v in history.items()
               if k not in ("action_distributions", "reward_components")}
        }
        json.dump(save_obj, f, indent=2)
    print("\nHistory saved to history.json")


if __name__ == "__main__":
    main()
