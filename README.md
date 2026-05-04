# RL Credit Limit Optimization

A Reinforcement Learning system that trains a **Dueling Double DQN** agent to make optimal credit limit decisions (Increase / Decrease / Maintain / Block) for credit card customers.

---

## Project Structure

```
.
├── data.py           # Synthetic dataset generation & transition models
├── env.py            # Gym-style RL environment (CreditLimitEnv)
├── model.py          # DQN architecture + Prioritized Replay Buffer
├── train.py          # Training loop, evaluation, plots & baselines
├── requirements.txt  # Python dependencies
└── run.sh            # Automated end-to-end bash script
```

**Outputs produced after running:**
```
outputs/
├── model.pt                  # Trained DQN model checkpoint
├── learning_curve.png        # Reward, loss, entropy, epsilon plots
├── baseline_comparison.png   # DQN vs baselines comparison
└── history.json              # Training metrics & evaluation results
```

---

## Quick Start (Ubuntu / Linux / Mac)

### Prerequisites
- Git
- Python 3.8+
- Bash

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Make the script executable
chmod +x run.sh

# 3. Run the full pipeline
bash run.sh
```

That's it. The script will automatically:
- Create a Python virtual environment
- Install all dependencies
- Train the DQN agent (2000 episodes)
- Save all outputs to the `outputs/` folder

---

## Running with Docker on Ubuntu 22.04 (Windows / Mac / Any OS)

Follow these steps exactly if you are on **Windows or Mac**, or want to replicate the evaluator's environment.

### Step 1 — Install Docker

- **Windows / Mac:** Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Ubuntu/Linux:** Run:
  ```bash
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo systemctl start docker
  sudo usermod -aG docker $USER   # log out and back in after this
  ```

### Step 2 — Pull the Ubuntu 22.04 image

Open a terminal (or Docker Desktop terminal) and run:

```bash
docker pull ubuntu:22.04
```

### Step 3 — Start an interactive container

```bash
docker run -it ubuntu:22.04 bash
```

You are now inside a fresh Ubuntu 22.04 container. All the following commands run **inside the container**.

### Step 4 — Install Git inside the container

```bash
apt-get update && apt-get install -y git
```

### Step 5 — Clone your repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### Step 6 — Run the pipeline

```bash
bash run.sh
```

The script handles everything else automatically — Python installation, virtual environment, dependencies, training, and saving outputs.

### Step 7 — Copy outputs to your host machine (optional)

Open a **new terminal on your host machine** (not inside the container) and run:

```bash
# Find your container ID
docker ps

# Copy outputs folder to your current directory
docker cp <container_id>:/path/to/<your-repo>/outputs ./outputs
```

---

## Manual Setup (without run.sh)

If you prefer to run steps manually:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run training
python train.py --episodes 2000 --save outputs/model.pt

# Evaluate a saved model
python train.py --eval-only outputs/model.pt
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.24 | Numerical computation |
| pandas | ≥ 2.0 | Data generation & analysis |
| scikit-learn | ≥ 1.3 | Transition models (GBT, LR, calibration) |
| torch | ≥ 2.0 | DQN neural network (PyTorch) |
| matplotlib | ≥ 3.7 | Training plots & visualizations |

---

## Model & Environment

| Parameter | Value |
|---|---|
| State space | 8-dimensional (credit limit, balance, utilization, payment ratio, late payments, txn frequency, risk score, age) |
| Action space | 4 discrete (Increase +20%, Decrease −20%, Maintain, Block) |
| Episode length | 12 steps (monthly decisions) |
| Algorithm | Dueling Double DQN + Prioritized Experience Replay |
| Training episodes | 2000 |
| Warmup episodes | 50 (random exploration) |

---

## Troubleshooting

**`bash: python3: command not found`**
The `run.sh` script handles this automatically via `apt-get`. If running manually: `sudo apt-get install python3 python3-venv python3-pip`

**`Permission denied` when running run.sh**
Run `chmod +x run.sh` first, then `bash run.sh`.

**Docker container exits immediately**
Make sure to use the `-it` flag: `docker run -it ubuntu:22.04 bash`

**Slow training on CPU**
Training 2000 episodes takes ~5–15 minutes on CPU. This is expected. A GPU will speed it up significantly if available.
