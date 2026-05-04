#!/usr/bin/env bash
# =============================================================================
#  run.sh — Full pipeline for RL Credit Limit Optimization
#  Compatible with: Ubuntu 22.04 (clean Docker environment)
#  Usage: bash run.sh
# =============================================================================

set -euo pipefail   # exit on error, undefined var, or pipe failure

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[run.sh]${NC} $*"; }
warn() { echo -e "${YELLOW}[run.sh]${NC} $*"; }
die()  { echo -e "${RED}[run.sh] ERROR:${NC} $*" >&2; exit 1; }

# ── 0. Locate project root (directory containing this script) ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
log "Working directory: $SCRIPT_DIR"

# ── 1. System-level prerequisites (Python 3, venv, pip) ───────────────────────
log "Checking system prerequisites …"

if ! command -v python3 &>/dev/null; then
    warn "python3 not found — attempting apt install …"
    apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Python version: $PYTHON_VERSION"

# Ensure venv module is available
if ! python3 -m venv --help &>/dev/null; then
    warn "python3-venv missing — installing …"
    apt-get update -qq && apt-get install -y -qq python3-venv
fi

# ── 2. Create virtual environment ─────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR — reusing."
else
    log "Creating virtual environment at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "Virtual environment activated: $(which python)"

# ── 3. Upgrade pip & install dependencies ─────────────────────────────────────
log "Upgrading pip …"
pip install --upgrade pip --quiet

log "Installing dependencies from requirements.txt …"
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# Verify key imports
python -c "import numpy, pandas, sklearn, torch, matplotlib" \
    || die "One or more required packages failed to import."
log "All dependencies installed and importable ✓"

# ── 4. Create output directory ─────────────────────────────────────────────────
OUTPUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUTPUT_DIR"
log "Outputs will be saved to: $OUTPUT_DIR"

# ── 5. Run the training pipeline ──────────────────────────────────────────────
log "Starting training pipeline (2000 episodes) …"
log "This may take several minutes on CPU."

python "$SCRIPT_DIR/train.py" \
    --episodes 2000 \
    --eval-every 100 \
    --warmup-episodes 50 \
    --seed 42 \
    --save "$OUTPUT_DIR/model.pt"

# ── 6. Move artefacts to outputs/ ─────────────────────────────────────────────
log "Collecting output artefacts …"

for f in learning_curve.png baseline_comparison.png history.json; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        mv "$SCRIPT_DIR/$f" "$OUTPUT_DIR/$f"
        log "  Moved $f → outputs/$f"
    fi
done

# ── 7. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Pipeline completed successfully ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output artefacts:"
ls -lh "$OUTPUT_DIR"
echo ""
log "Done."
