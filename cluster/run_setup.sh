#!/usr/bin/env bash
# Invoked by cluster/setup.sub inside the pytorch docker image. Downloads
# 3 suspects into ephemeral worker scratch and runs a quick smoke test
# (no-ood, no-align) to verify deps + GPU + target/CIFAR access.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

WORK_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}/tml-task2-smoke"
mkdir -p "$WORK_DIR/suspect_models"

# Docker image has no wget; use stdlib urllib via cluster/dl_suspects.py.
python3 cluster/dl_suspects.py \
    --start 0 --end 3 \
    --out-dir "$WORK_DIR/suspect_models"

pip install --quiet -r requirements.txt

python -m scripts.extract_signals \
    --device cuda \
    --suspects-dir "$WORK_DIR/suspect_models" \
    --num-suspects 3 \
    --no-ood \
    --no-align \
    --out checkpoints/smoke.csv

rm -rf "$WORK_DIR"
echo "SETUP OK"
