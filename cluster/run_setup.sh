#!/usr/bin/env bash
# Invoked by cluster/setup.sub inside the pytorch docker image.
# Installs Python deps and runs a 3-suspect smoke test (~30 s on V100).
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

pip install --quiet -r requirements.txt

python -m scripts.extract_signals \
    --device cuda \
    --num-suspects 3 \
    --no-ood \
    --no-align \
    --out checkpoints/smoke.csv

echo "SETUP OK"
