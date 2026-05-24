#!/usr/bin/env bash
# Run on the login node (CPU only, ~1 s). Concatenates all
# checkpoints/features_shard_*.csv, rank-averages, and writes
# submissions/submission.csv.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

python3 -c 'import pandas' 2>/dev/null || pip3 install --user --quiet pandas

python3 -m scripts.combine_and_submit \
    --features "checkpoints/features_shard_*.csv" \
    --out submissions/submission.csv

echo "COMBINE OK"
