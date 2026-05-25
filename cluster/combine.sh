#!/usr/bin/env bash
# Run on the login node (CPU only, ~1 s). Concatenates all
# checkpoints/features_shard_*.csv, rank-averages, and writes
# submissions/submission.csv.
#
# Uses a venv at ~/.tml-venv to satisfy PEP 668 cleanly (system Python on
# Debian 12+ is externally managed). To remove everything this script
# installs: rm -rf ~/.tml-venv.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

VENV="$HOME/.tml-venv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -c 'import pandas' 2>/dev/null \
    || "$VENV/bin/pip" install --quiet pandas

"$VENV/bin/python" -m scripts.combine_and_submit \
    --features "checkpoints/features_shard_*.csv" \
    --out submissions/submission.csv

echo "COMBINE OK"
