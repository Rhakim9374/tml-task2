#!/usr/bin/env bash
# Invoked by cluster/extract.sub inside the pytorch docker image.
# Full S1+S2+S3+S4 extraction over all 360 suspects, then rank-average combine.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

pip install --quiet -r requirements.txt

python -m scripts.extract_signals \
    --device cuda \
    --batch-size 512 \
    --out checkpoints/features.csv

python -m scripts.combine_and_submit \
    --features checkpoints/features.csv \
    --out submissions/submission.csv

echo "EXTRACT+COMBINE OK"
