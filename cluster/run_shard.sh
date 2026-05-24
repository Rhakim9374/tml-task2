#!/usr/bin/env bash
# Invoked by cluster/extract.sub inside the pytorch docker image.
# Processes a contiguous slice of suspects on one GPU and writes a per-shard
# features CSV. The slice is [SHARD_IDX * 360 / NUM_SHARDS,
#                            (SHARD_IDX + 1) * 360 / NUM_SHARDS).
# Args: $1 = SHARD_IDX (0..NUM_SHARDS-1)   $2 = NUM_SHARDS
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

SHARD_IDX=$1
NUM_SHARDS=$2
TOTAL=360
START=$(( SHARD_IDX * TOTAL / NUM_SHARDS ))
END=$(( (SHARD_IDX + 1) * TOTAL / NUM_SHARDS ))

pip install --quiet -r requirements.txt

python -m scripts.extract_signals \
    --device cuda \
    --batch-size 512 \
    --start "$START" \
    --end "$END" \
    --out "checkpoints/features_shard_${SHARD_IDX}.csv"

echo "SHARD ${SHARD_IDX}/${NUM_SHARDS} OK ([$START, $END))"
