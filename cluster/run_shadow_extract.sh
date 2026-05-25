#!/usr/bin/env bash
# Sharded feature extraction over the shadow set. Crucially, target is the
# REAL target_model — same as for the actual suspects — so the shadow feature
# distribution and the real suspect feature distribution are produced by an
# identical pipeline (the meta-classifier never has to cross a target gap).
#
# Args: $1 = SHARD_IDX, $2 = NUM_SHARDS, $3 = TOTAL_SHADOWS
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

SHARD_IDX=$1
NUM_SHARDS=$2
TOTAL=$3
START=$(( SHARD_IDX * TOTAL / NUM_SHARDS ))
END=$(( (SHARD_IDX + 1) * TOTAL / NUM_SHARDS ))

# Backward-compat: an earlier build_shadow_jobs.py named shadow outputs
# shadow_NNN.safetensors, but extract_signals.py expects suspect_NNN. Rename
# any legacy files in place; idempotent — no-op for fresh runs.
for f in shadows/suspects/shadow_*.safetensors; do
    [ -e "$f" ] || continue
    mv "$f" "${f/shadow_/suspect_}"
done

pip install --quiet -r requirements.txt

python -m scripts.extract_signals \
    --device cuda \
    --batch-size 512 \
    --target-path target_model/weights.safetensors \
    --target-dir target_model \
    --suspects-dir shadows/suspects \
    --num-suspects "$TOTAL" \
    --start "$START" \
    --end "$END" \
    --out "checkpoints/shadow_features_shard_${SHARD_IDX}.csv"

echo "SHADOW SHARD ${SHARD_IDX}/${NUM_SHARDS} OK ([$START, $END))"
