#!/usr/bin/env bash
# Invoked by cluster/extract.sub inside the pytorch docker image. Downloads
# this shard's 30 suspect safetensors into ephemeral worker scratch, then
# runs extract_signals over them. Suspects are NOT persisted — the container
# is destroyed when the job ends, so the 1.5 GB disappears with it.
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

# Per-shard ephemeral workspace inside the container. _CONDOR_SCRATCH_DIR is
# the HTCondor-provided job scratch; if it's unset for any reason, fall back
# to /tmp (also worker-local + ephemeral inside docker).
WORK_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}/tml-task2-shard-${SHARD_IDX}"
mkdir -p "$WORK_DIR/suspect_models"
echo "WORK_DIR=$WORK_DIR  (free: $(df -h "$WORK_DIR" | tail -1 | awk '{print $4}'))"

# Download only this shard's slice of suspects (~30 × 45 MB ≈ 1.4 GB)
BASE="https://huggingface.co/SprintML/tml26_task2/resolve/main"
for i in $(seq -f "%03g" "$START" $((END - 1))); do
    dst="$WORK_DIR/suspect_models/suspect_${i}.safetensors"
    if [ ! -s "$dst" ]; then
        wget -q --tries=5 "$BASE/suspect_models/suspect_${i}.safetensors" -O "$dst" \
            || { rm -f "$dst"; echo "FAILED: $i" >&2; exit 1; }
    fi
done
echo "downloaded $(ls "$WORK_DIR/suspect_models" | wc -l) suspects for shard $SHARD_IDX"

pip install --quiet -r requirements.txt

python -m scripts.extract_signals \
    --device cuda \
    --batch-size 512 \
    --suspects-dir "$WORK_DIR/suspect_models" \
    --start "$START" \
    --end "$END" \
    --out "checkpoints/features_shard_${SHARD_IDX}.csv"

# Free the ephemeral suspects so other jobs on this worker can use the space
rm -rf "$WORK_DIR"

echo "SHARD ${SHARD_IDX}/${NUM_SHARDS} OK ([$START, $END))"
