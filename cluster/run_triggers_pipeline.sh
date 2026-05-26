#!/usr/bin/env bash
# Boundary-trigger pipeline. Single ~35-45 min GPU job that produces
# submissions/triggers_ood_k500.csv — the dominant component in the blend
# that scored 0.703704.
#
# Stages:
#   1. Download all 360 real suspects into ephemeral worker scratch
#   2. Save argmax predictions for every SHADOW on train+ood+holdout+test
#      (~65k probes per shadow)
#   3. Save argmax predictions for every REAL suspect on the same 65k probes
#   4. Mine the top-500 most-discriminative OOD-probe triggers from shadows
#      (probes where 'suspect_pred == target_pred' has high AUC vs the binary
#      shadow label; weight each trigger by |AUC - 0.5|)
#   5. Score each real suspect by its weighted-mean trigger response
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

pip install --quiet -r requirements.txt

# --- Stage 1: download all 360 real suspects to ephemeral worker scratch ---
REAL_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}/tml-task2-real-suspects"
mkdir -p "$REAL_DIR"
echo "[stage1] downloading 360 real suspects to $REAL_DIR ..."
python3 cluster/dl_suspects.py --start 0 --end 360 --out-dir "$REAL_DIR" || true
N_GOT=$(ls "$REAL_DIR" 2>/dev/null | wc -l)
echo "[stage1] got $N_GOT / 360 suspects"
if [ "$N_GOT" -lt 350 ]; then
    echo "FATAL: only $N_GOT suspects downloaded (need >= 350); aborting" >&2
    exit 1
fi

# Save argmax predictions on all four probe splits so find_triggers can pick
# any subset; we use the OOD slice but generating the full matrix is cheap.
PROBE_SPLITS="train,ood,holdout,test"

# --- Stage 2: shadow predictions ---
python -m scripts.extract_predictions \
    --suspects-dir shadows/suspects \
    --num-suspects 195 \
    --splits "$PROBE_SPLITS" \
    --out shadows/predictions.npz

# --- Stage 3: real predictions (ephemeral suspect dir) ---
python -m scripts.extract_predictions \
    --suspects-dir "$REAL_DIR" \
    --num-suspects 360 \
    --splits "$PROBE_SPLITS" \
    --out real_predictions.npz

# --- Stage 4: mine top-500 OOD-probe triggers from the shadow predictions ---
python -m scripts.find_triggers \
    --shadow-preds shadows/predictions.npz \
    --labels shadows/labels.csv \
    --include-splits ood \
    --top-k 500 \
    --out shadows/triggers_ood.json

# --- Stage 5: score real suspects with the top-500 triggers ---
python -m scripts.score_with_triggers \
    --real-preds real_predictions.npz \
    --triggers shadows/triggers_ood.json \
    --top-k 500 \
    --out submissions/triggers_ood_k500.csv

# Free ephemeral suspects
rm -rf "$REAL_DIR"

echo "TRIGGER PIPELINE OK"
