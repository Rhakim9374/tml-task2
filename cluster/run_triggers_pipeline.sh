#!/usr/bin/env bash
# Comprehensive overnight boundary-trigger pipeline. Single ~35-45 min GPU job.
#
# Stages:
#   1. Download all 360 real suspects into ephemeral worker scratch
#   2. Save argmax predictions for every shadow on ALL 4 splits (train + ood +
#      holdout + test = 65k probes per suspect)
#   3. Save argmax predictions for every real suspect on the same 65k probes
#   4. Mine boundary triggers using SIX probe-set subsets:
#        - ood only         (5k probes, distillation-sensitive)
#        - holdout only     (10k probes, dataset-inference adjacent)
#        - test only        (10k probes, generalization-sensitive)
#        - train only       (40k probes, memorization-sensitive)
#        - ood+holdout+test (25k probes, all eval-time inputs)
#        - all              (65k probes, max coverage)
#   5. Score real suspects with each trigger set at TWO top-K (100, 500)
#      → 12 candidate submission CSVs in submissions/
#
# After job finishes the morning workflow is just:
#   show_distributions on all 12 + meta_lr to pick the best.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

pip install --quiet -r requirements.txt
pip install --quiet scikit-learn

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

# All four splits — gives find_triggers maximum candidate-probe diversity.
PROBE_SPLITS="train,ood,holdout,test"

# --- Stage 2: shadow predictions on the persistent ~/code/shadows/suspects ---
python -m scripts.extract_predictions \
    --suspects-dir shadows/suspects \
    --num-suspects 195 \
    --splits "$PROBE_SPLITS" \
    --out shadows/predictions.npz

# --- Stage 3: real-suspect predictions on the freshly-downloaded ephemeral dir ---
python -m scripts.extract_predictions \
    --suspects-dir "$REAL_DIR" \
    --num-suspects 360 \
    --splits "$PROBE_SPLITS" \
    --out real_predictions.npz

# --- Stage 4: find triggers for each of six probe-set subsets ---
declare -A SPLIT_SETS=(
    [ood]="ood"
    [holdout]="holdout"
    [test]="test"
    [train]="train"
    [eval]="ood,holdout,test"
    [all]="train,ood,holdout,test"
)
for variant in ood holdout test train eval all; do
    python -m scripts.find_triggers \
        --shadow-preds shadows/predictions.npz \
        --labels shadows/labels.csv \
        --include-splits "${SPLIT_SETS[$variant]}" \
        --top-k 1000 \
        --out "shadows/triggers_${variant}.json"
done

# --- Stage 5: score real suspects with each trigger set at two K values ---
for variant in ood holdout test train eval all; do
    for K in 100 500; do
        python -m scripts.score_with_triggers \
            --real-preds real_predictions.npz \
            --triggers "shadows/triggers_${variant}.json" \
            --top-k "$K" \
            --out "submissions/triggers_${variant}_k${K}.csv"
    done
done

# Free ephemeral suspects so we don't fill the worker disk for other jobs
rm -rf "$REAL_DIR"

# Quick summary table at the end of the log
echo ""
echo "==== Trigger pipeline summary ===="
for f in submissions/triggers_*.csv; do
    name=$(basename "$f" .csv)
    echo -n "$name: "
    python3 -c "
import pandas as pd
d = pd.read_csv('$f')['score']
print(f\"mean={d.mean():.3f}  std={d.std():.3f}  min={d.min():.3f}  p25={d.quantile(.25):.3f}  p50={d.quantile(.5):.3f}  p75={d.quantile(.75):.3f}  max={d.max():.3f}\")
"
done

echo "TRIGGER PIPELINE OK"
