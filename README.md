# TML26 Task 2 — reproducing our best submission (TPR@5%FPR = **0.703704**)

This repository contains exactly the code that produced
`submissions/blend_v5_a.csv`, the submission that scored **0.703704** on the leaderboard.

`blend_v5_a.csv` is a rank-average blend of three component submissions:

| Weight | Component CSV                       | Produced by                                                      |
|-------:|-------------------------------------|------------------------------------------------------------------|
| 1.5    | `submissions/triggers_ood_k500.csv` | Boundary triggers mined from shadows (`cluster/triggers.sub`)    |
| 1.0    | `submissions/submission_meta_lr.csv`| L2 logistic regression on extracted features (shadow-trained)    |
| 0.5    | `submissions/heur_v4_max.csv`       | Heuristic max-rank ensemble of six signal groups                 |

---

## 1. One-time setup on the cluster

The cluster has no `git-lfs`, so the target model and CIFAR data are fetched
over HTTP. Suspect safetensors are *never persisted* — each GPU job downloads
the slice it needs into its ephemeral `$_CONDOR_SCRATCH_DIR`.

```bash
ssh <atml_teamXXX>@conduit2.hpc.uni-saarland.de
git clone https://github.com/Rhakim9374/tml-task2.git code && cd code

# A virtualenv on the login node; only used for the small non-job steps
# (build_shadow_jobs, train_meta_classifier, combine_and_submit, blend, submit).
# Cluster jobs install requirements.txt themselves at the top of each script.
python3 -m venv ~/.tml-venv
~/.tml-venv/bin/pip install -r requirements.txt

# ~400 MB persistent: target weights + CIFAR-100 + CIFAR-10
bash cluster/fetch_data.sh
```

---

## 2. The exact ten-command recipe

```bash
cd ~/code

# --- Component A: heuristic max-rank ensemble --------------------------------

# (1) Extract every per-suspect feature for the 360 real suspects.
# 18 shards × 20 suspects each, each shard downloads its slice to /tmp.
# ~15-25 min wall on 18 GPUs.
condor_submit cluster/extract.sub
# → checkpoints/features_shard_{0..17}.csv

# --- Components B + C need 195 shadow models with known labels --------------

# (2) Generate the v4 shadow plan: 50 independent + 25 evil_twin
#     + 15 near_target_indep + 15 fine_tune + 25 partial_finetune + 30 distill
#     + 15 mixed_kd + 10 noise + 10 quant = 195 shadows.
~/.tml-venv/bin/python -m scripts.build_shadow_jobs
# → shadows/jobs.json, shadows/labels.csv

# (3) Train all 195 shadow ResNet-18s in parallel.
# ~30-45 min wall on 18 GPUs.
condor_submit cluster/shadow_suspects.sub
# → shadows/suspects/suspect_{000..194}.safetensors

# (4) Extract per-shadow features against the REAL target (same pipeline as
#     step 1, so shadow + real distributions are produced identically).
# 15 shards × 13 shadows each. ~5-7 min wall.
condor_submit cluster/shadow_extract.sub
# → checkpoints/shadow_features_shard_{0..14}.csv

# --- Component B: L2 logistic regression meta-classifier ---------------------

# (5) Fit LR on shadow features (with 5-fold CV report) and predict on real.
~/.tml-venv/bin/python -m scripts.train_meta_classifier
# → submissions/submission_meta_lr.csv

# --- Component C: boundary triggers ------------------------------------------

# (6) Save argmax predictions on train+ood+holdout+test for every shadow and
#     for every real suspect (downloaded ephemerally), mine the top-500 most-
#     discriminative OOD triggers (by |AUC - 0.5| of "suspect_pred == target_pred"
#     across shadows), then score each real suspect by its weighted-mean
#     trigger response. ~35-45 min on one GPU.
condor_submit cluster/triggers.sub
# → submissions/triggers_ood_k500.csv

# --- Component A finalize ----------------------------------------------------

# (7) Heuristic max-rank ensemble of six signal groups (S2 dropped — its
#     train/holdout gap is too sample-dependent to add over rank-aware groups).
~/.tml-venv/bin/python -m scripts.combine_and_submit \
    --features "checkpoints/features_shard_*.csv" \
    --groups "S1,S1f,S3,S4,S5,S6" \
    --out submissions/heur_v4_max.csv

# --- Final blend -------------------------------------------------------------

# (8) Rank-average the three component CSVs with the winning weights.
~/.tml-venv/bin/python -m scripts.blend_submissions \
    --inputs "submissions/triggers_ood_k500.csv:1.5,submissions/submission_meta_lr.csv:1.0,submissions/heur_v4_max.csv:0.5" \
    --out submissions/blend_v5_a.csv

# (9) Submit to the leaderboard.
export TML_API_KEY=<your api key>
~/.tml-venv/bin/python -m scripts.submit --file submissions/blend_v5_a.csv
```

If any job is `Held`, run `condor_q -hold <id>` for the reason
(usually a missing requirements install on a fresh worker — re-queue and the
in-script `pip install -r requirements.txt` fixes it).

---

## 3. What each signal computes

The feature extractor (`scripts/extract_signals.py`, called from steps 1 and 4)
computes 7 groups against the target on four probe splits — the target's 40k
training indices, the 10k holdout indices, the 10k CIFAR-100 test set, and a
fixed 5k CIFAR-10 OOD probe (`seed=0`):

| Group | Catches                                                  | Sub-features (per probe split unless noted)                                    |
|-------|----------------------------------------------------------|--------------------------------------------------------------------------------|
| S1    | Distillation, fine-tuning, function-preserving copies    | cosine / -KL / -L2 on softmax / top-1 agreement                                |
| S1f   | Inherited mistakes and confusion structure               | Agreement on target's mistakes; top-3/top-5 overlap; low-confidence agreement  |
| S2    | Memorization-driven fine-tunes                           | Train vs holdout vs test loss/conf gaps (one suspect-level row)                |
| S3    | Direct copies, permuted/scaled copies, light fine-tunes  | Raw L2/cos; sorted-SVD spectrum distance; L2/cos after Hungarian channel align |
| S4    | Function-preserving channel permutations, light tuning   | Linear CKA at stem / layer1..4 / penultimate / mean                            |
| S5    | Similar local decision boundary                          | Input-gradient cosine; FGSM ε=0.06 adversarial transfer rate                   |
| S6    | ModelDiff decision-distance fingerprint                  | DDV correlation / cosine / flip-agreement (K=4 random directions)              |

The heuristic ensemble (step 7) drops S2 because — on a real-suspect
distribution we can't whiten with a held-out set — its absolute gaps were
noisier than the rank-aware groups. The LR meta-classifier (step 5) uses
*all* groups including S2, because it whitens internally via `StandardScaler`.

---

## 4. Repository layout

```
src/
  data.py                    CIFAR-100 train_main / holdout / test splits +
                             target's exact BiasedRandomCrop augmentation
  model.py                   make_model() and safetensors loader
  signals/
    output_agreement.py      S1 + S1f
    dataset_inference.py     S2
    weight_align.py          S3 (raw + SVD + activation-aligned permutation)
    cka.py                   S4
    decision_boundary.py     S5
    ddv.py                   S6

scripts/
  extract_signals.py         One forward pass per suspect → row of S1..S6 features
  build_shadow_jobs.py       Writes shadows/jobs.json and shadows/labels.csv
  train_shadow.py            Trains one ResNet-18 shadow (train / distill / mixed_kd)
  derive_shadow.py           Function-preserving derivatives (noise / quant)
  run_shadow_job.py          Dispatcher: looks up job N in shadows/jobs.json and runs it
  extract_predictions.py     Saves per-suspect argmax predictions on probe sets
  find_triggers.py           Mines top-K |AUC - 0.5| triggers from shadow predictions
  score_with_triggers.py     Scores real suspects via the trigger weights
  train_meta_classifier.py   L2 logistic regression on extracted features
  combine_and_submit.py      Heuristic max-rank ensemble across feature groups
  blend_submissions.py       Rank-average blend of submission CSVs
  submit.py                  POST a submission CSV to the leaderboard

cluster/
  fetch_data.sh              HTTP download of target + CIFAR-100/10
  dl_suspects.py             stdlib urllib parallel downloader
  extract.sub / run_shard.sh                Real-suspect feature extract (18 shards)
  shadow_suspects.sub / run_shadow_job.sh   Shadow training (195 jobs)
  shadow_extract.sub / run_shadow_extract.sh  Shadow feature extract (15 shards)
  triggers.sub / run_triggers_pipeline.sh   End-to-end boundary trigger pipeline

requirements.txt
```

Cluster job scripts install `requirements.txt` themselves; the login-node
venv at `~/.tml-venv` only needs the same `requirements.txt` for the four
non-job commands (build_shadow_jobs, train_meta_classifier, combine_and_submit,
blend_submissions, submit).
