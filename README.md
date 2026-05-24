# TML26 task 2 — stolen-model detection

Multi-signal ensemble that scores each of 360 suspect CIFAR-100 ResNet-18s on
how likely it is to be stolen from / derived from the given target model.

## Approach (one paragraph)

Per suspect we extract four complementary signals against the target:
**S1** output agreement (KL / cosine / top-1 of logits) on the target's 40k
training indices, 10k holdout, the official CIFAR-100 test set, and 5k OOD
(CIFAR-10) images; **S2** dataset-inference style train/holdout/test loss and
confidence gaps; **S3** weight-space distance — raw L2/cosine, sorted-SVD
spectrum distance, and L2/cosine after an activation-matched permutation
alignment of channels; **S4** linear CKA at the stem, each ResNet stage, and
the 512-D penultimate feature. We rank-normalize every sub-feature across the
360 suspects, average within each signal group, then rank-average across the
four groups to produce the final score.

## Reproduce the leaderboard submission

```bash
# 1. clone + pull LFS weights (~16 GB total: 360 × ~45 MB suspects + target)
git clone <repo>
cd tml26_task2
git lfs pull

# 2. install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. extract per-suspect features (1 forward pass per suspect over ~65k probes)
python -m scripts.extract_signals \
    --device cuda \
    --batch-size 512 \
    --out checkpoints/features.csv

# 4. combine to a single score and write submissions/submission.csv
python -m scripts.combine_and_submit \
    --features checkpoints/features.csv \
    --out submissions/submission.csv

# 5. upload (60-min cooldown between successful submissions)
export TML_API_KEY=<your key>
python -m scripts.submit --file submissions/submission.csv
```

## Repository layout

```
src/
  data.py                    CIFAR-100 splits + biased crop matching target recipe
  model.py                   make_model() and safetensors loader
  signals/
    output_agreement.py      S1 — KL / cosine / top-1 on target vs suspect logits
    dataset_inference.py     S2 — train/holdout/test loss + confidence gaps
    weight_align.py          S3 — raw + permutation-aligned + SVD-spectrum
    cka.py                   S4 — linear CKA at stem / layer1..4 / penultimate
scripts/
  extract_signals.py         one GPU pass per suspect; writes features.csv
  combine_and_submit.py      rank-average within and across signal groups
  submit.py                  POST submission.csv to the leaderboard
target_model/
  weights.safetensors        ResNet-18 target (LFS)
  train_main_idx.json        40k CIFAR-100 indices the target was trained on
suspect_models/
  suspect_000.safetensors    360 suspect ResNet-18s (LFS)
  …
```

## Notes

- All weight files are git-lfs pointers locally; you must `git lfs pull` on the
  cluster before running.
- The pipeline is single-GPU. A V100 takes ≈10 s per suspect including
  alignment (~1 h end-to-end for all 360).
- `--no-ood` and `--no-align` flags exist for faster smoke tests but are not
  recommended for the final submission.
