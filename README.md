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

## Reproduce on the CISPA HPC cluster (the supported path)

The cluster has no `git-lfs`, so we fetch the safetensors via plain HTTP from
the HF model repo. Everything below is committed in `cluster/`; on the cluster
you only need `git pull` + four commands.

```bash
ssh <atml_teamXXX>@conduit2.hpc.uni-saarland.de
cd ~ && git clone https://github.com/Rhakim9374/tml-task2.git code && cd code

# 1. download target + 360 suspect safetensors (~16 GB, ~10-20 min, idempotent)
bash cluster/fetch_data.sh

# 2. install deps + smoke test on 3 suspects (~2 min total)
condor_submit cluster/setup.sub        # note the ClusterId it prints
condor_q                                # wait until done (Idle → Running → gone)
cat runlogs/setup.<ClusterId>.out       # expect: "SETUP OK"

# 3. full extract + combine (~1 h on V100)
condor_submit cluster/extract.sub
condor_q
tail -f runlogs/extract.<ClusterId>.out

# 4. submit to leaderboard
export TML_API_KEY=<your key>
python -m scripts.submit --file submissions/submission.csv
```

If a `condor_q` shows the job `Held`, run `condor_q -hold <id>` for the reason.

## Reproduce locally (laptop, no cluster)

```bash
git clone https://github.com/Rhakim9374/tml-task2.git && cd tml-task2
# manually drop target_model/ and suspect_models/ alongside the repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.extract_signals    --device cuda --batch-size 512 --out checkpoints/features.csv
python -m scripts.combine_and_submit --features checkpoints/features.csv  --out submissions/submission.csv
TML_API_KEY=<key> python -m scripts.submit --file submissions/submission.csv
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
cluster/
  fetch_data.sh              HTTP download of target + 360 suspects (no git-lfs)
  run_setup.sh / setup.sub   one-off HTCondor job: pip install + smoke test
  run_extract.sh / extract.sub  HTCondor job: full extract + combine (~1 h)
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
