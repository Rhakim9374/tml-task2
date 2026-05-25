#!/usr/bin/env bash
# Run one shadow-suspect job (indexed by HTCondor $(Process)). Looks the
# cmd up in shadows/jobs.json via scripts/run_shadow_job.py.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

JOB_IDX=$1

pip install --quiet -r requirements.txt

python -m scripts.run_shadow_job --job-idx "$JOB_IDX" --jobs-json shadows/jobs.json
