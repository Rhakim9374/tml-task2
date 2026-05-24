#!/usr/bin/env bash
# Download the target model + 360 suspect safetensors from the HF model repo
# via HTTP (cluster has no git-lfs). Idempotent: re-running skips files that
# already exist with non-zero size, so it is safe to interrupt and resume.
# Concurrency tuned with PARALLEL env var (default 16).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

PARALLEL="${PARALLEL:-16}"

mkdir -p target_model suspect_models checkpoints submissions runlogs

BASE="https://huggingface.co/SprintML/tml26_task2/resolve/main"

dl() {
    local url=$1 dst=$2
    if [ -s "$dst" ]; then return 0; fi
    wget -q --tries=5 --continue "$url" -O "$dst" \
        || { rm -f "$dst"; echo "FAILED: $url" >&2; return 1; }
    echo "  got $dst"
}
export BASE
export -f dl

echo "==> target"
dl "$BASE/target_model/weights.safetensors"  target_model/weights.safetensors
dl "$BASE/target_model/train_main_idx.json"  target_model/train_main_idx.json

echo "==> 360 suspects (parallel x $PARALLEL)"
seq -f "%03g" 0 359 | xargs -P "$PARALLEL" -I {} bash -c \
    'dl "$BASE/suspect_models/suspect_{}.safetensors" "suspect_models/suspect_{}.safetensors"'

echo "==> CIFAR datasets (worker nodes may not reach public internet)"
if [ ! -d cifar100_data/cifar-100-python ]; then
    mkdir -p cifar100_data
    dl "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" cifar100_data/cifar-100-python.tar.gz
    tar -xzf cifar100_data/cifar-100-python.tar.gz -C cifar100_data/
fi
if [ ! -d cifar10_data/cifar-10-batches-py ]; then
    mkdir -p cifar10_data
    dl "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" cifar10_data/cifar-10-python.tar.gz
    tar -xzf cifar10_data/cifar-10-python.tar.gz -C cifar10_data/
fi

n=$(ls suspect_models/suspect_*.safetensors 2>/dev/null | wc -l | tr -d ' ')
echo "FETCH OK ($n/360 suspect files present)"
