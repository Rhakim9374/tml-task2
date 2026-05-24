#!/usr/bin/env bash
# Download the target model + 360 suspect safetensors from the HF model repo
# via HTTP. The cluster has no git-lfs, so cloning the HF repo doesn't work.
# Idempotent: re-running skips files that already exist with non-zero size.
# Run from anywhere; the script cd's to the code dir on its own.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

mkdir -p target_model suspect_models checkpoints submissions runlogs

BASE="https://huggingface.co/SprintML/tml26_task2/resolve/main"

dl() {
    local url=$1 dst=$2
    if [ -s "$dst" ]; then return 0; fi
    wget -q --tries=5 --continue "$url" -O "$dst" \
        || { rm -f "$dst"; echo "FAILED: $url" >&2; exit 1; }
    echo "  got $dst"
}

echo "==> target"
dl "$BASE/target_model/weights.safetensors"  target_model/weights.safetensors
dl "$BASE/target_model/train_main_idx.json"  target_model/train_main_idx.json

echo "==> 360 suspects"
for i in $(seq -f "%03g" 0 359); do
    dl "$BASE/suspect_models/suspect_${i}.safetensors" \
       "suspect_models/suspect_${i}.safetensors"
done

echo "==> CIFAR datasets (in case worker nodes can't reach public internet)"
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
