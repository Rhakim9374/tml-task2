#!/usr/bin/env bash
# Download target model + CIFAR-100/10 only (~400 MB). The 360 suspect models
# (16 GB) are NOT downloaded here — under tight HPC user quotas we can't
# persistently store them. Each cluster/run_shard.sh job downloads its 30
# suspects into the docker container's ephemeral /tmp at extract time.
# Idempotent: re-running skips files that already exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

mkdir -p target_model checkpoints submissions runlogs

BASE="https://huggingface.co/SprintML/tml26_task2/resolve/main"

dl() {
    local url=$1 dst=$2
    if [ -s "$dst" ]; then return 0; fi
    wget -q --tries=5 --continue "$url" -O "$dst" \
        || { rm -f "$dst"; echo "FAILED: $url" >&2; exit 1; }
    echo "  got $dst"
}

echo "==> target model + indices (~45 MB)"
dl "$BASE/target_model/weights.safetensors"  target_model/weights.safetensors
dl "$BASE/target_model/train_main_idx.json"  target_model/train_main_idx.json

echo "==> CIFAR datasets (~330 MB)"
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

echo "FETCH OK (target + CIFAR present; suspects fetched per-shard at job time)"
