#!/usr/bin/env bash
set -euo pipefail

# Launch full-parameter SFT on 6 GPUs.
torchrun --nproc_per_node=6 --master_port=29501 \
  -m src.train.run_sft \
  --config configs/train/sft.yaml
