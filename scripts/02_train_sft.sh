#!/usr/bin/env bash
set -euo pipefail

# Launch full-parameter SFT on 6 GPUs.
RESUME_FROM_CHECKPOINT=${1:-}

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

torchrun --nproc_per_node=6 --master_port=29501 \
  -m src.train.run_sft \
  --config configs/train/sft.yaml \
  --model-config configs/model/qwen2_5_8b_sft.yaml \
  "${EXTRA_ARGS[@]}"
