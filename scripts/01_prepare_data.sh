#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash scripts/01_prepare_data.sh data/raw/verifybench.json

RAW_FILE=${1:-data/raw/verifybench.json}

python -m src.data.build_sft_dataset \
  --raw "${RAW_FILE}" \
  --out-dir data/processed \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
