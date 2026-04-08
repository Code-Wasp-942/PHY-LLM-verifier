#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash scripts/03_generate_pred.sh checkpoints/sft_full data/processed/test.jsonl outputs/predictions.jsonl

MODEL_PATH=${1:-checkpoints/sft_full}
INPUT_FILE=${2:-data/processed/test.jsonl}
OUTPUT_FILE=${3:-outputs/predictions.jsonl}

python -m src.eval.generate_predictions \
  --model "${MODEL_PATH}" \
  --input "${INPUT_FILE}" \
  --output "${OUTPUT_FILE}" \
  --trust-remote-code
