#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash scripts/04_eval.sh outputs/predictions.jsonl data/processed/test.jsonl

PRED_FILE=${1:-outputs/predictions.jsonl}
REF_FILE=${2:-data/processed/test.jsonl}

python -m src.eval.metrics --pred "${PRED_FILE}" --ref "${REF_FILE}"
