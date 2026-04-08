# LLM Physics Grader

This repo contains a starter pipeline for fine-tuning a Qwen 8B model on VerifyBench-style QA data.

## Goal

Given:

- question
- reference answer

Predict:

- final answer text

## Project Layout

- `configs/`: model and training configs
- `src/schema.py`: VerifyBench schema definitions
- `src/data/build_sft_dataset.py`: converts VerifyBench rows into SFT JSONL
- `src/train/run_sft.py`: full-parameter SFT training entry
- `src/eval/generate_predictions.py`: generation script for test set
- `src/eval/metrics.py`: text exact-match metrics
- `scripts/`: helper scripts for data prep, train, eval

## Quick Start

1. Create data:

```bash
bash scripts/01_prepare_data.sh
```

2. Launch 6-GPU SFT:

```bash
bash scripts/02_train_sft.sh
```

3. Run evaluation:

```bash
bash scripts/04_eval.sh
```

## Data Format

Processed `jsonl` rows look like:

```json
{
  "question_id": "Q_001",
  "prompt": "You are an accurate problem-solving assistant...",
  "target": "reference answer text",
  "metadata": {
    "completion_id": "...",
    "source": "..."
  }
}
```
