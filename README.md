# LLM Physics Grader

This repo contains a starter pipeline for fine-tuning a Qwen 8B model for physics rubric triggering.

## Goal

Given:

- question
- reference answer
- rubric points
- student answer

Predict:

- which rubric points are triggered
- final score

## Project Layout

- `configs/`: model and training configs
- `src/schema.py`: strict input/output schema
- `src/data/build_sft_dataset.py`: converts VerifyBench-like data into SFT JSONL
- `src/train/run_sft.py`: full-parameter SFT training entry
- `src/eval/metrics.py`: point-level metrics
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
  "input": {
    "question": "...",
    "reference_answer": "...",
    "rubric": [
      {"point_id": "P1", "desc": "...", "score": 2, "requires": []}
    ],
    "student_answer": "..."
  },
  "output": {
    "triggered_points": [
      {"point_id": "P1", "triggered": true, "evidence": "...", "confidence": 0.95}
    ],
    "final_score": 2
  }
}
```
