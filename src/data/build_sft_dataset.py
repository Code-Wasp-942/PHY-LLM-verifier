from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.schema import GradingOutput, GradingTaskInput, SFTSample


PROMPT_TEMPLATE = """You are a strict physics grading assistant.
Decide which rubric points are triggered by the student answer.

Rules:
1) Use only point_id values provided in rubric.
2) Give evidence from the student answer.
3) Output valid JSON only.

[Question]
{question}

[Reference Answer]
{reference_answer}

[Rubric]
{rubric}

[Student Answer]
{student_answer}

Return JSON:
{{
  "triggered_points": [{{"point_id": "...", "triggered": true, "evidence": "...", "confidence": 0.0}}],
  "final_score": 0
}}
"""


def _load_rows(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("train", "data", "samples", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported raw data format: {path}")


def _normalize_row(row: Dict) -> Tuple[GradingTaskInput, GradingOutput]:
    if "input" in row and "output" in row:
        task = GradingTaskInput(question_id=row["question_id"], **row["input"])
        out = GradingOutput(**row["output"])
        return task, out

    question_id = row.get("question_id") or row.get("id") or "unknown"
    task = GradingTaskInput(
        question_id=question_id,
        question=row["question"],
        reference_answer=row["reference_answer"],
        rubric=row["rubric"],
        student_answer=row["student_answer"],
    )
    out = GradingOutput(
        triggered_points=row["triggered_points"],
        final_score=row["final_score"],
    )
    return task, out


def _build_prompt(task: GradingTaskInput) -> str:
    rubric_json = json.dumps([point.model_dump() for point in task.rubric], ensure_ascii=False)
    return PROMPT_TEMPLATE.format(
        question=task.question,
        reference_answer=task.reference_answer,
        rubric=rubric_json,
        student_answer=task.student_answer,
    )


def _build_target(out: GradingOutput) -> str:
    return json.dumps(out.model_dump(), ensure_ascii=False)


def _split_rows(
    rows: List[SFTSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[SFTSample], List[SFTSample], List[SFTSample]]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)

    test_rows = shuffled[:test_size]
    val_rows = shuffled[test_size : test_size + val_size]
    train_rows = shuffled[test_size + val_size :]
    return train_rows, val_rows, test_rows


def _write_jsonl(path: Path, rows: Iterable[SFTSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset from raw VerifyBench-like rows")
    parser.add_argument("--raw", type=Path, required=True, help="Raw .json or .jsonl input")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_rows = _load_rows(args.raw)
    samples: List[SFTSample] = []
    for row in raw_rows:
        task, out = _normalize_row(row)
        sample = SFTSample(
            question_id=task.question_id,
            prompt=_build_prompt(task),
            target=_build_target(out),
            metadata={"question_id": task.question_id},
        )
        samples.append(sample)

    train_rows, val_rows, test_rows = _split_rows(
        rows=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    _write_jsonl(args.out_dir / "train.jsonl", train_rows)
    _write_jsonl(args.out_dir / "val.jsonl", val_rows)
    _write_jsonl(args.out_dir / "test.jsonl", test_rows)

    print(
        json.dumps(
            {
                "total": len(samples),
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
