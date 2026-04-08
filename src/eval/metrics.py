from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_pair(pred: Dict, ref: Dict) -> tuple[str, str]:
    predicted = pred.get("predicted_answer")
    if predicted is None:
        predicted = pred.get("output")
    if predicted is None:
        predicted = ""

    reference = ref.get("target")
    if reference is None:
        reference = ref.get("reference_answer")
    if reference is None:
        reference = ""

    return str(predicted), str(reference)


def compute_metrics(pred_rows: List[Dict], ref_rows: List[Dict]) -> Dict[str, float]:
    if len(pred_rows) != len(ref_rows):
        raise ValueError("Prediction and reference row counts must match")

    exact_match = 0
    normalized_match = 0

    for pred, ref in zip(pred_rows, ref_rows):
        predicted, reference = _extract_pair(pred, ref)
        if predicted == reference:
            exact_match += 1

        if _normalize_text(predicted) == _normalize_text(reference):
            normalized_match += 1

    total = len(ref_rows)

    return {
        "exact_match": exact_match / total if total else 0.0,
        "normalized_match": normalized_match / total if total else 0.0,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute grading metrics")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    args = parser.parse_args()

    pred_rows = _load_jsonl(args.pred)
    ref_rows = _load_jsonl(args.ref)
    results = compute_metrics(pred_rows=pred_rows, ref_rows=ref_rows)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
