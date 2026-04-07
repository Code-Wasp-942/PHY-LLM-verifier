from __future__ import annotations

import argparse
import json
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


def _to_point_map(triggered_points: List[Dict]) -> Dict[str, bool]:
    return {item["point_id"]: bool(item["triggered"]) for item in triggered_points}


def compute_metrics(pred_rows: List[Dict], ref_rows: List[Dict]) -> Dict[str, float]:
    if len(pred_rows) != len(ref_rows):
        raise ValueError("Prediction and reference row counts must match")

    tp = fp = fn = 0
    exact_match = 0
    score_abs_sum = 0.0

    for pred, ref in zip(pred_rows, ref_rows):
        pred_out = pred.get("output", pred)
        ref_out = ref.get("output", ref)

        pred_map = _to_point_map(pred_out["triggered_points"])
        ref_map = _to_point_map(ref_out["triggered_points"])

        all_point_ids = set(pred_map) | set(ref_map)
        for point_id in all_point_ids:
            pred_triggered = pred_map.get(point_id, False)
            ref_triggered = ref_map.get(point_id, False)
            if pred_triggered and ref_triggered:
                tp += 1
            elif pred_triggered and not ref_triggered:
                fp += 1
            elif (not pred_triggered) and ref_triggered:
                fn += 1

        pred_set = {point_id for point_id, triggered in pred_map.items() if triggered}
        ref_set = {point_id for point_id, triggered in ref_map.items() if triggered}
        if pred_set == ref_set:
            exact_match += 1

        score_abs_sum += abs(float(pred_out["final_score"]) - float(ref_out["final_score"]))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": f1,
        "exact_set_match": exact_match / len(ref_rows) if ref_rows else 0.0,
        "score_mae": score_abs_sum / len(ref_rows) if ref_rows else 0.0,
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
