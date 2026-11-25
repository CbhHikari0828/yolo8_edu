"""Choose a Varroa classifier and decision threshold using validation data."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from classification import build_metrics


NAMES = ["mite", "no_mite"]
THRESHOLDS = [index / 40 for index in range(12, 29)]


def metrics_at_threshold(samples: dict[str, tuple[str, float]], threshold: float) -> dict:
    matrix = np.zeros((2, 2), dtype=int)
    for actual, mite_score in samples.values():
        matrix[NAMES.index(actual), int(mite_score < threshold)] += 1
    return build_metrics(matrix, NAMES, "val")


def read_validation_predictions(path: Path) -> dict[str, tuple[str, float]]:
    samples = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            image = row["image"]
            if not image.startswith(f"val/{row['actual']}/") or image in samples:
                raise ValueError(f"Invalid or duplicate validation image: {image}")
            probability = float(row["mite_probability"])
            if not 0 <= probability <= 1:
                raise ValueError(f"Invalid mite probability: {probability}")
            samples[image] = (row["actual"], probability)
    return samples


def rank(entry: dict) -> tuple:
    report = entry["metrics"]
    return (
        report["accuracy"],
        report["macro_f1"],
        report["mite_recall"],
        -abs(entry["threshold"] - 0.5),
    )


def choose(runs: Path, baseline: str, candidates: list[str]) -> dict:
    candidate_reports = []
    original = None
    original_labels = None
    recall_floor = None
    for run_id in [baseline, *candidates]:
        report_dir = runs / "optimization-val" / run_id
        prediction = read_validation_predictions(report_dir / "val_predictions.csv")
        metadata = json.loads((report_dir / "val_metrics.json").read_text(encoding="utf-8"))
        if metadata["split"] != "val" or metadata["source_run"] != run_id:
            raise ValueError(f"Unexpected report for {run_id}")
        labels = {image: actual for image, (actual, _) in prediction.items()}
        unadjusted = metrics_at_threshold(prediction, 0.5)
        if unadjusted["confusion_matrix"] != metadata["confusion_matrix"]:
            raise ValueError(f"Validation predictions do not match the saved report for {run_id}")
        if original is None:
            recall_floor = unadjusted["mite_recall"]
            original_labels = labels
            original = {"run_id": run_id, "threshold": 0.5, "metrics": unadjusted}
        elif labels != original_labels:
            raise ValueError(f"Validation images and labels do not match for {run_id}")

        evaluated = [
            {"run_id": run_id, "threshold": threshold, "metrics": metrics_at_threshold(prediction, threshold)}
            for threshold in THRESHOLDS
        ]
        valid = [entry for entry in evaluated if entry["metrics"]["mite_recall"] >= recall_floor]
        candidate_reports.append({
            "run_id": run_id,
            "model": metadata["model"],
            "imgsz": metadata["imgsz"],
            "unadjusted": unadjusted,
            "selected": max(valid, key=rank) if valid else None,
        })

    eligible = [entry["selected"] for entry in candidate_reports if entry["selected"]]
    return {
        "split": "val",
        "thresholds": THRESHOLDS,
        "minimum_mite_recall": recall_floor,
        "baseline": original,
        "candidates": candidate_reports,
        "selection": max(eligible, key=rank),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("/runs"))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()
    summary = choose(args.runs, args.baseline, args.candidates)
    output = args.runs / "optimization_summary.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary["selection"], indent=2))


if __name__ == "__main__":
    main()
