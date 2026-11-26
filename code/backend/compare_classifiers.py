"""Compare classifiers on the same official Varroa test images."""

import argparse
import csv
import json
from pathlib import Path


def read_result(folder: Path, runs: Path) -> tuple[dict, dict]:
    metrics = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
    with (folder / "test_predictions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = {row["image"]: row["actual"] for row in rows}
    if len(labels) != len(rows) or len(rows) != metrics["samples"] or metrics["split"] != "test":
        raise ValueError(f"Incomplete or duplicate test predictions in {folder}")
    settings = json.loads((runs / metrics["source_run"] / "run_summary.json").read_text(encoding="utf-8"))
    return {"metrics": metrics, "settings": settings}, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("/runs"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline, baseline_labels = read_result(args.reference, args.runs)
    shared = ("imgsz", "preprocessing", "seed", "epochs", "batch")
    settings = {key: baseline["settings"][key] for key in shared}
    threshold = baseline["metrics"]["mite_threshold"]
    models = [baseline]
    for folder in args.candidates:
        candidate, candidate_labels = read_result(folder, args.runs)
        same_split_counts = candidate["settings"]["dataset_counts"] == baseline["settings"]["dataset_counts"]
        same_settings = all(candidate["settings"][key] == value for key, value in settings.items())
        if baseline_labels != candidate_labels or not (same_split_counts and same_settings):
            raise ValueError("The comparison requires identical test images and matching training settings")
        if candidate["metrics"]["mite_threshold"] != threshold:
            raise ValueError("The comparison requires the same image classification threshold")
        models.append(candidate)
    summary = {"samples": len(baseline_labels), "threshold": threshold, "protocol": settings,
               "models": models}
    output = args.output or args.runs / "classifier_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
