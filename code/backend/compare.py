"""Summarize matched validation runs and select a checkpoint for the held-out test."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = ("accuracy", "macro_f1", "mite_recall", "mite_false_negatives", "mite_false_negative_rate", "mite_false_positives")
SETTINGS = ("gpu", "torch", "cuda", "ultralytics", "pretrained_weights", "epochs", "imgsz", "batch", "workers", "dataset_counts")


def compare(runs: Path, default: list[str], pad: list[str]) -> dict:
    groups = {}
    shared = None
    for scheme, run_ids in (("default", default), ("pad", pad)):
        entries = []
        for run_id in run_ids:
            directory = runs / run_id
            summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
            metrics = json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8"))
            settings = {key: summary[key] for key in SETTINGS}
            if shared is None:
                shared = settings
            if summary["preprocessing"] != scheme or settings != shared or metrics["split"] != "val":
                raise ValueError(f"Run {run_id} does not match the paired validation experiment")
            fp = metrics["confusion_matrix"][1][0]
            entries.append({
                "run_id": run_id,
                "seed": summary["seed"],
                **{key: metrics[key] for key in METRICS if key != "mite_false_positives"},
                "mite_false_positives": fp,
            })
        entries.sort(key=lambda entry: entry["seed"])
        seeds = [entry["seed"] for entry in entries]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"Expected one run per seed for {scheme}")
        groups[scheme] = {
            "runs": entries,
            "mean": {key: mean(entry[key] for entry in entries) for key in METRICS},
            "std": {key: stdev(entry[key] for entry in entries) if len(entries) > 1 else None for key in METRICS},
        }
    if [entry["seed"] for entry in groups["default"]["runs"]] != [entry["seed"] for entry in groups["pad"]["runs"]]:
        raise ValueError("Both preprocessing methods must use the same seeds")

    scheme = max(groups, key=lambda key: (groups[key]["mean"]["mite_recall"], groups[key]["mean"]["macro_f1"]))
    winner = max(groups[scheme]["runs"], key=lambda row: (row["mite_recall"], row["macro_f1"], -row["seed"]))
    return {
        "split": "val",
        "settings": shared,
        "groups": groups,
        "selection": {
            "preprocessing": scheme,
            "run_id": winner["run_id"],
            "seed": winner["seed"],
            "rule": "Mean mite recall, then mean macro F1. Within the chosen method, validation mite recall, then macro F1.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("/runs"))
    parser.add_argument("--default", nargs="+", required=True)
    parser.add_argument("--pad", nargs="+", required=True)
    args = parser.parse_args()
    report = compare(args.runs, args.default, args.pad)
    output = args.runs / "study_summary.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
