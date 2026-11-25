"""Select a detection confidence on validation and evaluate the official test split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from classification import build_metrics
from detection import MAX_DETECTIONS, predict_upload, retire_detector_weights


THRESHOLDS = [index / 20 for index in range(2, 17)]


def collect_predictions(model, data: Path, split: str, imgsz: int, batch: int) -> list[dict]:
    image_dir = data / "images" / split
    label_dir = data / "labels" / split
    expected = {image.name for image in image_dir.glob("*.png")}
    predictions = []
    for result in model.predict(
        source=str(image_dir), stream=True, imgsz=imgsz, batch=batch, conf=0.001,
        device=0, rect=False, max_det=MAX_DETECTIONS, verbose=False,
    ):
        name = Path(result.path).name
        height, width = result.orig_shape
        actual = []
        for line in (label_dir / Path(name).with_suffix(".txt")).read_text(encoding="ascii").splitlines():
            _, x, y, w, h = map(float, line.split())
            actual.append([(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])
        detections = sorted(
            zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist()),
            key=lambda box: box[0],
            reverse=True,
        )
        predictions.append({"name": name, "actual": actual, "predictions": detections})
    if len(predictions) != len(expected) or {row["name"] for row in predictions} != expected:
        raise RuntimeError(f"Expected predictions for all {len(expected)} images in {split}")
    return predictions


def box_iou(first, second) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(right - left, 0) * max(bottom - top, 0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def assess(records: list[dict], threshold: float, split: str) -> dict:
    image_matrix = np.zeros((2, 2), dtype=int)
    true_boxes = predicted_boxes = matched = 0
    for record in records:
        expected = record["actual"]
        detected = [box for score, box in record["predictions"] if score >= threshold]
        image_matrix[int(not expected), int(not detected)] += 1
        true_boxes += len(expected)
        predicted_boxes += len(detected)
        remaining = set(range(len(expected)))
        for box in detected:
            if not remaining:
                break
            match = max(remaining, key=lambda index: box_iou(box, expected[index]))
            if box_iou(box, expected[match]) >= 0.5:
                matched += 1
                remaining.remove(match)
    return {
        "image": build_metrics(image_matrix, ["mite", "no_mite"], split),
        "boxes_at_iou50": {
            "matches": matched,
            "false_positives": predicted_boxes - matched,
            "missed": true_boxes - matched,
            "precision": matched / max(predicted_boxes, 1),
            "recall": matched / max(true_boxes, 1),
            "f1": 2 * matched / max(true_boxes + predicted_boxes, 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--retire-previous", action="store_true", help="Remove replaced detection checkpoints after evaluation")
    args = parser.parse_args()
    weights = args.run / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights), task="detect")

    validation = collect_predictions(model, args.data, "val", args.imgsz, args.batch)
    candidates = [(threshold, assess(validation, threshold, "val")) for threshold in THRESHOLDS]
    threshold, validation_metrics = max(candidates, key=lambda candidate: (
        candidate[1]["image"]["macro_f1"],
        candidate[1]["boxes_at_iou50"]["f1"],
        candidate[1]["image"]["mite_recall"],
        -abs(candidate[0] - 0.25),
    ))

    test = collect_predictions(model, args.data, "test", args.imgsz, args.batch)
    test_metrics = assess(test, threshold, "test")
    standard = model.val(
        data=str(args.data / "dataset.yaml"), split="test", imgsz=args.imgsz, batch=args.batch,
        device=0, workers=2, rect=False, conf=0.001, plots=False,
        project=str(args.run), name="test-localization",
    )
    test_metrics.update(map50=float(standard.box.map50), map50_95=float(standard.box.map))

    with (args.run / "detection_predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["image", "actual_boxes", "predicted_boxes"])
        for record in test:
            detected = [
                {"xyxy": box, "confidence": score} for score, box in record["predictions"] if score >= threshold
            ]
            writer.writerow([record["name"], json.dumps(record["actual"]), json.dumps(detected)])

    report = {
        "source_run": args.run.name,
        "model": "yolov8n",
        "imgsz": args.imgsz,
        "confidence_threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "data": json.loads((args.data / "source_summary.json").read_text(encoding="utf-8")),
    }
    sample = args.data / "images" / "test" / test[0]["name"]
    predict_upload(sample.read_bytes(), weights, report)
    temporary = args.run / "detection_report.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.run / "detection_report.json")
    print(json.dumps(report, indent=2))
    if args.retire_previous:
        for retired in retire_detector_weights(args.run.parent, weights):
            print(f"Removed replaced detector checkpoint: {retired}")


if __name__ == "__main__":
    main()
