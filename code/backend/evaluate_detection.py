"""Select a detection confidence on validation and evaluate the official test split."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import nms as ultralytics_nms
from ultralytics.utils.metrics import ap_per_class, box_iou as pairwise_iou

from classification import build_metrics
from detection import DEFAULT_NMS_IOU, MAX_DETECTIONS, predict_upload, retire_detector_weights


THRESHOLDS = [index / 20 for index in range(2, 17)]


def collect_predictions(model, data: Path, split: str, imgsz: int, batch: int, nms_iou: float = DEFAULT_NMS_IOU) -> list[dict]:
    image_dir = data / "images" / split
    label_dir = data / "labels" / split
    expected = {image.name for image in image_dir.glob("*.png")}
    predictions = []
    for result in model.predict(
        source=str(image_dir), stream=True, imgsz=imgsz, batch=batch, conf=0.001, iou=nms_iou,
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


def comparison_map(records: list[dict]) -> dict:
    """Compute AP with the same one-class matcher for every benchmark model."""
    match = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    correct, scores = [], []
    labels = sum(len(record["actual"]) for record in records)
    for record in records:
        if not record["predictions"]:
            continue
        confidence, boxes = zip(*record["predictions"])
        predicted = torch.zeros(len(boxes))
        actual = torch.zeros(len(record["actual"]))
        overlaps = pairwise_iou(torch.tensor(record["actual"]).reshape(-1, 4), torch.tensor(boxes))
        correct.append(BaseValidator.match_predictions(match, predicted, actual, overlaps).numpy())
        scores.extend(confidence)
    if not scores:
        return {"map50": 0.0, "map50_95": 0.0, "pr50": [[0.0, 0.0]]}
    _, _, _, _, _, average_precision, _, _, _, _, recall_steps, precision_steps = ap_per_class(
        np.concatenate(correct), np.asarray(scores), np.zeros(len(scores)), np.zeros(labels),
    )
    points = [[float(recall_steps[index]), float(precision_steps[0, index])]
              for index in [*range(0, len(recall_steps), 5), len(recall_steps) - 1]]
    return {"map50": float(average_precision[:, 0].mean()), "map50_95": float(average_precision.mean()), "pr50": points}


def evaluate(run: Path, data: Path, imgsz: int, batch: int, nms_iou: float,
             model_name: str = "yolov8n", predictor=None, *, publish: bool = True) -> dict:
    weights = run / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = predictor if predictor is not None else YOLO(str(weights), task="detect")
    # Initialize CUDA NMS before it is first called inside the timed prediction loop.
    nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda:0"), torch.ones(1, device="cuda:0"), 0.5)
    # Offline evaluation must not skip the remaining images in a slow batch.
    ultralytics_nms.non_max_suppression = partial(ultralytics_nms.non_max_suppression, max_time_img=10.0)

    print("Selecting confidence from validation images.", flush=True)
    validation = collect_predictions(model, data, "val", imgsz, batch, nms_iou)
    candidates = [(threshold, assess(validation, threshold, "val")) for threshold in THRESHOLDS]
    threshold, validation_metrics = max(candidates, key=lambda candidate: (
        candidate[1]["image"]["macro_f1"],
        candidate[1]["boxes_at_iou50"]["f1"],
        candidate[1]["image"]["mite_recall"],
        -abs(candidate[0] - 0.25),
    ))

    print(f"Evaluating test images at confidence {threshold:.2f}.", flush=True)
    test = collect_predictions(model, data, "test", imgsz, batch, nms_iou)
    test_metrics = assess(test, threshold, "test")
    if publish:
        standard = model.val(
            data=str(data / "dataset.yaml"), split="test", imgsz=imgsz, batch=batch,
            device=0, workers=2, rect=False, conf=0.001, iou=nms_iou, plots=False,
            project=str(run), name="test-localization",
        )
        test_metrics.update(map50=float(standard.box.map50), map50_95=float(standard.box.map))
    else:
        test_metrics.update(comparison_map(test))

    predictions = run / "detection_predictions.csv"
    pending_predictions = predictions.with_suffix(".tmp")
    with pending_predictions.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["image", "actual_boxes", "predicted_boxes"])
        for record in test:
            detected = [
                {"xyxy": box, "confidence": score} for score, box in record["predictions"] if score >= threshold
            ]
            writer.writerow([record["name"], json.dumps(record["actual"]), json.dumps(detected)])

    report = {
        "source_run": run.name,
        "model": model_name,
        "imgsz": imgsz,
        "serve": publish,
        "confidence_threshold": threshold,
        "nms_iou": nms_iou,
        "validation": validation_metrics,
        "test": test_metrics,
        "data": json.loads((data / "source_summary.json").read_text(encoding="utf-8")),
    }
    if publish:
        sample = data / "images" / "test" / test[0]["name"]
        predict_upload(sample.read_bytes(), weights, report)
    temporary = run / "detection_report.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending_predictions.replace(predictions)
    temporary.replace(run / "detection_report.json")
    print(f"Evaluated {test_metrics['image']['samples']} test images. mAP@0.50 {test_metrics['map50']:.4f}.", flush=True)
    print(f"Saved {run / 'detection_report.json'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU, help="IoU threshold for suppressing overlapping boxes")
    parser.add_argument("--benchmark", action="store_true", help="Evaluate a comparison model without publishing it")
    parser.add_argument("--retire-previous", action="store_true", help="Remove replaced detection checkpoints after evaluation")
    args = parser.parse_args()
    if not 0 < args.nms_iou <= 1:
        parser.error("--nms-iou must be greater than 0 and at most 1")
    evaluate(args.run, args.data, args.imgsz, args.batch, args.nms_iou, publish=not args.benchmark)
    if not args.benchmark:
        print(f"Published detector {args.run.name}.", flush=True)
    if args.retire_previous and not args.benchmark:
        weights = args.run / "weights" / "best.pt"
        try:
            for retired in retire_detector_weights(args.run.parent, weights):
                print(f"Removed replaced detector checkpoint: {retired}")
        except OSError as error:
            print(f"Could not clean up a replaced checkpoint: {error}", flush=True)


if __name__ == "__main__":
    main()
