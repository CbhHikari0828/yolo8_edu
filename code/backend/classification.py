"""Preprocessing and validation metrics for the Varroa experiment."""

import csv
import json
from functools import partial
from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics.models.yolo.classify import ClassificationTrainer


def padding_transform(size: int):
    return partial(
        ImageOps.pad,
        size=(size, size),
        method=Image.Resampling.BILINEAR,
        color=(114, 114, 114),
    )


class PaddingClassificationTrainer(ClassificationTrainer):
    """Keep the default augmentations after the resize step."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        operations = dataset.torch_transforms.transforms
        geometry = (
            (transforms.RandomResizedCrop,)
            if mode == "train"
            else (transforms.Resize, transforms.CenterCrop)
        )
        if tuple(type(operation) for operation in operations[: len(geometry)]) != geometry:
            raise RuntimeError(f"Unexpected Ultralytics {mode} preprocessing: {operations}")

        dataset.torch_transforms = transforms.Compose(
            [padding_transform(self.args.imgsz), *operations[len(geometry) :]]
        )
        return dataset


def build_metrics(matrix, names: list[str], split: str) -> dict:
    mite = names.index("mite")
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = matrix.diagonal()
    samples = int(actual.sum())
    if (actual == 0).any():
        raise RuntimeError(f"Expected both classes in {split}, got {actual.tolist()}")

    misses = int(actual[mite] - correct[mite])
    return {
        "split": split,
        "checkpoint": "weights/best.pt",
        "samples": samples,
        "accuracy": float(correct.sum() / samples),
        "macro_f1": float((2 * correct / (actual + predicted)).mean()),
        "mite_precision": float(correct[mite] / max(int(predicted[mite]), 1)),
        "mite_recall": float(correct[mite] / actual[mite]),
        "mite_f1": float(2 * correct[mite] / (actual[mite] + predicted[mite])),
        "mite_false_negatives": misses,
        "mite_false_positives": int(predicted[mite] - correct[mite]),
        "mite_false_negative_rate": float(misses / actual[mite]),
        "class_names": names,
        "confusion_matrix": matrix.tolist(),
    }


def write_metrics(metrics: dict, metrics_path: Path, matrix_path: Path) -> None:
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    names = metrics["class_names"]
    with matrix_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *names])
        for label, row in zip(names, metrics["confusion_matrix"]):
            writer.writerow([label, *row])


def save_validation_metrics(validator, run_dir: Path, expected_samples: int) -> None:
    names = [validator.names[index] for index in range(len(validator.names))]
    # Omit Ultralytics' unused background slot and export actual labels by row.
    classes = len(names)
    matrix = validator.confusion_matrix.matrix[:classes, :classes].T.astype(int)
    if int(matrix.sum()) != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} validation images, got {int(matrix.sum())}")
    report = build_metrics(matrix, names, split="val")
    write_metrics(report, run_dir / "validation_metrics.json", run_dir / "confusion_matrix.csv")
