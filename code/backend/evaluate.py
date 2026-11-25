"""Evaluate a saved classification experiment using its training preprocessing."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torchvision import datasets, transforms
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

from classification import build_metrics, padding_transform, write_metrics
from resnet_classification import load_resnet, resnet_transforms
from train import positive_int


def evaluate(run: Path, data: Path, split: str, output: Path, batch: int, mite_threshold: float) -> dict:
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")

    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    size = summary["imgsz"]
    geometry = summary["preprocessing"]
    resnet = summary.get("model") == "resnet18"
    preprocess = classify_transforms(size=size)
    if geometry == "pad":
        preprocess = transforms.Compose([padding_transform(size), *preprocess.transforms[2:]])
    elif geometry != "default":
        raise ValueError(f"Unknown preprocessing in run summary: {geometry}")
    if resnet:
        preprocess = resnet_transforms(size, geometry)

    data = data.resolve()
    dataset = datasets.ImageFolder(data / split, transform=preprocess)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")
    checkpoint = run / "weights" / "best.pt"
    if resnet:
        network, names = load_resnet(checkpoint, device)
    else:
        network = YOLO(str(checkpoint)).model.to(device).fuse(verbose=False).eval()
        names = [network.names[index] for index in range(len(network.names))]
    if names != dataset.classes:
        raise ValueError(f"Checkpoint classes {names} do not match dataset classes {dataset.classes}")
    mite = names.index("mite")
    no_mite = names.index("no_mite")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    matrix = np.zeros((len(names), len(names)), dtype=int)
    predictions = []
    with torch.inference_mode():
        for images, targets in loader:
            probabilities = network(images.to(device))
            if isinstance(probabilities, (tuple, list)):
                probabilities = probabilities[0]
            if resnet:
                probabilities = probabilities.softmax(dim=1)
            mite_scores = probabilities[:, mite]
            choices = torch.where(mite_scores >= mite_threshold, mite, no_mite)
            confidence = probabilities.gather(1, choices[:, None]).flatten()
            for actual, predicted, score, mite_score in zip(
                targets.tolist(), choices.tolist(), confidence.tolist(), mite_scores.tolist()
            ):
                sample_path = Path(dataset.samples[len(predictions)][0])
                matrix[actual, predicted] += 1
                predictions.append((sample_path.relative_to(data).as_posix(), names[actual], names[predicted], score, mite_score))

    if len(predictions) != len(dataset):
        raise RuntimeError(f"Evaluated {len(predictions)} of {len(dataset)} images")
    metrics = build_metrics(matrix, names, split)
    metrics.update(
        source_run=run.name,
        preprocessing=geometry,
        seed=summary["seed"],
        imgsz=size,
        model=summary.get("model", "yolov8n-cls"),
        mite_threshold=mite_threshold,
        gpu=torch.cuda.get_device_name(0),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="evaluation-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        write_metrics(metrics, staging / f"{split}_metrics.json", staging / f"{split}_confusion_matrix.csv")
        with (staging / f"{split}_predictions.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "actual", "predicted", "confidence", "mite_probability"])
            writer.writerows(predictions)
        staging.rename(output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/dataset"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=positive_int, default=64)
    parser.add_argument("--mite-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.mite_threshold <= 1:
        parser.error("--mite-threshold must be between 0 and 1")
    metrics = evaluate(args.run, args.data, args.split, args.output, args.batch, args.mite_threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
