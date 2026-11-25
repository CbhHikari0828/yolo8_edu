"""Train a Varroa detector, evaluate it, and publish its verified weights."""

import argparse
import subprocess
import sys
from pathlib import Path

from legacy_detection import SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", choices=("yolov8n", *SOURCES), default="yolov8n")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.model in SOURCES:
        command = [sys.executable, str(Path(__file__).with_name("legacy_detection.py")),
                   "--model", args.model, "--data", str(args.data), "--run", str(args.run),
                   "--weights", str(args.weights), "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
                   "--batch", str(args.batch), "--seed", str(args.seed)]
        subprocess.run([*command, "--mode", "train"], check=True)
        print("Training complete. Selecting confidence and evaluating the test split.", flush=True)
        subprocess.run([*command, "--mode", "evaluate"], check=True)
        return

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "yolo", "detect", "train", f"model={args.weights}", f"data={args.data / 'dataset.yaml'}",
        f"project={args.run.parent}", f"name={args.run.name}", f"epochs={args.epochs}",
        f"imgsz={args.imgsz}", f"batch={args.batch}", f"seed={args.seed}",
        "device=0", "workers=4", "amp=False", "mosaic=0", "plots=False",
    ]
    if args.fine_tune:
        command.extend(("optimizer=AdamW", "lr0=0.00005", "warmup_epochs=1", "hsv_v=0.50"))
    print(f"Training {args.run.name} from {args.weights}", flush=True)
    subprocess.run(command, check=True)
    print("Training complete. Selecting a confidence threshold and evaluating the test split.", flush=True)
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("evaluate_detection.py")),
        "--run", str(args.run), "--data", str(args.data), "--imgsz", str(args.imgsz),
        "--batch", str(args.batch), "--nms-iou", "0.45",
        "--benchmark" if args.benchmark else "--retire-previous",
    ], check=True)


if __name__ == "__main__":
    main()
