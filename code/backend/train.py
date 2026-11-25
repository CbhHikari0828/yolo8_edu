"""Train and validate a classifier on the Varroa dataset."""

import argparse
import gc
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
import ultralytics
from ultralytics import YOLO

from classification import PaddingClassificationTrainer, save_validation_metrics
from resnet_classification import PRETRAINED_FILE, train_resnet


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def prepare_dataset(
    source: Path, target: Path, samples_per_class: int | None
) -> dict[str, dict[str, int]]:
    counts = {}
    for split in ("train", "val"):
        counts[split] = {}
        for category in ("mite", "no_mite"):
            source_category = source / split / category
            images = sorted(source_category.glob("*.png"))[:samples_per_class]
            if not images:
                raise FileNotFoundError(f"No PNG images found in {source_category}")

            target_category = target / split / category
            target_category.mkdir(parents=True)
            for image in images:
                shutil.copyfile(image, target_category / image.name)
            counts[split][category] = len(images)
    return counts


def main(*, smoke: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/dataset"))
    parser.add_argument("--output", type=Path, default=Path("/runs"))
    parser.add_argument("--name", default="smoke" if smoke else "full")
    parser.add_argument("--model", choices=("yolov8n-cls", "yolov8s-cls", "resnet18"), default="yolov8n-cls")
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--imgsz", type=positive_int, default=160)
    parser.add_argument("--batch", type=positive_int, default=8 if smoke else 32)
    parser.add_argument("--workers", type=int, default=0 if smoke else 2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=positive_int, default=16 if smoke else None)
    parser.add_argument("--preprocessing", choices=("default", "pad"), default="default")
    parser.add_argument("--resume", action="store_true", help="Continue a YOLOv8 classification run from weights/last.pt")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.model == "resnet18" and args.imgsz < 64:
        parser.error("ResNet18 expects an image size of at least 64")
    if args.resume and args.model == "resnet18":
        parser.error("Only YOLOv8 classification runs support checkpoint resumption")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA. Check the Docker GPU configuration.")
    matrix = torch.ones((4, 4), device="cuda:0")
    (matrix @ matrix).cpu()
    gpu = torch.cuda.get_device_name(0)
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {gpu}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    resnet = args.model == "resnet18"
    weights = args.output / "pretrained" / (PRETRAINED_FILE if resnet else f"{args.model}.pt")
    weights.parent.mkdir(exist_ok=True)
    checkpoint = args.output / args.name / "weights" / "last.pt" if args.resume else weights
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # Ultralytics writes verification caches beside the class folders. Stage
    # only the train and val images on the container's writable filesystem.
    with TemporaryDirectory(prefix="varroa-train-") as temp_dir:
        dataset = Path(temp_dir)
        staging_started = perf_counter()
        counts = prepare_dataset(args.data, dataset, args.samples_per_class)
        staging_seconds = round(perf_counter() - staging_started, 2)
        print(f"Staged {counts} in {staging_seconds:.2f} s", flush=True)
        model = None if resnet else YOLO(str(checkpoint))
        if args.resume:
            restored = model.ckpt.get("epoch", -1) + 1
            if not (0 < restored < args.epochs) or model.ckpt.get("optimizer") is None:
                raise ValueError(f"Checkpoint does not contain resumable training state: {checkpoint}")
            if Path(model.ckpt["train_args"]["save_dir"]).resolve() != (args.output / args.name).resolve():
                raise ValueError(f"Checkpoint belongs to a different run: {checkpoint}")
        torch.cuda.reset_peak_memory_stats(0)
        training_started = perf_counter()
        if resnet:
            run_dir = train_resnet(args, dataset, weights)
        else:
            try:
                model.train(
                    trainer=PaddingClassificationTrainer if args.preprocessing == "pad" else None,
                    data=str(dataset), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, workers=args.workers, device=0, seed=args.seed,
                    amp=False, plots=False, project=str(args.output), name=args.name, resume=args.resume,
                )
                run_dir = model.trainer.save_dir
                save_validation_metrics(model.trainer.validator, run_dir, sum(counts["val"].values()))
            finally:
                # Release the trainer's workers while the temporary images are still available.
                model.trainer = None
                gc.collect()
        torch.cuda.synchronize(0)
        elapsed_seconds = round(perf_counter() - training_started, 2)

    for result in ("results.csv", "weights/best.pt"):
        output = run_dir / result
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Training did not produce {output}")
        print(f"Verified: {output}", flush=True)

    summary = {
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "dataset": str(args.data),
        "model": args.model,
        "pretrained_weights": str(weights),
        "preprocessing": args.preprocessing,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "dataset_counts": counts,
        "staging_seconds": staging_seconds,
        "training_and_validation_seconds": elapsed_seconds,
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
        "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
    }
    if args.resume:
        summary["resumed_from_epoch"] = restored
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved run summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
