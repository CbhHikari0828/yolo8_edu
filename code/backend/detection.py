"""Run the trained one-class mite detector on an uploaded image."""

import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_DETECTIONS = 20
MODEL_LOCK = Lock()


def completed_detector_runs(runs: Path):
    for path in runs.glob("mite-detection*/detection_report.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(report, dict) and report.get("source_run") == path.parent.name:
            yield path.parent, report


def active_detector(runs: Path) -> tuple[Path, dict] | None:
    selected = None
    newest = (0, "")
    for folder, report in completed_detector_runs(runs):
        weights = folder / "weights" / "best.pt"
        try:
            order = (weights.stat().st_mtime_ns, folder.name)
        except FileNotFoundError:
            continue
        if order > newest:
            newest = order
            selected = weights, report
    return selected


def retire_detector_weights(runs: Path, keep: Path) -> list[Path]:
    current = active_detector(runs)
    if current is None or current[0] != keep:
        return []
    cutoff = keep.stat().st_mtime_ns, keep.parent.parent.name
    retired = []
    for folder, _ in completed_detector_runs(runs):
        weights_dir = folder / "weights"
        if weights_dir == keep.parent or weights_dir.resolve() != runs.resolve() / folder.name / "weights":
            continue
        best = weights_dir / "best.pt"
        try:
            order = best.stat().st_mtime_ns, folder.name
        except FileNotFoundError:
            order = (0, "")
        if order >= cutoff:
            continue
        for file in (weights_dir / "last.pt", best):
            if file.is_file():
                file.unlink(missing_ok=True)
                retired.append(file)
    return retired


@lru_cache(maxsize=1)
def load_detector(weights: Path, _modified_ns: int):
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    if model.task != "detect":
        raise RuntimeError(f"Expected object detection weights: {weights}")
    return model


def predict_upload(contents: bytes, weights: Path, report: dict) -> dict:
    try:
        with Image.open(BytesIO(contents)) as original:
            if original.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("请选择 PNG、JPG 或 WebP 照片。")
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise OverflowError("图片像素超过 2500 万，请缩小后再试。")
            image = ImageOps.exif_transpose(original).convert("RGB")
    except Image.DecompressionBombError as error:
        raise OverflowError("图片像素超过 2500 万，请缩小后再试。") from error
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("图片无法解码，请选择完整的 PNG、JPG 或 WebP 照片。") from error

    import torch

    with MODEL_LOCK:
        model = load_detector(weights, weights.stat().st_mtime_ns)
        result = model.predict(
            image,
            imgsz=report["imgsz"],
            rect=False,
            conf=report["confidence_threshold"],
            # Ultralytics hides CUDA process-wide when passed the string "cpu".
            device=torch.device("cpu"),
            max_det=MAX_DETECTIONS,
            verbose=False,
        )[0]
        boxes = [
            {"xyxy": coordinates, "confidence": confidence}
            for coordinates, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
        ]

    return {
        "width": image.width,
        "height": image.height,
        "boxes": boxes,
        "confidence_threshold": report["confidence_threshold"],
        "source_run": report["source_run"],
    }
