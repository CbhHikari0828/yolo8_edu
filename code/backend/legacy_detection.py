"""Run pinned, upstream YOLOv3, YOLOv5 and YOLOv7 implementations."""

import argparse
import csv
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen
from zipfile import ZipFile, is_zipfile


SOURCES = {
    "yolov3-tiny": ("v3", "ultralytics/yolov3", "7eb23e3c1d387cdcefbad266046af369a9f40399", "v9.6.0", "models/yolov3-tiny.yaml", "data/hyps/hyp.scratch.yaml"),
    "yolov5n": ("v5", "ultralytics/yolov5", "915bbf294bb74c859f0b41f1c23bc395014ea679", "v7.0", "models/yolov5n.yaml", "data/hyps/hyp.scratch-low.yaml"),
    "yolov7-tiny": ("v7", "WongKinYiu/yolov7", "a207844b1ce82d204ab36d87d496728d3d2348e7", "v0.1", "cfg/training/yolov7-tiny.yaml", "data/hyp.scratch.tiny.yaml"),
}
V7_CURVE_FIELDS = ("epoch", "train/box_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "val/box_loss")


def download(url: str, destination: Path) -> None:
    pending = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    with urlopen(url, timeout=30) as source, pending.open("wb") as target:
        shutil.copyfileobj(source, target)
    if not is_zipfile(pending):
        raise RuntimeError(f"Incomplete download: {destination.name}")
    pending.replace(destination)


def prepare_upstream(model: str, weights: Path) -> Path:
    short, project, revision, release, _, _ = SOURCES[model]
    cache = weights.parent.parent / "legacy-sources"
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / f"{project.split('/')[-1]}-{revision}"
    if not (source / "train.py").is_file():
        archive = cache / f"{short}.zip"
        if not is_zipfile(archive):
            download(f"https://codeload.github.com/{project}/zip/{revision}", archive)
        with ZipFile(archive) as files:
            files.extractall(cache)
    if not is_zipfile(weights):
        weights.parent.mkdir(parents=True, exist_ok=True)
        download(f"https://github.com/{project}/releases/download/{release}/{model}.pt", weights)
    return source


def use_upstream(source: Path) -> None:
    import numpy as np
    from PIL import ImageFont

    for name, kind in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, kind)
    if not hasattr(ImageFont.FreeTypeFont, "getsize"):
        def text_size(font, text):
            left, _, right, bottom = font.getbbox(text)
            return right - left, bottom

        ImageFont.FreeTypeFont.getsize = text_size
    os.environ.update(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", WANDB_MODE="disabled",
                      YOLOv5_AUTOINSTALL="false", AUTOINSTALL="false", MPLBACKEND="Agg")
    sys.path.insert(0, str(source))
    os.chdir(source)


def read_v7_curves(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    epochs = []
    for index, line in enumerate(lines, start=1):
        columns = line.split()
        if len(columns) < 15:
            continue
        try:
            values = [index, *map(float, (columns[2], columns[-5], columns[-4], columns[-3]))]
        except ValueError:
            continue
        epochs.append(dict(zip(V7_CURVE_FIELDS, values)))
    return epochs


def convert_v7_curves(run: Path) -> None:
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=V7_CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(read_v7_curves(run / "results.txt"))


def train(args, source: Path) -> None:
    import yaml

    short, project, revision, release, cfg, hyp = SOURCES[args.model]
    if short in ("v3", "v7"):
        loss = source / "utils" / "loss.py"
        original = loss.read_text(encoding="utf-8")
        # Modern PyTorch requires integer bounds for an in-place clamp on indices.
        updated = original.replace("clamp_(0, gain[3] - 1)", "clamp_(0, gain[3].long() - 1)")
        updated = updated.replace("clamp_(0, gain[2] - 1)", "clamp_(0, gain[2].long() - 1)")
        if updated != original:
            loss.write_text(updated, encoding="utf-8")
    if short == "v7":
        reader = source / "utils" / "datasets.py"
        original = reader.read_text(encoding="utf-8")
        # YOLOv7 serializes its label cache with torch instead of NumPy.
        updated = original.replace(".with_suffix('.cache')", ".with_suffix('.yolov7.cache')")
        if updated != original:
            reader.write_text(updated, encoding="utf-8")
    args.run.mkdir(parents=True)
    data = {split: str(args.data / "images" / split) for split in ("train", "val", "test")}
    data.update(nc=1, names=["mite"])
    dataset = args.run / "legacy-dataset.yaml"
    dataset.write_text(yaml.safe_dump(data), encoding="utf-8")
    configuration = yaml.safe_load((source / hyp).read_text())
    configuration.update(mosaic=0.0, mixup=0.0, copy_paste=0.0)
    hyp_path = args.run / "hyp.yaml"
    hyp_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    flags = [
        "--weights", str(args.weights), "--cfg", str(source / cfg), "--data", str(dataset),
        "--hyp", str(hyp_path), "--epochs", str(args.epochs), "--batch-size", str(args.batch),
        "--img-size", str(args.imgsz), "--workers", "4", "--device", "0", "--project", str(args.run.parent),
        "--name", args.run.name, "--exist-ok",
    ]
    if short == "v7":
        from utils import general

        initialize_seed = general.init_seeds
        general.init_seeds = lambda *_: initialize_seed(args.seed)
        sys.argv = ["train.py", *flags]
        runpy.run_path(str(source / "train.py"), run_name="__main__")
        convert_v7_curves(args.run)
    else:
        from utils import general

        if hasattr(general, "check_git_info"):
            general.check_git_info = lambda: {"remote": project, "branch": None, "commit": revision}
        import train as upstream

        if short == "v5":
            flags.extend(("--seed", str(args.seed), "--noplots"))
            upstream.check_amp = lambda _model: False
        else:
            initialize_seed = upstream.init_seeds
            upstream.init_seeds = lambda *_: initialize_seed(args.seed)
        sys.argv = ["train.py", *flags]
        upstream.main(upstream.parse_opt())
    details = {key: getattr(args, key) for key in ("model", "imgsz", "epochs", "batch", "seed")}
    details.update(upstream=project, revision=revision, weights_release=release, model_weights=str(args.weights))
    (args.run / "args.yaml").write_text(yaml.safe_dump(details), encoding="utf-8")


class UpstreamPredictor:
    def __init__(self, weights: Path):
        import torch
        from models.experimental import attempt_load

        self.device = torch.device("cuda:0")
        self.model = attempt_load(str(weights), self.device).float().eval()

    def predict(self, source: str, imgsz: int, batch: int, conf: float, iou: float, max_det: int, **_):
        import cv2
        import numpy as np
        import torch
        from ultralytics.data.augment import LetterBox
        from torchvision.ops import nms
        from ultralytics.utils.ops import scale_boxes, xywh2xyxy

        images = sorted(Path(source).glob("*.png"))
        letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False)
        with torch.inference_mode():
            for offset in range(0, len(images), batch):
                paths = images[offset:offset + batch]
                originals = [cv2.imread(str(path)) for path in paths]
                batch_array = np.stack([letterbox(image=image) for image in originals])
                pixels = np.ascontiguousarray(batch_array[:, :, :, ::-1].transpose(0, 3, 1, 2))
                tensor = torch.from_numpy(pixels).to(self.device).float() / 255
                raw = self.model(tensor)[0]
                for path, original, predicted in zip(paths, originals, raw):
                    scores = predicted[:, 4] * predicted[:, 5]
                    selected = scores > conf
                    boxes = xywh2xyxy(predicted[selected, :4])
                    scores = scores[selected]
                    keep = nms(boxes, scores, iou)[:max_det]
                    boxes = scale_boxes((imgsz, imgsz), boxes[keep], original.shape[:2])
                    yield SimpleNamespace(path=path, orig_shape=original.shape[:2],
                                          boxes=SimpleNamespace(xyxy=boxes, conf=scores[keep]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--model", choices=SOURCES, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    for name in ("epochs", "imgsz", "batch", "seed"):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args()
    source = prepare_upstream(args.model, args.weights)
    if args.mode == "train":
        use_upstream(source)
        train(args, source)
    else:
        from evaluate_detection import evaluate

        use_upstream(source)
        predictor = UpstreamPredictor(args.run / "weights" / "best.pt")
        evaluate(args.run, args.data, args.imgsz, args.batch, 0.45, args.model, predictor, publish=False)


if __name__ == "__main__":
    main()
