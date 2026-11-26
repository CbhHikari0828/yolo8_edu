"""Train a Faster R-CNN with a ResNet18 FPN on the shared Varroa box dataset."""

import argparse
import csv
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from evaluate_detection import collect_predictions, comparison_map, evaluate, read_boxes
from resnet_classification import PRETRAINED, PRETRAINED_FILE


MODEL = "faster-rcnn-resnet18"
DEVICE = torch.device("cuda:0")
MICROBATCH = 4
NMS_IOU = 0.45
FIELDS = ("epoch", "train/box_loss", "val/box_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)")


def read_image(path: Path, size: int):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    height, width = image.shape[:2]
    gain = min(size / width, size / height)
    resized_width, resized_height = max(1, round(width * gain)), max(1, round(height * gain))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas[top : top + resized_height, left : left + resized_width] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    scale = (resized_width / width, resized_height / height) * 2
    offset = (left, top) * 2
    return torch.from_numpy(canvas).permute(2, 0, 1), (height, width), scale, offset


class MiteBoxes(Dataset):
    def __init__(self, data: Path, split: str, size: int):
        self.paths = sorted((data / "images" / split).glob("*.png"))
        self.labels = data / "labels" / split
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image, shape, scale, offset = read_image(path, self.size)
        boxes = torch.tensor(read_boxes(self.labels / path.with_suffix(".txt").name, shape), dtype=torch.float32)
        boxes = boxes.reshape(-1, 4) * torch.tensor(scale) + torch.tensor(offset)
        return image, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def collate(samples):
    return tuple(zip(*samples))


def network(size: int, weights: Path | None = None) -> FasterRCNN:
    backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=None, norm_layer=nn.BatchNorm2d, trainable_layers=5)
    if weights is not None:
        if not weights.is_file():
            weights.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(PRETRAINED.url, str(weights), hash_prefix="f37072fd")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        backbone.body.load_state_dict({name: tensor for name, tensor in state.items() if not name.startswith("fc.")})
    anchors = AnchorGenerator(sizes=((8,), (16,), (32,), (64,), (128,)), aspect_ratios=((0.5, 1, 2),) * 5)
    return FasterRCNN(
        backbone, num_classes=2, min_size=size, max_size=size, rpn_anchor_generator=anchors,
        rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=256, rpn_post_nms_top_n_test=256,
        box_batch_size_per_image=128,
    )


def to_device(images):
    return [image.to(DEVICE, dtype=torch.float32).div_(255) for image in images]


class Predictor:
    def __init__(self, model: FasterRCNN):
        self.model = model

    def predict(self, source: str, imgsz: int, batch: int, conf: float, iou: float, max_det: int, **_):
        self.model.eval()
        self.model.roi_heads.score_thresh = conf
        self.model.roi_heads.nms_thresh = iou
        self.model.roi_heads.detections_per_img = max_det
        paths = sorted(Path(source).glob("*.png"))
        with torch.inference_mode():
            for offset in range(0, len(paths), batch):
                batch_paths = paths[offset : offset + batch]
                images, shapes, scales, pads = zip(*(read_image(path, imgsz) for path in batch_paths))
                outputs = self.model(to_device(images))
                for path, shape, scale, pad, output in zip(batch_paths, shapes, scales, pads, outputs):
                    boxes = output["boxes"].cpu()
                    boxes = (boxes - torch.tensor(pad)) / torch.tensor(scale)
                    boxes[:, 0::2].clamp_(0, shape[1])
                    boxes[:, 1::2].clamp_(0, shape[0])
                    visible = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    yield SimpleNamespace(path=path, orig_shape=shape, boxes=SimpleNamespace(
                        xyxy=boxes[visible], conf=output["scores"].cpu()[visible],
                    ))


def box_loss_epoch(model, loader, optimizer=None) -> float:
    model.train()
    training = optimizer is not None
    if not training:
        # TorchVision returns detection losses in train mode. Freeze batch-norm statistics for validation.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    box_total = 0.0
    split = "train" if training else "val"
    with torch.set_grad_enabled(training):
        for index, (images, targets) in enumerate(loader, start=1):
            if training:
                optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(images), MICROBATCH):
                subset = images[offset : offset + MICROBATCH]
                labels = [{key: tensor.to(DEVICE) for key, tensor in target.items()}
                          for target in targets[offset : offset + MICROBATCH]]
                losses = model(to_device(subset), labels)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite Faster R-CNN loss in {split} batch {index}")
                if training:
                    (loss * (len(subset) / len(images))).backward()
                box_total += losses["loss_box_reg"].item() * len(subset)
            if training:
                optimizer.step()
            if index % 50 == 0 or index == len(loader):
                print(f"{split}: {index}/{len(loader)} batches", flush=True)
    return box_total / len(loader.dataset)


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    weights = None if args.scratch else args.run.parent / "pretrained" / PRETRAINED_FILE
    model = network(args.imgsz, weights).to(DEVICE)
    loaders = {}
    for split in ("train", "val"):
        dataset = MiteBoxes(args.data, split, args.imgsz)
        if not dataset:
            raise RuntimeError(f"No {split} images in {args.data}")
        loaders[split] = DataLoader(
            dataset, batch_size=args.batch, shuffle=split == "train", num_workers=4, collate_fn=collate,
            generator=torch.Generator().manual_seed(args.seed), persistent_workers=True,
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = args.run / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    settings = {key: getattr(args, key) for key in ("epochs", "imgsz", "batch", "seed")}
    settings.update(
        model=MODEL, initialization="scratch" if args.scratch else "pretrained",
        backbone_weights=None if args.scratch else PRETRAINED.url,
        gradient_microbatch=min(args.batch, MICROBATCH), optimizer="SGD",
    )
    (args.run / "args.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    best = (-1.0, -1.0)
    predictor = Predictor(model)
    with (args.run / "results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(FIELDS)
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}", flush=True)
            train_loss = box_loss_epoch(model, loaders["train"], optimizer)
            val_loss = box_loss_epoch(model, loaders["val"])
            scores = comparison_map(collect_predictions(predictor, args.data, "val", args.imgsz, min(MICROBATCH, args.batch), NMS_IOU))
            fitness = scores["map50_95"], scores["map50"]
            if fitness > best:
                best = fitness
                torch.save(model.state_dict(), checkpoint)
            writer.writerow((epoch, train_loss, val_loss, scores["map50"], scores["map50_95"]))
            output.flush()
            schedule.step()
            print(f"Validation mAP@0.50 {scores['map50']:.4f}, mAP@0.50:0.95 {scores['map50_95']:.4f}", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    evaluate(args.run, args.data, args.imgsz, min(MICROBATCH, args.batch), NMS_IOU, MODEL, predictor, publish=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    for name in ("epochs", "imgsz", "batch", "seed"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
