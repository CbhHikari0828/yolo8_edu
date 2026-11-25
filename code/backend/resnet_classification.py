"""Train and load the TorchVision ResNet18 classification baseline."""

import csv
from pathlib import Path

import torch
import yaml
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from ultralytics.data.augment import classify_augmentations, classify_transforms
from ultralytics.utils.torch_utils import init_seeds

from classification import build_metrics, padding_transform, write_metrics


PRETRAINED = ResNet18_Weights.IMAGENET1K_V1
PRETRAINED_FILE = Path(PRETRAINED.url).name
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def resnet_transforms(size: int, preprocessing: str, *, training: bool = False):
    if training:
        transform = classify_augmentations(
            size=size, mean=MEAN, std=STD, scale=(0.5, 1.0), hflip=0.5,
            auto_augment="randaugment", hsv_s=0.7, hsv_v=0.4, erasing=0.4,
        )
    else:
        transform = classify_transforms(size=size, mean=MEAN, std=STD)
    if preprocessing == "pad":
        transform = transforms.Compose([padding_transform(size), *transform.transforms[1 if training else 2 :]])
    return transform


def load_resnet(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    names = state["names"]
    model = resnet18(weights=None, num_classes=len(names))
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), names


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = torch.zeros((), device=device)
    targets, predictions = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.detach() * len(labels)
            targets.append(labels)
            predictions.append(logits.detach().argmax(dim=1))
    classes = len(loader.dataset.classes)
    pairs = torch.cat(targets) * classes + torch.cat(predictions)
    matrix = torch.bincount(pairs, minlength=classes**2).view(classes, classes).cpu().numpy()
    return total_loss.item() / len(loader.dataset), matrix


def train_resnet(args, dataset: Path, weights: Path) -> Path:
    init_seeds(args.seed, deterministic=True)
    device = torch.device("cuda:0")
    run = args.output / args.name
    (run / "weights").mkdir(parents=True)
    if not weights.is_file():
        torch.hub.download_url_to_file(PRETRAINED.url, str(weights), hash_prefix="f37072fd")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    learning_rate = 0.0003
    weight_decay = 0.0005
    config = {**vars(args), "optimizer": "AdamW", "lr0": learning_rate, "weight_decay": weight_decay,
              "lr_schedule": "cosine", "pretrained_weights": PRETRAINED.url, "normalization": "ImageNet"}
    for key in ("data", "output"):
        config[key] = str(config[key])
    (run / "args.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    training = datasets.ImageFolder(dataset / "train", transform=resnet_transforms(args.imgsz, args.preprocessing, training=True))
    validation = datasets.ImageFolder(dataset / "val", transform=resnet_transforms(args.imgsz, args.preprocessing))
    if training.classes != validation.classes:
        raise RuntimeError("Training and validation classes do not match")
    model.fc = torch.nn.Linear(model.fc.in_features, len(training.classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader_options = {"batch_size": args.batch, "num_workers": args.workers, "pin_memory": True,
                      "persistent_workers": args.workers > 0}
    train_loader = torch.utils.data.DataLoader(training, shuffle=True, **loader_options)
    val_loader = torch.utils.data.DataLoader(validation, **loader_options)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_accuracy = -1.0
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as file:
        results = csv.writer(file)
        results.writerow(["epoch", "train/loss", "val/loss", "metrics/accuracy_top1"])
        for epoch in range(1, args.epochs + 1):
            train_loss, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_loss, matrix = run_epoch(model, val_loader, loss_fn, device)
            scheduler.step()
            report = build_metrics(matrix, training.classes, "val")
            accuracy = report["accuracy"]
            results.writerow([epoch, train_loss, val_loss, accuracy])
            file.flush()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                checkpoint = run / "weights" / "best.pt"
                pending = checkpoint.with_suffix(".tmp")
                torch.save({"state_dict": model.state_dict(), "names": training.classes}, pending)
                pending.replace(checkpoint)
                write_metrics(report, run / "validation_metrics.json", run / "confusion_matrix.csv")
            print(f"Epoch {epoch}/{args.epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  val accuracy {accuracy:.4f}", flush=True)
    return run
