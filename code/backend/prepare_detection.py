"""Convert official Varroa boxes to the one-class YOLO detection dataset."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from locations import CLASSES, png_size, visible_boxes


SPLITS = ("train", "val", "test")


def prepare(data: Path, annotations: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Dataset already exists: {output}")

    entries = {}
    repeated = 0
    for line in annotations.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        filename, label, *coordinates = line.split()
        split, _, name = filename.replace("\\", "/").partition("/")
        name = name.rsplit("/", 1)[-1]
        if split not in SPLITS or label not in CLASSES or len(coordinates) % 4 or bool(coordinates) != (label != "0"):
            raise ValueError(f"Unexpected annotation: {line}")
        key = (split, CLASSES[label], name)
        if key in entries:
            repeated += 1
            continue
        entries[key] = [list(map(int, coordinates[offset : offset + 4])) for offset in range(0, len(coordinates), 4)]

    images = {
        (split, category, image.name): image
        for split in SPLITS
        for category in ("mite", "no_mite")
        for image in (data / split / category).glob("*.png")
    }
    if not images or entries.keys() != images.keys():
        raise ValueError(
            f"Images and annotations differ: {len(images.keys() - entries.keys())} unannotated images, "
            f"{len(entries.keys() - images.keys())} missing images"
        )

    report = {"counts": {}, "duplicate_annotations": repeated, "skipped_positive_without_valid_box": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="detection-data-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        for split in SPLITS:
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
            report["counts"][split] = {"mite": 0, "no_mite": 0, "boxes": 0}

        for (split, category, name), source in images.items():
            with source.open("rb") as file:
                width, height = png_size(file.read(24))
            boxes = visible_boxes(entries[(split, category, name)], width, height)
            if category == "mite" and not boxes:
                report["skipped_positive_without_valid_box"].append(f"{split}/{name}")
                continue

            shutil.copyfile(source, staging / "images" / split / name)
            labels = []
            for x1, y1, x2, y2 in boxes:
                values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
                labels.append("0 " + " ".join(f"{value:.8f}" for value in values))
            label_path = staging / "labels" / split / Path(name).with_suffix(".txt")
            label_path.write_text("".join(f"{line}\n" for line in labels), encoding="ascii")
            report["counts"][split][category] += 1
            report["counts"][split]["boxes"] += len(boxes)

        configuration = {split: f"images/{split}" for split in SPLITS}
        configuration["names"] = ["mite"]
        (staging / "dataset.yaml").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        (staging / "source_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.annotations, args.output), indent=2))


if __name__ == "__main__":
    main()
