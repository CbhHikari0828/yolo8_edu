"""Index the original Varroa training annotations for image lookup."""

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


CLASSES = {"0": "no_mite", "1": "mite", "3": "mite"}


def png_size(contents: bytes) -> tuple[int, int]:
    if contents[:8] != b"\x89PNG\r\n\x1a\n" or contents[12:16] != b"IHDR":
        raise ValueError("Invalid PNG header")
    return struct.unpack(">II", contents[16:24])


def visible_boxes(coordinates, width: int, height: int) -> list[list[int]]:
    boxes = []
    for x1, y1, x2, y2 in coordinates:
        left, top, right, bottom = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
        if left < right and top < bottom:
            boxes.append([left, top, right, bottom])
    return boxes


def build_index(dataset: Path, annotations: Path) -> dict:
    source = defaultdict(list)
    for line in annotations.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        filename, label, *coordinates = line.split()
        name = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if label not in CLASSES or len(coordinates) % 4 or bool(coordinates) != (label != "0"):
            raise ValueError(f"Unexpected annotation: {line}")
        boxes = [list(map(int, coordinates[offset : offset + 4])) for offset in range(0, len(coordinates), 4)]
        source[f"{CLASSES[label]}/{name}"].append((label, boxes))

    train = dataset / "train"
    images = {path.relative_to(train).as_posix(): path for path in train.glob("*/*.png")}
    if not images or source.keys() != images.keys():
        raise ValueError(
            f"Training images and annotations differ: {len(images.keys() - source.keys())} "
            f"unannotated images, {len(source.keys() - images.keys())} missing images"
        )

    records = []
    for relative, entries in source.items():
        path = images[relative]
        contents = path.read_bytes()
        width, height = png_size(contents)
        label, coordinates = entries[0]
        boxes = visible_boxes(coordinates, width, height)
        records.append({
            "id": len(records),
            "name": path.name,
            "category": CLASSES[label],
            "raw_label": label,
            "width": width,
            "height": height,
            "boxes": boxes,
            "undrawable_boxes": len(coordinates) - len(boxes),
            "duplicate_annotations": len(entries) - 1,
            "sha256": hashlib.sha256(contents).hexdigest(),
        })
    return {"split": "train", "source": annotations.name, "records": records}


@lru_cache(maxsize=1)
def read_index(path: Path, _modified_ns: int):
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    hashes = defaultdict(list)
    for record in records:
        hashes[record["sha256"]].append(record)
    return records, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = build_index(args.data, args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Indexed {len(index['records'])} training images from {args.annotations.name}")


if __name__ == "__main__":
    main()
