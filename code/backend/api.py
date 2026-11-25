"""Serve local Varroa experiment results and launch one CUDA training job at a time."""

import csv
import json
import math
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread
from typing import Literal
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from detection import MAX_UPLOAD_BYTES, active_detector, completed_detector_runs, predict_upload
from legacy_detection import read_v7_curves
from locations import read_index


RUNS = Path(os.environ.get("VARROA_RUNS_DIR", "/runs")).resolve()
DATA = Path(os.environ.get("VARROA_DATA_DIR", "/dataset")).resolve()
JOBS = RUNS / ".jobs"
TEST_DIRS = {"baseline": RUNS / "final-test", "optimized": RUNS / "optimized-test"}
LOCATIONS = RUNS / "train-locations.json"
DETECTION_DATA = RUNS / "detection-dataset"
TestVariant = Literal["baseline", "optimized"]
RUN_FILES = (
    "results.csv", "run_summary.json", "args.yaml", "validation_metrics.json",
    "confusion_matrix.csv", "weights/best.pt", "detection_report.json", "detection_predictions.csv",
    "classifier-test/test_metrics.json", "classifier-test/test_confusion_matrix.csv", "classifier-test/test_predictions.csv",
)
TEST_FILES = ("test_metrics.json", "test_confusion_matrix.csv", "test_predictions.csv")
JOB_LOCK = Lock()
active_process: subprocess.Popen | None = None
escape_codes = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TrainingRequest(BaseModel):
    preprocessing: Literal["default", "pad"] = "default"
    model: Literal["yolov8n-cls", "yolov8s-cls", "resnet18"] = "yolov8n-cls"
    epochs: int = Field(default=10, ge=1, le=200)
    imgsz: int = Field(default=160, ge=32, le=640, multiple_of=32)
    batch: int = Field(default=32, ge=1, le=64)
    seed: int = Field(default=42, ge=0, le=4294967295)
    samples_per_class: int | None = Field(default=None, ge=1, le=128)


class DetectionTrainingRequest(BaseModel):
    model: Literal["yolov3-tiny", "yolov5n", "yolov7-tiny", "yolov8n"] = "yolov8n"
    initialization: Literal["pretrained", "active"] = "pretrained"
    benchmark: bool = False
    epochs: int = Field(default=18, ge=1, le=200)
    imgsz: int = Field(default=384, ge=64, le=640, multiple_of=32)
    batch: int = Field(default=16, ge=1, le=64)
    seed: int = Field(default=2026, ge=0, le=4294967295)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_job(record: dict) -> None:
    path = JOBS / f"{record['id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@asynccontextmanager
async def lifespan(_app):
    JOBS.mkdir(parents=True, exist_ok=True)
    for path in JOBS.glob("*.json"):
        job = read_json(path)
        if job and job["status"] == "running":
            write_job({**job, "status": "interrupted", "interruption_reason": "backend_restarted"})
    yield


app = FastAPI(title="Varroa experiment dashboard", lifespan=lifespan)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def with_class_details(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    (tp, fn), (fp, tn) = metrics["confusion_matrix"]
    return {
        **metrics,
        "mite_precision": tp / max(tp + fp, 1),
        "mite_f1": 2 * tp / max(2 * tp + fn + fp, 1),
        "mite_false_positives": fp,
    }


def run_path(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise HTTPException(404, "实验不存在")
    path = (RUNS / run_id).resolve()
    if path.parent != RUNS:
        raise HTTPException(404, "实验不存在")
    return path


def recent_log(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as file:
        header = file.read(2)
        end = file.seek(0, os.SEEK_END)
        offset = max(0, end - 16000)
        encoding = "utf-8-sig"
        if header in (b"\xff\xfe", b"\xfe\xff"):
            offset = max(2, offset + offset % 2)
            encoding = "utf-16-le" if header == b"\xff\xfe" else "utf-16-be"
        file.seek(offset)
        tail = file.read().decode(encoding, errors="replace")
    lines = escape_codes.sub("", tail).splitlines()
    return "\n".join(line for line in lines[-45:] if line.strip())


def training_epochs(path: Path) -> list[dict]:
    if not path.is_file():
        return [epoch for epoch in read_v7_curves(path.with_suffix(".txt")) if all(map(math.isfinite, epoch.values()))]
    epochs = []
    for row in read_csv(path):
        try:
            metrics = {key.strip(): float(value) for key, value in row.items() if key and value}
        except ValueError:
            continue
        if "epoch" not in metrics or not all(math.isfinite(value) for value in metrics.values()):
            continue
        for upstream, common in (("metrics/mAP_0.5", "metrics/mAP50(B)"),
                                 ("metrics/mAP_0.5:0.95", "metrics/mAP50-95(B)")):
            if upstream in metrics:
                metrics[common] = metrics[upstream]
        epochs.append(metrics)
    if epochs and epochs[0]["epoch"] == 0:
        for row in epochs:
            row["epoch"] += 1
    return epochs


def run_record(run_id: str, *, detailed: bool = False) -> dict:
    folder = run_path(run_id)
    summary = read_json(folder / "run_summary.json")
    job = read_json(JOBS / f"{run_id}.json")
    report = read_json(folder / "detection_report.json") if run_id.startswith("mite-detection") else None
    if summary is None and job is None and report is None:
        raise HTTPException(404, "实验不存在")
    task = "detect" if report else "classify"
    if job:
        task = job.get("task", task)
    settings = summary or (job["settings"] if job else {})
    if task == "detect" and job is None:
        config = folder / "args.yaml"
        args = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}) if config.is_file() else {}
        settings = {key: args.get(key) for key in ("epochs", "imgsz", "batch", "seed")}
        settings["model"] = report["model"]
        if args.get("model"):
            initial = Path(args["model"])
            settings["initialization"] = "active" if initial.name == "best.pt" else "pretrained"
            if settings["initialization"] == "active":
                settings["initial_source_run"] = initial.parent.parent.name
    record = {
        "id": run_id,
        "task": task,
        "origin": "web" if job else "cli",
        "status": job["status"] if job else "completed",
        "settings": settings,
        "validation": report["validation"] if report else with_class_details(read_json(folder / "validation_metrics.json")),
        "files": [file for file in RUN_FILES if (folder / file).is_file()],
        "created_at": job["created_at"] if job else datetime.fromtimestamp(folder.stat().st_mtime, timezone.utc).isoformat(),
    }
    if report:
        record.update(test=report["test"], confidence_threshold=report["confidence_threshold"],
                      nms_iou=report.get("nms_iou", 0.70))
    if detailed:
        log = JOBS / f"{run_id}.log" if job else RUNS / f"{run_id}.log"
        record.update(epochs=training_epochs(folder / "results.csv"), log=recent_log(log))
        record["can_resume"] = (record["status"] == "interrupted" and task == "classify"
                                and settings.get("model", "yolov8n-cls") in ("yolov8n-cls", "yolov8s-cls")
                                and (folder / "weights" / "last.pt").is_file())
        if task == "detect":
            detector = active_detector(RUNS)
            record["serving"] = detector is not None and detector[1]["source_run"] == run_id
            if not job and not record["log"]:
                record["log"] = recent_log(RUNS / f"{run_id}-train.log")
    return record


def monitor_job(record: dict, process: subprocess.Popen) -> None:
    global active_process
    exit_code = process.wait()
    with JOB_LOCK:
        status = "completed" if exit_code == 0 else "failed"
        write_job({**record, "status": status, "exit_code": exit_code})
        if active_process is process:
            active_process = None


def require_training_slot() -> None:
    if active_process is not None and active_process.poll() is None:
        raise HTTPException(409, "已有训练任务运行中")


def spawn_training(run_id: str, task: str, settings: dict, command: list[str], *, previous: dict | None = None) -> dict:
    global active_process
    record = {
        "id": run_id, "task": task, "origin": "web", "status": "running", "settings": settings,
        "created_at": previous["created_at"] if previous else datetime.now(timezone.utc).isoformat(),
    }
    write_job(record)
    try:
        with (JOBS / f"{run_id}.log").open("ab" if previous else "wb") as output:
            if previous:
                output.write(b"\nResuming training from the last completed epoch.\n")
                output.flush()
            active_process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    except OSError as error:
        write_job({**record, "status": "failed"})
        raise HTTPException(500, "无法启动训练进程") from error
    Thread(target=monitor_job, args=(record, active_process), daemon=True).start()
    return record


def classification_command(settings: TrainingRequest, run_id: str) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).with_name("train.py")),
        "--data", str(DATA), "--output", str(RUNS), "--name", run_id,
        "--preprocessing", settings.preprocessing, "--model", settings.model, "--epochs", str(settings.epochs),
        "--imgsz", str(settings.imgsz), "--batch", str(settings.batch), "--seed", str(settings.seed),
    ]
    if settings.samples_per_class is not None:
        command.extend(("--samples-per-class", str(settings.samples_per_class)))
    return command


@app.get("/api/health")
def health():
    return {"status": "ready"}


@app.get("/api/dataset")
@lru_cache(maxsize=1)
def dataset_overview():
    counts = {
        split: {label: sum(1 for _ in (DATA / split / label).glob("*.png")) for label in ("mite", "no_mite")}
        for split in ("train", "val", "test")
    }
    return {"counts": counts, "labels": {"0": "no_mite", "1": "mite", "3": "mite"}}


@app.get("/api/detection")
def detection_status():
    detector = active_detector(RUNS)
    if detector is None:
        return {"ready": False}
    _, report = detector
    return {
        "ready": True, "source_run": report["source_run"], "model": report["model"],
        "imgsz": report["imgsz"], "threshold": report["confidence_threshold"], "test": report["test"],
    }


@app.post("/api/detection")
async def detect_image(request: Request):
    contents = bytearray()
    async for chunk in request.stream():
        if len(contents) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "请上传不超过 10 MB 的照片。")
        contents.extend(chunk)
    if not contents:
        raise HTTPException(400, "请先选择一张照片。")
    image = bytes(contents)
    for _ in range(2):
        detector = active_detector(RUNS)
        if detector is None:
            break
        weights, report = detector
        try:
            return await run_in_threadpool(predict_upload, image, weights, report)
        except FileNotFoundError:
            # A completed detector may have been replaced while this upload was queued.
            continue
        except OverflowError as error:
            raise HTTPException(413, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    raise HTTPException(503, "瓦螨检测模型正在准备。")


def training_locations():
    try:
        modified_ns = LOCATIONS.stat().st_mtime_ns
    except FileNotFoundError as error:
        raise HTTPException(503, "本机还未生成训练集位置索引，请按启动说明准备原始标注。") from error
    return read_index(LOCATIONS, modified_ns)


@app.get("/api/locations/samples")
def location_samples(category: Literal["mite", "no_mite"] = "mite", q: str = Query(default="", max_length=160)):
    records, _ = training_locations()
    query = q.strip().casefold()
    return [
        record for record in records
        if record["category"] == category and query in record["name"].casefold()
    ][:12]


@app.get("/api/locations/lookup")
def location_lookup(
    sha256: str = Query(pattern=r"^[0-9a-f]{64}$"),
    name: str = Query(default="", max_length=255),
):
    _, hashes = training_locations()
    candidates = hashes.get(sha256, ())
    if not candidates:
        raise HTTPException(404, "这张图未匹配到训练集原图，请选择未裁剪或另存的原始 PNG。")
    for record in candidates:
        if record["name"] == name:
            return record
    first = candidates[0]
    if any((record["category"], record["boxes"]) != (first["category"], first["boxes"]) for record in candidates):
        raise HTTPException(409, "相同图片有不同标注，请使用原文件名或从训练集样本中选择。")
    return first


@app.get("/api/locations/images/{index}")
def location_image(index: int):
    records, _ = training_locations()
    if not 0 <= index < len(records):
        raise HTTPException(404, "训练图片不存在")
    record = records[index]
    train = DATA / "train"
    file = (train / record["category"] / record["name"]).resolve()
    if not file.is_relative_to(train) or file.suffix.lower() != ".png" or not file.is_file():
        raise HTTPException(404, "训练图片不存在")
    return FileResponse(file, media_type="image/png")


@app.get("/api/runs")
def list_runs():
    ids = {path.parent.name for path in RUNS.glob("*/run_summary.json")}
    ids.update(path.stem for path in JOBS.glob("*.json"))
    ids.update(folder.name for folder, _ in completed_detector_runs(RUNS))
    records = [run_record(run_id) for run_id in sorted(ids) if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id)]
    return sorted(records, key=lambda record: record["created_at"], reverse=True)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return run_record(run_id, detailed=True)


@app.post("/api/runs", status_code=201)
def start_run(settings: TrainingRequest):
    with JOB_LOCK:
        require_training_slot()
        if settings.model == "resnet18" and settings.imgsz < 64:
            raise HTTPException(422, "ResNet18 的输入尺寸至少为 64。")
        run_id = f"{settings.preprocessing}-{uuid4().hex[:10]}"
        return spawn_training(run_id, "classify", settings.model_dump(), classification_command(settings, run_id))


@app.post("/api/runs/{run_id}/resume", status_code=201)
def resume_run(run_id: str):
    with JOB_LOCK:
        require_training_slot()
        folder = run_path(run_id)
        previous = read_json(JOBS / f"{run_id}.json")
        if not previous or previous["status"] != "interrupted" or previous.get("task", "classify") != "classify":
            raise HTTPException(409, "请选择已中断的 YOLOv8 分类任务。")
        settings = TrainingRequest.model_validate(previous["settings"])
        if settings.model == "resnet18" or not (folder / "weights" / "last.pt").is_file():
            raise HTTPException(409, "该任务没有可用的 YOLOv8 分类检查点。")
        return spawn_training(run_id, "classify", settings.model_dump(),
                              [*classification_command(settings, run_id), "--resume"], previous=previous)


@app.post("/api/runs/detection", status_code=201)
def start_detection_run(settings: DetectionTrainingRequest):
    with JOB_LOCK:
        require_training_slot()
        if not all((DETECTION_DATA / name).is_file() for name in ("dataset.yaml", "source_summary.json")):
            raise HTTPException(409, "缺少瓦螨位置数据。请按启动文档用原始 gt.csv 生成检测数据集。")
        run_id = f"mite-detection-{uuid4().hex[:10]}"
        weights = RUNS / "pretrained" / f"{settings.model}.pt"
        benchmark = settings.benchmark or settings.model != "yolov8n"
        recorded = {**settings.model_dump(), "benchmark": benchmark}
        fine_tune = settings.initialization == "active"
        if fine_tune:
            if benchmark:
                raise HTTPException(422, "定位对照请使用各模型的官方预训练权重。")
            detector = active_detector(RUNS)
            if detector is None:
                raise HTTPException(409, "当前没有可微调的检测模型，请选择官方预训练权重。")
            weights, report = detector
            recorded["initial_source_run"] = report["source_run"]
        command = [
            sys.executable, str(Path(__file__).with_name("train_detection.py")),
            "--data", str(DETECTION_DATA), "--run", str(RUNS / run_id), "--weights", str(weights),
            "--model", settings.model,
            "--epochs", str(settings.epochs), "--imgsz", str(settings.imgsz),
            "--batch", str(settings.batch), "--seed", str(settings.seed),
        ]
        if fine_tune:
            command.append("--fine-tune")
        if benchmark:
            command.append("--benchmark")
        return spawn_training(run_id, "detect", recorded, command)


def download(folder: Path, filename: str, allowed: tuple[str, ...]):
    path = (folder / filename).resolve()
    if filename not in allowed or not path.is_relative_to(folder) or not path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, filename=path.name)


@app.get("/api/runs/{run_id}/files/{filename:path}")
def run_file(run_id: str, filename: str):
    return download(run_path(run_id), filename, RUN_FILES)


@app.get("/api/study")
def study():
    return read_json(RUNS / "study_summary.json")


@app.get("/api/study/file")
def study_file():
    return download(RUNS, "study_summary.json", ("study_summary.json",))


@app.get("/api/optimization")
def optimization():
    return read_json(RUNS / "optimization_summary.json")


@app.get("/api/classifier-comparison")
def classifier_comparison():
    return read_json(RUNS / "classifier_comparison.json")


@app.get("/api/detector-comparison")
def detector_comparison():
    groups = {}
    for folder, report in completed_detector_runs(RUNS):
        if report.get("serve") is not False:
            continue
        record = run_record(folder.name)
        if record["status"] != "completed":
            continue
        settings = record["settings"]
        protocol = tuple(settings.get(key) for key in ("epochs", "imgsz", "batch", "seed"))
        if None in protocol:
            continue
        record["epochs"] = training_epochs(folder / "results.csv")
        models = groups.setdefault(protocol, {})
        model = report["model"]
        if model not in models or record["created_at"] > models[model]["created_at"]:
            models[model] = record
    return [{"protocol": dict(zip(("epochs", "imgsz", "batch", "seed"), key)),
             "models": [models[name] for name in sorted(models)]}
            for key, models in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))]


@app.get("/api/evaluation")
def evaluation(variant: TestVariant = "baseline"):
    folder = TEST_DIRS[variant]
    metrics = read_json(folder / "test_metrics.json")
    if metrics is None:
        return None
    mistakes = {"misses": [], "false_alarms": []}
    for index, row in enumerate(read_csv(folder / "test_predictions.csv")):
        if row["actual"] != row["predicted"]:
            kind = "misses" if row["actual"] == "mite" else "false_alarms"
            mistakes[kind].append({**row, "confidence": float(row["confidence"]), "index": index})
    examples = {
        kind: sorted(rows, key=lambda row: row["confidence"], reverse=True)[:12]
        for kind, rows in mistakes.items()
    }
    return {"metrics": metrics, "examples": examples}


@app.get("/api/evaluation/images/{index}")
def evaluation_image(index: int, variant: TestVariant = "baseline"):
    rows = read_csv(TEST_DIRS[variant] / "test_predictions.csv")
    if not 0 <= index < len(rows):
        raise HTTPException(404, "图片不存在")
    file = (DATA / rows[index]["image"]).resolve()
    if not file.is_relative_to(DATA / "test") or file.suffix.lower() != ".png" or not file.is_file():
        raise HTTPException(404, "图片不存在")
    return FileResponse(file, media_type="image/png")


@app.get("/api/evaluation/files/{filename}")
def evaluation_file(filename: str, variant: TestVariant = "baseline"):
    return download(TEST_DIRS[variant], filename, TEST_FILES)
