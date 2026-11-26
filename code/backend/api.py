"""Serve local Varroa experiment results and launch one CUDA training job at a time."""

import csv
import json
import logging
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
from typing import Literal, get_args
from urllib.parse import quote
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from detection import MAX_UPLOAD_BYTES, active_detector, completed_detector_runs, predict_upload
from legacy_detection import read_v7_curves
from locations import read_index


RUNS = Path(os.environ.get("VARROA_RUNS_DIR", "/runs")).resolve()
DATA = Path(os.environ.get("VARROA_DATA_DIR", "/dataset")).resolve()
JOBS = RUNS / ".jobs"
COMPARISONS = RUNS / ".comparisons"
TEST_DIRS = {"baseline": RUNS / "final-test", "optimized": RUNS / "optimized-test"}
LOCATIONS = RUNS / "train-locations.json"
DETECTION_DATA = RUNS / "detection-dataset"
TestVariant = Literal["baseline", "optimized"]
DetectionModel = Literal["yolov3-tiny", "yolov5n", "yolov7-tiny", "yolov8n", "faster-rcnn-resnet18"]
RUN_FILES = (
    "results.csv", "run_summary.json", "args.yaml", "validation_metrics.json",
    "confusion_matrix.csv", "weights/best.pt", "detection_report.json", "detection_predictions.csv",
    "classifier-test/test_metrics.json", "classifier-test/test_confusion_matrix.csv", "classifier-test/test_predictions.csv",
)
TEST_FILES = ("test_metrics.json", "test_confusion_matrix.csv", "test_predictions.csv")
JOB_LOCK = Lock()
active_process: subprocess.Popen | None = None
active_comparison: str | None = None
escape_codes = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
logger = logging.getLogger(__name__)


class TrainingRequest(BaseModel):
    preprocessing: Literal["default", "pad"] = "default"
    model: Literal["yolov8n-cls", "yolov8s-cls", "resnet18"] = "yolov8n-cls"
    epochs: int = Field(default=10, ge=1, le=200)
    imgsz: int = Field(default=160, ge=32, le=640, multiple_of=32)
    batch: int = Field(default=32, ge=1, le=64)
    seed: int = Field(default=42, ge=0, le=4294967295)
    samples_per_class: int | None = Field(default=None, ge=1, le=128)


class DetectionTrainingRequest(BaseModel):
    model: DetectionModel = "yolov8n"
    initialization: Literal["pretrained", "scratch", "active"] = "pretrained"
    benchmark: bool = False
    epochs: int = Field(default=18, ge=1, le=200)
    imgsz: int = Field(default=384, ge=64, le=640, multiple_of=32)
    batch: int = Field(default=16, ge=1, le=64)
    seed: int = Field(default=2026, ge=0, le=4294967295)


class ComparisonRequest(BaseModel):
    task: Literal["classify", "detect"]
    epochs: int = Field(ge=1, le=200)
    imgsz: int = Field(ge=64, le=640, multiple_of=32)
    batch: int = Field(ge=1, le=64)
    seed: int = Field(ge=0, le=4294967295)
    initialization: Literal["scratch", "pretrained"] = "scratch"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_job(record: dict, directory: Path = JOBS) -> None:
    path = directory / f"{record['id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@asynccontextmanager
async def lifespan(_app):
    for directory in (JOBS, COMPARISONS):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.json"):
            job = read_json(path)
            if job and job["status"] == "running":
                write_job({**job, "status": "interrupted", "interruption_reason": "backend_restarted"}, directory)
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
        if args.get("initialization") or args.get("pretrained") is False:
            settings["initialization"] = args.get("initialization", "scratch")
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
    if job and job.get("comparison_id"):
        record["comparison_id"] = job["comparison_id"]
    if detailed:
        log = JOBS / f"{run_id}.log" if job else RUNS / f"{run_id}.log"
        record.update(epochs=training_epochs(folder / "results.csv"), log=recent_log(log))
        record["can_resume"] = (record["status"] == "interrupted" and task == "classify"
                                and "comparison_id" not in record
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
    if active_process is not None or active_comparison is not None:
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


def require_detection_data() -> None:
    if not all((DETECTION_DATA / name).is_file() for name in ("dataset.yaml", "source_summary.json")):
        raise HTTPException(409, "缺少瓦螨位置数据。请按启动文档用原始 gt.csv 生成检测数据集。")


def detection_command(settings: DetectionTrainingRequest, run_id: str, weights: Path, *, benchmark: bool) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).with_name("train_detection.py")),
        "--data", str(DETECTION_DATA), "--run", str(RUNS / run_id), "--weights", str(weights),
        "--model", settings.model, "--epochs", str(settings.epochs), "--imgsz", str(settings.imgsz),
        "--batch", str(settings.batch), "--seed", str(settings.seed),
    ]
    if settings.initialization == "active":
        command.append("--fine-tune")
    if settings.initialization == "scratch":
        command.append("--scratch")
    if benchmark:
        command.append("--benchmark")
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
        if (not previous or previous["status"] != "interrupted" or previous.get("task", "classify") != "classify"
                or previous.get("comparison_id")):
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
        require_detection_data()
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
        command = detection_command(settings, run_id, weights, benchmark=benchmark)
        return spawn_training(run_id, "detect", recorded, command)


def comparison_record(record: dict) -> dict:
    fallback = "queued" if record["status"] == "running" else "skipped"
    steps = []
    for step in record["runs"]:
        job = read_json(JOBS / f"{step['id']}.json")
        steps.append({**step, "status": job["status"] if job else fallback})
    return {**record, "runs": steps}


def execute_comparison_run(job: dict, commands: list[list[str]]) -> None:
    write_job(job)
    try:
        with (JOBS / f"{job['id']}.log").open("wb") as output:
            for command in commands:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        write_job({**job, "status": "failed", "exit_code": getattr(error, "returncode", None)})
        raise
    write_job({**job, "status": "completed", "exit_code": 0})


def execute_comparison(record: dict, steps: list[tuple[dict, list[list[str]]]]) -> None:
    global active_comparison
    status = "failed"
    failed_run = None
    try:
        for job, commands in steps:
            failed_run = job["id"]
            execute_comparison_run(job, commands)
        failed_run = None
        if record["task"] == "classify":
            results = [str(RUNS / step["id"] / "classifier-test") for step in record["runs"]]
            output = COMPARISONS / record["id"] / "classifier_comparison.json"
            with (COMPARISONS / f"{record['id']}.log").open("wb") as log:
                subprocess.run([
                    sys.executable, str(Path(__file__).with_name("compare_classifiers.py")),
                    "--runs", str(RUNS), "--reference", results[0], "--candidates", *results[1:],
                    "--output", str(output),
                ], stdout=log, stderr=subprocess.STDOUT, check=True)
        status = "completed"
    except Exception:
        logger.exception("Comparison %s failed", record["id"])
    finally:
        with JOB_LOCK:
            try:
                write_job({**record, "status": status, "failed_run": failed_run}, COMPARISONS)
            finally:
                active_comparison = None


@app.post("/api/comparison-jobs", status_code=201)
def start_comparison(settings: ComparisonRequest):
    global active_comparison
    with JOB_LOCK:
        require_training_slot()
        protocol = settings.model_dump(exclude={"task", "initialization"})
        if settings.task == "detect":
            require_detection_data()
            protocol["initialization"] = settings.initialization
            models = get_args(DetectionModel)
            prefix = "mite-detection"
        else:
            models = ("yolov8n-cls", "yolov8s-cls", "resnet18")
            prefix = "default"

        serial = uuid4().hex[:10]
        comparison_id = f"comparison-{serial}"
        created_at = datetime.now(timezone.utc).isoformat()
        steps = []
        runs = []
        for index, model in enumerate(models, start=1):
            run_id = f"{prefix}-{serial}-{index}"
            if settings.task == "detect":
                training = DetectionTrainingRequest(**protocol, model=model, benchmark=True)
                commands = [detection_command(training, run_id, RUNS / "pretrained" / f"{model}.pt", benchmark=True)]
            else:
                training = TrainingRequest(**protocol, model=model)
                commands = [
                    classification_command(training, run_id),
                    [sys.executable, str(Path(__file__).with_name("evaluate.py")),
                     "--run", str(RUNS / run_id), "--data", str(DATA), "--split", "test",
                     "--output", str(RUNS / run_id / "classifier-test"), "--batch", str(settings.batch)],
                ]
            runs.append({"id": run_id, "model": model})
            steps.append(({
                "id": run_id, "task": settings.task, "origin": "web", "status": "running",
                "settings": training.model_dump(), "comparison_id": comparison_id, "created_at": created_at,
            }, commands))

        record = {
            "id": comparison_id, "task": settings.task, "status": "running", "created_at": created_at,
            "protocol": protocol, "runs": runs,
        }
        write_job(record, COMPARISONS)
        active_comparison = comparison_id
        try:
            Thread(target=execute_comparison, args=(record, steps), daemon=True).start()
        except RuntimeError as error:
            active_comparison = None
            write_job({**record, "status": "failed"}, COMPARISONS)
            raise HTTPException(500, "无法启动对照实验") from error
        return comparison_record(record)


@app.get("/api/comparison-jobs")
def list_comparisons():
    records = [comparison_record(record) for path in COMPARISONS.glob("comparison-*.json")
               if (record := read_json(path))]
    return sorted(records, key=lambda record: record["created_at"], reverse=True)


@app.get("/api/comparison-jobs/{comparison_id}/result")
def comparison_result(comparison_id: str):
    if not re.fullmatch(r"comparison-[0-9a-f]{10}", comparison_id):
        raise HTTPException(404, "对照实验不存在")
    record = read_json(COMPARISONS / f"{comparison_id}.json")
    if record is None:
        raise HTTPException(404, "对照实验不存在")
    if record["status"] != "completed":
        raise HTTPException(409, "对照实验尚未完成")
    if record["task"] == "classify":
        report = read_json(COMPARISONS / comparison_id / "classifier_comparison.json")
        if report is None:
            raise HTTPException(404, "对照报告不存在")
        return report
    models = [run_record(step["id"]) for step in record["runs"]]
    for model in models:
        model["epochs"] = training_epochs(run_path(model["id"]) / "results.csv")
    return {"protocol": record["protocol"], "models": models}


def test_detection_image(image: str) -> Path:
    folder = (DETECTION_DATA / "images" / "test").resolve()
    path = (folder / image).resolve()
    if path.parent != folder or path.suffix != ".png" or not path.is_file():
        raise HTTPException(404, "测试图不存在")
    return path


@app.get("/api/box-comparison")
def box_comparison(run: list[str] = Query(min_length=2, max_length=5), index: int | None = Query(None, ge=1)):
    predictions = {}
    for run_id in run:
        folder = run_path(run_id)
        if not run_id.startswith("mite-detection") or not (folder / "detection_report.json").is_file():
            raise HTTPException(404, "实验没有定位评估结果")
        rows = read_csv(folder / "detection_predictions.csv")
        if not rows:
            raise HTTPException(404, "实验没有逐图检测记录")
        predictions[run_id] = {row["image"]: row for row in rows}
    reference = predictions[run[0]]
    if len(predictions) != len(run) or any(rows.keys() != reference.keys() for rows in predictions.values()):
        raise HTTPException(409, "请选择使用同一批测试图的不同实验")
    images = sorted(reference)
    if index is None:
        chosen = max(images, key=lambda name: (
            reference[name]["actual_boxes"] != "[]",
            sum(rows[name]["predicted_boxes"] != "[]" for rows in predictions.values()),
        ))
        index = images.index(chosen) + 1
    if index > len(images):
        raise HTTPException(404, "测试图序号超出范围")
    name = images[index - 1]
    with Image.open(test_detection_image(name)) as image:
        width, height = image.size
    return {
        "index": index, "total": len(images), "width": width, "height": height, "name": name,
        "url": f"/api/box-comparison/images/{quote(name, safe='')}",
        "expected": json.loads(reference[name]["actual_boxes"]),
        "predictions": {run_id: json.loads(rows[name]["predicted_boxes"]) for run_id, rows in predictions.items()},
    }


@app.get("/api/box-comparison/images/{image}")
def box_comparison_image(image: str):
    return FileResponse(test_detection_image(image), media_type="image/png")


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
    keys = ("epochs", "imgsz", "batch", "seed", "initialization")
    groups = {}
    for folder, report in completed_detector_runs(RUNS):
        if report.get("serve") is not False:
            continue
        record = run_record(folder.name)
        if record["status"] != "completed" or record.get("comparison_id"):
            continue
        settings = record["settings"]
        protocol = tuple(settings.get(key) for key in keys[:-1]) + (settings.get("initialization", "pretrained"),)
        if None in protocol:
            continue
        record["epochs"] = training_epochs(folder / "results.csv")
        models = groups.setdefault(protocol, {})
        model = report["model"]
        if model not in models or record["created_at"] > models[model]["created_at"]:
            models[model] = record
    return [{"protocol": dict(zip(keys, key)),
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
