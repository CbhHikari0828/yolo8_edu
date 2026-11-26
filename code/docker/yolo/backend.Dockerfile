FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY backend/classification.py backend/resnet_classification.py backend/train.py backend/train_smoke.py backend/compare.py backend/compare_classifiers.py backend/evaluate.py backend/optimize.py backend/locations.py backend/prepare_detection.py backend/detection.py backend/legacy_detection.py backend/evaluate_detection.py backend/train_detection.py backend/faster_rcnn.py backend/api.py ./
RUN mkdir -p "$YOLO_CONFIG_DIR"

# Set this before the CUDA warm-up creates a cuBLAS handle.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
