"""Run a one-epoch YOLOv8 classification smoke test on CUDA."""

from train import main


if __name__ == "__main__":
    main(smoke=True)
