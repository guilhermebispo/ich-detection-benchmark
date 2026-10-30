"""YOLOv8 (Ultralytics 8.3): dataset lists, training with the shared settings, and prediction.

Uses the same PNG files and the YOLO label files exported by notebook 04 (`images/` and `labels/`
side by side, which is where Ultralytics looks for labels). Requires ultralytics==8.3.*.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import training_config as cfg
from .data_io import image_key


def write_dataset(train_entries, root, out_dir) -> Path:
    """Write a dataset YAML whose lists hold absolute image paths.

    The lists exported by notebook 04 use `./images/...`, which Ultralytics resolves relative to the list
    file and not to the dataset root, so new lists are written here. The `val` list repeats the training
    images: per-epoch validation is turned off and the held-out fold is never shown to the trainer.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(root).resolve()
    (out / "train.txt").write_text("".join(f"{root / e['image']}\n" for e in train_entries))
    (out / "data.yaml").write_text(f"train: {out / 'train.txt'}\nval: {out / 'train.txt'}\nnames:\n  0: hemorrhage\n")
    return out / "data.yaml"


def train_arguments(data_yaml, out_dir, name: str, epochs: int, mosaic: bool, batch_size: int = cfg.BATCH_SIZE,
                    seed: int = cfg.SEED, workers: int = 2) -> dict:
    """Arguments passed to `YOLO.train`, kept apart so the tests can check them without Ultralytics."""
    o = cfg.OPTIMIZER
    return dict(
        data=str(data_yaml), project=str(out_dir), name=name, exist_ok=True, epochs=epochs,
        imgsz=cfg.IMAGE_SIZE, batch=batch_size, nbs=batch_size, seed=seed, deterministic=True, workers=workers,
        optimizer=o["name"], lr0=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"],
        warmup_epochs=cfg.WARMUP_EPOCHS, cos_lr=True, lrf=cfg.FINAL_LR_FRACTION, amp=True, val=False,
        plots=False, **cfg.YOLO_OPTIMISATION, **(cfg.YOLO_MOSAIC if mosaic else cfg.YOLO_NO_AUGMENTATION))


def train(data_yaml, out_dir, name: str, epochs: int, mosaic: bool, batch_size: int = cfg.BATCH_SIZE,
          seed: int = cfg.SEED, model: str = cfg.YOLO_MODEL, workers: int = 2, log=print) -> Path:
    """Train for a fixed number of epochs and return `last.pt` (no selection on validation data).

    Ultralytics saves `last.pt` every epoch; calling again with the same `out_dir`/`name` resumes.
    """
    from ultralytics import YOLO

    run = Path(out_dir) / name
    last = run / "weights" / "last.pt"
    t0 = time.time()
    if last.exists():
        log(f"resuming YOLO from {last}")
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as e:            # raised when the run had already finished
            log(f"nothing to resume: {e}")
    else:
        YOLO(model).train(**train_arguments(data_yaml, out_dir, name, epochs, mosaic, batch_size, seed, workers))
    log(f"YOLO ({name}) finished in {time.time() - t0:.0f} s")
    return last


def predict(weights, entries, root, batch_size: int = cfg.BATCH_SIZE) -> dict:
    """{image: (boxes (N, 4), scores (N,))} with the shared inference settings."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    paths = [str(Path(root) / e["image"]) for e in entries]
    result = {}
    for start in range(0, len(paths), 256):
        for r in model.predict(paths[start:start + 256], imgsz=cfg.IMAGE_SIZE, conf=cfg.SCORE_THRESHOLD,
                               iou=cfg.NMS_IOU, max_det=cfg.MAX_DETECTIONS, batch=batch_size, verbose=False,
                               stream=True):
            result[image_key(r.path)] = (r.boxes.xyxy.cpu().numpy().reshape(-1, 4).astype(float),
                                         r.boxes.conf.cpu().numpy().reshape(-1).astype(float))
    return result
