"""Read the ground truth exported by notebook 04 and the predictions of either detector.

Ground truth: the MONAI split files (`cq500/monai/fold<k>.json`, `ctich/monai/external.json`), whose
entries carry `image`, `box` (x1, y1, x2, y2 in pixels of the 512 x 512 image) and `exam`.

Predictions: one CSV per model and image set, columns `image,x1,y1,x2,y2,score`, boxes in the same
pixel space. An image with no detections simply has no rows. Images are keyed by file name, so
`images/a.png` and `a.png` refer to the same image.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PREDICTION_COLUMNS = ["image", "x1", "y1", "x2", "y2", "score"]


def image_key(path: str) -> str:
    return Path(path).name


def load_monai_split(path, section: str = "validation") -> dict:
    """{image: (exam, boxes (M, 4))} for one section of a MONAI split file."""
    entries = json.loads(Path(path).read_text())[section]
    return {image_key(e["image"]): (e["exam"], np.asarray(e["box"], dtype=float).reshape(-1, 4)) for e in entries}


def load_predictions(path) -> dict:
    """{image: (boxes (N, 4), scores (N,))} from a predictions CSV."""
    rows = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(PREDICTION_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for r in reader:
            rows[image_key(r["image"])].append([float(r[k]) for k in PREDICTION_COLUMNS[1:]])
    return {k: (np.asarray(v)[:, :4], np.asarray(v)[:, 4]) for k, v in rows.items()}


def save_predictions(path, predictions: dict) -> None:
    """Write {image: (boxes, scores)} in the shared CSV format."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PREDICTION_COLUMNS)
        for image in sorted(predictions):
            boxes, scores = predictions[image]
            for b, s in zip(np.asarray(boxes).reshape(-1, 4), np.asarray(scores).reshape(-1)):
                w.writerow([image, *(f"{v:.3f}" for v in b), f"{s:.6f}"])
