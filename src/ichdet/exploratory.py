"""Exploratory endpoints added after the registered analysis (not part of the statistical plan).

Requested at the supervision meeting of 5 Nov 2026. Nothing here changes the frozen matching, AP or
bootstrap: every metric is computed from the same `DetectionTable` and the same exam weights, so the
bootstrap intervals are comparable with those of the registered endpoints.

Endpoints:
- AP at IoU 0.75 (stricter localisation than the primary AP@0.5);
- precision-recall curve at IoU 0.5 and exam-level ROC curve (the AUC is already a registered endpoint);
- precision, recall and F1 of boxes, and exam-level sensitivity and specificity, at the operating point of
  one false positive per exam;
- box-level sensitivity by box size (COCO area ranges, in pixels of the 512 x 512 input);
- training loss per epoch, read from the logs of each job.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .final import detected_ground_truth, score_at_fp_rate
from .metrics import COCO_IOU_THRESHOLDS, COCO_RECALL_POINTS, DetectionTable
from .stats import paired_bootstrap, percentile_ci

# COCO area ranges (pixels^2). At 0.488 mm per pixel, 32 px = 15.6 mm and 96 px = 46.8 mm.
SIZE_BINS = (("small", 0.0, 32.0 ** 2), ("medium", 32.0 ** 2, 96.0 ** 2), ("large", 96.0 ** 2, np.inf))
ROC_FPR_GRID = np.linspace(0.0, 1.0, 101)
METRICS = ("ap50", "ap75", "map50_95", "exam_auc", "froc_1fp", "precision_1fp", "recall_1fp", "f1_1fp",
           "exam_sensitivity_1fp", "exam_specificity_1fp")


def _iou_index(table: DetectionTable, iou: float) -> int:
    hits = np.flatnonzero(np.isclose(table.iou_thresholds, iou))
    if not len(hits):
        raise ValueError(f"table was built without IoU {iou}")
    return int(hits[0])


def ap_at(table: DetectionTable, iou: float, exam_weights=None) -> float:
    """COCO AP at one IoU threshold (101-point interpolation)."""
    return float(table.average_precision(exam_weights, [_iou_index(table, iou)])[0])


def at_threshold(table: DetectionTable, score: float, exam_weights=None, iou: float = 0.5) -> dict:
    """Box and exam metrics when only detections with score >= `score` are kept.

    Box level: precision = TP / (TP + FP), recall = TP / reference boxes, F1. Exam level: an exam is
    flagged when its highest box score reaches `score`; sensitivity over exams with reference boxes and
    specificity over exams without them. Exam weights follow `DetectionTable` (bootstrap counts).
    """
    w = table._weights(exam_weights)
    t = _iou_index(table, iou)
    keep = table.scores >= score
    wd = w[table.exam_of_det]
    tp = float(np.sum(wd * (table.tp[t] & keep)))
    fp = float(np.sum(wd * (~table.tp[t] & keep)))
    n_pos = float(np.sum(w * table.n_gt))
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / n_pos if n_pos > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    positive = table.n_gt > 0
    flagged = table.exam_score >= score
    wp, wn = float(np.sum(w[positive])), float(np.sum(w[~positive]))
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "exam_sensitivity": float(np.sum(w[positive & flagged]) / wp) if wp else float("nan"),
        "exam_specificity": float(np.sum(w[~positive & ~flagged]) / wn) if wn else float("nan"),
    }


def metrics_of(table: DetectionTable, threshold: float, exam_weights=None) -> dict:
    """All exploratory scalar metrics of one model, plus the registered ones for reference."""
    op = at_threshold(table, threshold, exam_weights)
    w = table._weights(exam_weights)
    if not np.sum(w * table.n_gt):        # a resample without any reference box (frozen FROC would fail)
        return {k: float("nan") for k in METRICS}
    return {
        "ap50": table.ap50(exam_weights), "ap75": ap_at(table, 0.75, exam_weights),
        "map50_95": table.map50_95(exam_weights), "exam_auc": table.exam_auc(exam_weights),
        "froc_1fp": table.froc((1.0,), exam_weights=exam_weights)[1.0],
        "precision_1fp": op["precision"], "recall_1fp": op["recall"], "f1_1fp": op["f1"],
        "exam_sensitivity_1fp": op["exam_sensitivity"], "exam_specificity_1fp": op["exam_specificity"],
    }


def summarise(tables_by_arm: dict, thresholds_by_arm: dict, weights_of, strata_sizes, n_resamples: int,
              seed: int, reference_arm: str | None = None) -> dict:
    """Mean over fold models of every metric, with 95% bootstrap intervals, per arm and paired differences.

    Args:
        tables_by_arm: `{arm: [DetectionTable of each fold model]}`, all built on the same exam lists.
        thresholds_by_arm: `{arm: [score threshold of each fold model]}`, fixed before resampling (the
            operating point of 1 FP per exam on the unresampled data).
        weights_of: Maps the per-stratum bootstrap weights to one weight array per table (as in
            `evaluate._summary`).
        strata_sizes: Number of exams in each bootstrap stratum.
        n_resamples, seed: Bootstrap settings (the same draws for all arms, so differences are paired).
        reference_arm: If given, differences `arm - reference_arm` are also reported.
    """
    arms = list(tables_by_arm)

    def statistic(w):
        ws = weights_of(w)
        out = []
        for arm in arms:
            per_model = [metrics_of(t, s, wi) for t, s, wi in zip(tables_by_arm[arm], thresholds_by_arm[arm], ws)]
            out.extend(float(np.nanmean([m[k] for m in per_model])) for k in METRICS)
        return out

    point, reps = paired_bootstrap(statistic, strata_sizes, n_resamples, seed)
    n = len(METRICS)
    result = {"metrics": list(METRICS), "seed": seed, "n_resamples": n_resamples, "arms": {}}
    for i, arm in enumerate(arms):
        block = {}
        for j, k in enumerate(METRICS):
            c = i * n + j
            block[k] = {"value": float(point[c]), "ci95": percentile_ci(reps[:, c], 0.95)}
        block["thresholds_1fp"] = [float(s) for s in thresholds_by_arm[arm]]
        result["arms"][arm] = block
    if reference_arm is not None:
        r = arms.index(reference_arm)
        result["differences"] = {}
        for i, arm in enumerate(arms):
            if arm == reference_arm:
                continue
            result["differences"][f"{arm}_minus_{reference_arm}"] = {
                k: {"value": float(point[i * n + j] - point[r * n + j]),
                    "ci95": percentile_ci(reps[:, i * n + j] - reps[:, r * n + j], 0.95)}
                for j, k in enumerate(METRICS)}
    return result


def pr_curve(table: DetectionTable, iou: float = 0.5) -> list[float]:
    """Interpolated precision at the 101 COCO recall points (its mean is the AP at `iou`)."""
    t = _iou_index(table, iou)
    n_pos = float(np.sum(table.n_gt))
    if n_pos == 0 or not len(table.scores):
        return [0.0] * len(COCO_RECALL_POINTS)
    tp_cum = np.cumsum(table.tp[t]).astype(float)
    fp_cum = np.cumsum(~table.tp[t]).astype(float)
    recall = tp_cum / n_pos
    precision = np.maximum.accumulate((tp_cum / (tp_cum + fp_cum))[::-1])[::-1]
    idx = np.searchsorted(recall, COCO_RECALL_POINTS, side="left")
    return np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0).tolist()


def roc_curve(table: DetectionTable) -> list[float]:
    """Exam-level ROC (score = highest box score): sensitivity at each false-positive rate of `ROC_FPR_GRID`.

    The step curve is evaluated at the grid, taking the highest sensitivity reached without exceeding each
    false-positive rate, so curves of several fold models can be averaged vertically.
    """
    positive = table.n_gt > 0
    s_pos, s_neg = table.exam_score[positive], table.exam_score[~positive]
    cuts = np.concatenate([[np.inf], np.unique(table.exam_score)[::-1]])
    tpr = np.array([np.mean(s_pos >= c) for c in cuts])
    fpr = np.array([np.mean(s_neg >= c) for c in cuts])
    return [float(tpr[fpr <= f].max()) for f in ROC_FPR_GRID]


def size_bin(box) -> str:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    return next(name for name, lo, hi in SIZE_BINS if lo <= area < hi)


def size_sensitivity(folds, thresholds) -> dict:
    """Box-level sensitivity per size bin at each fold model's own operating point; counts summed over folds.

    Args:
        folds: One `(ground_truth, predictions)` pair per fold model.
        thresholds: Score threshold of each fold model (same order as `folds`).
    """
    counts = {name: [0, 0] for name, _, _ in SIZE_BINS}
    for (gt, pred), threshold in zip(folds, thresholds):
        for image, (_, boxes) in gt.items():
            boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
            if not len(boxes):
                continue
            db, ds = pred.get(image, (np.zeros((0, 4)), np.zeros(0)))
            for box, hit in zip(boxes, detected_ground_truth(db, ds, boxes, threshold)):
                c = counts[size_bin(box)]
                c[0] += int(hit)
                c[1] += 1
    return {k: {"detected": d, "boxes": n, "sensitivity": d / n if n else float("nan")}
            for k, (d, n) in counts.items()}


def thresholds_1fp(tables) -> list[float]:
    return [score_at_fp_rate(t, 1.0) for t in tables]


def loss_history(directory, arm: str) -> dict:
    """Training loss per epoch of one job, from its own log.

    RetinaNet (`history.json`): `loss` = classification (focal) + box regression (smooth L1). YOLO
    (Ultralytics `results.csv`): `box_loss` (CIoU), `cls_loss` (BCE), `dfl_loss` (distribution focal) and
    their sum. The two families of losses are on different scales and are not comparable between
    detectors; they show the convergence of each arm.
    """
    d = Path(directory)
    if arm == "retinanet":
        h = json.loads((d / "history.json").read_text())
        return {"epoch": [int(r["epoch"]) for r in h], "loss": [float(r["loss"]) for r in h]}
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(d / "results.csv"))]
    out = {"epoch": [int(float(r["epoch"])) for r in rows]}
    for k in ("box_loss", "cls_loss", "dfl_loss"):
        out[k] = [float(r[f"train/{k}"]) for r in rows]
    out["loss"] = [b + c + f for b, c, f in zip(out["box_loss"], out["cls_loss"], out["dfl_loss"])]
    return out


def build_tables(ground_truth_list, predictions_list) -> list[DetectionTable]:
    return [DetectionTable.build(g, p, COCO_IOU_THRESHOLDS) for g, p in zip(ground_truth_list, predictions_list)]
