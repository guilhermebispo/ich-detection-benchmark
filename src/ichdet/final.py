"""Final evaluation (notebook 07): the analyses of the frozen statistical plan, run once after the 15 jobs.

The primary comparison and the test-set comparison are computed by the frozen `evaluate` module. This
module only assembles the inputs of the other pre-specified analyses (§6 and §7 of the plan) from the
frozen building blocks (`metrics`, `stats`, `evaluate`), without changing how boxes are matched, how AP is
computed or how exams are resampled:

- mosaic effect (C - B, A - C) and test-set comparisons: `evaluate.compare_crossval` and `compare_test`;
- generalisation drop (validation - test AP@0.5 of each arm), with independent bootstraps of the two sets;
- per-subtype sensitivity at 1 false positive per exam (exploratory, counts only);
- sensitivity analyses: pooled out-of-fold AP, equivocal exams added as negatives, exams with BHX boxes
  but a negative CQ500 majority removed;
- inference latency on CPU (median per slice).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from . import evaluate
from .experiment import job_dir
from .metrics import COCO_IOU_THRESHOLDS, DetectionTable, iou_matrix
from .stats import paired_bootstrap, percentile_ci

ARM_LABEL = {"retinanet": "A", "yolo_no_mosaic": "B", "yolo_mosaic": "C"}


def reading(ci90, margin: float = evaluate.MARGIN) -> str:
    """Pre-specified reading of the 90% confidence interval of a difference (section 5 of the plan).

    Args:
        ci90: Lower and upper limits of the 90% interval of the difference in AP@0.5.
        margin: Equivalence margin.

    Returns:
        `"equivalent"` if the interval lies inside (-margin, margin), `"relevant difference"` if it
        excludes zero otherwise, and `"inconclusive"` if it includes zero and crosses a margin.
    """
    lo, hi = ci90
    if -margin < lo and hi < margin:
        return "equivalent"
    if lo > 0 or hi < 0:
        return "relevant difference"
    return "inconclusive"


def weights_path(out_root, fold: int, arm: str) -> Path:
    """Path of the last-epoch weights of one training job.

    Args:
        out_root: Root directory of the experiment outputs.
        fold: Held-out fold (1 to 5).
        arm: Arm name.

    Returns:
        `last.pt` in the job directory for RetinaNet, `weights/last.pt` for YOLO.
    """
    d = job_dir(out_root, fold, arm)
    return d / "last.pt" if arm == "retinanet" else d / "weights" / "last.pt"


def predict(arm: str, weights, entries, images_root) -> dict:
    """Predictions of one trained model with the frozen inference code and settings.

    Args:
        arm: Arm name.
        weights: Path of the model weights.
        entries: MONAI split entries to predict (`image` paths relative to `images_root`).
        images_root: Directory that the entries' image paths are relative to.

    Returns:
        `{image: (boxes (N, 4) as [x1, y1, x2, y2] in pixels, scores (N,))}`.
    """
    if arm == "retinanet":
        from . import retinanet
        return retinanet.predict(weights, entries, images_root)
    from . import yolo
    return yolo.predict(weights, entries, images_root)


def subset(ground_truth: dict, predictions: dict, drop_exams) -> tuple[dict, dict]:
    """Remove the slices of some exams from a ground truth and its predictions.

    Args:
        ground_truth: `{image: (exam, boxes (M, 4))}`.
        predictions: `{image: (boxes (N, 4), scores (N,))}`.
        drop_exams: Exam identifiers to remove.

    Returns:
        `(ground_truth, predictions)` restricted to the remaining slices.
    """
    drop = set(drop_exams)
    gt = {k: v for k, v in ground_truth.items() if v[0] not in drop}
    return gt, {k: v for k, v in predictions.items() if k in gt}


def merge(*pairs) -> tuple[dict, dict]:
    """Union of several `(ground_truth, predictions)` pairs over disjoint sets of images.

    Args:
        *pairs: `(ground_truth, predictions)` dictionaries as in `subset`.

    Returns:
        `(ground_truth, predictions)` containing all images.

    Raises:
        AssertionError: If two pairs share an image.
    """
    gt, pred = {}, {}
    for g, p in pairs:
        assert not set(g) & set(gt), "image sets overlap"
        gt.update(g)
        pred.update(p)
    return gt, pred


def compare_pooled(folds, n_resamples=evaluate.N_RESAMPLES, seed=evaluate.SEED, margin=evaluate.MARGIN) -> dict:
    """Sensitivity analysis 1: AP@0.5 of the pooled out-of-fold predictions (all exams in one table).

    The bootstrap is the same as in the primary analysis: exams resampled within each fold, same seed.

    Args:
        folds: One `(ground_truth, predictions_A, predictions_B)` triple per fold, as for
            `evaluate.compare_crossval`.
        n_resamples: Number of bootstrap resamples.
        seed: Seed of the bootstrap.
        margin: Equivalence margin.

    Returns:
        The same summary as `evaluate.compare_crossval` (AP@0.5 of each detector with 95% intervals, the
        equivalence test of A - B and the secondary metrics), computed on the pooled table.
    """
    gt, pa = merge(*[(g, a) for g, a, _ in folds])
    _, pb = merge(*[(g, b) for g, _, b in folds])
    table_a = DetectionTable.build(gt, pa, COCO_IOU_THRESHOLDS)
    table_b = DetectionTable.build(gt, pb, COCO_IOU_THRESHOLDS)
    fold_exams = [sorted({e for e, _ in g.values()}) for g, _, _ in folds]
    position = {e: i for i, e in enumerate(table_a.exams)}
    order = np.array([position[e] for exams in fold_exams for e in exams])

    def weights_of(w):
        pooled = np.empty(len(order))
        pooled[order] = np.concatenate(w)
        return [pooled]

    sizes = [len(e) for e in fold_exams]
    result = evaluate._summary([table_a], [table_b], weights_of, sizes, n_resamples, seed, margin)
    result.update(design="pooled out-of-fold predictions, bootstrap stratified by fold", exams=int(sum(sizes)))
    return result


def _mean_ap50_replicates(tables, weights_of, strata_sizes, n_resamples, seed):
    def statistic(w):
        return [float(np.nanmean([t.ap50(wi) for t, wi in zip(tables, weights_of(w))]))]

    point, reps = paired_bootstrap(statistic, strata_sizes, n_resamples, seed)
    return float(point[0]), reps[:, 0]


def generalisation_drop(validation_folds, test_ground_truth, test_models, n_resamples=evaluate.N_RESAMPLES,
                        seed=evaluate.SEED) -> dict:
    """Drop from validation to test AP@0.5 of one arm (section 6.6 of the plan).

    The two sets are resampled independently; the test bootstrap uses `seed + 1`.

    Args:
        validation_folds: One `(ground_truth, predictions)` pair per fold.
        test_ground_truth: Ground truth of the test set, `{image: (exam, boxes (M, 4))}`.
        test_models: Predictions of each fold model on the test set.
        n_resamples: Number of bootstrap resamples.
        seed: Seed of the validation bootstrap.

    Returns:
        Dictionary with the validation and test AP@0.5 (means over the fold models), the drop, its 95%
        percentile interval and the two seeds.
    """
    val_tables = [DetectionTable.build(g, p, COCO_IOU_THRESHOLDS) for g, p in validation_folds]
    test_tables = [DetectionTable.build(test_ground_truth, p, COCO_IOU_THRESHOLDS) for p in test_models]
    val, val_reps = _mean_ap50_replicates(val_tables, lambda w: w, [len(t.exams) for t in val_tables],
                                          n_resamples, seed)
    n = len(test_tables[0].exams)
    test, test_reps = _mean_ap50_replicates(test_tables, lambda w: [w[0]] * len(test_tables), [n],
                                            n_resamples, seed + 1)
    drop = val_reps - test_reps
    return {"validation_ap50": val, "test_ap50": test, "drop": val - test,
            "ci95": percentile_ci(drop, 0.95), "seeds": [seed, seed + 1]}


def score_at_fp_rate(table: DetectionTable, fp_per_exam: float = 1.0, iou: float = 0.5) -> float:
    """Score threshold of the FROC operating point at a given false-positive rate.

    Args:
        table: Detections of one model on one set of exams.
        fp_per_exam: Mean number of false positives per exam allowed.
        iou: IoU threshold for a true positive.

    Returns:
        Lowest score whose mean number of false positives per exam does not exceed `fp_per_exam` (the
        operating point of `DetectionTable.froc`), or `inf` if even the first detection exceeds it.
    """
    t = int(np.flatnonzero(np.isclose(table.iou_thresholds, iou))[0])
    fp_rate = np.cumsum(~table.tp[t]) / len(table.exams)
    ok = np.flatnonzero(fp_rate <= fp_per_exam)
    return float(table.scores[ok[-1]]) if len(ok) else float("inf")


def detected_ground_truth(det_boxes, det_scores, gt_boxes, min_score: float, iou: float = 0.5,
                          max_dets: int = 100) -> np.ndarray:
    """Which ground-truth boxes of one slice are detected above a score threshold.

    Uses the greedy rule of `metrics.match_image` (detections by descending score, truncated to
    `max_dets`, each matched to the unmatched box with the highest IoU >= `iou`), keeping track of which
    box was taken.

    Args:
        det_boxes: Detected boxes (N, 4) as [x1, y1, x2, y2] in pixels.
        det_scores: Detection scores (N,).
        gt_boxes: Ground-truth boxes (M, 4) as [x1, y1, x2, y2] in pixels.
        min_score: Detections below this score are ignored.
        iou: IoU threshold for a match.
        max_dets: Maximum number of detections kept per slice.

    Returns:
        Boolean array (M,), `True` for each matched ground-truth box.
    """
    det_boxes = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    det_scores = np.asarray(det_scores, dtype=float).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-det_scores, kind="mergesort")[:max_dets]
    det_boxes, det_scores = det_boxes[order], det_scores[order]
    taken = np.zeros(len(gt_boxes), dtype=bool)
    if len(det_scores) == 0 or len(gt_boxes) == 0:
        return taken
    ious = iou_matrix(det_boxes, gt_boxes)
    for d in range(len(det_scores)):
        if det_scores[d] < min_score:
            break
        best, best_iou = -1, min(iou, 1 - 1e-10)
        for g in range(len(gt_boxes)):
            if not taken[g] and ious[d, g] >= best_iou:
                best, best_iou = g, ious[d, g]
        if best >= 0:
            taken[best] = True
    return taken


def box_subtypes(ground_truth: dict, boxes_csv) -> dict:
    """Haemorrhage subtype of every ground-truth box, read from the exported box table.

    Boxes are matched by coordinates, since the MONAI lists round them to 0.001 pixel.

    Args:
        ground_truth: `{image: (exam, boxes (M, 4))}`.
        boxes_csv: `boxes_512.csv` written by the export notebook (one row per box: image file name,
            subtype and x1, y1, x2, y2 in pixels).

    Returns:
        `{image: [subtype of each box, in the order of the ground truth]}`.

    Raises:
        ValueError: If a box does not match exactly one row of the table.
    """
    rows = {}
    with open(boxes_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["arquivo"], []).append((r["tipo"], [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    result = {}
    for image, (_, boxes) in ground_truth.items():
        types = []
        for b in np.asarray(boxes).reshape(-1, 4):
            match = [t for t, c in rows.get(image, []) if np.max(np.abs(np.asarray(c) - b)) < 0.01]
            if len(match) != 1:
                raise ValueError(f"{image}: box {b.tolist()} matches {len(match)} rows of boxes_512.csv")
            types.append(match[0])
        result[image] = types
    return result


def subtype_sensitivity(folds, subtypes: dict, fp_per_exam: float = 1.0) -> dict:
    """Box-level sensitivity per haemorrhage subtype at a fixed false-positive rate (section 6.7, exploratory).

    Each fold model uses its own operating point on its own held-out fold; detected and total boxes are
    summed over folds.

    Args:
        folds: One `(ground_truth, predictions)` pair per fold.
        subtypes: Output of `box_subtypes`.
        fp_per_exam: Mean number of false positives per exam at the operating point.

    Returns:
        `{subtype: {"detected", "boxes", "sensitivity"}}`.
    """
    counts = {}
    for gt, pred in folds:
        threshold = score_at_fp_rate(DetectionTable.build(gt, pred, COCO_IOU_THRESHOLDS), fp_per_exam)
        for image, (_, boxes) in gt.items():
            if not len(np.asarray(boxes).reshape(-1, 4)):
                continue
            db, ds = pred.get(image, (np.zeros((0, 4)), np.zeros(0)))
            hit = detected_ground_truth(db, ds, boxes, threshold)
            for kind, h in zip(subtypes[image], hit):
                c = counts.setdefault(kind, [0, 0])
                c[0] += int(h)
                c[1] += 1
    return {k: {"detected": d, "boxes": n, "sensitivity": d / n} for k, (d, n) in sorted(counts.items())}


def cpu_latency(arm: str, weights, entries, images_root, n_slices: int = 100, warmup: int = 5) -> dict:
    """Inference time on CPU, one slice at a time, with the shared inference settings (section 6.8).

    Args:
        arm: Arm name.
        weights: Path of the model weights.
        entries: MONAI split entries to time (the first `warmup + n_slices` are used).
        images_root: Directory that the entries' image paths are relative to.
        n_slices: Number of timed slices.
        warmup: Number of initial slices excluded from the timing.

    Returns:
        Median and 90th percentile of the seconds per slice, number of timed slices and number of CPU
        threads used by PyTorch.
    """
    import torch

    from . import training_config as cfg

    entries = list(entries)[:warmup + n_slices]
    times = []
    if arm == "retinanet":
        from . import retinanet
        detector = retinanet.build_detector(pretrained_backbone=False)
        detector.network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["network"])
        detector.eval()
        data = retinanet._Slices(entries, images_root, False)
        with torch.no_grad():
            for i in range(len(entries)):
                t0 = time.perf_counter()                 # reading the PNG counts, as in YOLO's predict
                detector([data[i][0]], use_inferer=False)
                times.append(time.perf_counter() - t0)
    else:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        for e in entries:
            path = str(Path(images_root) / e["image"])
            t0 = time.perf_counter()
            model.predict(path, imgsz=cfg.IMAGE_SIZE, conf=cfg.SCORE_THRESHOLD, iou=cfg.NMS_IOU,
                          max_det=cfg.MAX_DETECTIONS, device="cpu", verbose=False)
            times.append(time.perf_counter() - t0)
    measured = np.asarray(times[warmup:])
    return {"median_seconds_per_slice": float(np.median(measured)),
            "p90_seconds_per_slice": float(np.percentile(measured, 90)),
            "slices": int(len(measured)), "torch_threads": torch.get_num_threads()}
