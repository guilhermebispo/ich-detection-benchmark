"""Pre-specified comparisons of two detectors (A = RetinaNet, B = YOLO), as in the statistical protocol.

Validation (CQ500/BHX, 5 folds): each fold's model is evaluated on its own held-out exams; the primary
statistic is the mean over folds of AP@0.5, and the bootstrap resamples exams within each fold.

Test (CT-ICH): the 5 fold models of each detector predict every test exam; the statistic is the mean
AP@0.5 of the 5 models, and the bootstrap resamples test exams (the same draws for all models).
"""

from __future__ import annotations

import numpy as np

from .metrics import COCO_IOU_THRESHOLDS, DetectionTable
from .stats import paired_bootstrap, percentile_ci, tost

MARGIN = 0.05
N_RESAMPLES = 10_000
SEED = 20261029


def _summary(tables_a, tables_b, weights_of, strata_sizes, n_resamples, seed, margin):
    """Shared core: tables_x[i] is evaluated with weights_of(w)[i]; returns primary and secondary results."""

    def mean_ap50(tables, w):
        return float(np.nanmean([t.ap50(wi) for t, wi in zip(tables, weights_of(w))]))

    def primary(w):
        a, b = mean_ap50(tables_a, w), mean_ap50(tables_b, w)
        return a, b, a - b

    point, reps = paired_bootstrap(primary, strata_sizes, n_resamples, seed)
    result = {
        "primary_endpoint": "mean AP@0.5 (single class: hemorrhage)",
        "A": {"ap50": point[0], "ci95": percentile_ci(reps[:, 0], 0.95)},
        "B": {"ap50": point[1], "ci95": percentile_ci(reps[:, 1], 0.95)},
        "equivalence_A_minus_B": tost(point[2], reps[:, 2], margin),
        "seed": seed,
    }
    ones = weights_of([np.ones(n) for n in strata_sizes])
    for name, tables in (("A", tables_a), ("B", tables_b)):
        result[name]["per_model_ap50"] = [t.ap50(w) for t, w in zip(tables, ones)]
        result[name]["map50_95"] = float(np.nanmean([t.map50_95(w) for t, w in zip(tables, ones)]))
        froc = [t.froc(exam_weights=w) for t, w in zip(tables, ones)]
        result[name]["froc_sensitivity"] = {f: float(np.mean([d[f] for d in froc])) for f in froc[0]}
        result[name]["exam_auc"] = float(np.nanmean([t.exam_auc(w) for t, w in zip(tables, ones)]))
    return result


def compare_crossval(folds, n_resamples=N_RESAMPLES, seed=SEED, margin=MARGIN) -> dict:
    """folds: list of (ground_truth, predictions_A, predictions_B), one per validation fold."""
    tables_a = [DetectionTable.build(gt, pa, COCO_IOU_THRESHOLDS) for gt, pa, _ in folds]
    tables_b = [DetectionTable.build(gt, pb, COCO_IOU_THRESHOLDS) for gt, _, pb in folds]
    for a, b in zip(tables_a, tables_b):
        assert a.exams == b.exams
    sizes = [len(t.exams) for t in tables_a]
    result = _summary(tables_a, tables_b, lambda w: w, sizes, n_resamples, seed, margin)
    result.update(design="validation, stratified by fold", folds=len(folds), exams=int(sum(sizes)))
    return result


def compare_test(ground_truth, models_a, models_b, n_resamples=N_RESAMPLES, seed=SEED, margin=MARGIN) -> dict:
    """models_x: list of predictions on the test set, one per fold model of detector x."""
    tables_a = [DetectionTable.build(ground_truth, p, COCO_IOU_THRESHOLDS) for p in models_a]
    tables_b = [DetectionTable.build(ground_truth, p, COCO_IOU_THRESHOLDS) for p in models_b]
    n = len(tables_a[0].exams)
    result = _summary(tables_a, tables_b, lambda w: [w[0]] * len(tables_a), [n], n_resamples, seed, margin)
    result.update(design="test, same exams for every model", models_per_detector=len(models_a), exams=n)
    return result
