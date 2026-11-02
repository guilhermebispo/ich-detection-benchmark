"""Analyses of the final evaluation (notebook 07) built on the frozen evaluator."""

import numpy as np
import pytest

from ichdet import evaluate, final
from ichdet.metrics import COCO_IOU_THRESHOLDS, DetectionTable, match_image


def _fold(rng, exams, prefix, n_slices=4):
    """Synthetic fold: half the exams with one box per slice; noisy detections of two 'models'."""
    gt, pa, pb = {}, {}, {}
    for e in range(exams):
        exam = f"{prefix}-{e}"
        for s in range(n_slices):
            image = f"{exam}_{s}.png"
            boxes = np.array([[100, 100, 160, 170]], float) + rng.integers(0, 50) if e % 2 == 0 else np.zeros((0, 4))
            gt[image] = (exam, boxes)
            for pred, jitter in ((pa, 4), (pb, 12)):
                shift = np.tile(rng.normal(0, jitter, (len(boxes), 2)), 2)
                corner = rng.uniform(0, 400, (2, 2))
                db = np.vstack([boxes + shift, np.hstack([corner, corner + 40])])
                pred[image] = (db, rng.uniform(0, 1, len(db)))
    return gt, pa, pb


@pytest.fixture
def folds():
    rng = np.random.default_rng(7)
    return [_fold(rng, 6, f"F{k}") for k in range(3)]


def test_reading_follows_the_pre_specified_table():
    assert final.reading([-0.03, 0.04]) == "equivalent"
    assert final.reading([0.02, 0.09]) == "relevant difference"
    assert final.reading([-0.12, -0.06]) == "relevant difference"
    assert final.reading([-0.02, 0.07]) == "inconclusive"
    assert final.reading([-0.06, 0.01]) == "inconclusive"


def test_detected_ground_truth_follows_the_frozen_matching(folds):
    gt, pa, _ = folds[0]
    for image, (_, boxes) in gt.items():
        db, ds = pa[image]
        _, tp = match_image(db, ds, boxes, [0.5])
        assert final.detected_ground_truth(db, ds, boxes, -np.inf).sum() == tp.sum()


def test_operating_point_reproduces_the_froc_sensitivity(folds):
    gt, pa, _ = folds[0]
    table = DetectionTable.build(gt, pa, COCO_IOU_THRESHOLDS)
    threshold = final.score_at_fp_rate(table, 1.0)
    hits = sum(final.detected_ground_truth(*pa[i], b, threshold).sum() for i, (_, b) in gt.items())
    assert hits / table.n_gt.sum() == pytest.approx(table.froc((1.0,))[1.0])


def test_subtype_sensitivity_counts_every_box(folds):
    subtypes = {i: ["Subdural" if i.startswith("F0") else "Epidural"] * len(b) for f in folds for i, (_, b) in f[0].items()}
    result = final.subtype_sensitivity([(g, a) for g, a, _ in folds], subtypes)
    assert sum(r["boxes"] for r in result.values()) == sum(len(b) for f in folds for _, b in f[0].values())
    assert set(result) == {"Subdural", "Epidural"}


def test_pooled_comparison_of_a_single_fold_equals_the_primary_analysis(folds):
    pooled = final.compare_pooled(folds[:1], n_resamples=200)
    primary = evaluate.compare_crossval(folds[:1], n_resamples=200)
    assert pooled["equivalence_A_minus_B"] == primary["equivalence_A_minus_B"]


def test_pooled_comparison_keeps_the_fold_strata(folds):
    pooled = final.compare_pooled(folds, n_resamples=200)
    assert pooled["exams"] == 18 and np.isfinite(pooled["equivalence_A_minus_B"]["difference"])


def test_generalisation_drop_is_validation_minus_test(folds):
    test_gt, ta, _ = folds[2]
    r = final.generalisation_drop([(g, a) for g, a, _ in folds[:2]], test_gt, [ta, ta], n_resamples=200)
    assert r["drop"] == pytest.approx(r["validation_ap50"] - r["test_ap50"])
    assert r["ci95"][0] <= r["drop"] <= r["ci95"][1]


def test_subset_and_merge(folds):
    gt, pa, _ = folds[0]
    g, p = final.subset(gt, pa, ["F0-0"])
    assert all(v[0] != "F0-0" for v in g.values()) and set(p) == set(g)
    with pytest.raises(AssertionError):
        final.merge((gt, pa), (gt, pa))


def test_box_subtypes_match_by_coordinates(tmp_path):
    csv = tmp_path / "boxes_512.csv"
    csv.write_text("arquivo,tipo,x1,y1,x2,y2\na.png,Subdural,1.0001,2,3,4\na.png,Epidural,10,20,30,40\n")
    gt = {"a.png": ("E", np.array([[10, 20, 30, 40], [1, 2, 3, 4]], float)), "b.png": ("E", np.zeros((0, 4)))}
    assert final.box_subtypes(gt, csv) == {"a.png": ["Epidural", "Subdural"], "b.png": []}
    with pytest.raises(ValueError):
        final.box_subtypes({"a.png": ("E", np.array([[0, 0, 5, 5]], float))}, csv)
