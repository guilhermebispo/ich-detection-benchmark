"""Exploratory endpoints (meeting of 5 Nov 2026): consistency with the frozen evaluator."""

import json

import numpy as np
import pytest

from ichdet import exploratory as ex
from ichdet.final import score_at_fp_rate
from ichdet.metrics import COCO_IOU_THRESHOLDS, DetectionTable
from test_final import _fold


@pytest.fixture
def folds():
    rng = np.random.default_rng(11)
    return [_fold(rng, 8, f"F{k}") for k in range(3)]


def _table(gt, pred):
    return DetectionTable.build(gt, pred, COCO_IOU_THRESHOLDS)


def test_ap_at_matches_the_frozen_average_precision(folds):
    gt, pa, _ = folds[0]
    t = _table(gt, pa)
    assert ex.ap_at(t, 0.5) == pytest.approx(t.ap50())
    assert ex.ap_at(t, 0.75) == pytest.approx(t.average_precision(thresholds=[5])[0])
    with pytest.raises(ValueError):
        ex.ap_at(t, 0.3)


def test_mean_of_the_precision_recall_curve_is_ap50(folds):
    for gt, pa, pb in folds:
        for p in (pa, pb):
            t = _table(gt, p)
            assert np.mean(ex.pr_curve(t)) == pytest.approx(t.ap50())


def test_recall_at_the_1fp_threshold_is_the_frozen_froc_sensitivity(folds):
    gt, pa, _ = folds[1]
    t = _table(gt, pa)
    op = ex.at_threshold(t, score_at_fp_rate(t, 1.0))
    assert op["recall"] == pytest.approx(t.froc((1.0,))[1.0])
    assert op["f1"] == pytest.approx(2 * op["precision"] * op["recall"] / (op["precision"] + op["recall"]))


def test_exam_sensitivity_and_specificity_by_hand():
    box = np.array([[10, 10, 50, 50]], float)
    gt = {"p1_0.png": ("p1", box), "p2_0.png": ("p2", box), "n1_0.png": ("n1", np.zeros((0, 4))),
          "n2_0.png": ("n2", np.zeros((0, 4)))}
    pred = {"p1_0.png": (box, np.array([0.9])), "p2_0.png": (box, np.array([0.2])),
            "n1_0.png": (box, np.array([0.8])), "n2_0.png": (box, np.array([0.1]))}
    op = ex.at_threshold(_table(gt, pred), 0.5)
    assert op["exam_sensitivity"] == 0.5 and op["exam_specificity"] == 0.5
    assert op["precision"] == 0.5 and op["recall"] == 0.5
    # weights: counting p1 twice and removing n1
    op = ex.at_threshold(_table(gt, pred), 0.5, exam_weights=[0, 1, 2, 1])   # exams sorted: n1, n2, p1, p2
    assert op["exam_specificity"] == 1.0 and op["exam_sensitivity"] == pytest.approx(2 / 3)


def test_roc_curve_is_monotone_and_reaches_one(folds):
    gt, pa, _ = folds[2]
    curve = ex.roc_curve(_table(gt, pa))
    assert len(curve) == len(ex.ROC_FPR_GRID)
    assert all(b >= a for a, b in zip(curve, curve[1:]))
    assert curve[-1] == 1.0
    # area under the step curve on the grid is close to the frozen exam AUC
    assert np.trapezoid(curve, ex.ROC_FPR_GRID) == pytest.approx(_table(gt, pa).exam_auc(), abs=0.1)


def test_size_sensitivity_adds_up_to_the_froc_sensitivity(folds):
    pairs = [(gt, pa) for gt, pa, _ in folds]
    tables = [_table(g, p) for g, p in pairs]
    thresholds = ex.thresholds_1fp(tables)
    sizes = ex.size_sensitivity(pairs, thresholds)
    detected = sum(v["detected"] for v in sizes.values())
    boxes = sum(v["boxes"] for v in sizes.values())
    assert boxes == sum(int(t.n_gt.sum()) for t in tables)
    expected = sum(t.froc((1.0,))[1.0] * t.n_gt.sum() for t in tables)
    assert detected == pytest.approx(expected)


def test_size_bins_follow_coco_areas():
    assert ex.size_bin([0, 0, 31, 31]) == "small"
    assert ex.size_bin([0, 0, 32, 32]) == "medium"
    assert ex.size_bin([0, 0, 96, 96]) == "large"


def test_summary_points_are_means_over_fold_models_and_differences_are_paired(folds):
    tables = {"a": [_table(g, pa) for g, pa, _ in folds], "b": [_table(g, pb) for g, _, pb in folds]}
    thr = {k: ex.thresholds_1fp(v) for k, v in tables.items()}
    sizes = [len(t.exams) for t in tables["a"]]
    r = ex.summarise(tables, thr, lambda w: w, sizes, n_resamples=50, seed=1, reference_arm="b")
    for arm in ("a", "b"):
        assert r["arms"][arm]["ap50"]["value"] == pytest.approx(np.mean([t.ap50() for t in tables[arm]]))
        assert r["arms"][arm]["ap75"]["value"] == pytest.approx(np.mean([ex.ap_at(t, 0.75) for t in tables[arm]]))
        lo, hi = r["arms"][arm]["ap50"]["ci95"]
        assert lo <= r["arms"][arm]["ap50"]["value"] <= hi
    d = r["differences"]["a_minus_b"]["ap50"]["value"]
    assert d == pytest.approx(r["arms"]["a"]["ap50"]["value"] - r["arms"]["b"]["ap50"]["value"])


def test_loss_history_reads_both_logs(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "history.json").write_text(json.dumps([{"epoch": 1, "loss": 1.5, "seconds": 9},
                                                             {"epoch": 2, "loss": 1.1, "seconds": 9}]))
    assert ex.loss_history(tmp_path / "r", "retinanet") == {"epoch": [1, 2], "loss": [1.5, 1.1]}
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / "results.csv").write_text(
        "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss\n1,10,1.0,2.0,0.5\n2,20,0.8,1.5,0.4\n")
    h = ex.loss_history(tmp_path / "y", "yolo_no_mosaic")
    assert h["epoch"] == [1, 2] and h["loss"] == pytest.approx([3.5, 2.7])
