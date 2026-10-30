import numpy as np
import pytest

from ichdet.data_io import load_predictions, save_predictions
from ichdet.evaluate import compare_crossval, compare_test
from ichdet.stats import bootstrap_weights, projected_half_width, required_exams, tost

from test_metrics import random_case


def test_bootstrap_weights_keep_stratum_sizes_and_are_reproducible():
    a = list(bootstrap_weights([5, 3], 4, seed=1))
    b = list(bootstrap_weights([5, 3], 4, seed=1))
    for wa, wb in zip(a, b):
        assert [w.sum() for w in wa] == [5, 3]
        assert all(np.array_equal(x, y) for x, y in zip(wa, wb))


def test_tost_decision_rule():
    rng = np.random.default_rng(0)
    assert tost(0.0, rng.normal(0, 0.01, 5000), margin=0.05)["equivalent"]
    assert not tost(0.04, rng.normal(0.04, 0.02, 5000), margin=0.05)["equivalent"]


def test_sample_size_is_consistent_with_projection():
    n = required_exams(pilot_se=0.03, pilot_exams=87, margin=0.05)
    assert projected_half_width(0.03, 87, n) <= 0.05 < projected_half_width(0.03, 87, n - 1)


def test_same_detector_is_equivalent_to_itself():
    folds = []
    for k in range(3):
        gt, preds = random_case(k)
        folds.append((gt, preds, preds))
    r = compare_crossval(folds, n_resamples=200, seed=1)
    assert r["equivalence_A_minus_B"]["difference"] == 0
    assert r["equivalence_A_minus_B"]["ci90"] == [0, 0]
    assert r["equivalence_A_minus_B"]["equivalent"]


def test_test_set_comparison_runs_with_several_models():
    gt, p1 = random_case(1)
    _, p2 = random_case(2)
    p2 = {k: v for k, v in p2.items() if k in gt}
    r = compare_test(gt, [p1, p2], [p2, p1], n_resamples=100, seed=1)
    assert r["A"]["ap50"] == pytest.approx(r["B"]["ap50"])     # same models, swapped order


def test_predictions_csv_round_trip(tmp_path):
    _, preds = random_case(3)
    save_predictions(tmp_path / "p.csv", preds)
    back = load_predictions(tmp_path / "p.csv")
    assert set(back) == set(preds)
    for k in preds:
        np.testing.assert_allclose(back[k][0], preds[k][0], atol=1e-3)
        np.testing.assert_allclose(back[k][1], preds[k][1], atol=1e-6)


def test_pilot_summary_budget_and_projection():
    from ichdet.pilot import summarize

    comparison = {"equivalence_A_minus_B": {"difference": 0.01, "bootstrap_se": 0.04}}
    s = summarize(comparison, {"retinanet": 360.0, "yolo_no_mosaic": 180.0}, pilot_exams=87)
    hours = s["estimated_training_gpu_hours_full_experiment"]
    assert hours["retinanet"] == 25.0 and hours["yolo_mosaic"] == hours["yolo_no_mosaic"] == 12.5   # 50 epochs x 5 folds
    p = s["precision_projection"]
    assert p["projected_ci90_half_width"] == pytest.approx(1.645 * 0.04 * (87 / 436) ** 0.5, rel=1e-3)
    assert p["informative"] is True
