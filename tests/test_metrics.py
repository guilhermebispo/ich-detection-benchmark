import numpy as np
import pytest

from ichdet.metrics import COCO_IOU_THRESHOLDS, DetectionTable, iou_matrix, match_image


def box(x, y, w, h):
    return [x, y, x + w, y + h]


def test_iou_known_values():
    a = np.array([box(0, 0, 10, 10)])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([box(5, 0, 10, 10)]))[0, 0] == pytest.approx(50 / 150)
    assert iou_matrix(a, np.array([box(20, 20, 5, 5)]))[0, 0] == 0.0


def test_match_prefers_higher_score_and_one_gt_per_detection():
    gt = [box(0, 0, 10, 10)]
    scores, tp = match_image([box(0, 0, 10, 10), box(1, 0, 10, 10)], [0.6, 0.9], gt, [0.5])
    assert list(scores) == [0.9, 0.6]
    assert tp.tolist() == [[True, False]]       # the duplicate is a false positive


def test_perfect_detector_scores_one_and_empty_detector_zero():
    gt = {"a.png": ("e1", [box(10, 10, 20, 20)]), "b.png": ("e2", np.zeros((0, 4)))}
    perfect = {"a.png": ([box(10, 10, 20, 20)], [0.9])}
    assert DetectionTable.build(gt, perfect).ap50() == pytest.approx(1.0)
    assert DetectionTable.build(gt, {}).ap50() == 0.0


def test_false_positive_on_normal_exam_lowers_ap():
    gt = {"a.png": ("e1", [box(10, 10, 20, 20)]), "b.png": ("e2", np.zeros((0, 4)))}
    preds = {"a.png": ([box(10, 10, 20, 20)], [0.5]), "b.png": ([box(0, 0, 5, 5)], [0.9])}
    # ranked: false positive (0.9), then the true positive (0.5) -> precision 0.5 at every recall point
    assert DetectionTable.build(gt, preds).ap50() == pytest.approx(0.5)


def test_predictions_outside_ground_truth_are_rejected():
    with pytest.raises(ValueError):
        DetectionTable.build({"a.png": ("e1", [])}, {"z.png": ([box(0, 0, 1, 1)], [0.5])})


def test_inverted_boxes_are_rejected():
    with pytest.raises(ValueError):
        DetectionTable.build({"a.png": ("e1", [])}, {"a.png": ([[10, 0, 5, 5]], [0.5])})


def random_case(seed, n_images=60, n_exams=12):
    rng = np.random.default_rng(seed)
    gt, preds = {}, {}
    for i in range(n_images):
        exam = f"e{i % n_exams}"
        k = 0 if i % n_exams < 3 else rng.integers(0, 3)      # exams e0-e2 are normal
        g = [box(*rng.uniform(0, 400, 2), *rng.uniform(10, 80, 2)) for _ in range(k)]
        gt[f"img{i}.png"] = (exam, np.array(g).reshape(-1, 4))
        dets, scores = [], []
        for b in g:                                  # noisy copies of the truth
            if rng.random() < 0.8:
                d = np.array(b) + rng.normal(0, 6, 4)
                dets.append([min(d[0], d[2]), min(d[1], d[3]), max(d[0], d[2]), max(d[1], d[3])])
                scores.append(rng.uniform(0.3, 1))
        for _ in range(rng.integers(0, 3)):          # false positives
            dets.append(box(*rng.uniform(0, 400, 2), *rng.uniform(10, 80, 2)))
            scores.append(rng.uniform(0, 0.8))
        if dets:
            preds[f"img{i}.png"] = (np.array(dets), np.array(scores))
    return gt, preds


def pycocotools_ap(gt, preds):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images = sorted(gt)
    ids = {im: i + 1 for i, im in enumerate(images)}
    anns, dets = [], []
    for im in images:
        for b in gt[im][1]:
            x1, y1, x2, y2 = b
            anns.append({"id": len(anns) + 1, "image_id": ids[im], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        for b, s in zip(*preds.get(im, (np.zeros((0, 4)), np.zeros(0)))):
            x1, y1, x2, y2 = b
            dets.append({"image_id": ids[im], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    coco = COCO()
    coco.dataset = {"images": [{"id": i} for i in ids.values()], "annotations": anns,
                    "categories": [{"id": 1, "name": "hemorrhage"}]}
    coco.createIndex()
    ev = COCOeval(coco, coco.loadRes(dets), "bbox")
    ev.params.areaRng, ev.params.areaRngLbl = [[0, 1e10]], ["all"]
    ev.evaluate()
    ev.accumulate()
    precision = ev.eval["precision"][:, :, 0, 0, -1]        # (T, R) for area "all", maxDets 100
    return precision.mean(axis=1)


@pytest.mark.parametrize("seed", range(20))
def test_matches_pycocotools(seed):
    pytest.importorskip("pycocotools")
    gt, preds = random_case(seed)
    ours = DetectionTable.build(gt, preds, COCO_IOU_THRESHOLDS).average_precision()
    np.testing.assert_allclose(ours, pycocotools_ap(gt, preds), atol=1e-9)


def test_exam_weights_equal_duplicating_exams():
    gt, preds = random_case(7)
    table = DetectionTable.build(gt, preds)
    w = np.random.default_rng(0).integers(0, 4, len(table.exams)).astype(float)
    dup_gt, dup_preds = {}, {}
    for e, k in zip(table.exams, w.astype(int)):
        for c in range(k):
            for im, (ex, g) in gt.items():
                if ex == e:
                    dup_gt[f"{c}_{im}"] = (f"{e}_{c}", g)
                    if im in preds:
                        dup_preds[f"{c}_{im}"] = preds[im]
    np.testing.assert_allclose(table.average_precision(w), DetectionTable.build(dup_gt, dup_preds).average_precision(), atol=1e-12)
    assert table.exam_auc(w) == pytest.approx(DetectionTable.build(dup_gt, dup_preds).exam_auc())


def test_froc_and_exam_auc_simple_case():
    gt = {"a.png": ("e1", [box(10, 10, 20, 20)]), "b.png": ("e2", np.zeros((0, 4)))}
    preds = {"a.png": ([box(10, 10, 20, 20)], [0.9]), "b.png": ([box(0, 0, 5, 5)], [0.3])}
    table = DetectionTable.build(gt, preds)
    assert table.froc(fp_per_exam=(0.25, 1)) == {0.25: 1.0, 1.0: 1.0}
    assert table.exam_auc() == 1.0
