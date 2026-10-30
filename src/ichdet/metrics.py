"""Box matching and detection metrics, shared by RetinaNet and YOLO.

Matching and average precision follow the COCO rules (pycocotools, area "all"): per image, detections
are sorted by score, truncated to `max_dets`, and each one is matched to the unmatched ground-truth box
with the highest IoU, if that IoU reaches the threshold. AP is the mean of the interpolated precision at
101 recall points. The tests check the results against pycocotools.

Every metric accepts one weight per exam. A weight of 2 counts the exam twice and a weight of 0 removes
it, which is how the patient-level bootstrap resamples exams without rebuilding the tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COCO_IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)
COCO_RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N, 4) and b (M, 4), both as (x1, y1, x2, y2) in pixels."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_image(det_boxes, det_scores, gt_boxes, iou_thresholds=COCO_IOU_THRESHOLDS, max_dets=100):
    """Match one image's detections to its ground truth.

    Returns (scores, tp): the kept scores in descending order and a boolean array (T, N) telling, for
    each IoU threshold, whether each kept detection is a true positive.
    """
    det_boxes = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    det_scores = np.asarray(det_scores, dtype=float).reshape(-1)
    order = np.argsort(-det_scores, kind="mergesort")[:max_dets]
    det_boxes, det_scores = det_boxes[order], det_scores[order]
    thresholds = np.atleast_1d(np.asarray(iou_thresholds, dtype=float))
    tp = np.zeros((len(thresholds), len(det_scores)), dtype=bool)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    if len(det_scores) == 0 or len(gt_boxes) == 0:
        return det_scores, tp
    ious = iou_matrix(det_boxes, gt_boxes)
    for t, thr in enumerate(thresholds):
        taken = np.zeros(len(gt_boxes), dtype=bool)
        for d in range(len(det_scores)):
            best, best_iou = -1, min(thr, 1 - 1e-10)
            for g in range(len(gt_boxes)):
                if not taken[g] and ious[d, g] >= best_iou:
                    best, best_iou = g, ious[d, g]
            if best >= 0:
                taken[best] = True
                tp[t, d] = True
    return det_scores, tp


@dataclass
class DetectionTable:
    """All detections of one model on one set of exams, matched once and sorted by score.

    exams: exam ids, in a fixed order; weights passed to the metrics follow this order.
    exam_of_det: index into `exams` for each detection (sorted by descending score).
    scores: detection scores, descending.
    tp: (T, N) true-positive flags for each IoU threshold in `iou_thresholds`.
    n_gt: ground-truth boxes per exam.
    exam_score: highest detection score per exam (0 when the model found nothing), for exam-level AUC.
    """

    exams: list[str]
    exam_of_det: np.ndarray
    scores: np.ndarray
    tp: np.ndarray
    n_gt: np.ndarray
    exam_score: np.ndarray
    iou_thresholds: np.ndarray

    @classmethod
    def build(cls, ground_truth, predictions, iou_thresholds=COCO_IOU_THRESHOLDS, max_dets=100):
        """ground_truth: {image: (exam, gt_boxes)}; predictions: {image: (boxes, scores)}.

        Every image in `ground_truth` is evaluated, including images without boxes (their detections are
        false positives). Predictions for images outside `ground_truth` are an error, because they mean
        that the model and the evaluation used different image lists.
        """
        extra = set(predictions) - set(ground_truth)
        if extra:
            raise ValueError(f"{len(extra)} predicted images are not in the ground truth, e.g. {sorted(extra)[:3]}")
        # pycocotools silently ignores unmatched boxes with negative area; here they are an error
        for source, items in (("ground truth", ((k, v[1]) for k, v in ground_truth.items())),
                              ("predictions", ((k, v[0]) for k, v in predictions.items()))):
            for image, boxes in items:
                b = np.asarray(boxes, dtype=float).reshape(-1, 4)
                if np.any(b[:, 2] < b[:, 0]) or np.any(b[:, 3] < b[:, 1]):
                    raise ValueError(f"{source}: box with x2 < x1 or y2 < y1 in {image}")
        thresholds = np.atleast_1d(np.asarray(iou_thresholds, dtype=float))
        exams = sorted({exam for exam, _ in ground_truth.values()})
        index = {e: i for i, e in enumerate(exams)}
        n_gt = np.zeros(len(exams))
        exam_score = np.zeros(len(exams))
        all_scores, all_tp, all_exam = [], [], []
        for image in sorted(ground_truth):
            exam, gt_boxes = ground_truth[image]
            e = index[exam]
            n_gt[e] += len(np.asarray(gt_boxes).reshape(-1, 4))
            boxes, scores = predictions.get(image, (np.zeros((0, 4)), np.zeros(0)))
            kept, tp = match_image(boxes, scores, gt_boxes, thresholds, max_dets)
            if len(kept):
                exam_score[e] = max(exam_score[e], kept[0])
            all_scores.append(kept)
            all_tp.append(tp)
            all_exam.append(np.full(len(kept), e))
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        tp = np.concatenate(all_tp, axis=1) if all_tp else np.zeros((len(thresholds), 0), dtype=bool)
        exam_of_det = np.concatenate(all_exam).astype(int) if all_exam else np.zeros(0, dtype=int)
        order = np.argsort(-scores, kind="mergesort")
        return cls(exams, exam_of_det[order], scores[order], tp[:, order], n_gt, exam_score, thresholds)

    def _weights(self, exam_weights):
        if exam_weights is None:
            return np.ones(len(self.exams))
        w = np.asarray(exam_weights, dtype=float)
        if w.shape != (len(self.exams),):
            raise ValueError(f"expected {len(self.exams)} exam weights, got {w.shape}")
        return w

    def average_precision(self, exam_weights=None, thresholds=None) -> np.ndarray:
        """COCO AP (101-point interpolation) for each IoU threshold (or the given threshold indices);
        NaN when there is no ground truth."""
        w = self._weights(exam_weights)
        ts = range(len(self.iou_thresholds)) if thresholds is None else thresholds
        n_pos = float(np.sum(w * self.n_gt))
        if n_pos == 0:
            return np.full(len(ts), np.nan)
        wd = w[self.exam_of_det]
        result = np.zeros(len(ts))
        if len(wd) == 0:
            return result
        for i, t in enumerate(ts):
            tp_cum = np.cumsum(wd * self.tp[t])
            fp_cum = np.cumsum(wd * ~self.tp[t])
            recall = tp_cum / n_pos
            denom = tp_cum + fp_cum
            precision = np.divide(tp_cum, denom, out=np.zeros_like(tp_cum), where=denom > 0)
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            idx = np.searchsorted(recall, COCO_RECALL_POINTS, side="left")
            q = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
            result[i] = q.mean()
        return result

    def ap50(self, exam_weights=None) -> float:
        """AP at IoU 0.5 (the primary endpoint, with a single class this is mAP@0.5)."""
        t = int(np.flatnonzero(np.isclose(self.iou_thresholds, 0.5))[0])
        return float(self.average_precision(exam_weights, [t])[0])

    def map50_95(self, exam_weights=None) -> float:
        """Mean AP over IoU 0.50:0.05:0.95 (needs the ten COCO thresholds in the table)."""
        if not np.allclose(self.iou_thresholds, COCO_IOU_THRESHOLDS):
            raise ValueError("table was built without the COCO IoU thresholds")
        return float(np.mean(self.average_precision(exam_weights)))

    def froc(self, fp_per_exam=(0.5, 1, 2, 4), iou=0.5, exam_weights=None) -> dict[float, float]:
        """Box-level sensitivity at given mean false positives per exam (FROC operating points)."""
        w = self._weights(exam_weights)
        t = int(np.flatnonzero(np.isclose(self.iou_thresholds, iou))[0])
        wd = w[self.exam_of_det]
        n_pos, n_exams = float(np.sum(w * self.n_gt)), float(np.sum(w))
        sens = np.concatenate([[0.0], np.cumsum(wd * self.tp[t]) / n_pos]) if n_pos else np.zeros(1)
        fp_rate = np.concatenate([[0.0], np.cumsum(wd * ~self.tp[t]) / n_exams])
        return {float(f): float(sens[fp_rate <= f].max()) for f in fp_per_exam}

    def exam_auc(self, exam_weights=None) -> float:
        """AUC of exam-level detection: score = highest box score, label = exam has a ground-truth box."""
        w = self._weights(exam_weights)
        pos, neg = self.n_gt > 0, self.n_gt == 0
        wp, wn = w[pos], w[neg]
        if wp.sum() == 0 or wn.sum() == 0:
            return float("nan")
        sp, sn = self.exam_score[pos][:, None], self.exam_score[neg][None, :]
        wins = (sp > sn) + 0.5 * (sp == sn)
        return float((wp[:, None] * wn[None, :] * wins).sum() / (wp.sum() * wn.sum()))
