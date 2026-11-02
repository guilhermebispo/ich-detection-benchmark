# Statistical analysis plan

Pre-specified analysis plan for Article 1 (RetinaNet × YOLO for intracranial haemorrhage detection on
head CT). It was written and **frozen on 29 October 2026, before any model was trained on the full
experiment**; only the pilot (§8) had been run. After freezing, this file changes only in §10
(deviations) and in the *Registration* line below.

| Item | Value |
|---|---|
| Version | 1.1 (29 Oct 2026; 1.0 plus pilot results and YOLO pipeline fixes, §8) |
| Frozen | 29 Oct 2026: this file at git tag `artigo1-protocolo-v1.1` |
| Registration | OSF Registries, Open-Ended Registration, https://osf.io/yu48s (registered 29 Oct 2026, 17:50; under embargo; project https://osf.io/gkzu8) |
| Code that implements it | `src/ichdet/` (`evaluate.py`, `metrics.py`, `stats.py`, `training_config.py`, `yolo.py`, `retinanet.py`), v0.1.1, frozen with this plan |

---

## 1. Question and hypothesis

**Question.** When RetinaNet (MONAI) and YOLO (Ultralytics) are trained and evaluated under the same
conditions — same images, same patient-level folds, same input, same training budget, same evaluator —
is there a relevant difference in detection performance?

**Hypothesis (equivalence).** The difference in the primary endpoint, RetinaNet − YOLO, lies within
±0.05. The literature suggests no clear winner between the two architectures (Cheng et al., 2023:
0.34–0.38; Lee et al., 2026: depends on subtype), while the evaluation protocol alone changes results
by 0.45 (Tapia et al., 2024); see the related-work section of the article.

## 2. Data and units of analysis

| Set | Role | Exams | With haemorrhage | Normal | Source |
|---|---|---|---|---|---|
| CQ500 + BHX boxes (India) | Training and validation, 5-fold cross-validation | 436 | 215 | 221 | `data/particao_5partes.csv` |
| CT-ICH (Iraq) | Test (different hospital and country) | 75 | 36 | 39 | `data/ctich_contagens.json` |

- **Unit of resampling: the exam.** Each CQ500 and CT-ICH exam belongs to a different patient, so
  "exam" and "patient" are the same unit here. Slices of one exam are never split across folds.
- Folds (validation exams / slices): 87/3,034, 89/3,073, 85/2,921, 87/3,000, 88/3,001
  (`data/export_512_summary.json`).
- **Excluded from the primary analysis:** 23 equivocal exams (1 or 2 of 3 radiologists reported
  haemorrhage, no boxes) — used only in a sensitivity analysis; 32 exams without a usable thick series
  or with boxes only on thin or contrast series.
- Reference standard: BHX hand-drawn boxes (CQ500) and boxes derived from the CT-ICH masks, one class
  ("haemorrhage"); boxes are per slice, in the 512 × 512 pixel space of the exported images.

## 3. Detectors and arms

| Arm | Model | Role |
|---|---|---|
| **A** | RetinaNet, MONAI 1.5.2 (detection code unchanged from 1.4.0), official LUNA16 recipe adapted to 2D (ATSS, hard-negative sampling, Smooth L1); ResNet-50 backbone with ImageNet weights | Primary |
| **B** | YOLOv8m, Ultralytics 8.3, COCO weights, **without mosaic** | Primary comparator |
| C | YOLOv8m as B, **with mosaic** (closed in the last 10 epochs) | Secondary: effect of mosaic |

B is the primary comparator because it receives exactly the same augmentation as A; C measures what
mosaic, which only YOLO uses by default, adds on top (Kothala et al., 2023).

**Shared training settings** (`src/ichdet/training_config.py`):

| Setting | Value |
|---|---|
| Input | The same PNG file for both: 512 × 512, three HU windows (brain 40/80, subdural 80/200, bone 600/2800) as channels |
| Epochs / batch | 50 / 16; one optimiser step per batch (no gradient accumulation: Ultralytics `nbs` = 16) |
| Optimiser | SGD, Nesterov, lr 0.01, momentum 0.9, weight decay 5 × 10⁻⁴ |
| Schedule | Linear warm-up for 1 epoch (every lr from 0, momentum fixed at 0.9), cosine decay to 1% of the initial lr |
| Augmentation | Horizontal flip (p = 0.5) only; C adds mosaic. Ultralytics' default Albumentations filters are removed |
| Mixed precision | Yes |
| Checkpoint used | **Last epoch.** No early stopping and no checkpoint selection: the held-out fold is never seen during training |
| Inference | Score ≥ 0.001, NMS IoU 0.5, at most 100 boxes per slice |
| Seed | 2026; one training run per fold and arm |

RetinaNet anchors: three base shapes (32 × 32, 45 × 23, 23 × 45 px) scaled ×1, ×2, ×4, ×8 across the FPN
levels; 32 px is about the 5th percentile of the longest box side in BHX (14.9 mm at 0.488 mm/px).
Model size: YOLOv8m (26 M parameters) is the YOLOv8 size closest to RetinaNet-R50 (about 34 M).
Remaining implementation difference, kept as part of each library: Ultralytics applies weight decay only
to convolution weights (not to biases and normalisation layers), while the RetinaNet code applies it to
all parameters.

## 4. Primary endpoint

**AP@0.5** (average precision at IoU 0.5; with a single class this equals mAP@0.5), computed by the
project's own evaluator for both detectors, following the COCO rules: detections of each slice sorted by
score and truncated to 100; greedy matching to the unmatched ground-truth box with the highest IoU ≥ 0.5;
precision interpolated at 101 recall points. Slices without haemorrhage are included, so false positives
on normal exams lower the AP.

**Validation statistic:** the mean over the 5 folds of the AP@0.5 of each fold's model on its own held-out
exams. Averaging per fold avoids pooling scores of five different models, whose calibration can differ.

**Difference:** Δ = AP@0.5(A) − AP@0.5(B).

## 5. Equivalence test

- **Margin:** ±0.05 in AP@0.5, fixed a priori (rationale in §1).
- **Test:** two one-sided tests (TOST) at α = 0.05 each, by the interval rule: equivalence if the 90%
  confidence interval of Δ lies entirely inside (−0.05, +0.05).
- **Interval:** percentile bootstrap, **paired** (both detectors evaluated on the same resampled exams),
  **stratified by fold** (exams resampled with replacement within each fold), 10,000 resamples, seed
  20261029. Exams drawn k times count k times (implemented as exam weights).
- **Pre-specified reading of the result:**

| 90% CI of Δ | Conclusion |
|---|---|
| Entirely inside (−0.05, +0.05) | Equivalent |
| Excludes 0 and extends beyond the margin | Relevant difference (report direction) |
| Includes 0 but crosses a margin | Inconclusive (not "no difference") |

The 95% CI of Δ and of each detector's AP are also reported.

## 6. Secondary endpoints (descriptive)

Reported with 95% bootstrap CIs, no hypothesis tests and no multiplicity adjustment; only §5 is
confirmatory.

1. mAP@0.5:0.95 (COCO IoU thresholds 0.50:0.05:0.95).
2. FROC: box-level sensitivity at 0.5, 1, 2 and 4 false positives per exam (IoU 0.5).
3. Exam-level AUC: exam score = highest box score; positive = exam with at least one box.
4. **Mosaic effect:** C − B and A − C, same bootstrap as §5.
5. **Test set (CT-ICH):** the 5 fold models of each arm predict every test exam; statistic = mean
   AP@0.5 of the 5 models; bootstrap over the 75 test exams (same draws for all models); Δ with 90% and
   95% CI. The test set is used once, after all models are final.
6. **Generalisation drop:** validation − test AP@0.5 for each arm (independent bootstraps of the two sets).
7. Per-subtype sensitivity at 1 false positive per exam (exploratory; epidural has 6 exams, so no
   separate inference).
8. Inference latency on CPU (median per slice), descriptive.

## 7. Sensitivity analyses

1. Pooled out-of-fold AP@0.5 (all 436 exams together) instead of the mean over folds.
2. Equivocal exams (23) added to the validation as negatives.
3. Exams labelled positive by BHX but not by the majority of the CQ500 reads removed (count to be taken
   from `data/` when the analysis is run).

## 8. Pilot and expected precision

The sample size is fixed by the data (436 validation exams, 75 test exams), so the question is whether
the study is informative: whether the 90% CI of Δ will be narrower than the margin.

- **Pilot** (notebook `05_piloto.ipynb`): A and B trained once on folds 2–5 for 5 epochs, evaluated on
  fold 1 (87 exams).
- **Allowed uses:** fix pipeline errors; measure GPU time per epoch; estimate the bootstrap SE of Δ.
  **Not allowed:** tuning hyperparameters or changing anything in §3–§7 because of pilot results.
- **Projection** (`ichdet.pilot`, `ichdet.stats`): half-width of the final 90% CI ≈ 1.645 × SE_pilot ×
  √(87 / 436), since the final statistic averages 5 folds of the pilot's size. If the projected
  half-width is not below 0.05, this is declared as a limitation **before** the full experiment.
- Caveat: pilot models are under-trained, so the SE is a planning estimate only.

**Pilot results (run on 29/10/2026, NVIDIA L4, Colab; `data/pilot_summary.json`).**

| Item | Value |
|---|---|
| Pipeline | Ran end to end: unzip, both trainings, prediction, evaluator, summary. Counts matched notebook 04 (training 11,995 slices; validation 3,034 slices from 87 exams) |
| Training time per epoch (11,995 slices, batch 16) | RetinaNet 587 s; YOLOv8m 141 s |
| Prediction time (3,034 slices) | RetinaNet 514 s; YOLOv8m 48 s |
| Projected training time, full experiment (50 epochs × 5 folds) | RetinaNet 40.7 h; YOLO without mosaic 9.8 h; YOLO with mosaic 9.8 h (assumed equal); **total 60.4 GPU-hours (L4)**, plus about 2 h of prediction |
| Bootstrap SE of Δ (pilot fold, 87 exams) | 0.033 |
| Projected half-width of the final 90% CI (436 exams) | **0.025**, half the margin: the study is informative |
| Exams needed for 90% CI half-width < 0.05 if the true Δ is 0 | 106 (436 available) |

Pilot AP values (RetinaNet 0.14, YOLO 0.38 after 5 epochs) are **not** interpreted; models were under-
trained and, per the rules above, nothing in §3–§7 was changed because of them.

**Pipeline errors found in the pilot log and fixed** (allowed use; neither depends on the pilot AP):
1. *Gradient accumulation in YOLO.* Ultralytics' nominal batch size (`nbs` = 64) made YOLO accumulate
   gradients over 4 batches (effective batch 64, a quarter of the optimiser steps) and rescale weight
   decay, while RetinaNet steps every batch of 16. Fixed with `nbs` = batch size.
2. *Extra YOLO augmentation.* With `albumentations` installed (as in Colab), Ultralytics adds Blur,
   MedianBlur, ToGray and CLAHE (p = 0.01 each), which §3 does not allow. Removed with an empty
   `augmentations` list (checked: the pipeline then contains no Albumentations transform).
3. *Warm-up.* Ultralytics starts biases at lr 0.1 and momentum at 0.8 during warm-up; set to the
   RetinaNet warm-up (every lr from 0, momentum 0.9).

These fixes make the code match §3 as written. The pilot was not repeated: they change the number of
optimiser steps, not the time per epoch, and the SE is only a planning estimate.

## 9. Implementation and verification

- One evaluator for both detectors (`src/ichdet/`), independent of the metrics built into MONAI and
  Ultralytics, which differ in interpolation, matching and NMS defaults.
- **Tests** (`tests/`, `pytest`): 38 tests, including AP compared with pycocotools on 20 random cases
  (identical to 10⁻⁹ at all 10 IoU thresholds), exam weights equal to duplicating exams (the basis of
  the bootstrap), CSV round trip, TOST decision rule and sample-size consistency, the pilot budget projection, and
  the YOLO training arguments (no gradient accumulation, shared warm-up, horizontal flip as the only
  augmentation).
- **Finding during verification:** pycocotools silently ignores unmatched boxes with negative area; our
  evaluator rejects invalid boxes (x2 < x1 or y2 < y1) with an error instead.
- End-to-end smoke test (CPU, synthetic images): both detectors train, resume from checkpoint, predict
  and are compared by the evaluator.
- Speed: 10,000 resamples for 5 folds × 2 detectors take under a minute.

## 10. Deviations

Any change after freezing (to this plan or to the frozen `ichdet` 0.1.1 code that affects results) is
listed here with date, what changed, why, and whether it was decided before or after seeing results. The
YOLO fixes of §8 were made before freezing.

| Date | Change | Reason | Before/after results |
|---|---|---|---|
| 31 Oct 2026 | **Deviation 1 (amendment v1.2).** `retinanet.py`: boxes are swapped from [x1, y1, x2, y2] to MONAI's [row1, col1, row2, col2] on input and back on output (`_swap_xy`); nothing else changes. New fingerprint `a6f3c0fc5798879d` (v1.1: `c3752d247a0173f1`), git tag `artigo1-protocolo-v1.2`. The five RetinaNet jobs are retrained with this code | Implementation error: MONAI reads box coordinate *i* along tensor axis *i*, so in v1.1 every RetinaNet target lay at the position mirrored across the image diagonal. The YOLO arms, the evaluator and the statistics are unaffected and byte-identical to v1.1 | **After** the results of the first run (all arms, validation and test) were known |

### Analysis after deviation 1

- **Corrected arm (A′).** Five RetinaNet jobs, same folds, settings, epochs and seed as in §3, trained with
  the v1.2 code into a separate output folder; the v1.1 RetinaNet models and predictions are kept.
- **Primary and secondary analyses.** Exactly as in §4–7, with A′ in place of A and the YOLO predictions
  of the first run (arms B and C, code unchanged). The same reading table, margin, bootstrap and seed.
- **Test set.** The CT-ICH results of arms B and C were computed once in the first run and are not
  recomputed. The test-set comparisons involving A′ are computed once, after the five A′ jobs, and stored
  the same way.
- **Reporting.** The manuscript reports the corrected comparison as the main result, states that it
  follows an amendment made after the first results were known, and gives the first-run RetinaNet
  results and the cause of the error. Because the YOLO results were known before A′ was trained, the
  corrected comparison is not blind; no setting of A′ is tuned, so the only change between A and A′ is the
  box axis order.

## 11. Draft *Methods* paragraph (statistical analysis)

> The primary endpoint was the average precision at an IoU threshold of 0.5 (AP@0.5), computed with a
> single evaluator for both detectors following the COCO definition (101-point interpolation, at most 100
> detections per slice), with slices without haemorrhage included. For cross-validation, AP@0.5 was
> computed for each fold's model on its held-out exams and averaged over the five folds. Equivalence
> between RetinaNet and YOLO was assessed with two one-sided tests at α = 0.05, using an equivalence
> margin of ±0.05 fixed before training: equivalence was concluded if the 90% confidence interval of the
> difference lay entirely within the margin. Confidence intervals were obtained by a paired, fold-
> stratified percentile bootstrap over exams (10,000 resamples), so that slices of the same patient were
> resampled together. Secondary endpoints (mAP@0.5:0.95, FROC sensitivity, exam-level AUC, the effect of
> mosaic augmentation and performance on the CT-ICH test set) are reported descriptively with 95%
> confidence intervals. The analysis plan was registered before the experiment [OSF link].

## References

- Cheng et al. (2023); Lee et al. (2026); Tapia et al. (2024); Kothala et al. (2023) — full references in the article.
- Lin T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV — AP definition.
- Schuirmann D.J. (1987). A comparison of the two one-sided tests procedure and the power approach for
  assessing the equivalence of average bioavailability. J Pharmacokinet Biopharm 15:657–680 — TOST.
- Efron B., Tibshirani R.J. (1993). An Introduction to the Bootstrap. Chapman & Hall — percentile bootstrap.
