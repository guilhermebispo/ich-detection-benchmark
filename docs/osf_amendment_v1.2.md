# OSF registration update — amendment v1.2 (deviation 1)

Text submitted as an update to the registration osf.io/yu48s ("Update" on the registration page; the
original version stays visible). Prepared on 31 Oct 2026, before the corrected RetinaNet jobs were started.

**Status:** accepted. The registration page shows "This is an update to the original registration. This
update was made on Oct 31, 2026", followed by this text as the reason for update (checked on 1 Nov 2026; the
registration is still embargoed).

---

**Summary of changes.** Correction of an implementation error in the RetinaNet arm; retraining of the five
RetinaNet models. No other part of the plan changes.

**What happened.** After the registered analysis was run (all three arms, validation and test), the
RetinaNet arm showed a very low AP@0.5 that did not improve between the 5-epoch pilot and the 50-epoch
run. Inspection showed that MONAI reads box coordinate *i* along spatial axis *i* of the image tensor
(rows first), whereas the exported boxes are [x1, y1, x2, y2] with x along columns. Every RetinaNet
training target was therefore placed at the position mirrored across the image diagonal. The YOLO arms,
the evaluator and the statistical code are not affected.

**Change.** `ichdet/retinanet.py` swaps the box axes on input to MONAI and back on output; nothing else in
the RetinaNet configuration changes (folds, epochs, optimiser, schedule, augmentation, anchors, seed).
Code: `ichdet` 0.4.0, git tag `artigo1-protocolo-v1.2`, fingerprint of the frozen modules
`a6f3c0fc5798879d` (registered version: `c3752d247a0173f1`). Only `retinanet.py` differs from the
registered code.

**Analysis.** The five RetinaNet models are retrained with the corrected code. The primary, secondary and
sensitivity analyses are repeated exactly as registered, with the corrected RetinaNet arm and the YOLO
predictions of the first run. The CT-ICH results of the YOLO arms are not recomputed; the test-set
comparisons involving the corrected RetinaNet are computed once.

**Timing and reporting.** This amendment was made after the first results were known. The manuscript will
report the corrected comparison as the main result, state that it follows this amendment, and give the
first-run RetinaNet results with the cause of the error. Since the YOLO results were known before the
corrected RetinaNet was trained, the corrected comparison is not blind; no RetinaNet setting is tuned.
