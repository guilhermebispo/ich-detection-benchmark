# RetinaNet versus YOLO for intracranial haemorrhage detection on head CT

Code for a pre-registered, controlled comparison of RetinaNet (MONAI) and YOLOv8 (Ultralytics) for the
localisation of intracranial haemorrhage on non-contrast head CT. Both detectors read the same images,
with the same patient-level folds, training budget, optimiser and augmentation, and are scored by a single
evaluator. Training and validation use CQ500 with the BHX boxes (five-fold cross-validation by patient);
testing uses CT-ICH, an independent dataset from another country.

> **If you use this code, please cite the article** (see [Citation](#citation) and
> [`CITATION.cff`](CITATION.cff); GitHub's *Cite this repository* button gives the reference).

**Status.** The article is in preparation. The analysis plan was frozen and registered on OSF Registries
before the full experiment (registration under embargo until submission). The modules that determine the
results are frozen with the plan, and their fingerprint is checked by the tests. The notebooks, the derived
data files and the evidence documents will be added after the final evaluation.

## Layout

| Path | Content |
|---|---|
| `src/ichdet/` | Python package: a single evaluator for both detectors (`metrics.py`, `stats.py`, `evaluate.py`), training and prediction (`retinanet.py`, `yolo.py`), shared settings (`training_config.py`), pilot summary (`pilot.py`) and the job runner of the full experiment (`experiment.py`) |
| `tests/` | Automated tests (`pytest`), including cross-checks against pycocotools |

## Installation and tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest
```

Training and prediction also need `torch`, `torchvision`, `monai==1.5.2` and `ultralytics>=8.3,<8.4`; the
experiments were run on Google Colab with NVIDIA L4 GPUs.

## Quick start: evaluating your own detector

The evaluator does not depend on MONAI or Ultralytics. Any detector can be scored with the same box
matching, average precision, patient-level bootstrap and equivalence test used in the article, so that new
results are directly comparable.

**Ground truth** is a JSON file with a `validation` list; each entry is one slice, with its image path, the
reference boxes `[x1, y1, x2, y2]` in pixels (an empty list for slices without haemorrhage) and the exam it
belongs to:

```json
{"validation": [
  {"image": "images/exam01_s012.png", "box": [[100.0, 120.0, 160.0, 190.0]], "exam": "exam01"},
  {"image": "images/exam01_s013.png", "box": [], "exam": "exam01"}
]}
```

**Predictions** are a CSV file with one row per detected box, in the same pixel space; slices without
detections have no rows:

```text
image,x1,y1,x2,y2,score
exam01_s012.png,98.4,118.9,161.2,191.7,0.913
```

Images are matched by file name. Every slice in the ground truth is evaluated, including slices without
haemorrhage, so false positives on normal exams lower the score.

```python
from ichdet import data_io, evaluate
from ichdet.metrics import DetectionTable

# One detector on one set of slices
gt = data_io.load_monai_split("fold1.json")
pred = data_io.load_predictions("detector_a_fold1.csv")
table = DetectionTable.build(gt, pred)
print(table.ap50(), table.map50_95(), table.froc())   # AP@0.5, mAP@0.5:0.95, FROC sensitivity

# Two detectors in cross-validation: one (ground truth, detector A, detector B) triple per fold
folds = [(data_io.load_monai_split(f"fold{k}.json"),
          data_io.load_predictions(f"detector_a_fold{k}.csv"),
          data_io.load_predictions(f"detector_b_fold{k}.csv")) for k in (1, 2, 3, 4, 5)]
result = evaluate.compare_crossval(folds)              # 10,000 paired bootstrap resamples by default
eq = result["equivalence_A_minus_B"]
print(result["A"]["ap50"], result["B"]["ap50"])        # mean AP@0.5 over folds
print(eq["difference"], eq["ci90"], eq["equivalent"])  # A - B, its 90% CI, equivalence within ±0.05
```

`compare_crossval` resamples whole exams within each fold, so slices of the same patient are always drawn
together. For a separate test set predicted by several models, use `evaluate.compare_test`. The result also
reports 95% intervals, mAP@0.5:0.95, FROC sensitivity at 0.5, 1, 2 and 4 false positives per exam, and
exam-level AUC.

## Frozen code

`ichdet.experiment.FROZEN_MODULES` lists the modules that determine the results (data reading, matching and
average precision, bootstrap and equivalence test, training settings, RetinaNet and YOLO training and
prediction). Their fingerprint, `ichdet.experiment.frozen_fingerprint`, must equal the value registered with
the analysis plan (`c3752d247a0173f1`); `tests/test_experiment.py` checks it. Because these files must
stay byte-identical to the registered version, the docstring of `training_config.py` still refers to the
analysis plan by its original file name in the research repository
(`evidencias/F3-05-statistical-protocol.md`); the plan itself will be published here as an English
document.

## Data

No images are distributed here. The datasets must be obtained from their providers, under their own terms:

| Dataset | Use | Access |
|---|---|---|
| CQ500 (qure.ai) | Training and validation images | Public download, CC BY-NC-SA 4.0 and end-user licence agreement |
| BHX v1.1 (PhysioNet) | Bounding boxes on CQ500 | Open access, CC BY-SA 4.0 |
| CT-ICH v1.3.1 (PhysioNet) | Test set | Credentialed access under the PhysioNet Restricted Health Data Use Agreement; not redistributed |

See [`DATA_LICENSES.md`](DATA_LICENSES.md) for the references and licence notices.

## Citation

```bibtex
@unpublished{bispo2026ichdetection,
  title  = {RetinaNet versus {YOLO} for intracranial haemorrhage detection on head {CT}:
            a pre-registered equivalence study with cross-dataset testing},
  author = {Bispo, Guilherme Dantas and Serrano, Andr{\'e} Luiz Marques and Gon{\c{c}}alves, Vin{\'i}cius Pereira},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

## Licence

GNU Affero General Public License v3.0 ([`LICENSE`](LICENSE)). YOLO training and prediction use the
Ultralytics library, which is distributed under the same licence.

## Author and maintainer

The code and this repository are written and maintained by Guilherme Dantas Bispo, Department of
Electrical Engineering, University of Brasília (UnB), Brazil (guilherme.bispo@redes.unb.br,
ORCID [0000-0002-4938-2076](https://orcid.org/0000-0002-4938-2076)). The authors of the article are listed
in the citation above.
