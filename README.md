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
