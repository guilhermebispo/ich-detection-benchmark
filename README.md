# RetinaNet versus YOLOv8m for intracranial haemorrhage detection on head CT

Code, notebooks and derived data for a pre-registered, controlled comparison of RetinaNet (MONAI) and
YOLOv8m (Ultralytics) for the localisation of intracranial haemorrhage on non-contrast head CT. Both
detectors read the same images, with the same patient-level folds, training budget, optimiser and
augmentation, and are scored by a single evaluator. Training and validation use CQ500 with the BHX boxes
(five-fold cross-validation by patient); testing uses CT-ICH, an independent dataset from another country.

> **If you use this code, please cite the article** (see [Citation](#citation) and
> [`CITATION.cff`](CITATION.cff); GitHub's *Cite this repository* button gives the reference).

**Status.** The article is in preparation. The analysis plan was frozen and registered on OSF Registries
before the full experiment ([doi:10.17605/OSF.IO/YU48S](https://doi.org/10.17605/OSF.IO/YU48S)). The full pipeline, from the
download of the datasets to the figures of the article, is in this repository. The version used in the article
is archived on Zenodo: v1.0.0, [doi:10.5281/zenodo.24370357](https://doi.org/10.5281/zenodo.24370357) (all
versions: [doi:10.5281/zenodo.24370356](https://doi.org/10.5281/zenodo.24370356)).

## Layout

| Path | Content |
|---|---|
| `src/ichdet/` | Python package: a single evaluator for both detectors (`metrics.py`, `stats.py`, `evaluate.py`), training and prediction (`retinanet.py`, `yolo.py`), shared settings (`training_config.py`), the job runner of the experiment (`experiment.py`), the pilot summary (`pilot.py`), the remaining pre-specified analyses (`final.py`) and the exploratory endpoints added after the registered analysis (`exploratory.py`) |
| `tests/` | Automated tests (`pytest`), including cross-checks against pycocotools and tests of the RetinaNet box axes |
| `notebooks/` | The twelve Google Colab notebooks that ran the study, in order (see [Pipeline](#pipeline)) |
| `data/` | Derived data: cohort and series selection, the five folds, the BHX reference boxes used, counts, training records and the aggregated results (see [Derived data](#derived-data)) |
| `docs/` | The registered statistical analysis plan (`statistical_protocol.md`) and its amendment (`osf_amendment_v1.2.md`) |
| `tools/make_figures.py` | Draws the result figures and the tables of the article and its supplementary material from the files in `data/` |

## Pipeline

The notebooks run on Google Colab with the datasets on a private Google Drive folder. They are meant to be
run in this order:

| Notebook | What it does | Runtime |
|---|---|---|
| `01_dados_cq500_bhx` | Downloads CQ500, BHX and CT-ICH, checks checksums, repairs a corrupted CQ500 archive, links CQ500 examinations to BHX | CPU |
| `02_preparo_imagens` | Inventory of all series and selection of one thick-slice series per examination | CPU |
| `03_caixas_particao_ctich` | Final reference boxes, split into five patient-level folds, CT-ICH masks converted into boxes | CPU |
| `04_exportacao_monai_yolo` | Resampling to 0.488 mm, 512 × 512 crop, three HU windows, MONAI and YOLO labels, automated tests | CPU |
| `05_piloto` | Five-epoch pilot on one fold (pipeline check and planning of the experiment) | GPU |
| `06_treino_completo` | First run: the 15 training jobs (3 arms × 5 folds) | GPU, ~60 h |
| `07_avaliacao_final` | Evaluation of the first run, including the single evaluation of the test set | GPU |
| `08_retreino_retinanet` | Retraining of the five RetinaNet models with the corrected box axes (deviation 1) | GPU, ~41 h |
| `09_avaliacao_correcao` | Evaluation of the corrected RetinaNet against the YOLO predictions of the first run | GPU |
| `10_figura_exemplos` | Figure of qualitative examples, chosen by a rule fixed before looking at the images | CPU |
| `11_figura_preprocessamento` | Preprocessing figure of the article (same slice and geometry as notebook 04, without titles) | CPU |
| `12_metricas_exploratorias` | Exploratory endpoints from the stored predictions (AP@0.75, ROC and precision-recall curves, metrics at one false positive per examination, recall by box size, training loss); nothing is trained or predicted | CPU |

To run them, copy `src/ichdet` to `MyDrive/ich-benchmark/codigo/ichdet/` on your Drive. The notebooks
create the data folders under `MyDrive/ich-benchmark/dados/`. Long jobs resume from the last saved epoch
after a disconnection, and two Colab sessions can share the job list (see notebooks 06 and 08).

**Language of the notebooks.** The documentation cells and the code comments are in English. The code is
kept exactly as it was run: identifiers, printed messages, folder names and the column names of the files
that the notebooks exchange are in Portuguese, because changing them would change the code that produced
the results. A check in the authors' export script compares the syntax tree of every code cell with the
notebook that was run; the only difference allowed is the name of the Drive root folder. The notebooks are
published without outputs.

## Frozen code and the deviation from the plan

`ichdet.experiment.FROZEN_MODULES` lists the modules that determine the results (data reading, matching and
average precision, bootstrap and equivalence test, training settings, RetinaNet and YOLO training and
prediction). Their fingerprint, `ichdet.experiment.frozen_fingerprint`, proves which version ran:

| Version | Fingerprint | Used by | In this repository |
|---|---|---|---|
| Protocol v1.1 (registered plan) | `c3752d247a0173f1` | Notebooks 05 to 07 (first run) | Commit `1f61311` |
| Protocol v1.2 (amendment) | `a6f3c0fc5798879d` | Notebooks 08 to 12 | Current version (`ichdet` 0.5.0; 0.5.0 only adds the non-frozen `exploratory.py`) |

After the first run had been evaluated, the RetinaNet arm turned out to have been trained with transposed
boxes: MONAI reads box coordinate *i* along spatial axis *i* of the image tensor (rows first), whereas the
exported boxes are `[x1, y1, x2, y2]` with `x` along the columns. Version 0.4.0 swaps the axes on the way into
and out of MONAI (`retinanet._swap_xy`) and changes nothing else; `tests/test_retinanet.py` fails on the old
code. The correction was registered on OSF as an amendment before the five RetinaNet models were retrained
(`docs/osf_amendment_v1.2.md` and section 10 of `docs/statistical_protocol.md`). The YOLO arms, the evaluator
and the statistical code are byte-identical in both versions, so the YOLO models of the first run were kept.

Notebooks 06 and 07 check for fingerprint `c3752d247a0173f1` and therefore stop at their first cell with the
current package: to rerun them as they were run, use the package at commit `1f61311`. `tests/test_experiment.py`
checks the current fingerprint and, when the tags `artigo1-protocolo-v1.1` and `artigo1-protocolo-v1.2` are
present, that only `retinanet.py` changed between the two versions. The docstring of `training_config.py`
refers to the plan by its original file name in the research repository
(`evidencias/F3-05-statistical-protocol.md`); it cannot be edited without changing the fingerprint. The plan is
published here as `docs/statistical_protocol.md`, with the internal cross-references of the research
repository removed; the registered version on OSF is authoritative.

## Derived data

| File | Content |
|---|---|
| `data/cq500_series_escolhidas.csv` | One row per CQ500 examination: group, number of positive reads (`votos_ich`), selected series and whether it is the BHX series |
| `data/particao_5partes.csv` | Fold (`parte`) and haemorrhage subtypes (`tipos`) of each examination of the analysis |
| `data/cq500_caixas.csv` | The BHX boxes used as reference standard, per slice of the selected series (BHX licence, CC BY-SA 4.0) |
| `data/cq500_contagens.json`, `data/bhx_x_cq500.json` | Cohort counts and the agreement between BHX boxes and CQ500 reads |
| `data/export_512_summary.json` | Preprocessing parameters, slice and box counts per fold, results of the export tests |
| `data/ctich_contagens.json` | CT-ICH counts only (examinations, slices, boxes); nothing derived from its images or masks is distributed |
| `data/pilot_summary.json` | Pilot experiment |
| `data/training_status.json`, `data/sessions.jsonl` | Training jobs and Colab sessions of the first run |
| `data/final_results.json`, `data/results_validation.json`, `data/results_test.json` | Results of the first run (RetinaNet with transposed boxes; the YOLO results are the final ones) |
| `data/correcao_retinanet/` | Training records and final results with the corrected RetinaNet, the selection of the qualitative examples and the exploratory endpoints (`exploratory_results.json`, not part of the registered plan) |

Column names and some values are in Portuguese, as written by the notebooks: `exame` (examination), `parte`
(fold), `grupo` (group: `com sangramento` = with haemorrhage, `normal`, `duvidoso` = equivocal), `tipo` /
`tipos` (subtype), `votos_ich` (number of CQ500 readers reporting haemorrhage), `corte` (slice), `arquivo`
(file). In the result files, arm A is RetinaNet, B is YOLOv8m without mosaic and C is YOLOv8m with mosaic.

To draw the figures and the tables of the article again, run:

```bash
.venv/bin/python tools/make_figures.py data out
```

## Installation and tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest
```

Training and prediction also need `torch`, `torchvision`, `monai==1.5.2` and `ultralytics>=8.3,<8.4`; the
figure script needs `matplotlib`. The experiments were run on Google Colab with NVIDIA L4 GPUs.

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
  title  = {RetinaNet versus {YOLOv8m} for intracranial haemorrhage detection on head {CT}:
            a pre-registered equivalence study with cross-dataset testing},
  author = {Bispo, Guilherme Dantas and Serrano, Andr{\'e} Luiz Marques and Rocha Filho, Geraldo Pereira and Gon{\c{c}}alves, Vin{\'i}cius Pereira},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

The code itself (version used in the article):

```bibtex
@software{bispo2026ichdetectioncode,
  author    = {Bispo, Guilherme Dantas},
  title     = {ich-detection-benchmark: {RetinaNet} versus {YOLOv8m} for intracranial haemorrhage
               detection on head {CT}},
  version   = {v1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24370357}
}
```

## Licence

GNU Affero General Public License v3.0 ([`LICENSE`](LICENSE)). YOLO training and prediction use the
Ultralytics library, which is distributed under the same licence. The files in `data/` derived from BHX keep
the BHX licence (CC BY-SA 4.0).

## Author and maintainer

The code and this repository are written and maintained by Guilherme Dantas Bispo, Department of
Electrical Engineering, University of Brasília (UnB), Brazil (guilherme.bispo@redes.unb.br,
ORCID [0000-0002-4938-2076](https://orcid.org/0000-0002-4938-2076)). The authors of the article are listed
in the citation above.
