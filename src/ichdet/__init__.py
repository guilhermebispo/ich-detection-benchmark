"""ichdet: evaluation code for Article 1 (RetinaNet vs YOLO for intracranial hemorrhage detection on head CT).

One evaluator for both detectors: the same box matching, the same average precision and the same
patient-level bootstrap, so that differences come from the detectors and not from how each library
computes its metrics. The statistical analysis plan that this code implements was frozen with it and
registered on OSF Registries before the full experiment.
"""

__version__ = "0.3.0"
