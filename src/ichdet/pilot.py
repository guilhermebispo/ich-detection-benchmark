"""Pilot summary: compute budget of the full experiment and expected precision of the primary analysis.

The pilot trains each detector once (fold 1 held out, PILOT_EPOCHS epochs). Its allowed uses are fixed in
the protocol: fixing the pipeline, estimating GPU time, and estimating the bootstrap standard error of the
difference to project the width of the final 90% interval. Hyperparameters are not tuned on it.
"""

from __future__ import annotations

from . import training_config as cfg
from .stats import projected_half_width, required_exams

FOLDS = 5
PLANNED_VALIDATION_EXAMS = 436       # 215 with hemorrhage + 221 normal (equivocal exams excluded)
ARMS = ("retinanet", "yolo_no_mosaic", "yolo_mosaic")


def summarize(comparison: dict, seconds_per_epoch: dict, pilot_exams: int, margin: float = 0.05) -> dict:
    """comparison: output of evaluate.compare_crossval on the pilot fold.

    seconds_per_epoch: measured training time per epoch for "retinanet" and "yolo_no_mosaic"; the mosaic
    arm is assumed to cost the same as the no-mosaic arm.
    """
    eq = comparison["equivalence_A_minus_B"]
    se = eq["bootstrap_se"]
    per_epoch = dict(seconds_per_epoch)
    per_epoch.setdefault("yolo_mosaic", per_epoch["yolo_no_mosaic"])
    hours = {arm: per_epoch[arm] * cfg.EPOCHS * FOLDS / 3600 for arm in ARMS}
    precision = {
        "pilot_exams": pilot_exams,
        "pilot_bootstrap_se_of_difference": se,
        "planned_exams": PLANNED_VALIDATION_EXAMS,
        "projected_ci90_half_width": projected_half_width(se, pilot_exams, PLANNED_VALIDATION_EXAMS) if se else None,
        "margin": margin,
    }
    if se:
        precision["exams_needed_if_true_difference_is_0"] = required_exams(se, pilot_exams, margin, 0.0)
        if abs(eq["difference"]) < margin:
            precision["exams_needed_at_pilot_difference"] = required_exams(se, pilot_exams, margin, eq["difference"])
        precision["informative"] = precision["projected_ci90_half_width"] < margin
    return {
        "pilot_epochs": cfg.PILOT_EPOCHS,
        "full_epochs": cfg.EPOCHS,
        "seconds_per_epoch": per_epoch,
        "estimated_training_gpu_hours_full_experiment": {**{k: round(v, 1) for k, v in hours.items()},
                                                          "total": round(sum(hours.values()), 1)},
        "precision_projection": precision,
        "note": "Pilot models are under-trained (few epochs); the SE is a planning estimate, not a result.",
    }
