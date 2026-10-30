"""Full experiment (F3-06): the 15 training jobs (5 folds x 3 arms), run in Colab sessions that may stop.

Each job trains one arm on four folds of CQ500/BHX, predicts the held-out fold and writes a marker
(`job.json`). A job with a marker is skipped; an interrupted one resumes from its last saved epoch. A lock
file with a heartbeat lets two Colab sessions share the job list without running the same job twice.

Nothing here changes how models are trained, predicted or evaluated: that is the code frozen with the
statistical plan (FROZEN_MODULES, git tag `artigo1-protocolo-v1.1`), and `frozen_fingerprint` lets each
run prove it used that code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import threading
import time
from pathlib import Path

from . import training_config as cfg

FOLDS = (1, 2, 3, 4, 5)
ARMS = ("retinanet", "yolo_no_mosaic", "yolo_mosaic")
PRIMARY_ARMS = ("retinanet", "yolo_no_mosaic")
# Modules that determine the results; any change after freezing must be listed as a deviation (§10 of the plan)
FROZEN_MODULES = ("data_io.py", "evaluate.py", "metrics.py", "retinanet.py", "stats.py", "training_config.py",
                  "yolo.py")
STALE_SECONDS = 15 * 60          # a lock without a heartbeat for this long belongs to a session that died
HEARTBEAT_SECONDS = 60
# YOLO settings checked in the args.yaml that Ultralytics saves with every run
YOLO_CHECKED = ("epochs", "batch", "nbs", "imgsz", "seed", "optimizer", "lr0", "momentum", "weight_decay",
                "warmup_epochs", "warmup_bias_lr", "warmup_momentum", "cos_lr", "lrf", "val")


def frozen_fingerprint(package_dir) -> str:
    """Fingerprint of the frozen modules, used to prove that a run used the registered code.

    Args:
        package_dir: Directory containing the `ichdet` modules.

    Returns:
        First 16 hexadecimal digits of the SHA-256 of each frozen module's name and contents, in the order
        of `FROZEN_MODULES`.

    Raises:
        FileNotFoundError: If a frozen module is missing from `package_dir`.
    """
    h = hashlib.sha256()
    for name in FROZEN_MODULES:
        h.update(name.encode())
        h.update((Path(package_dir) / name).read_bytes())
    return h.hexdigest()[:16]


def job_order(folds=FOLDS, arms=ARMS) -> list[tuple[int, str]]:
    """Order in which the training jobs are run.

    The jobs of the primary comparison (RetinaNet and YOLO without mosaic, fold by fold) come first and the
    mosaic arm last, so that the primary analysis is complete if compute runs out.

    Args:
        folds: Fold numbers to include.
        arms: Arm names to include (`"retinanet"`, `"yolo_no_mosaic"`, `"yolo_mosaic"`).

    Returns:
        List of `(fold, arm)` pairs.
    """
    first = [(f, a) for f in folds for a in arms if a in PRIMARY_ARMS]
    return first + [(f, a) for f in folds for a in arms if a not in PRIMARY_ARMS]


def job_dir(out_root, fold: int, arm: str) -> Path:
    """Output directory of one job.

    Args:
        out_root: Root directory of the experiment outputs.
        fold: Held-out fold (1 to 5).
        arm: Arm name.

    Returns:
        `out_root/fold<fold>/<arm>`.
    """
    return Path(out_root) / f"fold{fold}" / arm


def epochs_done(directory, arm: str) -> int:
    """Number of training epochs already recorded for one job.

    Args:
        directory: Job directory (see `job_dir`).
        arm: Arm name; RetinaNet epochs are read from `history.json`, YOLO epochs from the Ultralytics
            `results.csv`.

    Returns:
        Number of completed epochs (0 if training has not started).
    """
    d = Path(directory)
    if arm == "retinanet":
        f = d / "history.json"
        return len(json.loads(f.read_text())) if f.exists() else 0
    f = d / "results.csv"
    return len(list(csv.DictReader(open(f)))) if f.exists() else 0


def seconds_per_epoch(directory, arm: str) -> float:
    """Median training time per epoch of one job.

    The first epoch is skipped (it includes data loading and warm-up), and the median is robust to the
    longer epochs that follow a resumed session.

    Args:
        directory: Job directory (see `job_dir`).
        arm: Arm name; for YOLO, per-epoch times are differences of the cumulative `time` column of
            `results.csv`.

    Returns:
        Median seconds per epoch.

    Raises:
        FileNotFoundError: If the training log of the job does not exist.
    """
    d = Path(directory)
    if arm == "retinanet":
        times = [h["seconds"] for h in json.loads((d / "history.json").read_text())]
    else:                                  # Ultralytics results.csv: "time" = seconds elapsed at the end of each epoch
        elapsed = [float(r["time"]) for r in csv.DictReader(open(d / "results.csv"))]
        times = [b - a for a, b in zip([0.0] + elapsed, elapsed)]
    return float(statistics.median(times[1:] if len(times) > 1 else times))


def check_yolo_args(run_dir, mosaic: bool, epochs: int, batch_size: int = cfg.BATCH_SIZE) -> dict:
    """Check the arguments recorded by Ultralytics for a YOLO job against the frozen settings.

    Args:
        run_dir: YOLO run directory containing the `args.yaml` written by Ultralytics.
        mosaic: Whether the run belongs to the mosaic arm.
        epochs: Expected number of epochs.
        batch_size: Expected batch size.

    Returns:
        The checked settings and their expected values.

    Raises:
        ValueError: If any checked setting differs from the frozen value.
    """
    import yaml
    from .yolo import train_arguments

    saved = yaml.safe_load((Path(run_dir) / "args.yaml").read_text())
    expected = train_arguments("data.yaml", "out", "run", epochs, mosaic, batch_size)
    keys = YOLO_CHECKED + tuple(cfg.YOLO_MOSAIC if mosaic else cfg.YOLO_NO_AUGMENTATION)
    wrong = {k: (saved.get(k), expected[k]) for k in keys if saved.get(k) != expected[k]}
    if wrong:
        raise ValueError(f"YOLO run {run_dir} differs from the frozen settings (saved, expected): {wrong}")
    return {k: expected[k] for k in keys}


class JobLock:
    """Lock file with a heartbeat thread, shared by Colab sessions working on the same job list.

    Another session may take the job once the heartbeat is older than `stale_after` seconds.

    Args:
        path: Path of the lock file (JSON with the session name and the time of the last heartbeat).
        session: Name of the session that holds or requests the lock.
        stale_after: Seconds without a heartbeat after which the lock is considered free.
        beat: Seconds between heartbeats.
    """

    def __init__(self, path, session: str, stale_after: float = STALE_SECONDS, beat: float = HEARTBEAT_SECONDS):
        self.path, self.session, self.stale_after, self.beat = Path(path), session, stale_after, beat
        self._stop = threading.Event()
        self._thread = None

    def holder(self):
        """Current holder of the lock, if it is another session with a recent heartbeat.

        Returns:
            `(session, seconds since its last heartbeat)` for a live lock held by another session, or
            `None` if the lock is free, stale, unreadable or held by this session.
        """
        if not self.path.exists():
            return None
        try:
            info = json.loads(self.path.read_text())
        except (OSError, ValueError):          # half-written file: treat as free
            return None
        age = time.time() - info["heartbeat"]
        return None if info["session"] == self.session or age > self.stale_after else (info["session"], age)

    def _write(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"session": self.session, "heartbeat": time.time()}))
        tmp.replace(self.path)

    def _run(self):
        while not self._stop.wait(self.beat):
            self._write()

    def acquire(self) -> bool:
        """Take the lock and start the heartbeat thread.

        Returns:
            `True` if the lock was taken, `False` if another session holds a live lock.
        """
        if self.holder():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def release(self):
        """Stop the heartbeat thread and delete the lock file."""
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.path.unlink(missing_ok=True)


def status(out_root, folds=FOLDS, arms=ARMS, session: str = "") -> list[dict]:
    """State of every job of the experiment.

    Args:
        out_root: Root directory of the experiment outputs.
        folds: Fold numbers to include.
        arms: Arm names to include.
        session: Name of the calling session (its own locks do not count as "running").

    Returns:
        One dictionary per job, in `job_order`, with `fold`, `arm`, `epochs` and `state`: `"done"`
        (`job.json` written), `"running"` (live lock of another session), `"partial"` (some epochs saved)
        or `"pending"`.
    """
    rows = []
    for fold, arm in job_order(folds, arms):
        d = job_dir(out_root, fold, arm)
        n = epochs_done(d, arm)
        if (d / "job.json").exists():
            state = "done"
        elif JobLock(d / "running.json", session).holder():
            state = "running"
        else:
            state = "partial" if n else "pending"
        rows.append({"fold": fold, "arm": arm, "state": state, "epochs": n})
    return rows


def _train_and_predict(arm, training, validation, images_root, directory, epochs):
    """Train one arm with the frozen code and predict its held-out fold.

    Args:
        arm: Arm name.
        training: Training entries of the MONAI split file.
        validation: Held-out entries of the MONAI split file.
        images_root: Directory that the entries' image paths are relative to.
        directory: Job directory.
        epochs: Number of training epochs.

    Returns:
        `(predictions, info)`: predictions as `{image: (boxes (N, 4) as [x1, y1, x2, y2] in pixels,
        scores (N,))}`, and extra job information (prediction time and, for YOLO, the checked arguments).
    """
    if arm == "retinanet":
        from . import retinanet
        checkpoint = retinanet.train(training, images_root, directory, epochs=epochs)
        t0 = time.time()
        return retinanet.predict(checkpoint, validation, images_root), {"prediction_seconds": time.time() - t0}
    from . import yolo
    mosaic = arm == "yolo_mosaic"
    data = yolo.write_dataset(training, images_root, directory.parent / "yolo_data")
    weights = yolo.train(data, directory.parent, arm, epochs=epochs, mosaic=mosaic)
    checked = check_yolo_args(directory, mosaic, epochs)
    t0 = time.time()
    return yolo.predict(weights, validation, images_root), {"prediction_seconds": time.time() - t0,
                                                            "yolo_args_checked": checked}


def run_job(fold: int, arm: str, split_json, images_root, out_root, epochs: int = cfg.EPOCHS,
            session: str = "", environment: dict | None = None, take_over: bool = False, log=print) -> str:
    """Run one training job to the end: train, predict the held-out fold and write `job.json`.

    Args:
        fold: Held-out fold (1 to 5).
        arm: Arm name.
        split_json: MONAI split file of the fold (`training` and `validation` entries).
        images_root: Directory that the entries' image paths are relative to.
        out_root: Root directory of the experiment outputs.
        epochs: Number of training epochs.
        session: Name of the calling session.
        environment: Software and hardware description recorded in `job.json`.
        take_over: Ignore another session's lock. Meant for a single Colab session, whose own lock from
            before a disconnection would otherwise block the job until its heartbeat goes stale.
        log: Function used for progress messages.

    Returns:
        `"done"` if the job had already finished, `"locked"` if another session holds it, or `"finished"`
        if it finished now.

    Raises:
        RuntimeError: If the number of recorded epochs differs from `epochs` after training.
        ValueError: If a YOLO run's recorded arguments differ from the frozen settings.
    """
    from .data_io import save_predictions

    directory = job_dir(out_root, fold, arm)
    if (directory / "job.json").exists():
        return "done"
    lock = JobLock(directory / "running.json", session, stale_after=0 if take_over else STALE_SECONDS)
    if not lock.acquire():
        who, age = lock.holder()
        log(f"fold {fold} / {arm}: running in another session ({who}, heartbeat {age:.0f} s ago); skipped")
        return "locked"
    try:
        split = json.loads(Path(split_json).read_text())
        training, validation = split["training"], split["validation"]
        log(f"fold {fold} / {arm}: training ({epochs_done(directory, arm)}/{epochs} epochs already saved)")
        t0 = time.time()
        predictions, extra = _train_and_predict(arm, training, validation, images_root, directory, epochs)
        n = epochs_done(directory, arm)
        if n != epochs:
            raise RuntimeError(f"fold {fold} / {arm}: {n} epochs recorded, {epochs} expected")
        save_predictions(directory / "pred_validation.csv", predictions)
        info = {
            "fold": fold, "arm": arm, "epochs": n, "training_images": len(training),
            "validation_images": len(validation), "validation_exams": len({e["exam"] for e in validation}),
            "seconds_per_epoch": round(seconds_per_epoch(directory, arm), 1),
            "boxes_predicted": int(sum(len(s) for _, s in predictions.values())),
            "session_seconds": round(time.time() - t0), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session": session, "environment": environment or {}, **extra,
        }
        info["prediction_seconds"] = round(info["prediction_seconds"])
        (directory / "job.json").write_text(json.dumps(info, indent=2))
        log(f"fold {fold} / {arm}: finished ({info['seconds_per_epoch']:.0f} s per epoch)")
        return "finished"
    finally:
        lock.release()
