"""Job list, locks, resumable jobs and the frozen-code fingerprint of the full experiment (F3-06)."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from ichdet import experiment as ex
from ichdet.yolo import train_arguments

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ichdet"
FROZEN_TAG = "artigo1-protocolo-v1.2"          # amendment: RetinaNet box axes (deviation 1, section 10 of the plan)
ORIGINAL_TAG = "artigo1-protocolo-v1.1"        # plan as first registered; the first run used this code


def _frozen_blobs(tag):
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=PACKAGE, capture_output=True, text=True)
    if root.returncode != 0:
        pytest.skip("not a git checkout")
    prefix = PACKAGE.relative_to(root.stdout.strip()).as_posix()
    blobs = {}
    for name in ex.FROZEN_MODULES:
        r = subprocess.run(["git", "show", f"{tag}:{prefix}/{name}"], cwd=PACKAGE, capture_output=True)
        if r.returncode != 0:
            pytest.skip(f"tag {tag} not available")
        blobs[name] = r.stdout
    return blobs


def test_frozen_modules_are_unchanged_since_the_plan_was_frozen():
    """Fails if result-determining code changed after freezing (that must be recorded as a deviation)."""
    for name, content in _frozen_blobs(FROZEN_TAG).items():
        assert (PACKAGE / name).read_bytes() == content, f"{name} changed after freezing"


def test_only_retinanet_changed_since_the_first_registration():
    """The amendment touched the RetinaNet input/output only; evaluator, statistics and YOLO are as in v1.1."""
    for name, content in _frozen_blobs(ORIGINAL_TAG).items():
        if name != "retinanet.py":
            assert (PACKAGE / name).read_bytes() == content, f"{name} changed since {ORIGINAL_TAG}"


def test_frozen_fingerprint_matches_the_registered_value():
    """The fingerprint of the amended plan (v1.1 was c3752d247a0173f1); holds in any copy of the package."""
    assert ex.frozen_fingerprint(PACKAGE) == "a6f3c0fc5798879d"


def test_primary_jobs_come_first():
    order = ex.job_order()
    assert len(order) == 15 and len(set(order)) == 15
    assert all(arm in ex.PRIMARY_ARMS for _, arm in order[:10])
    assert order[:2] == [(1, "retinanet"), (1, "yolo_no_mosaic")] and order[10] == (1, "yolo_mosaic")
    assert ex.job_order(folds=[3], arms=["yolo_mosaic", "retinanet"]) == [(3, "retinanet"), (3, "yolo_mosaic")]


def test_lock_blocks_other_sessions_until_its_heartbeat_is_stale(tmp_path):
    path = tmp_path / "running.json"
    a = ex.JobLock(path, "session-a", beat=0.05)
    assert a.acquire()
    assert not ex.JobLock(path, "session-b").acquire()
    assert ex.JobLock(path, "session-a").holder() is None            # the same session may take it back
    time.sleep(0.2)
    assert json.loads(path.read_text())["heartbeat"] > time.time() - 0.15   # heartbeat keeps it fresh
    a.release()
    assert not path.exists()
    path.write_text(json.dumps({"session": "dead", "heartbeat": time.time() - ex.STALE_SECONDS - 1}))
    assert ex.JobLock(path, "session-b").holder() is None


def _fake_training(epochs_written):
    def train_and_predict(arm, training, validation, images_root, directory, epochs):
        directory.mkdir(parents=True, exist_ok=True)
        history = [{"epoch": i + 1, "loss": 1.0, "seconds": 10.0 + i} for i in range(epochs_written)]
        (directory / "history.json").write_text(json.dumps(history))
        preds = {Path(e["image"]).name: (np.array([[1.0, 1.0, 5.0, 5.0]]), np.array([0.9])) for e in validation}
        return preds, {"prediction_seconds": 1.2}
    return train_and_predict


def _split(tmp_path):
    entries = [{"image": f"images/e{k}_{i}.png", "exam": f"e{k}", "box": [], "label": []}
               for k in range(4) for i in range(2)]
    path = tmp_path / "fold1.json"
    path.write_text(json.dumps({"training": entries[2:], "validation": entries[:2]}))
    return path


def test_run_job_writes_marker_and_skips_finished_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_train_and_predict", _fake_training(epochs_written=3))
    out = tmp_path / "out"
    assert ex.run_job(1, "retinanet", _split(tmp_path), tmp_path, out, epochs=3, session="s", log=lambda *_: None) == "finished"
    info = json.loads((out / "fold1" / "retinanet" / "job.json").read_text())
    assert info["epochs"] == 3 and info["validation_exams"] == 1 and info["boxes_predicted"] == 2
    assert info["seconds_per_epoch"] == 11.5                         # median without the first epoch
    assert (out / "fold1" / "retinanet" / "pred_validation.csv").exists()
    assert not (out / "fold1" / "retinanet" / "running.json").exists()
    assert ex.run_job(1, "retinanet", _split(tmp_path), tmp_path, out, epochs=3, session="s") == "done"
    states = {(r["fold"], r["arm"]): r["state"] for r in ex.status(out)}
    assert states[(1, "retinanet")] == "done" and states[(2, "retinanet")] == "pending"


def test_run_job_refuses_to_finish_with_missing_epochs(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_train_and_predict", _fake_training(epochs_written=2))
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="2 epochs recorded, 3 expected"):
        ex.run_job(1, "retinanet", _split(tmp_path), tmp_path, out, epochs=3, session="s", log=lambda *_: None)
    assert not (out / "fold1" / "retinanet" / "job.json").exists()
    assert not (out / "fold1" / "retinanet" / "running.json").exists()   # lock released on error
    assert ex.status(out)[0]["state"] == "partial"


def test_run_job_skips_a_job_locked_by_another_session(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_train_and_predict", _fake_training(epochs_written=3))
    out = tmp_path / "out"
    lock = ex.JobLock(ex.job_dir(out, 1, "retinanet") / "running.json", "other")
    assert lock.acquire()
    try:
        assert ex.run_job(1, "retinanet", _split(tmp_path), tmp_path, out, epochs=3, session="s", log=lambda *_: None) == "locked"
        assert ex.status(out, session="s")[0]["state"] == "running"
        assert ex.run_job(1, "retinanet", _split(tmp_path), tmp_path, out, epochs=3, session="s", take_over=True,
                          log=lambda *_: None) == "finished"
    finally:
        lock.release()


def test_yolo_epoch_times_come_from_the_cumulative_results_column(tmp_path):
    rows = "epoch,time\n1,150\n2,290\n3,431\n4,572\n"
    (tmp_path / "results.csv").write_text(rows)
    assert ex.epochs_done(tmp_path, "yolo_no_mosaic") == 4
    assert ex.seconds_per_epoch(tmp_path, "yolo_no_mosaic") == 141.0


@pytest.mark.parametrize("mosaic", [False, True])
def test_yolo_args_check_accepts_the_frozen_settings_and_rejects_changes(tmp_path, mosaic):
    saved = train_arguments("/drive/data.yaml", "/drive/out", "run", epochs=50, mosaic=mosaic)
    (tmp_path / "args.yaml").write_text(yaml.safe_dump(saved))
    checked = ex.check_yolo_args(tmp_path, mosaic, epochs=50)
    assert checked["nbs"] == 16 and checked["mosaic"] == (1.0 if mosaic else 0.0)
    (tmp_path / "args.yaml").write_text(yaml.safe_dump({**saved, "nbs": 64}))
    with pytest.raises(ValueError, match="nbs"):
        ex.check_yolo_args(tmp_path, mosaic, epochs=50)
