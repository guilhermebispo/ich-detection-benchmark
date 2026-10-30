"""The YOLO arguments must reproduce the shared settings, not Ultralytics' defaults."""

from ichdet import training_config as cfg
from ichdet.yolo import train_arguments


def test_yolo_takes_one_optimiser_step_per_batch_with_the_shared_warm_up():
    args = train_arguments("data.yaml", "out", "run", epochs=50, mosaic=False)
    assert args["batch"] == args["nbs"] == cfg.BATCH_SIZE          # no gradient accumulation, no decay rescaling
    assert args["weight_decay"] == cfg.OPTIMIZER["weight_decay"]
    assert args["warmup_bias_lr"] == 0.0 and args["warmup_momentum"] == cfg.OPTIMIZER["momentum"]
    assert args["optimizer"] == "SGD" and args["cos_lr"] and args["lrf"] == cfg.FINAL_LR_FRACTION


def test_yolo_augmentation_is_horizontal_flip_only_unless_mosaic():
    plain = train_arguments("data.yaml", "out", "run", epochs=50, mosaic=False)
    assert plain["augmentations"] == []                            # removes the default albumentations
    assert plain["fliplr"] == cfg.HFLIP_PROBABILITY and plain["mosaic"] == 0.0
    off = ("mixup", "copy_paste", "scale", "translate", "degrees", "shear", "perspective",
           "hsv_h", "hsv_s", "hsv_v", "flipud")
    assert all(plain[k] == 0.0 for k in off)
    mosaic = train_arguments("data.yaml", "out", "run", epochs=50, mosaic=True)
    assert {k: v for k, v in mosaic.items() if plain.get(k) != v} == {"mosaic": 1.0, "close_mosaic": 10}


def test_yolo_never_validates_during_training():
    assert train_arguments("data.yaml", "out", "run", epochs=50, mosaic=False)["val"] is False
