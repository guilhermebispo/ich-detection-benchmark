"""Training settings fixed before the experiment and shared by both detectors.

Everything that can be equal is equal: input (the same 512 x 512 PNG with three HU windows), epochs,
batch, optimiser, learning-rate schedule, augmentation, NMS and the number of detections kept. What
differs is what defines each detector (anchors, assigner, losses), plus mosaic in the YOLO+mosaic arm.
The rationale is in `evidencias/F3-05-statistical-protocol.md`.
"""

IMAGE_SIZE = 512
EPOCHS = 50                    # full experiment; the pilot uses PILOT_EPOCHS
PILOT_EPOCHS = 5
BATCH_SIZE = 16
SEED = 2026

OPTIMIZER = {"name": "SGD", "lr": 0.01, "momentum": 0.9, "nesterov": True, "weight_decay": 5e-4}
WARMUP_EPOCHS = 1              # linear warm-up, then cosine decay
FINAL_LR_FRACTION = 0.01       # learning rate at the last epoch = 1% of the initial one

HFLIP_PROBABILITY = 0.5        # the only augmentation in the primary comparison

# Inference and evaluation, identical for both
SCORE_THRESHOLD = 0.001
NMS_IOU = 0.5
MAX_DETECTIONS = 100

# RetinaNet (MONAI 1.5.2; detection code unchanged from 1.4.0): official LUNA16 recipe adapted to 2D
RETINANET = {
    "backbone": "ResNet-50, ImageNet weights (torchvision)",
    "returned_layers": [1, 2, 3],                      # FPN levels at strides 4, 8, 16 (+ max-pool level at 32)
    # 32 px ~ 5th percentile of the longest box side in BHX (14.9 mm); scaled x1, x2, x4, x8 across levels
    "base_anchor_shapes": [[32, 32], [45, 23], [23, 45]],
    "atss_candidates": 4,
    "hard_negative_sampler": {"batch_size_per_image": 64, "positive_fraction": 0.3, "pool_size": 20, "min_neg": 16},
}

# YOLO (Ultralytics 8.3): COCO-pretrained YOLOv8m, the size closest to RetinaNet-R50 (26 M vs 34 M parameters)
YOLO_MODEL = "yolov8m.pt"
YOLO_NO_AUGMENTATION = {      # Ultralytics augmentations turned off, except the shared horizontal flip
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "scale": 0.0, "translate": 0.0, "degrees": 0.0,
    "shear": 0.0, "perspective": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "flipud": 0.0,
    "fliplr": HFLIP_PROBABILITY,
    # Ultralytics also applies Blur, MedianBlur, ToGray and CLAHE (p = 0.01 each) when albumentations is
    # installed, as it is in Colab; an empty custom list removes them (found in the pilot log)
    "augmentations": [],
}
# Ultralytics defaults that would make YOLO's optimisation differ from RetinaNet's (found in the pilot log):
# nbs=64 accumulates gradients over 64/16 = 4 batches (effective batch 64) and rescales weight decay; the
# warm-up starts biases at lr 0.1 and momentum at 0.8. Here: one step per batch of 16, and the same
# warm-up as RetinaNet (every lr linearly from 0, momentum fixed at 0.9). `nbs` is set to the batch size
# in `yolo.train_arguments`.
YOLO_OPTIMISATION = {"warmup_bias_lr": 0.0, "warmup_momentum": OPTIMIZER["momentum"]}
YOLO_MOSAIC = {**YOLO_NO_AUGMENTATION, "mosaic": 1.0, "close_mosaic": 10}   # secondary arm: adds mosaic only
