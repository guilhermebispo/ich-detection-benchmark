"""RetinaNet (MONAI 1.5.2) in 2D: build, train with resumable checkpoints, and predict.

Input: the entries of a MONAI split file (`image`, `box`, `label`, `exam`) and the folder that holds
`images/`. Requires torch, torchvision and monai==1.5.2; imported lazily so that the evaluator does not
depend on them.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import training_config as cfg
from .data_io import image_key


def build_detector(pretrained_backbone: bool = True):
    import torchvision
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

    r = cfg.RETINANET
    levels = len(r["returned_layers"]) + 1
    anchors = AnchorGeneratorWithAnchorShape(feature_map_scales=[2**i for i in range(levels)],
                                             base_anchor_shapes=r["base_anchor_shapes"])
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
    backbone = torchvision.models.resnet50(weights=weights)
    backbone.in_planes = backbone.inplanes          # name MONAI's FPN builder reads (2048 for ResNet-50)
    extractor = resnet_fpn_feature_extractor(backbone, spatial_dims=2, pretrained_backbone=False,
                                             returned_layers=r["returned_layers"], trainable_backbone_layers=None)
    size_divisible = [s * 2 * 2 ** max(r["returned_layers"]) for s in extractor.body.conv1.stride]
    network = RetinaNet(spatial_dims=2, num_classes=1, num_anchors=anchors.num_anchors_per_location()[0],
                        feature_extractor=extractor, size_divisible=size_divisible)
    detector = RetinaNetDetector(network=network, anchor_generator=anchors, spatial_dims=2)
    detector.set_atss_matcher(num_candidates=r["atss_candidates"], center_in_gt=False)
    detector.set_hard_negative_sampler(**r["hard_negative_sampler"])
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(score_thresh=cfg.SCORE_THRESHOLD, topk_candidates_per_level=1000,
                                         nms_thresh=cfg.NMS_IOU, detections_per_img=cfg.MAX_DETECTIONS)
    return detector


class _Slices:
    """Dataset over split entries: float image (3, H, W) in [0, 1], boxes (M, 4), labels (M,)."""

    def __init__(self, entries, root, train: bool):
        self.entries, self.root, self.train = entries, Path(root), train

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        import torch
        from PIL import Image

        e = self.entries[i]
        image = np.asarray(Image.open(self.root / e["image"]).convert("RGB"), dtype=np.float32) / 255.0
        boxes = np.asarray(e["box"], dtype=np.float32).reshape(-1, 4)
        if self.train and torch.rand(1).item() < cfg.HFLIP_PROBABILITY:   # torch seeds each worker
            image = image[:, ::-1]
            boxes = np.stack([image.shape[1] - boxes[:, 2], boxes[:, 1], image.shape[1] - boxes[:, 0], boxes[:, 3]], 1)
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                {"box": torch.from_numpy(boxes), "label": torch.zeros(len(boxes), dtype=torch.long)},
                image_key(e["image"]))


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def _lr_factor(step, steps_per_epoch, epochs):
    """Linear warm-up for WARMUP_EPOCHS, then cosine decay to FINAL_LR_FRACTION."""
    warm = cfg.WARMUP_EPOCHS * steps_per_epoch
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, epochs * steps_per_epoch - warm)
    return cfg.FINAL_LR_FRACTION + (1 - cfg.FINAL_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress))


def train(entries, root, out_dir, epochs: int, batch_size: int = cfg.BATCH_SIZE, seed: int = cfg.SEED,
          num_workers: int = 2, pretrained_backbone: bool = True, log=print) -> Path:
    """Train for a fixed number of epochs and return the last checkpoint (no selection on validation data).

    A checkpoint is written after every epoch to `out_dir/last.pt`; calling again resumes from it, which
    is how a disconnected Colab session continues. `out_dir/history.json` keeps loss and time per epoch.
    """
    import torch

    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = build_detector(pretrained_backbone).to(device)
    o = cfg.OPTIMIZER
    optimizer = torch.optim.SGD(detector.network.parameters(), lr=o["lr"], momentum=o["momentum"],
                                nesterov=o["nesterov"], weight_decay=o["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loader = torch.utils.data.DataLoader(_Slices(entries, root, True), batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, collate_fn=_collate, drop_last=True,
                                         generator=torch.Generator().manual_seed(seed))
    start, history = 0, []
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        detector.network.load_state_dict(state["network"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, history = state["epoch"], state["history"]
        log(f"resuming RetinaNet from epoch {start}")
    steps = len(loader)
    for epoch in range(start, epochs):
        detector.train()
        t0, total = time.time(), 0.0
        for i, (images, targets, _) in enumerate(loader):
            for g in optimizer.param_groups:
                g["lr"] = o["lr"] * _lr_factor(epoch * steps + i, steps, epochs)
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                losses = detector(images, targets)
                loss = losses[detector.cls_key] + losses[detector.box_reg_key]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        history.append({"epoch": epoch + 1, "loss": total / steps, "seconds": round(time.time() - t0, 1)})
        log(f"RetinaNet epoch {epoch + 1}/{epochs}: loss {history[-1]['loss']:.4f}, {history[-1]['seconds']:.0f} s")
        torch.save({"network": detector.network.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    return out / "last.pt"


def predict(checkpoint, entries, root, batch_size: int = cfg.BATCH_SIZE, num_workers: int = 2) -> dict:
    """{image: (boxes (N, 4), scores (N,))} for the given entries, with the shared inference settings."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = build_detector(pretrained_backbone=False).to(device)
    detector.network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["network"])
    detector.eval()
    loader = torch.utils.data.DataLoader(_Slices(entries, root, False), batch_size=batch_size,
                                         num_workers=num_workers, collate_fn=_collate)
    result = {}
    with torch.no_grad():
        for images, _, names in loader:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                outputs = detector([x.to(device) for x in images], use_inferer=False)
            for name, o in zip(names, outputs):
                result[name] = (o[detector.target_box_key].float().cpu().numpy().reshape(-1, 4),
                                o[detector.pred_score_key].float().cpu().numpy().reshape(-1))
    return result
