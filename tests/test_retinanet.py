"""Box axes between the split files ([x1, y1, x2, y2], x along columns) and MONAI (deviation 1, plan v1.2)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
from PIL import Image  # noqa: E402

from ichdet import retinanet, training_config as cfg  # noqa: E402

LESION = (slice(80, 100), slice(10, 30))          # rows 80-100, columns 10-30: away from the diagonal
BOX = [10.0, 80.0, 30.0, 100.0]                   # the same lesion as exported: [x1, y1, x2, y2]


def _entry(tmp_path):
    image = np.zeros((128, 128, 3), np.uint8)
    image[LESION] = 255
    (tmp_path / "images").mkdir()
    Image.fromarray(image).save(tmp_path / "images" / "a.png")
    return {"image": "images/a.png", "box": [BOX], "label": [0], "exam": "e"}


def _supervised_region(image, box):
    """Pixels that MONAI assigns to a box: coordinate i runs along spatial axis i of the (C, H, W) tensor."""
    b = box.round().int().tolist()
    return image[0, b[0]:b[2], b[1]:b[3]]


def test_monai_reads_box_coordinate_0_along_image_rows():
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

    anchors = AnchorGeneratorWithAnchorShape(feature_map_scales=[1], base_anchor_shapes=[[8, 8]])(
        torch.zeros(1, 3, 64, 128), [torch.zeros(1, 1, 16, 32)])[0]
    centre0, centre1 = (anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2
    assert float(centre0.max()) < 64 <= float(centre1.max())


@pytest.mark.parametrize("flip", [False, True])
def test_training_targets_cover_the_lesion(tmp_path, monkeypatch, flip):
    monkeypatch.setattr(cfg, "HFLIP_PROBABILITY", 1.0 if flip else 0.0)
    image, target, _ = retinanet._Slices([_entry(tmp_path)], tmp_path, train=True)[0]
    region = _supervised_region(image, target["box"][0])
    assert region.numel() == 20 * 20 and float(region.mean()) == 1.0


def test_swap_is_its_own_inverse_and_restores_the_exported_order():
    boxes = np.array([BOX, [1.0, 2.0, 3.0, 4.0]])
    assert np.array_equal(retinanet._swap_xy(retinanet._swap_xy(boxes)), boxes)
    assert retinanet._swap_xy(boxes)[0].tolist() == [80.0, 10.0, 100.0, 30.0]
