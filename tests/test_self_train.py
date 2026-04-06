"""Tests for self-training relabeling logic."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import supervision as sv

from scripts.self_train import relabel_patches


def _make_fake_patches(tmpdir: Path, n: int = 3) -> Path:
    """Create fake JPEG patches for testing."""
    from PIL import Image

    patch_dir = tmpdir / "patches"
    patch_dir.mkdir()
    for i in range(n):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        )
        img.save(patch_dir / f"{i:04d}.jpg")
    return patch_dir


def _make_mock_detections(n_boxes: int):
    """Create a mock sv.Detections with n_boxes of 30x30 boxes."""
    if n_boxes == 0:
        return sv.Detections.empty()
    xyxy = np.array(
        [[10 + i * 5, 10, 40 + i * 5, 40] for i in range(n_boxes)],
        dtype=np.float32,
    )
    return sv.Detections(xyxy=xyxy)


@patch("scripts.self_train.rfdetr", create=True)
def test_relabel_produces_coco_json(mock_rfdetr):
    """Relabeling should produce valid COCO JSON with correct structure."""
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_mock_detections(2)
    mock_rfdetr.RFDETRBase.return_value = mock_model

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        patch_dir = _make_fake_patches(tmpdir, n=3)
        output = tmpdir / "annotations.json"
        checkpoint = tmpdir / "fake.pt"
        checkpoint.touch()

        coco = relabel_patches(checkpoint, patch_dir, 0.5, output)

        assert output.exists()
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) == 3
        assert len(coco["annotations"]) == 6  # 2 per patch x 3
        assert coco["categories"][0]["name"] == "tree"


@patch("scripts.self_train.rfdetr", create=True)
def test_confidence_threshold_passed(mock_rfdetr):
    """The confidence threshold should be passed to model.predict()."""
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_mock_detections(1)
    mock_rfdetr.RFDETRBase.return_value = mock_model

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        patch_dir = _make_fake_patches(tmpdir, n=1)
        output = tmpdir / "annotations.json"
        checkpoint = tmpdir / "fake.pt"
        checkpoint.touch()

        relabel_patches(checkpoint, patch_dir, 0.8, output)

        # Verify predict was called with threshold=0.8
        _, kwargs = mock_model.predict.call_args
        assert kwargs["threshold"] == 0.8


@patch("scripts.self_train.rfdetr", create=True)
def test_empty_detections_handled(mock_rfdetr):
    """Patches with no detections should have images but no annotations."""
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_mock_detections(0)
    mock_rfdetr.RFDETRBase.return_value = mock_model

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        patch_dir = _make_fake_patches(tmpdir, n=2)
        output = tmpdir / "annotations.json"
        checkpoint = tmpdir / "fake.pt"
        checkpoint.touch()

        coco = relabel_patches(checkpoint, patch_dir, 0.7, output)

        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 0
