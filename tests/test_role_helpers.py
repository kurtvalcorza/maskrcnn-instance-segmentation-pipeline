"""Offline tests for the public validation and evaluation stage helpers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from maskrcnn_instance_segmentation_pipeline import (
    DETECTION_THRESHOLD,
    INPUT_SCHEMA,
    LABELS,
    MASK_THRESHOLD,
    MAX_DETECTIONS,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

SIZE = (640, 480)


def _mask(box):
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    x0, y0, x1, y1 = (int(v) for v in box)
    mask[y0:y1, x0:x1] = True
    return mask


STOP = [48.0, 88.0, 172.0, 212.0]
CLOCK = [410.0, 70.0, 550.0, 210.0]
REFERENCES = {
    "stop sign": {"boxes": [STOP], "masks": [_mask(STOP)]},
    "clock": {"boxes": [CLOCK], "masks": [_mask(CLOCK)]},
}


def _image(width: int = 850, height: int = 1100) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(detections: list[dict], masks: list[np.ndarray]) -> dict:
    return {
        "detections": detections,
        "masks": masks,
        "threshold": DETECTION_THRESHOLD,
        "width": SIZE[0],
        "height": SIZE[1],
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), names=["page.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["labels"] == list(LABELS)
    assert manifest["schema"]["max_detections"] == MAX_DETECTIONS
    assert manifest["schema"]["mask_threshold"] == MASK_THRESHOLD
    assert manifest["inputs"] == [{"id": "page.png", "mode": "RGB", "size": [850, 1100]}]
    assert manifest["threshold"] == DETECTION_THRESHOLD
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_threshold() -> None:
    manifest = validate_inputs(_image(), threshold=0.5)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["threshold"] == 0.5


def test_validate_inputs_rejects_like_detect() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8))
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="threshold"):
        validate_inputs(_image(), threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        validate_inputs(_image(), threshold=True)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), names=["a", "b"])


def test_evaluation_report_not_measurable_without_references() -> None:
    detection = {"box": [52.0, 92.0, 168.0, 209.0], "label": "stop sign", "score": 0.977, "mask_area": 1}
    report = evaluation_report(_result([detection], [_mask(detection["box"])]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_detections"] == 1
    assert "mask_iou" in report["needs"] and "box_iou" in report["needs"]
    assert report["baselines"] == []
    assert report["threshold"] == DETECTION_THRESHOLD and report["mask_threshold"] == MASK_THRESHOLD
    assert "softmax" in report["decision_rule"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_references() -> None:
    detections = [
        {"box": STOP, "label": "stop sign", "score": 0.977, "mask_area": 1},
        {"box": CLOCK, "label": "traffic light", "score": 0.931, "mask_area": 1},  # wrong label: no match
        {"box": [420.0, 80.0, 540.0, 200.0], "label": "clock", "score": 0.965, "mask_area": 1},
    ]
    masks = [_mask(STOP), _mask(CLOCK), _mask([420.0, 80.0, 540.0, 200.0])]
    report = evaluation_report(_result(detections, masks), REFERENCES, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "synthetic"
    assert [metric["reference"] for metric in report["metrics"]] == ["stop sign-0", "clock-0"]
    stop, clock = report["metrics"]
    assert stop["box_iou"] == pytest.approx(1.0) and stop["mask_iou"] == pytest.approx(1.0)
    assert stop["matched_score"] == 0.977 and stop["n_detected_same_label"] == 1
    assert 0.0 < clock["box_iou"] < 1.0 and 0.0 < clock["mask_iou"] < 1.0  # the clock, not the traffic light
    assert clock["matched_score"] == 0.965
    assert all(metric["estimation"] for metric in report["metrics"])


def test_evaluation_report_handles_zero_detections() -> None:
    report = evaluation_report(_result([], []), REFERENCES)
    assert report["n_detections"] == 0
    assert [(m["box_iou"], m["mask_iou"]) for m in report["metrics"]] == [(0.0, 0.0), (0.0, 0.0)]
    assert [metric["matched_score"] for metric in report["metrics"]] == [None, None]


def test_evaluation_report_rejects_unknown_reference_label() -> None:
    with pytest.raises(ValueError, match="unknown reference label"):
        evaluation_report(_result([], []), {"table": {"boxes": [[0.0, 0.0, 1.0, 1.0]], "masks": [None]}})
