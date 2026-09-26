"""End-to-end adaptation on a random-weight Mask R-CNN at a small input size: no checkpoint, no network.

The model is the same torchvision ``maskrcnn_resnet50_fpn_v2`` architecture the pinned checkpoint loads
into, with its transform shrunk to 64 px so a CPU step takes about a second. These tests exercise the
real fine-tuning losses, the held BatchNorm statistics, box and mask evaluation, adapter export and
adapter reload. They say nothing about segmentation quality.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from safetensors.torch import save_file  # noqa: E402

from maskrcnn_instance_segmentation_pipeline import (  # noqa: E402
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_KEY,
    SIGN_CLASSES,
    MaskRcnnPipeline,
    build_model,
    sign_dataset,
    split_dataset,
)
from maskrcnn_instance_segmentation_pipeline import pipeline as pipeline_module  # noqa: E402

SIDE = 64


def _small_pipeline(seed: int = 0, class_names=SIGN_CLASSES) -> MaskRcnnPipeline:
    torch.manual_seed(seed)
    model = build_model(len(class_names) + 1)
    model.transform.min_size = (SIDE,)
    model.transform.max_size = SIDE
    return MaskRcnnPipeline(model=model.eval(), device="cpu", class_names=tuple(class_names), source="small")


def _body_batchnorm_state(pipe):
    return {
        name: value.clone()
        for name, value in pipe.model.state_dict().items()
        if name.startswith(pipeline_module.BACKBONE_PREFIX) and ("running_" in name or "num_batches" in name)
    }


@pytest.fixture(scope="module")
def data():
    records = sign_dataset(6, seed=5)
    return split_dataset(records, train_fraction=0.67, seed=0)


def test_finetune_evaluate_export_and_reload(tmp_path, data):
    train, held = data
    pipe = _small_pipeline()
    body_before = {
        k: v.clone()
        for k, v in pipe.model.state_dict().items()
        if k.startswith(pipeline_module.BACKBONE_PREFIX)
    }
    heads_before = {
        k: v.clone() for k, v in pipe.model.state_dict().items() if k.startswith("roi_heads.mask_predictor.")
    }
    assert _body_batchnorm_state(pipe), "the v2 ResNet body should carry BatchNorm running statistics"
    baseline = pipe.evaluate(held)
    assert 0.0 <= baseline["ap"] <= 1.0 and 0.0 <= baseline["mask_ap"] <= 1.0 and baseline["adapted"] is False

    run = pipe.finetune(train, epochs=1, batch_size=2, learning_rate=1e-3, seed=1)
    assert run["epochs"] == 1 and len(run["epoch_losses"]) == 1
    assert torch.isfinite(torch.tensor(run["final_loss"]))
    assert 0 < run["trainable_parameters"] < run["total_parameters"]
    assert run["frozen_prefixes"] == [pipeline_module.BACKBONE_PREFIX]
    after = pipe.model.state_dict()
    # Frozen means frozen: weights *and* BatchNorm running statistics of the ResNet body are unchanged.
    assert all(torch.equal(value, after[key]) for key, value in body_before.items())
    assert any(not torch.equal(value, after[key]) for key, value in heads_before.items())

    adapted = pipe.evaluate(held)
    assert set(adapted) >= {"ap", "ap50", "mask_ap", "mask_ap50", "per_class_mask_ap50", "estimation"}
    assert adapted["adapted"]

    path = tmp_path / "adapter.safetensors"
    descriptor = pipe.save_artifact(path, notes="small")
    assert descriptor["format"] == ARTIFACT_FORMAT and descriptor["bytes"] == path.stat().st_size
    metadata = MaskRcnnPipeline.read_artifact_metadata(path)
    assert metadata["class_names"] == list(SIGN_CLASSES)
    assert metadata["frozen_prefixes"] == [pipeline_module.BACKBONE_PREFIX]

    fresh = _small_pipeline()  # same seed: identical frozen body, different from `pipe` elsewhere
    fresh.apply_artifact(path)
    assert fresh.adapted and fresh.source == "artifact:adapter.safetensors"
    reloaded = fresh.model.state_dict()
    assert all(torch.equal(value, reloaded[key]) for key, value in pipe.model.state_dict().items())
    image = held[0]["image"]
    first = pipe.detect(image, threshold=0.0)
    second = fresh.detect(image, threshold=0.0)
    assert [(d["label"], round(d["score"], 5), d["mask_area"]) for d in first["detections"]] == [
        (d["label"], round(d["score"], 5), d["mask_area"]) for d in second["detections"]
    ]
    assert all(m.shape == (image.height, image.width) for m in first["masks"])


def test_unfrozen_finetune_updates_the_body(data):
    train, _held = data
    pipe = _small_pipeline()
    before = {
        k: v.clone()
        for k, v in pipe.model.state_dict().items()
        if k.startswith(pipeline_module.BACKBONE_PREFIX)
    }
    run = pipe.finetune(train[:2], epochs=1, batch_size=2, seed=1, freeze_backbone=False)
    assert run["frozen_prefixes"] == [] and run["trainable_parameters"] == run["total_parameters"]
    after = pipe.model.state_dict()
    assert any(not torch.equal(value, after[key]) for key, value in before.items())


def test_apply_artifact_refuses_mismatched_adapters(tmp_path, data):
    train, _held = data
    pipe = _small_pipeline()
    pipe.finetune(train[:2], epochs=1, batch_size=2, seed=1)
    path = tmp_path / "adapter.safetensors"
    pipe.save_artifact(path)

    other_vocabulary = _small_pipeline(class_names=("a", "b", "c"))
    with pytest.raises(ValueError, match="class_names differ"):
        other_vocabulary.apply_artifact(path)

    forged = tmp_path / "forged.safetensors"
    tensor = {"roi_heads.box_predictor.cls_score.bias": torch.zeros(4)}
    save_file(tensor, str(forged), metadata={"format": "other", "model_id": MODEL_ID, "model_key": MODEL_KEY})
    with pytest.raises(ValueError, match="artifact format"):
        _small_pipeline().apply_artifact(forged)

    partial = tmp_path / "partial.safetensors"
    metadata = {
        "format": ARTIFACT_FORMAT,
        "model_id": MODEL_ID,
        "model_revision": pipeline_module.MODEL_REVISION,
        "model_key": MODEL_KEY,
        "class_names": json.dumps(list(SIGN_CLASSES)),
        "frozen_prefixes": json.dumps([pipeline_module.BACKBONE_PREFIX]),
        "base_state_digest": "",
        "notes": "",
    }
    save_file(tensor, str(partial), metadata=metadata)
    with pytest.raises(ValueError, match="missing trainable tensors"):
        _small_pipeline().apply_artifact(partial)


def test_rehead_replaces_both_predictors_for_the_new_vocabulary():
    torch.manual_seed(0)
    model = build_model()
    prefixes = pipeline_module._rehead(model, len(SIGN_CLASSES))
    assert prefixes == ("roi_heads.box_predictor.", "roi_heads.mask_predictor.")
    assert model.roi_heads.box_predictor.cls_score.out_features == len(SIGN_CLASSES) + 1
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 4 * (len(SIGN_CLASSES) + 1)
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == len(SIGN_CLASSES) + 1


def test_build_model_matches_the_published_parameter_count():
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

    model = build_model()
    assert (
        sum(p.numel() for p in model.parameters())
        == MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["num_params"]
    )
