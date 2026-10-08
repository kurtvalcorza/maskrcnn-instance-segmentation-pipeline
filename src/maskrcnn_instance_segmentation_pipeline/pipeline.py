"""COCO instance segmentation and bounded fine-tuning with torchvision's Mask R-CNN ResNet-50 FPN v2 weights.

The model is torchvision's own ``maskrcnn_resnet50_fpn_v2`` architecture, built with ``weights=None`` so
that nothing is downloaded at construction, then loaded from one digest-verified local checkpoint file
(``weights/<key>/``) with ``torch.load(..., weights_only=True)``, which refuses arbitrary pickled objects.

Until ``tools/pin_snapshot.py`` has recorded the checkpoint's SHA-256 and byte size, the package refuses to
stage, verify or load weights: an unpinned checkpoint is never trusted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "torchvision/maskrcnn_resnet50_fpn_v2"
# The pinned identity of a URL-hosted checkpoint is the SHA-256 of its bytes. torchvision names the file
# after the first 8 hex digits of that digest, so the URL is content-addressed; the full digest is pinned.
MODEL_REVISION = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"
MODEL_LICENSE = "bsd-3-clause"
MODEL_KEY = "maskrcnn-resnet50-fpn-v2"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_FORMAT = "maskrcnn-adapter-v1"
UNPINNED = "unpinned"
PIN_COMMAND = "python tools/pin_snapshot.py"
WEIGHTS_FILE = "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
WEIGHTS_URL = f"https://download.pytorch.org/models/{WEIGHTS_FILE}"
WEIGHTS_SHA256_PREFIX = "73cbd019"
TORCHVISION_WEIGHTS = "MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1"

# torchvision's COCO category list for these weights (MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta
# ["categories"]), in label-id order: 91 slots. Id 0 is the background class and the 10 "N/A" ids are COCO
# category numbers that COCO 2017 never annotated, which leaves 80 object classes. LABELS drops the
# background, so model label id k is LABELS[k - 1].
COCO_CATEGORIES = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip
LABELS = COCO_CATEGORIES[1:]

# Score threshold: the value torchvision's visualization-utilities gallery uses to keep Mask R-CNN instances
# (score_threshold = .75, shown there with the v1 maskrcnn_resnet50_fpn weights), with its mask probability
# cut (proba_threshold = 0.5). Scores are softmax class probabilities after torchvision's per-class NMS;
# neither value was re-derived for the v2 weights or calibrated for any deployment, and the deployment owns
# tuning them on labelled images.
DETECTION_THRESHOLD = 0.75
MASK_THRESHOLD = 0.5
EVAL_DETECTION_THRESHOLD = 0.05
MAX_DETECTIONS = 100  # torchvision's box_detections_per_img default
MAX_EVAL_DETECTIONS = 100
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_CLASSES = 1000
MAX_RECORDS = 5000

# Training defaults for the bounded tutorial adaptation (torchvision's detection fine-tuning tutorial
# uses SGD, learning rate 0.005, momentum 0.9 and weight decay 0.0005).
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_SEED = 20260924
BACKBONE_PREFIX = "backbone.body."

COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pinned() -> bool:
    """True once MODEL_REVISION is the checkpoint's 64-hex SHA-256."""
    revision = MODEL_REVISION
    return len(revision) == 64 and all(c in "0123456789abcdef" for c in revision)


def _require_pinned(action: str) -> None:
    if not is_pinned():
        raise RuntimeError(
            f"refusing to {action}: {MODEL_ID} has no pinned checkpoint digest yet (MODEL_REVISION = "
            f"{MODEL_REVISION!r}); run `{PIN_COMMAND}` to record the SHA-256 and byte size"
        )


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local checkpoint against its DIMER manifest; raise naming the first mismatch."""
    _require_pinned("verify the checkpoint")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        if not entry.get("sha256") or entry.get("bytes") is None:
            raise ValueError(f"{entry['path']}: manifest records no sha256/bytes; run `{PIN_COMMAND}`")
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"checkpoint file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _url_download(entry: Mapping[str, Any], root: Path) -> None:
    """Fetch one manifest entry from its URL; torch.hub checks the filename's SHA-256 prefix on the way in."""
    from torch.hub import download_url_to_file

    download_url_to_file(
        entry["url"], str(root / entry["path"]), hash_prefix=WEIGHTS_SHA256_PREFIX, progress=False
    )


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[Mapping[str, Any], Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the checkpoint). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    _require_pinned("stage the checkpoint")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"checkpoint at {root} is missing {[e['path'] for e in missing]}; pass allow_download=True to "
            "fetch it"
        )
    fetch = downloader or _url_download
    for entry in missing:
        if entry.get("url") != WEIGHTS_URL:
            raise ValueError(f"manifest URL {entry.get('url')!r} != package URL {WEIGHTS_URL!r}")
        fetch(entry, root)
    return [entry["path"] for entry in missing]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy pixel boxes."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks of the same shape."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def average_precision(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    kind: str = "box",
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Compact COCO-style average precision over a scored dataset, on boxes or on masks.

    For each class and IoU threshold, detections are matched greedily to references by descending score;
    each detection matches at most one reference, and precision is sampled at 101 recall points. Classes
    with no reference are left out of the mean. There are no area ranges and no crowd handling.
    """
    if kind not in ("box", "mask"):
        raise ValueError(f"kind must be 'box' or 'mask', got {kind!r}")
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} prediction lists but {len(references)} references")
    recall_points = np.linspace(0.0, 1.0, 101)
    per_threshold: dict[float, dict[str, float]] = {}

    def overlap(det: Mapping[str, Any], reference: Mapping[str, Any], j: int) -> float:
        if kind == "box":
            return box_iou(det["box"], reference["boxes"][j])
        return mask_iou(det["mask"], reference["masks"][j])

    for threshold in iou_thresholds:
        per_class: dict[str, float] = {}
        for name in class_names:
            scored: list[tuple[float, bool]] = []
            n_references = 0
            for dets, reference in zip(predictions, references, strict=True):
                ref_ids = [j for j, label in enumerate(reference["labels"]) if label == name]
                n_references += len(ref_ids)
                claimed = dict.fromkeys(ref_ids, False)
                for det in sorted((d for d in dets if d["label"] == name), key=lambda d: -float(d["score"])):
                    best, best_iou = -1, 0.0
                    for j in ref_ids:
                        if claimed[j]:
                            continue
                        value = overlap(det, reference, j)
                        if value > best_iou:
                            best, best_iou = j, value
                    hit = best >= 0 and best_iou >= threshold
                    if hit:
                        claimed[best] = True
                    scored.append((float(det["score"]), hit))
            if n_references == 0:
                continue
            if not scored:
                per_class[name] = 0.0
                continue
            scored.sort(key=lambda pair: -pair[0])
            true_positives = np.cumsum([1 if hit else 0 for _s, hit in scored])
            false_positives = np.cumsum([0 if hit else 1 for _s, hit in scored])
            recall = true_positives / n_references
            precision = np.maximum.accumulate(
                (true_positives / np.maximum(true_positives + false_positives, 1))[::-1]
            )[::-1]
            sampled = np.zeros_like(recall_points)
            indices = np.searchsorted(recall, recall_points, side="left")
            valid = indices < len(precision)
            sampled[valid] = precision[indices[valid]]
            per_class[name] = float(sampled.mean())
        per_threshold[threshold] = per_class
    means = {t: (float(np.mean(list(v.values()))) if v else 0.0) for t, v in per_threshold.items()}
    return {
        "ap": float(np.mean(list(means.values()))) if means else 0.0,
        "ap50": means.get(0.5, 0.0),
        "ap75": means.get(0.75, 0.0),
        "per_class_ap50": per_threshold.get(0.5, {}),
        "iou_thresholds": [float(t) for t in iou_thresholds],
        "kind": kind,
        "n_images": len(references),
        "n_references": sum(len(r["labels"]) for r in references),
    }


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(value: Any, name: str = "threshold") -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


def _check_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(class_names)
    if not 1 <= len(names) <= MAX_CLASSES:
        raise ValueError(f"class_names must hold 1..{MAX_CLASSES} names, got {len(names)}")
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"class names must be non-empty strings, got {name!r}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"class_names contains duplicates: {duplicates}")
    return names


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one image as PIL.Image.Image (any mode, converted to RGB): a photograph or a rendered scene",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "threshold": [0.0, 1.0],
    "mask_threshold": MASK_THRESHOLD,
    "labels": list(LABELS),
    "max_detections": MAX_DETECTIONS,
    "preprocessing": (
        "image converted to RGB and to a float tensor in [0, 1]; the model's own transform resizes it so the "
        "shorter side is 800 px and the longer at most 1333 px and normalises it with the ImageNet mean and "
        "standard deviation; boxes and masks are returned in input pixels"
    ),
}

DATASET_SCHEMA: dict[str, Any] = {
    "record": (
        "{'image': PIL.Image.Image, 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...], "
        "'masks': [bool array (height, width), ...]}"
    ),
    "boxes": "xyxy pixel coordinates inside the image, x0 < x1 and y0 < y1, one per instance",
    "masks": "one non-empty boolean mask per instance, the size of the image, lying inside its box",
    "labels": "one name per instance, each one of class_names",
    "records": [1, MAX_RECORDS],
}


def _check_inputs(image: Any, threshold: Any) -> tuple[Image.Image, float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request."""
    return validate_image(image), _check_threshold(threshold)


def validate_inputs(
    image: Image.Image,
    *,
    threshold: float = DETECTION_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict)."""
    _rgb, checked = _check_inputs(image, threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (detect takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "threshold": checked,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    epochs: int = DEFAULT_EPOCHS,
) -> dict[str, Any]:
    """Validation stage for labelled instance-segmentation records: raise on the first broken record, else
    return the dataset manifest. Classes that never occur are reported as findings, not silently accepted."""
    names = _check_class_names(class_names)
    if not records:
        raise ValueError("dataset must hold at least one record")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset holds {len(records)} records > MAX_RECORDS {MAX_RECORDS}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise ValueError(f"epochs must be an int in 1..100, got {epochs!r}")
    valid = set(names)
    per_class = dict.fromkeys(names, 0)

    def _pixel_hint(box_values: Sequence[float]) -> str:
        # MRC-m1: boxes in [0, 1] are a common BYOD mistake; say so, not only which rule failed.
        if all(0.0 <= v <= 1.0 for v in box_values):
            return (
                "; every coordinate of this box is between 0 and 1, which looks like normalised coordinates: "
                "boxes must be xyxy pixel coordinates (multiply x by the image width and y by its height)"
            )
        return ""

    total = 0
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping) or not {"image", "boxes", "labels", "masks"} <= set(record):
            raise ValueError(f"record {idx} must be a mapping with 'image', 'boxes', 'labels' and 'masks'")
        try:
            img = validate_image(record["image"])
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"record {idx}: {exc}") from exc
        boxes, labels, masks = record["boxes"], record["labels"], record["masks"]
        if not len(boxes) == len(labels) == len(masks):
            raise ValueError(f"record {idx}: {len(boxes)} boxes, {len(labels)} labels and {len(masks)} masks")
        width, height = img.size
        for b_idx, (box, mask) in enumerate(zip(boxes, masks, strict=True)):
            if len(box) != 4:
                raise ValueError(f"record {idx} box {b_idx} must have 4 elements, got {len(box)}")
            x0, y0, x1, y1 = (float(v) for v in box)
            if not all(np.isfinite([x0, y0, x1, y1])):
                raise ValueError(f"record {idx} box {b_idx} has a non-finite coordinate")
            if not (0.0 <= x0 < x1 <= width and 0.0 <= y0 < y1 <= height):
                raise ValueError(
                    f"record {idx} box {b_idx} [{x0}, {y0}, {x1}, {y1}] is empty or outside image bounds "
                    f"{(width, height)}" + _pixel_hint((x0, y0, x1, y1))
                )
            array = np.asarray(mask)
            if array.shape != (height, width):
                raise ValueError(
                    f"record {idx} mask {b_idx} has shape {array.shape}, image is {(height, width)}"
                )
            if array.dtype != bool and not np.isin(array, (0, 1)).all():
                raise ValueError(f"record {idx} mask {b_idx} must be boolean or 0/1")
            ys, xs = np.nonzero(array)
            if not len(xs):
                raise ValueError(f"record {idx} mask {b_idx} is empty")
            if xs.min() < x0 - 1 or xs.max() + 1 > x1 + 1 or ys.min() < y0 - 1 or ys.max() + 1 > y1 + 1:
                raise ValueError(
                    f"record {idx} mask {b_idx} extends outside its box" + _pixel_hint((x0, y0, x1, y1))
                )
        for label in labels:
            if label not in valid:
                raise ValueError(f"record {idx} has unknown class {label!r}; expected one of {list(names)}")
            per_class[label] += 1
        total += len(labels)
    return {
        "schema": dict(DATASET_SCHEMA),
        "n_records": len(records),
        "n_instances": total,
        "class_names": list(names),
        "instances_per_class": per_class,
        "epochs": epochs,
        "findings": [
            f"class {name!r} has no instance in this dataset" for name, n in per_class.items() if n == 0
        ],
        "verdict": "accepted",
    }


def _open_named(idx: int, kind: str, path: Path, root: Path) -> Image.Image:
    """Open one BYOD file fully, or raise a ValueError that names the file and the fix (MRC-m1)."""
    relative = path.relative_to(root) if root in path.parents else path
    if not path.is_file():
        raise ValueError(
            f"annotations.json entry {idx}: {kind} file {str(relative)!r} is missing from {root}; "
            "add the file or correct its name in annotations.json"
        )
    try:
        with Image.open(path) as handle:
            handle.load()
            return handle.copy()
    except (OSError, SyntaxError, ValueError) as exc:  # PIL.UnidentifiedImageError is an OSError
        raise ValueError(
            f"annotations.json entry {idx}: {kind} file {str(relative)!r} is not an image Pillow can "
            f"read ({type(exc).__name__}); save it as PNG (masks: single-channel PNG, non-zero inside "
            "the instance)"
        ) from exc


def read_detection_records(directory: str | Path) -> list[dict[str, Any]]:
    """Read BYOD records from ``<directory>/annotations.json`` plus the image and mask files it names.

    ``annotations.json`` is a list of ``{"file": "name.png", "boxes": [[x0, y0, x1, y1], ...],
    "labels": [...], "masks": ["name_mask0.png", ...]}`` objects; each mask file is a single-channel PNG
    whose non-zero pixels are the instance. File names must stay inside ``directory``; absolute paths and
    ``..`` are refused before any file is opened. The result still has to pass ``validate_dataset``.
    """
    root = Path(directory).resolve()
    index_path = root / "annotations.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found; expected annotations.json next to the images")
    with open(index_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list) or not entries:
        raise ValueError("annotations.json must hold a non-empty list of records")
    if len(entries) > MAX_RECORDS:
        raise ValueError(f"annotations.json lists {len(entries)} records > MAX_RECORDS {MAX_RECORDS}")

    def resolve(idx: int, name: Any) -> Path:
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"annotations.json entry {idx}: file {name!r} must be relative to {root}")
        resolved = (root / relative).resolve()
        if root not in resolved.parents:
            raise ValueError(f"annotations.json entry {idx}: file {name!r} resolves outside {root}")
        return resolved

    records: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or not {"file", "boxes", "labels", "masks"} <= set(entry):
            raise ValueError(f"annotations.json entry {idx} must have 'file', 'boxes', 'labels' and 'masks'")
        image_path = resolve(idx, entry["file"])
        mask_paths = [resolve(idx, name) for name in entry["masks"]]
        image = _open_named(idx, "image", image_path, root).convert("RGB")
        masks = [np.asarray(_open_named(idx, "mask", path, root).convert("L")) > 0 for path in mask_paths]
        records.append(
            {
                "id": str(entry["file"]),
                "image": image,
                "boxes": entry["boxes"],
                "labels": entry["labels"],
                "masks": masks,
            }
        )
    return records


def evaluation_report(
    result: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Single-image evaluation stage: per-object box_iou and mask_iou against same-label detections."""
    detections = list(result["detections"])
    masks = list(result.get("masks") or [])
    labels = tuple(result.get("class_names") or LABELS)
    base = {
        "task": f"instance segmentation over {len(labels)} class slots on one image",
        "decision_rule": (
            "an instance survives when its softmax class probability reaches the threshold after "
            "torchvision's "
            "per-class non-maximum suppression; its mask is the pixels whose probability is at least "
            f"{MASK_THRESHOLD}; "
            "neither value is a calibrated probability for the deployment's images"
        ),
        "threshold": result.get("threshold", DETECTION_THRESHOLD),
        "mask_threshold": MASK_THRESHOLD,
        "sample_kind": sample_kind,
        "n_detections": len(detections),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not references:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference boxes or masks were supplied for the evaluated image",
            "needs": (
                "labelled instance masks per class on your own images, scored per object with box_iou and "
                "mask_iou "
                "and aggregated into box and mask average precision at stated IoU thresholds"
            ),
        }
    metrics = []
    for label, reference in references.items():
        if label not in labels:
            raise ValueError(f"unknown reference label {label!r}; expected one of {len(labels)} classes")
        same = [i for i, det in enumerate(detections) if det["label"] == label]
        for index, box in enumerate(reference["boxes"]):
            box_ious = [box_iou(detections[i]["box"], box) for i in same]
            best = same[int(np.argmax(box_ious))] if box_ious else None
            ref_mask = reference.get("masks", [None] * len(reference["boxes"]))[index]
            metrics.append(
                {
                    "id": "box_iou+mask_iou",
                    "reference": f"{label}-{index}",
                    "box_iou": max(box_ious) if box_ious else 0.0,
                    "mask_iou": mask_iou(masks[best], ref_mask)
                    if best is not None and ref_mask is not None and masks
                    else 0.0,
                    "matched_score": detections[best]["score"] if best is not None else None,
                    "n_detected_same_label": len(same),
                    "estimation": "one reference instance per object on a single image, no dispersion "
                    "estimate",
                }
            )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": f"{len(metrics)} reference instance(s) on one tutorial image; geometry sanity evidence, "
        "not a benchmark",
        "needs": "a labelled image set from the deployment domain for any average-precision or "
        "precision/recall claim",
    }


def build_model(num_classes: int = len(COCO_CATEGORIES)) -> Any:
    """torchvision's maskrcnn_resnet50_fpn_v2 architecture with random weights and nothing downloaded."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    return maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=num_classes)


def _rehead(model: Any, num_labels: int) -> tuple[str, ...]:
    """Replace the box and mask predictors for ``num_labels`` classes plus background."""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, num_labels + 1)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, num_labels + 1)
    return ("roi_heads.box_predictor.", "roi_heads.mask_predictor.")


def _hold_batchnorm(module: Any) -> None:
    """Keep every BatchNorm layer under ``module`` in eval mode, so frozen layers keep their statistics."""
    import torch

    for sub in module.modules():
        if isinstance(sub, torch.nn.modules.batchnorm._BatchNorm):
            sub.eval()


@dataclass
class MaskRcnnPipeline:
    """COCO-class instance segmentation and transfer fine-tuning over Mask R-CNN ResNet-50 FPN v2."""

    model: Any
    device: str
    class_names: tuple[str, ...] = LABELS
    source: str = "snapshot"
    base_state_digest: str | None = None
    adapted: bool = False
    reinitialised: tuple[str, ...] = field(default_factory=tuple)
    frozen_prefixes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        class_names: Sequence[str] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> MaskRcnnPipeline:
        _require_pinned("load the model")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no checkpoint manifest at {root}; stage {MODEL_ID} under weights/{MODEL_KEY} "
                "(allow_download=True fetches the manifest-listed file)"
            )
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        names = _check_class_names(class_names) if class_names is not None else LABELS

        import torch

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model(len(COCO_CATEGORIES))
        state = torch.load(root / WEIGHTS_FILE, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        reinitialised: tuple[str, ...] = ()
        if class_names is not None:
            torch.manual_seed(seed)
            reinitialised = _rehead(model, len(names))
        model = model.to(resolved_device).eval()
        return cls(
            model=model,
            device=resolved_device,
            class_names=names,
            source=str(root),
            base_state_digest=MODEL_REVISION,
            reinitialised=reinitialised,
        )

    def _run(self, image: Image.Image, threshold: float) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        import torch
        from torchvision.transforms.functional import to_tensor

        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            output = self.model([to_tensor(image).to(self.device)])[0]
        if was_training:
            self.model.train()
        detections, masks = [], []
        for box, label, score, mask in zip(
            output["boxes"], output["labels"], output["scores"], output["masks"], strict=True
        ):
            if float(score) < threshold:
                continue
            idx = int(label) - 1
            name = self.class_names[idx] if 0 <= idx < len(self.class_names) else f"class_{int(label)}"
            binary = (mask[0] >= MASK_THRESHOLD).cpu().numpy()
            detections.append(
                {
                    "box": [float(v) for v in box.tolist()],
                    "label": name,
                    "score": float(score),
                    "mask_area": int(binary.sum()),
                }
            )
            masks.append(binary)
        return detections, masks

    def detect(self, image: Image.Image, *, threshold: float = DETECTION_THRESHOLD) -> dict[str, Any]:
        """Segment instances on one image; boxes are xyxy input pixels, masks boolean (height, width)."""
        rgb, checked = _check_inputs(image, threshold)
        detections, masks = self._run(rgb, checked)
        if len(detections) > MAX_DETECTIONS:
            raise RuntimeError(
                f"backend returned {len(detections)} detections > MAX_DETECTIONS {MAX_DETECTIONS}"
            )
        for det, mask in zip(detections, masks, strict=True):
            if (
                set(det) != {"box", "label", "score", "mask_area"}
                or len(det["box"]) != 4
                or det["label"] not in self.class_names
            ):
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")
            if mask.shape != (rgb.height, rgb.width):
                raise RuntimeError(f"backend returned a mask of shape {mask.shape} for a {rgb.size} image")
        order = sorted(range(len(detections)), key=lambda i: -detections[i]["score"])
        return {
            "detections": [detections[i] for i in order],
            "masks": [masks[i] for i in order],
            "threshold": checked,
            "mask_threshold": MASK_THRESHOLD,
            "width": rgb.width,
            "height": rgb.height,
            "class_names": list(self.class_names),
            "adapted": self.adapted,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int = DEFAULT_SEED,
        freeze_backbone: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning on ``records`` with torchvision's own Mask R-CNN losses.

        The loss is the sum torchvision returns in training mode: RPN objectness and box regression, ROI
        classification and box regression, and the per-pixel mask cross-entropy. With ``freeze_backbone`` the
        ResNet-50 body keeps its weights and its BatchNorm statistics (its BatchNorm layers are held in eval
        mode); the FPN, the RPN and the ROI heads are trained. Mutates this pipeline in place.

        A pipeline that is already adapted (fine-tuned, or loaded with an adapter) is refused: training it
        again would stack a second schedule on the first while the returned record describes only the
        second. Build a fresh re-headed pipeline with ``from_pretrained(class_names=..., seed=...)`` instead.
        """
        if self.adapted:
            raise RuntimeError(
                "Cannot fine-tune: this pipeline was already adapted (fine-tuned, or loaded with an "
                "adapter), so a second call would continue from the adapted weights while the run record "
                "described only the new schedule. Rebuild it with MaskRcnnPipeline.from_pretrained("
                "weights_dir=..., class_names=..., seed=...) and fine-tune that (the tutorial's "
                "reset_to_pretrained() does this)."
            )
        import torch
        from torchvision.transforms.functional import to_tensor

        validate_dataset(records, self.class_names, epochs=epochs)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, int | float)
            or not 0.0 < float(learning_rate) <= 1.0
        ):
            raise ValueError(f"learning_rate must be a number in (0, 1], got {learning_rate!r}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not (freeze_backbone and name.startswith(BACKBONE_PREFIX))
        self.frozen_prefixes = (BACKBONE_PREFIX,) if freeze_backbone else ()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            trainable, lr=float(learning_rate), momentum=DEFAULT_MOMENTUM, weight_decay=DEFAULT_WEIGHT_DECAY
        )
        class_to_id = {name: i + 1 for i, name in enumerate(self.class_names)}

        epoch_losses: list[float] = []
        for epoch in range(epochs):
            self.model.train()
            if freeze_backbone:
                _hold_batchnorm(self.model.backbone.body)
            order = rng.permutation(len(records))
            running, n_batches = 0.0, 0
            for start in range(0, len(records), batch_size):
                batch = [records[int(i)] for i in order[start : start + batch_size]]
                images = [to_tensor(validate_image(r["image"])).to(self.device) for r in batch]
                targets = [
                    {
                        "boxes": torch.tensor(
                            [[float(v) for v in b] for b in r["boxes"]],
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        "labels": torch.tensor(
                            [class_to_id[n] for n in r["labels"]], dtype=torch.int64, device=self.device
                        ),
                        "masks": torch.as_tensor(
                            np.stack([np.asarray(m, dtype=np.uint8) for m in r["masks"]]), device=self.device
                        ),
                    }
                    for r in batch
                ]
                optimizer.zero_grad(set_to_none=True)
                losses = self.model(images, targets)
                loss = sum(losses.values())
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu())
                n_batches += 1
            epoch_losses.append(running / max(1, n_batches))
            if progress is not None:
                progress({"epoch": epoch + 1, "epochs": epochs, "loss": epoch_losses[-1]})
        self.model.eval()
        self.adapted = True
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "momentum": DEFAULT_MOMENTUM,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "optimizer": "SGD",
            "seed": seed,
            "precision": "float32",
            "freeze_backbone": freeze_backbone,
            "frozen_prefixes": list(self.frozen_prefixes),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "loss": "torchvision Mask R-CNN training losses (RPN objectness + RPN box + ROI class + ROI "
            "box + mask)",
            "device": self.device,
            "class_names": list(self.class_names),
            "reinitialised_prefixes": list(self.reinitialised),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        threshold: float = EVAL_DETECTION_THRESHOLD,
        iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
        max_detections: int = MAX_EVAL_DETECTIONS,
    ) -> dict[str, Any]:
        """Score a labelled dataset: box and mask average precision plus evaluation metadata."""
        checked = _check_threshold(threshold, "threshold")
        predictions = []
        for record in records:
            result = self.detect(record["image"], threshold=checked)
            dets = [{**d, "mask": m} for d, m in zip(result["detections"], result["masks"], strict=True)][
                :max_detections
            ]
            predictions.append(dets)
        box = average_precision(
            predictions, records, self.class_names, kind="box", iou_thresholds=iou_thresholds
        )
        mask = average_precision(
            predictions, records, self.class_names, kind="mask", iou_thresholds=iou_thresholds
        )
        return {
            "ap": box["ap"],
            "ap50": box["ap50"],
            "ap75": box["ap75"],
            "per_class_ap50": box["per_class_ap50"],
            "mask_ap": mask["ap"],
            "mask_ap50": mask["ap50"],
            "mask_ap75": mask["ap75"],
            "per_class_mask_ap50": mask["per_class_ap50"],
            "n_images": box["n_images"],
            "n_references": box["n_references"],
            "threshold": checked,
            "mask_threshold": MASK_THRESHOLD,
            "max_detections": max_detections,
            "adapted": self.adapted,
            "class_names": list(self.class_names),
            "estimation": f"one pass over {len(records)} held-out images; no resampling, no dispersion "
            "estimate",
            "implementation": (
                "package-local average_precision on boxes and on masks: greedy score-ordered matching and "
                "101-point "
                "interpolation, without pycocotools area ranges or crowd handling"
            ),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def save_artifact(self, path: str | Path, *, notes: str | None = None) -> dict[str, Any]:
        """Write the adapted tensors as one SafeTensors file with the provenance in its metadata.

        Tensors under ``frozen_prefixes`` are left out: they equal the verified base checkpoint, which
        ``load_artifact`` loads first. The artifact is therefore an adapter bound to the base checkpoint.
        """
        from safetensors.torch import save_file

        if not self.adapted:
            raise RuntimeError("nothing to export: the pipeline has not been fine-tuned")
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in self.model.state_dict().items()
            if not any(name.startswith(prefix) for prefix in self.frozen_prefixes)
        }
        metadata = {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": MODEL_KEY,
            "class_names": json.dumps(list(self.class_names)),
            "frozen_prefixes": json.dumps(list(self.frozen_prefixes)),
            "base_state_digest": self.base_state_digest or "",
            "notes": notes or "",
        }
        save_file(tensors, str(artifact_path), metadata=metadata)
        return {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
            "format": ARTIFACT_FORMAT,
            "class_names": list(self.class_names),
            "tensors": len(tensors),
            "frozen_prefixes": list(self.frozen_prefixes),
            "base_state_digest": self.base_state_digest,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @staticmethod
    def read_artifact_metadata(path: str | Path) -> dict[str, Any]:
        """Read and check the artifact's provenance header without loading any tensor."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            metadata = dict(handle.metadata() or {})
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {metadata.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if (metadata.get("model_id"), metadata.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(
                f"artifact was built on {metadata.get('model_id')}@{metadata.get('model_revision')}, "
                f"package pins {MODEL_ID}@{MODEL_REVISION}"
            )
        if metadata.get("model_key") != MODEL_KEY:
            raise ValueError(
                f"artifact was built on {metadata.get('model_key')!r}, package pins {MODEL_KEY!r}"
            )
        return {
            **metadata,
            "class_names": json.loads(metadata["class_names"]),
            "frozen_prefixes": json.loads(metadata["frozen_prefixes"]),
        }

    def apply_artifact(self, path: str | Path) -> None:
        """Load adapter tensors onto this (base) pipeline; refuse any tensor set that does not fit."""
        from safetensors.torch import load_file

        metadata = self.read_artifact_metadata(path)
        if tuple(metadata["class_names"]) != tuple(self.class_names):
            raise ValueError("artifact class_names differ from this pipeline's class_names")
        expected_base = metadata.get("base_state_digest") or None
        if expected_base and self.base_state_digest and expected_base != self.base_state_digest:
            raise ValueError("artifact was exported against a different base checkpoint digest")
        tensors = load_file(str(path), device="cpu")
        prefixes = tuple(metadata["frozen_prefixes"])
        result = self.model.load_state_dict(tensors, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"artifact carries tensors the model does not have: {result.unexpected_keys[:5]}"
            )
        stray = [key for key in result.missing_keys if not any(key.startswith(p) for p in prefixes)]
        if stray:
            raise ValueError(f"artifact is missing trainable tensors: {stray[:5]}")
        self.model.eval()
        self.adapted = True
        self.frozen_prefixes = prefixes
        self.source = f"artifact:{Path(path).name}"

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        *,
        weights_dir: str | Path | None = None,
        device: str | None = None,
    ) -> MaskRcnnPipeline:
        """Rebuild an adapted pipeline: verified base checkpoint first, then the adapter tensors."""
        metadata = cls.read_artifact_metadata(path)
        pipe = cls.from_pretrained(
            device=device, weights_dir=weights_dir, class_names=metadata["class_names"]
        )
        pipe.apply_artifact(path)
        return pipe
