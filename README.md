# Mask R-CNN ResNet-50 FPN v2 instance segmentation pipeline

DIMER pipeline for **Mask R-CNN with a ResNet-50 FPN backbone and torchvision's v2 weights** (`torchvision/maskrcnn_resnet50_fpn_v2`), the reference two-stage instance segmentation model, trained on the COCO categories. The pipeline loads the checkpoint only from a digest-verified local file, returns pixel-space boxes and boolean masks with the model's softmax score under a caller-owned threshold, and adds a bounded fine-tuning workflow that re-heads the box and mask predictors onto a new class vocabulary and exports a SafeTensors adapter.

> **The upstream checkpoint is pinned** (pinned 2026-09-25) to the SHA-256 of its bytes, `73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`. The manifest records the download URL, that digest and the byte size (185,828,065); the digest starts with the `73cbd019` prefix in the file name, and the file strict-loaded into the architecture. Default-path execution recorded on 2026-09-25 (Kaggle T4); REL12 BYOD exercise pending before promotion (see [Release status](#release-status)).

## Upstream alignment

- Model: `torchvision/maskrcnn_resnet50_fpn_v2` (builder `maskrcnn_resnet50_fpn_v2`, weights `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`)
- Revision: `73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`, the SHA-256 of the checkpoint file
- Weights host: `https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` (not the Hugging Face Hub)
- Upstream code license: BSD-3-Clause (torchvision); the COCO training data carries its own terms
- Upstream task: instance segmentation over the COCO categories (91 slots including background, 80 trained with instances)
- Repository adaptation: bounded gradient fine-tuning of the FPN, RPN and ROI heads, with the ResNet-50 body frozen by default

## Quick start

```python
from PIL import Image
from maskrcnn_instance_segmentation_pipeline import MaskRcnnPipeline, sign_dataset, split_dataset, SIGN_CLASSES

pipe = MaskRcnnPipeline.from_pretrained(allow_download=True)  # stages + verifies weights/maskrcnn-resnet50-fpn-v2
result = pipe.detect(Image.open("street.jpg"), threshold=0.75)     # threshold is caller-owned
for det, mask in zip(result["detections"], result["masks"]):       # sorted by score; boxes are xyxy pixels
    print(det["label"], det["box"], round(det["score"], 3), mask.sum())

train, held_out = split_dataset(sign_dataset(40), train_fraction=0.75)
adapter = MaskRcnnPipeline.from_pretrained(class_names=SIGN_CLASSES)
print(adapter.evaluate(held_out)["mask_ap50"])                     # baseline
adapter.finetune(train)                                            # 10 epochs, ResNet-50 body frozen
print(adapter.evaluate(held_out)["mask_ap50"])                     # adapted
adapter.save_artifact("outputs/maskrcnn_adapter.safetensors")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`, and run `pytest` for the offline test suite. No weights are needed: `tests/test_tiny_model.py` builds the full architecture with random weights and a 64 px input size to exercise fine-tuning, evaluation and adapter reload.

## Pinning the checkpoint

The checkpoint is pinned (see [Upstream alignment](#upstream-alignment)). To re-pin it, from the repository root with network access to download.pytorch.org:

1. Run `python tools/pin_snapshot.py`. It downloads the manifest URL, checks that the file's SHA-256 starts with `73cbd019` (the prefix in its name), strict-loads it into `maskrcnn_resnet50_fpn_v2`, moves it into `weights/maskrcnn-resnet50-fpn-v2/`, and writes the digest and byte size into the manifest and the digest into `MODEL_REVISION`.
2. Commit, then run `python tools/build_notebook.py` and commit the regenerated notebook.
3. Update the digest and byte size cited in `README.md`, `MODEL_CARD.md`, `STATUS.md`, `docs/WEIGHTS.md`, `tutorials/README.md` and `docs/release-verification.md`.
4. Run `python tools/validate_release_assets.py` and `pytest`. A new pin invalidates any recorded execution, so the status returns to Candidate until the new checkpoint is run.

## Weights layout

```
weights/maskrcnn-resnet50-fpn-v2/
  dimer-base-manifest.json                     # modelId, revision, url, bytes + SHA-256 (1 file)
  maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth   # git-ignored; staged from download.pytorch.org
```

## Input ceilings and thresholds

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_DETECTIONS = 100` (torchvision's `box_detections_per_img`), `COCO_CATEGORIES` (91 slots in label-id order, background first) and `LABELS` (the 90 non-background slots, 10 of them `"N/A"`), `DETECTION_THRESHOLD = 0.75` and `MASK_THRESHOLD = 0.5` (the values torchvision's visualization gallery uses for Mask R-CNN). Adaptation datasets hold 1–5,000 records, each instance with a boolean mask. See `MODEL_CARD.md` for who owns the thresholds and what the score means.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/maskrcnn-instance-segmentation-pipeline/blob/main/tutorials/maskrcnn_instance_segmentation_colab.ipynb)

`tutorials/maskrcnn_instance_segmentation_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.2 and is **standalone** (§4): `tools/build_notebook.py` generates it, and it carries the package modules, the model identity, the manifest and the hash-locked runtime (`tutorials/requirements-colab.lock.txt`), so it runs without this repository. Section 1 builds a separate uv environment from that lock and runs every later cell there, so `Run all` needs no restart; the notebook supports Linux x86_64 runtimes only (Colab, Kaggle, Linux Jupyter). Its default `Run all` path segments a drawn COCO scene, probes a blank and a noise image, validates a 40-image drawn sign dataset, measures a baseline, fine-tunes, evaluates the held-out split with box and mask AP, segments unseen images, and exports and reloads the adapter. BYOD image and dataset branches are off by default. See `tutorials/README.md` and `docs/release-verification.md`.

## Release status

**Candidate.** The checkpoint is pinned (SHA-256 `73cbd0190fcb…`). Default-path execution recorded on 2026-09-25 (Kaggle T4) for an earlier notebook revision (blob `a8f04d114788`, commit `8bd5bba`, in-kernel install) with both BYOD branches off; it passed only after a manual restart after the install cell, so it is not a one-pass Run all and not promotion evidence. The current blob `a5e0533913f5` (commit `568eead`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI sequential execution, 16/16 code cells, 174.2 s; adapted box `ap` 0.9072, `mask_ap` 0.9823). The pretrained model found 3 of 4 drawn COCO objects (sports ball missed); on one seeded split of 10 synthetic held-out sign images the fine-tune moved box `ap` 0.0347 → 0.8693 and `mask_ap` 0.0 → 1.0; one runtime. REL12 BYOD exercise pending before promotion: release step 7 has not been run. Static checks, unit tests and the small-model test do not constitute notebook execution evidence; `docs/release-verification.md` defines the release gate.

## Documentation

- `MODEL_CARD.md`: MODEL_CARD_SPEC 1.2 card, provenance, input/output contract.
- `docs/WEIGHTS.md`: weight provenance, pinning and hosting notes.
- `STATUS.md`: release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). torchvision, which defines the architecture and publishes the weights, is BSD-3-Clause; its README notes that pre-trained weights may carry terms derived from their training data. See `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
