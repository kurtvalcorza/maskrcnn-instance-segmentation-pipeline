"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.1 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.1 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
Every value a reader can change is a `# @param` form field, and each file-reading BYOD branch has a
location field that bypasses the upload dialog when set (EXE1, EXE2).

The checkpoint is torchvision's, hosted on download.pytorch.org rather than the Hugging Face Hub, so the
template declares `weights_host`.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "maskrcnn_instance_segmentation_pipeline",
    "repo_name": "maskrcnn-instance-segmentation-pipeline",
    "stem": "maskrcnn_instance_segmentation",
    "notebook_name": "maskrcnn_instance_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "MaskRcnnPipeline",
    "weights_key": "maskrcnn-resnet50-fpn-v2",
    "weights_host": {
        "name": "download.pytorch.org",
        "size": "one file that torchvision's weight metadata lists as 177.219 MB",
    },
    "modules": [
        "samples.py",
        "pipeline.py",
    ],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "torchvision", "numpy", "PIL"],
    "title": "Mask R-CNN ResNet-50 FPN v2 (COCO) — DIMER instance segmentation and bounded fine-tuning (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/maskrcnn-instance-segmentation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/maskrcnn-instance-segmentation-pipeline/blob/main/tutorials/maskrcnn_instance_segmentation_colab.ipynb",
        ),
        (
            "torchvision",
            "https://img.shields.io/badge/torchvision-maskrcnn__resnet50__fpn__v2-ee4c2c?style=flat&logo=pytorch&logoColor=white",
            "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-pytorch%2Fvision-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/pytorch/vision",
        ),
        (
            "arXiv",
            "https://img.shields.io/badge/arXiv-1703.06870-b31b1b.svg",
            "https://arxiv.org/abs/1703.06870",
        ),
        (
            "License",
            "https://img.shields.io/badge/License-Apache--2.0-green.svg",
            "https://github.com/kurtvalcorza/maskrcnn-instance-segmentation-pipeline/blob/main/LICENSE",
        ),
    ],
    "capability": "instance segmentation over the COCO classes with torchvision's Mask R-CNN ResNet-50 FPN v2, and a bounded fine-tune that re-heads the box and mask predictors onto your own class vocabulary, evaluates the result on a held-out split with box and mask average precision, and exports a reloadable SafeTensors adapter",
    "intro": (
        "Mask R-CNN is a two-stage instance segmentation model. A ResNet-50 backbone with a feature pyramid network (FPN) produces feature maps "
        "at five scales; a region proposal network (RPN) proposes candidate boxes; and for each surviving region the ROI heads predict a class, "
        "a refined box, and a 28×28 mask that is pasted back into the image. This notebook uses torchvision's **v2** weights "
        "(`MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`): torchvision describes the v2 builder as the improved Mask R-CNN from the "
        "*Benchmarking Detection Transfer Learning with Vision Transformers* paper, trained with an enhanced recipe, and reports 47.4 box AP and "
        "41.8 mask AP on COCO val2017 for these weights. The model's own transform resizes each image so its shorter side is 800 px, and "
        "every instance whose class probability reaches a caller-owned threshold is returned as an xyxy box plus a boolean mask in input pixels.\n\n"
        "**The default path really adapts the model:** it re-heads Mask R-CNN onto a three-class traffic sign vocabulary that does not exist in "
        "COCO (`stop-sign`, `yield-sign`, `speed-limit-sign`), measures a pre-adaptation baseline, runs a bounded fine-tune with the ResNet-50 "
        "body frozen, scores the result on a held-out split with box and mask average precision (AP@[.50:.95] and AP50), runs the adapted model "
        "on unseen images, exports the changed tensors as a SafeTensors adapter, and reloads that adapter onto a fresh copy of the verified base "
        "model to check that it reproduces the same instances. Every number you see is measured in this notebook runtime."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the pinned checkpoint; run COCO instance "
        "segmentation on a drawn scene and score per-object `box_iou` and `mask_iou`; probe the model with a blank and a noise image; build and "
        "validate a labelled instance-segmentation dataset over a new three-class sign vocabulary; split it and measure a pre-adaptation baseline; "
        "run a bounded fine-tune with Mask R-CNN's own training losses; score the adapted model on the held-out split with box and mask AP; run "
        "inference on unseen images; and export, reload and verify the adapter."
    ),
    "exclusions": (
        "real-world traffic sign segmentation (the adaptation dataset is drawn in code, so the model learns these renderings and nothing about "
        "road photographs); COCO benchmark results (the average-precision helper here is a compact implementation without pycocotools area "
        "ranges or crowd handling, and it is run on synthetic data only); full-schedule Mask R-CNN training (the v2 weights come from a long "
        "multi-GPU recipe, recorded in torchvision's weight metadata as pytorch/vision pull request 5773; the tutorial runs a few epochs on 30 images); semantic or panoptic segmentation; keypoints; video "
        "tracking."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). A CUDA GPU such as a Colab or Kaggle T4 is the documented runtime for the fine-tuning stages and is used automatically when present; the notebook also runs on CPU, much more slowly. Runtimes are not measured in this revision. The pinned `torch==2.14.0` wheel and the checkpoint are the largest downloads.",
        "- **Knowledge:** basic Python, PIL and NumPy; bounding boxes as xyxy pixel coordinates; binary masks as boolean arrays; intersection-over-union (IoU) for boxes and for masks; and how to read average precision (AP50 and AP@[.50:.95]).",
        "- **Data:** the default path generates everything in code with `samples.py` and downloads no dataset: one 640×480 COCO demonstration scene and a 40-image labelled sign dataset, each object with an exact mask. BYOD is optional and off by default. Expected BYOD input: one image, or a directory holding `annotations.json` — a list of `{'file': 'name.png', 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...], 'masks': ['name_mask0.png', ...]}` objects — and the image and mask files it names (one single-channel PNG per instance, non-zero pixels inside the instance). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the pinned checkpoint, "
        "runs COCO instance segmentation on a drawn scene, validates the 40-image sign dataset, splits it into training and held-out parts, "
        "measures the pre-adaptation baseline, **runs the bounded fine-tune**, re-evaluates on the held-out split, segments unseen images, exports "
        "the adapter, reloads it onto a fresh base model to verify the instances, and writes machine-readable outputs with provenance. Nothing is "
        "skipped behind a default-off flag, and no clone or DIMER worker is required (NOTEBOOK_SPEC 2.1 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional BYOD branches are included, and both are off by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image through the same validation, segmentation and evaluation-report stages as the sample scene. "
        "`USE_BYOD_DATASET` takes your own labelled instance masks through the full adaptation workflow — validate, split, baseline, fine-tune, "
        "evaluate, export and reload — under NOTEBOOK_SPEC 2.1 DAT14. Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_DIR` to read from a location "
        "without an upload dialog (EXE2)."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. COCO scene
        {
            "md": (
                "## 4. What the pretrained model does on a drawn scene\n\n"
                "Before adapting anything, inspect the model you start from. The carried `samples` module draws a deterministic street scene with "
                "four objects, each with a reference box and an exact reference mask: a **stop sign**, a **traffic light**, an analogue **clock**, "
                "and an orange **sports ball**. All four are COCO classes.\n\n"
                "The detection threshold is a **caller-owned request parameter**, not a pipeline constant. Each score is the instance's **softmax "
                "class probability after torchvision's per-class non-maximum suppression (NMS), not a calibrated** probability for your images. "
                "The default `0.75` is the value torchvision's visualization-utilities gallery uses to keep Mask R-CNN instances (shown there "
                "with the v1 weights, not re-derived for v2), and it is passed explicitly on every call. A mask pixel belongs to the instance when its predicted probability is at least `MASK_THRESHOLD` (0.5).\n\n"
                "The scene is drawn, not photographed, so a miss here is a finding about renderings, not about photographs. The evaluation report "
                "records every miss with `box_iou = 0.0` and `mask_iou = 0.0`. COCO mean average precision needs a labelled image set; on one scene "
                "the verdict is `sample-sanity`."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "OUTPUTS = Path('outputs')\n\n"
                'threshold = 0.75  # @param {{type:"number"}}\n\n'
                "print({{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_DETECTIONS': MAX_DETECTIONS,\n"
                "       'label_slots': len(LABELS), 'unannotated_slots': LABELS.count('N/A'), 'DETECTION_THRESHOLD': DETECTION_THRESHOLD,\n"
                "       'MASK_THRESHOLD': MASK_THRESHOLD}})\n\n"
                "scene, references = tutorial_scene()\n"
                "buffer = io.BytesIO()\n"
                "scene.save(buffer, format='PNG')\n"
                "print({{'sample_kind': 'synthetic', 'size': list(scene.size), 'sha256': hashlib.sha256(buffer.getvalue()).hexdigest()[:16],\n"
                "       'references': {{label: len(ref['boxes']) for label, ref in references.items()}}}})\n\n"
                "input_manifest = validate_inputs(scene, threshold=threshold, names=['tutorial-scene'])\n"
                "try:\n"
                "    validate_inputs(scene, threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'probe': 'threshold=1.5', 'rejected': str(exc)}})\n"
                "print({{'verdict': input_manifest['verdict'], 'findings': input_manifest['findings'], 'inputs': input_manifest['inputs']}})\n\n"
                "coco_result = pipe.detect(scene, threshold=threshold)\n"
                "for det in coco_result['detections']:\n"
                "    print(f\"{{det['label']:>14s}} {{det['score']:.3f}}  mask {{det['mask_area']:>6d}} px  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n\n"
                "coco_report = evaluation_report(coco_result, references, sample_kind='synthetic')\n"
                "print({{'verdict': coco_report['verdict'], 'n_detections': coco_report['n_detections']}})\n"
                "for metric in coco_report['metrics']:\n"
                "    print(f\"  {{metric['reference']:>18s}}  box_iou {{metric['box_iou']:.3f}}  mask_iou {{metric['mask_iou']:.3f}}  same-label {{metric['n_detected_same_label']}}\")\n"
                "hits = sum(1 for m in coco_report['metrics'] if m['mask_iou'] >= 0.5)\n"
                "print(f'{{hits}}/{{len(coco_report[\"metrics\"])}} drawn objects matched at mask IoU >= 0.5')\n\n"
                "overlay = np.asarray(scene).copy()\n"
                "for mask in coco_result['masks']:\n"
                "    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)\n"
                "Image.fromarray(overlay)"
            ),
        },
        # ---------------------------------------------------------------- 5. Degenerate inputs
        {
            "md": (
                "## 5. Degenerate input probes: blank canvas and noise\n\n"
                "Ask a segmentation model about structure-free input before trusting it. A model that returns confident instances on a blank canvas "
                "or on uniform noise will return them on empty real frames too. The cell counts instances on both images at the default threshold "
                "and at the evaluation threshold `EVAL_DETECTION_THRESHOLD` (0.05) that average precision is computed at.\n\n"
                "**What to look for:** any instance above the default threshold on these two images is a false positive by construction."
            ),
            "code": (
                "degenerate = {{}}\n"
                "for name, image in (('blank', blank_scene()), ('noise', noise_scene(0))):\n"
                "    standard = pipe.detect(image, threshold=threshold)['detections']\n"
                "    lenient = pipe.detect(image, threshold=EVAL_DETECTION_THRESHOLD)['detections']\n"
                "    degenerate[name] = {{\n"
                "        'at_default_threshold': len(standard),\n"
                "        'at_evaluation_threshold': len(lenient),\n"
                "        'top': [(d['label'], round(d['score'], 3), d['mask_area']) for d in lenient[:3]],\n"
                "    }}\n"
                "print(json.dumps(degenerate, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 6. Dataset & Validation
        {
            "md": (
                "## 6. Labelled adaptation dataset and validation\n\n"
                "Suppose your task needs sign classes that COCO does not have. `sign_dataset` draws a deterministic 40-image dataset over "
                "`SIGN_CLASSES`: `stop-sign`, `yield-sign` and `speed-limit-sign`. Every sign is drawn twice, once onto the image and once onto its "
                "own mask, so each reference mask is exact and each reference box is the tight box of its mask.\n\n"
                "**Keep the two vocabularies apart.** COCO has `stop sign` (with a space). The adaptation vocabulary uses `stop-sign` (hyphenated), "
                "`yield-sign` and `speed-limit-sign`. They are different class identities, and the adapted model answers in the new names only.\n\n"
                "`validate_dataset` checks every record before any model runs: the record keys, the image size ceilings, that every box is finite, "
                "non-empty and inside its image, that every mask has the image's shape, is boolean, is not empty and lies inside its box, and that "
                "every label is in the vocabulary. It returns a dataset manifest with the instance count per class and a finding for any class "
                "that has no instance."
            ),
            "code": (
                'N_IMAGES = 40  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'EPOCHS = 10  # @param {{type:"integer"}}\n\n'
                "records = sign_dataset(N_IMAGES, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SIGN_CLASSES, epochs=EPOCHS)\n"
                "print(json.dumps({{k: v for k, v in dataset_manifest.items() if k != 'schema'}}, indent=2))\n\n"
                "preview = Image.new('RGB', (480, 320))\n"
                "for index, record in enumerate(records[:3]):\n"
                "    preview.paste(record['image'].resize((160, 160)), (160 * index, 0))\n"
                "    union = np.any(np.stack(record['masks']), axis=0)\n"
                "    preview.paste(Image.fromarray((union * 255).astype(np.uint8)).convert('RGB').resize((160, 160)), (160 * index, 160))\n"
                "preview"
            ),
        },
        # ---------------------------------------------------------------- 7. Split & Baseline
        {
            "md": (
                "## 7. Split, re-head, and measure the pre-adaptation baseline\n\n"
                "The dataset is split at random into a training part (75%, 30 images) and a held-out part (25%, 10 images). A random split is "
                "valid here because every image is drawn independently; records that share a photograph or a camera session must be split by that "
                "group instead. The held-out images are never shown to the optimizer, and no hyperparameter is selected on them.\n\n"
                "`from_pretrained(class_names=SIGN_CLASSES)` loads the verified checkpoint and replaces only the two class-specific predictors with "
                "freshly initialised layers: the box predictor (`roi_heads.box_predictor`, now 3 classes plus background) and the mask predictor "
                "(`roi_heads.mask_predictor`, one 28×28 mask channel per class). The backbone, the FPN, the RPN and the shared ROI layers keep their "
                "COCO weights.\n\n"
                "**The baseline is expected to be near zero.** Randomly initialised predictors have no information about the new classes, so "
                "this number is the floor the fine-tune has to beat, not a property of Mask R-CNN."
            ),
            "code": (
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 0  # @param {{type:"integer"}}\n\n'
                "train_records, held_out = split_dataset(records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "overlap = {{r['id'] for r in train_records}} & {{r['id'] for r in held_out}}\n"
                "assert not overlap, f'split leaked records: {{sorted(overlap)}}'\n"
                "print({{'train': len(train_records), 'held_out': len(held_out),\n"
                "       'train_instances': sum(len(r['boxes']) for r in train_records),\n"
                "       'held_out_instances': sum(len(r['boxes']) for r in held_out)}})\n\n"
                "adapter = MaskRcnnPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "print({{'class_names': list(adapter.class_names), 'device': adapter.device, 'reinitialised': list(adapter.reinitialised)}})\n\n"
                "baseline = adapter.evaluate(held_out)\n"
                "print(json.dumps({{key: round(baseline[key], 4) for key in ('ap', 'ap50', 'mask_ap', 'mask_ap50')}} | {{'n_references': baseline['n_references']}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 8. Bounded Fine-Tuning
        {
            "md": (
                "## 8. Bounded fine-tuning\n\n"
                "This cell runs the real adaptation step in this runtime. It is gradient fine-tuning with Mask R-CNN's own training losses, which "
                "torchvision returns when the model is called with targets: RPN objectness and RPN box regression, ROI classification and ROI box "
                "regression, and a per-pixel binary cross-entropy on the 28×28 mask of each positive region. The optimised loss is their sum.\n\n"
                "- **The ResNet-50 body is frozen.** Parameters under `backbone.body.` keep their COCO values, and its BatchNorm layers are held in "
                "evaluation mode so their running statistics do not drift either. The FPN, the RPN and the ROI heads are trained. The cell prints the "
                "trainable and total parameter counts.\n"
                "- **Schedule:** `EPOCHS` epochs of SGD at learning rate `0.005`, momentum 0.9 and weight decay 0.0005 (the values of torchvision's "
                "detection fine-tuning tutorial), batch size 2, float32, seed `SEED`.\n\n"
                "**Read the loss as optimisation evidence only.** A falling loss says the optimizer is fitting the training images; the held-out "
                "average precision in the next section is the task evidence."
            ),
            "code": (
                'LEARNING_RATE = 0.005  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 2  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = True  # @param {{type:"boolean"}}\n\n'
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_backbone=FREEZE_BACKBONE,\n"
                "    progress=lambda row: print(f\"epoch {{row['epoch']}}/{{row['epochs']}}  loss {{row['loss']:.4f}}\"),\n"
                ")\n"
                "print(json.dumps({{key: run[key] for key in ('freeze_backbone', 'trainable_parameters', 'total_parameters', 'epochs',\n"
                "                                         'batch_size', 'learning_rate', 'optimizer', 'precision', 'device')}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate Held-Out
        {
            "md": (
                "## 9. Evaluate on the held-out split\n\n"
                "`evaluate` re-runs on the same held-out images with the same thresholds as the baseline, so the two rows are comparable. It scores "
                "every instance twice: once by box IoU (`ap`, `ap50`, `ap75`) and once by mask IoU (`mask_ap`, `mask_ap50`, `mask_ap75`). `ap50` is "
                "average precision at IoU 0.50; `ap` averages AP over the ten IoU thresholds 0.50–0.95, so it also rewards tight boxes and masks. A "
                "model can find every sign with a good box and still draw a poor mask, which only the mask rows show. These are tutorial metrics "
                "from one pass over 10 synthetic images, with no dispersion estimate."
            ),
            "code": (
                "adapted = adapter.evaluate(held_out)\n"
                "print(f\"{{'metric':<10s}} {{'baseline':>10s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in ('ap', 'ap50', 'ap75', 'mask_ap', 'mask_ap50', 'mask_ap75'):\n"
                '    print(f"{{key:<10s}} {{baseline[key]:>10.4f}} {{adapted[key]:>10.4f}} {{adapted[key] - baseline[key]:>+10.4f}}")\n'
                "print('per-class mask AP50:', {{k: round(v, 4) for k, v in adapted['per_class_mask_ap50'].items()}})"
            ),
        },
        # ---------------------------------------------------------------- 10. New-data inference
        {
            "md": (
                "## 10. Inference on unseen images\n\n"
                "Three new images come from a seed the dataset never used (`NEW_DATA_SEED = 99`). The adapted pipeline segments the new sign "
                "classes, and each reference instance is compared with the best same-label instance by box IoU and by mask IoU."
            ),
            "code": (
                'NEW_DATA_SEED = 99  # @param {{type:"integer"}}\n\n'
                "new_records = sign_dataset(3, seed=NEW_DATA_SEED)\n"
                "new_data_rows = []\n"
                "for record in new_records:\n"
                "    out = adapter.detect(record['image'], threshold=threshold)\n"
                "    box_ious, mask_ious = [], []\n"
                "    for box, label, mask in zip(record['boxes'], record['labels'], record['masks'], strict=True):\n"
                "        same = [i for i, d in enumerate(out['detections']) if d['label'] == label]\n"
                "        box_ious.append(round(max((box_iou(out['detections'][i]['box'], box) for i in same), default=0.0), 3))\n"
                "        mask_ious.append(round(max((mask_iou(out['masks'][i], mask) for i in same), default=0.0), 3))\n"
                "    row = {{'id': record['id'], 'truth': record['labels'],\n"
                "           'detections': [(d['label'], round(d['score'], 3)) for d in out['detections']],\n"
                "           'same_label_box_iou': box_ious, 'same_label_mask_iou': mask_ious}}\n"
                "    new_data_rows.append(row)\n"
                "    print(json.dumps(row))"
            ),
        },
        # ---------------------------------------------------------------- 11. Export, reload & verify
        {
            "md": (
                "## 11. Adapter export, fresh reload, and equivalence check\n\n"
                "`save_artifact` writes `outputs/maskrcnn_adapter.safetensors`: every tensor the fine-tune could change, plus a metadata header naming "
                "the base model, its pinned checkpoint SHA-256, the class names and the frozen prefixes. The frozen ResNet-50 body is left out because "
                "it equals the verified base checkpoint.\n\n"
                "`load_artifact` then builds a **fresh** pipeline from the verified base checkpoint, loads the adapter tensors onto it, and refuses "
                "an adapter whose format, base identity, base digest or tensor set does not fit. The cell compares the reloaded instances with the "
                "in-memory model's on an unseen image, with a stated tolerance: loading succeeding is not the check, reproducing the boxes, scores "
                "and masks is."
            ),
            "code": (
                "artifact_path = OUTPUTS / 'maskrcnn_adapter.safetensors'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='Mask R-CNN ResNet-50 FPN v2 sign adaptation tutorial adapter')\n"
                "print(json.dumps(descriptor, indent=2))\n\n"
                "reloaded = MaskRcnnPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'class_names': list(reloaded.class_names)}})\n\n"
                "TOLERANCE = 1e-3\n"
                "MASK_AGREEMENT = 0.99\n"
                "test_img = new_records[0]['image']\n"
                "out_orig = adapter.detect(test_img, threshold=EVAL_DETECTION_THRESHOLD)\n"
                "out_reloaded = reloaded.detect(test_img, threshold=EVAL_DETECTION_THRESHOLD)\n"
                "assert len(out_orig['detections']) == len(out_reloaded['detections'])\n"
                "for d1, d2, m1, m2 in zip(out_orig['detections'], out_reloaded['detections'], out_orig['masks'], out_reloaded['masks'], strict=True):\n"
                "    assert d1['label'] == d2['label']\n"
                "    assert np.allclose(d1['box'], d2['box'], atol=TOLERANCE)\n"
                "    assert abs(d1['score'] - d2['score']) <= TOLERANCE\n"
                "    assert (not m1.any() and not m2.any()) or mask_iou(m1, m2) >= MASK_AGREEMENT\n"
                "reload_check = {{'instances_compared': len(out_orig['detections']), 'tolerance': TOLERANCE, 'mask_iou_at_least': MASK_AGREEMENT, 'equivalent': True}}\n"
                "print(reload_check)"
            ),
        },
        # ---------------------------------------------------------------- 12. Outputs & provenance
        {
            "md": (
                "## 12. Write machine-readable outputs and provenance\n\n"
                "The cell writes:\n"
                "- `outputs/maskrcnn_instance_segmentation_input_manifest.json`\n"
                "- `outputs/maskrcnn_instance_segmentation_evaluation_report.json`\n"
                "- `outputs/maskrcnn_instance_segmentation_result.json` (identity, runtime versions, device, dataset manifest, split, baseline and adapted metrics, "
                "fine-tuning configuration, new-data rows, adapter descriptor and reload check)\n"
                "- `outputs/maskrcnn_instance_segmentation_detections.csv` (one row per instance, with its mask area)\n"
                "- `outputs/maskrcnn_instance_segmentation_masks.npz` (the demonstration scene's boolean masks)\n"
                "- `outputs/maskrcnn_instance_segmentation_annotated.png`\n"
                "- `outputs/maskrcnn_adapter.safetensors` (written in Section 11)"
            ),
            "code": (
                "import csv\n"
                "from PIL import ImageDraw\n\n"
                "with open(OUTPUTS / '{stem}_input_manifest.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(input_manifest, f, indent=2)\n\n"
                "with open(OUTPUTS / '{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(coco_report, f, indent=2)\n\n"
                "np.savez_compressed(OUTPUTS / '{stem}_masks.npz', *coco_result['masks'])\n\n"
                "annotated_pixels = np.asarray(scene).copy()\n"
                "for mask in coco_result['masks']:\n"
                "    annotated_pixels[mask] = (0.5 * annotated_pixels[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)\n"
                "annotated = Image.fromarray(annotated_pixels)\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in coco_result['detections']:\n"
                "    x0, y0, x1, y1 = det['box']\n"
                "    draw.rectangle([x0, y0, x1, y1], outline='red', width=3)\n"
                "    draw.text((x0 + 4, y0 + 4), f\"{{det['label']}} {{det['score']:.2f}}\", fill='red')\n"
                "annotated.save(OUTPUTS / '{stem}_annotated.png')\n\n"
                "with open(OUTPUTS / '{stem}_detections.csv', 'w', newline='', encoding='utf-8') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1', 'mask_area'])\n"
                "    for rank, d in enumerate(coco_result['detections']):\n"
                "        writer.writerow(['tutorial-scene', rank, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box']), d['mask_area']])\n"
                "    for row, record in zip(new_data_rows, new_records, strict=True):\n"
                "        for rank, d in enumerate(adapter.detect(record['image'], threshold=threshold)['detections']):\n"
                "            writer.writerow([row['id'], rank, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box']), d['mask_area']])\n\n"
                "result_export = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'weights_url': WEIGHTS_URL,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,\n"
                "                'cuda': torch.cuda.is_available()}},\n"
                "    'device': pipe.device,\n"
                "    'threshold': threshold,\n"
                "    'mask_threshold': MASK_THRESHOLD,\n"
                "    'coco_detections': coco_result['detections'],\n"
                "    'degenerate_probes': degenerate,\n"
                "    'adaptation': {{\n"
                "        'dataset': {{k: v for k, v in dataset_manifest.items() if k != 'schema'}},\n"
                "        'dataset_seed': DATASET_SEED,\n"
                "        'split': {{'train': len(train_records), 'held_out': len(held_out), 'seed': SEED, 'holdout': HOLDOUT}},\n"
                "        'finetune': run,\n"
                "        'baseline': baseline,\n"
                "        'adapted': adapted,\n"
                "        'new_data': new_data_rows,\n"
                "        'artifact': descriptor,\n"
                "        'reload_check': reload_check,\n"
                "    }},\n"
                "}}\n"
                "with open(OUTPUTS / '{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_export, f, indent=2, default=str)\n\n"
                "for p in sorted(OUTPUTS.iterdir()):\n"
                "    if p.is_file():\n"
                "        print(f'  {{p.name:<44s}} {{p.stat().st_size:>12,d}} bytes')"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: Bring Your Own Data (BYOD)\n\n"
                "Both branches are off by default, so `Run all` never stops here. Before you turn one on, read the contract:\n\n"
                "- **Image branch** (`USE_BYOD_IMAGE`): one image file PIL can open, each side between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` "
                "(4096) px. It runs through `validate_inputs`, `detect` and `evaluation_report` — the report is `not-measurable`, because no "
                "reference instances come with it.\n"
                "- **Dataset branch** (`USE_BYOD_DATASET`): a directory with `annotations.json` (a list of `{{'file', 'boxes', 'labels', 'masks'}}` "
                "objects: xyxy pixel boxes and one mask PNG name per instance) and the image and mask files it names, at least 2 records and at most "
                "5,000. File names must stay inside the directory. `BYOD_CLASS_NAMES` is a comma-separated vocabulary; leave it empty to use the "
                "sorted set of labels in the annotations. The branch runs the same validate → split → baseline → fine-tune → evaluate → export → "
                "reload stages as the sample.\n\n"
                "Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_DIR` to read from a mounted or local location; leave them empty on Colab to get an upload "
                "dialog instead. Uploaded files are written under `outputs/byod/` in this runtime and are not sent anywhere else. The first lines of "
                "the cell show the validator refusing two malformed inputs with messages that name the failed rule."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'BYOD_IMAGE_PATH = ""  # @param {{type:"string"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_DATASET_DIR = ""  # @param {{type:"string"}}\n'
                'BYOD_CLASS_NAMES = ""  # @param {{type:"string"}}\n\n'
                "_probe_image = blank_scene()\n"
                "for desc, probe in (\n"
                "    ('non-image object', lambda: validate_inputs('/not/an/image.png')),\n"
                "    ('empty mask', lambda: validate_dataset([{{'image': _probe_image, 'boxes': [[0, 0, 10, 10]], 'labels': [SIGN_CLASSES[0]],\n"
                "                                               'masks': [np.zeros((_probe_image.height, _probe_image.width), dtype=bool)]}}], SIGN_CLASSES)),\n"
                "):\n"
                "    try:\n"
                "        probe()\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print(f'refused as expected: {{desc}} -> {{type(exc).__name__}}: {{exc}}')\n\n"
                "BYOD_DIR = OUTPUTS / 'byod'\n\n"
                "def _upload_into(target):\n"
                "    from google.colab import files  # type: ignore[import-not-found]\n"
                "    target.mkdir(parents=True, exist_ok=True)\n"
                "    for name, data in files.upload().items():\n"
                "        (target / Path(name).name).write_bytes(data)\n"
                "    return target\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    image_path = Path(BYOD_IMAGE_PATH) if BYOD_IMAGE_PATH else next(iter(sorted(_upload_into(BYOD_DIR / 'image').iterdir())))\n"
                "    with Image.open(image_path) as handle:\n"
                "        byod_image = handle.convert('RGB')\n"
                "    print(validate_inputs(byod_image, threshold=threshold, names=[image_path.name])['verdict'])\n"
                "    byod_result = pipe.detect(byod_image, threshold=threshold)\n"
                "    for det in byod_result['detections'][:20]:\n"
                "        print(f\"{{det['label']:>16s}} {{det['score']:.3f}}  mask {{det['mask_area']:>7d}} px  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n"
                "    print(evaluation_report(byod_result, None, sample_kind='byod')['verdict'])\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True to run segmentation on your own image.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    dataset_dir = Path(BYOD_DATASET_DIR) if BYOD_DATASET_DIR else _upload_into(BYOD_DIR / 'dataset')\n"
                "    byod_records = read_detection_records(dataset_dir)\n"
                "    byod_names = [n.strip() for n in BYOD_CLASS_NAMES.split(',') if n.strip()] or sorted({{l for r in byod_records for l in r['labels']}})\n"
                "    byod_manifest = validate_dataset(byod_records, byod_names, epochs=EPOCHS)\n"
                "    print(json.dumps({{k: v for k, v in byod_manifest.items() if k != 'schema'}}, indent=2))\n"
                "    byod_train, byod_held = split_dataset(byod_records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "    byod_pipe = MaskRcnnPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_names, seed=SEED)\n"
                "    byod_baseline = byod_pipe.evaluate(byod_held)\n"
                "    byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_backbone=FREEZE_BACKBONE)\n"
                "    byod_adapted = byod_pipe.evaluate(byod_held)\n"
                "    print({{key: (round(byod_baseline[key], 4), round(byod_adapted[key], 4)) for key in ('ap50', 'mask_ap50')}})\n"
                "    byod_descriptor = byod_pipe.save_artifact(OUTPUTS / 'byod_maskrcnn_adapter.safetensors', notes='BYOD adaptation adapter')\n"
                "    MaskRcnnPipeline.load_artifact(OUTPUTS / 'byod_maskrcnn_adapter.safetensors', weights_dir=WEIGHTS_DIR)\n"
                "    print('BYOD adapter exported and reloaded:', byod_descriptor['sha256'][:16])\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt Mask R-CNN on your own labelled images.')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**What this notebook established, in this runtime.** The pinned torchvision checkpoint for `torchvision/maskrcnn_resnet50_fpn_v2` was "
        "verified against a committed SHA-256 manifest before loading. The pretrained model was run on a drawn scene and on two structure-free "
        "probes, and its boxes and masks were compared with the drawn references by IoU. A three-class sign vocabulary that COCO does not contain "
        "was then adapted by replacing the box and mask predictors, training the FPN, RPN and ROI heads with the ResNet-50 body frozen, and "
        "scoring the held-out split with box and mask average precision before and after. The adapter was exported, reloaded onto a fresh base "
        "model and checked against the in-memory instances.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set and the "
        "pinned checkpoint together reproduce these stages in a fresh runtime, without the repository being cloned or installed and without "
        "any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the adapted model "
        "generalises beyond the synthetic images it was fitted to. The held-out AP is measured on 10 drawn images and carries no dispersion "
        "estimate. The scores are not calibrated probabilities.\n\n"
        "**Reproducibility.** Seeds are form fields (`DATASET_SEED`, `SEED`, `NEW_DATA_SEED`), the run is float32 with no data augmentation, "
        "and the new predictors are initialised under `SEED`. GPU kernels (ROI Align in particular) are not forced to be deterministic, so "
        "repeated GPU runs can differ in the last digits of the loss and the scores.\n\n"
        "**Try next.** Change `FREEZE_BACKBONE` to `False` and compare held-out mask AP and runtime, or change `EPOCHS` and watch where "
        "held-out mask AP stops improving. To transfer the workflow, point `BYOD_DATASET_DIR` at a small labelled set from your own domain.\n\n"
        "## References\n\n"
        "- He, K., Gkioxari, G., Dollár, P. and Girshick, R. (2017). *Mask R-CNN.* [arXiv:1703.06870](https://arxiv.org/abs/1703.06870).\n"
        "- Li, Y., Xie, S., Chen, X., Dollár, P., He, K. and Girshick, R. (2021). *Benchmarking Detection Transfer Learning with Vision Transformers.* [arXiv:2111.11429](https://arxiv.org/abs/2111.11429) — the paper torchvision cites for the v2 builder.\n"
        "- torchvision model documentation: [maskrcnn_resnet50_fpn_v2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html); upstream repository [pytorch/vision](https://github.com/pytorch/vision) — BSD-3-Clause.\n"
        "- Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312).\n"
        "- Repository model card: https://github.com/kurtvalcorza/maskrcnn-instance-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/maskrcnn-instance-segmentation-pipeline`](https://github.com/kurtvalcorza/maskrcnn-instance-segmentation-pipeline) — source repository for this pipeline."
    ),
}
