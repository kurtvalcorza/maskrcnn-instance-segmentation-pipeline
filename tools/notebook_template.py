"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.2 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
Every value a reader can change is a `# @param` form field, and each file-reading BYOD branch has a
location field that bypasses the upload dialog when set (EXE1, EXE2).

The checkpoint is torchvision's, hosted on download.pytorch.org rather than the Hugging Face Hub, so the
template declares `weights_host`.

Review 2026-10-02 (MRC-M1..M3, MRC-m1..m6), following the sibling detr-detection-pipeline fix (c3c81cb): the
notebook runs in the fleet's uv isolated environment (generator /2.1, no in-kernel install, no restart), carries
the GUIDED layer (audience, task contract, how to use, roadmap, predictions, worked answers, a change-one-thing
activity, troubleshooting, glossary, conclusion template), rebuilds the re-headed model before every fine-tune
(`reset_to_pretrained()`; `finetune` itself refuses an adapted pipeline), keeps every schedule field in Section 8,
explains the saturated 1.0 rows instead of comparing on them (Kurt's task-at-ceiling rule), and gives BYOD named
refusals, a results file and a reload comparison.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "maskrcnn_instance_segmentation_pipeline",
    "repo_name": "maskrcnn-instance-segmentation-pipeline",
    "stem": "maskrcnn_instance_segmentation",
    "notebook_name": "maskrcnn_instance_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # GDL11 (NOTEBOOK_SPEC 2.2 §3.5): Sections 1-3 labelled Infrastructure and the carried module source collapsed.
    "infrastructure_labels": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (generator /2.1): managed CPython, a size- and SHA-256-verified
    # uv wheel, and a lock compiled from the pyproject pins with `uv pip compile pyproject.toml --python-version 3.12
    # --python-platform x86_64-manylinux_2_28 --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt`,
    # every transitive version constrained to the detr-detection-pipeline c3c81cb lock, which passed a one-pass Colab
    # T4 Run all; all 33 entries carry that lock's versions and hashes unchanged.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
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
        "model to check that it reproduces the same instances. Every number you see is measured in this notebook runtime. "
        "One result is worth knowing in advance: on these drawn signs the adapted model reaches the **ceiling** of most held-out metrics — in "
        "the recorded run every mask AP row read 1.0 — so this notebook teaches the adaptation *workflow* and how to read a saturated result, "
        "not the size of a gain. Section 9 explains what a 1.0 row does and does not show, and Section 14 compares settings on the one "
        "held-out number that still had headroom (box AP@[.50:.95]).\n\n"
        "**Who this is for.** A learner who knows basic Python and PIL, has met bounding boxes and intersection-over-union, and wants to see "
        "how a pretrained instance segmentation model is re-headed and fine-tuned for a new class vocabulary, and how the result is measured "
        "without fooling themselves. No prior experience with Mask R-CNN or fine-tuning is assumed; *instance*, *mask*, *NMS*, *re-heading* and "
        "*average precision* are explained where they are first used and again in the **Glossary** at the end.\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Inference | Adaptation |\n"
        "|---|---|---|\n"
        "| Input | one RGB image, sides 16..4,096 px, and a score threshold you choose | drawn training images, each with xyxy pixel boxes, one boolean mask per instance and labels from a three-class sign vocabulary |\n"
        "| Model | Mask R-CNN ResNet-50 FPN v2 with its 91-slot COCO head | the same model with fresh box and mask predictors for the three classes; FPN, RPN and ROI heads trained, ResNet-50 body frozen |\n"
        "| Output | instances (label, softmax score, xyxy box, boolean mask) at or above the threshold, at most `MAX_DETECTIONS` (100) | held-out box and mask AP before and after, instances on unseen images, and a SafeTensors adapter that reloads to the same instances |\n\n"
        "**How to use this notebook.** Choose a runtime (a GPU runtime such as a Colab T4 is much faster; CPU works, slowly — see "
        "Prerequisites), then **Runtime → Run all**. Run all completes in one pass: Section 1 builds a separate locked environment and installs "
        "nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the "
        "carried code (collapsed) and the model verification — and can be run without study. The learning path starts in Section 4. Form "
        "fields (`# @param`) are the only values meant to be edited; the defaults reproduce the recorded path. Every training setting "
        "(`EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `FREEZE_BACKBONE`) sits in the Section 8 cell, and Section 8 always starts from the freshly "
        "re-headed model (`reset_to_pretrained()`), so re-running it after a change compares the new setting against the same starting point. "
        "Before each of Sections 4–11 runs you are asked to **predict** what they will print; the following section opens with "
        "**What to notice** and a collapsible **Check your reasoning** block with a worked answer from a recorded run. Section 14 is a "
        "**change-one-thing activity**; it and each optional experiment name the cell to change and the cell to re-run from (**Runtime → Run "
        "after**). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Your notes are optional and are not "
        "required submissions. Two environment variables change how Section 1 behaves, and neither is needed on Colab or Kaggle: "
        "`DIMER_NOTEBOOK_CI_PREINSTALLED=1` (set by an executor that has already installed exactly these pins) skips the isolated "
        "environment and runs every cell in the kernel, and Section 1 prints that it did so; `DIMER_ISOLATED_ENV` names the folder the "
        "environment is built in (default `dimer_isolated_env`).\n\n"
        "**Roadmap:** 1–3 infrastructure → 4 the pretrained model on a drawn scene *(scores, thresholds, box and mask IoU)* → 5 blank and noise "
        "probes *(evaluation practice)* → 6 build and validate the sign dataset → 7 split, re-head and measure the baseline → 8 bounded "
        "fine-tune → 9 held-out box and mask AP, and what a 1.0 row means → 10 unseen images → 11 export and reload the adapter → 12 outputs → "
        "13 optional BYOD → 14 **change one thing: unfreeze the backbone** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) describe the path *image → ResNet-50 + FPN features → region proposals → per-region class, box "
        "and 28×28 mask* and say why a threshold decides what is returned (Section 4); (2) explain why an instance on a blank or noise image is "
        "a false positive, and how instances that appear only at the evaluation threshold enter average precision (Section 5); (3) explain why "
        "a new vocabulary needs new box and mask predictors and why their baseline is near zero (Sections 6–7); (4) read a training loss as "
        "optimisation evidence and held-out box and mask AP as task evidence (Sections 8–9); (5) identify a saturated metric — a 1.0 row — "
        "and explain why it cannot rank two settings (Section 9); (6) check that an exported adapter reproduces the evaluated model "
        "(Section 11); and (7) predict and then measure what unfreezing the backbone changes when both runs start from the same re-headed "
        "model (Section 14). Along the way the notebook builds a locked runtime, stages and digest-verifies the pinned checkpoint, and writes "
        "machine-readable outputs with provenance."
    ),
    "exclusions": (
        "real-world traffic sign segmentation (the adaptation dataset is drawn in code, so the model learns these renderings and nothing about "
        "road photographs); COCO benchmark results (the average-precision helper here is a compact implementation without pycocotools area "
        "ranges or crowd handling, and it is run on synthetic data only); full-schedule Mask R-CNN training (the v2 weights come from a long "
        "multi-GPU recipe, recorded in torchvision's weight metadata as pytorch/vision pull request 5773; the tutorial runs a few epochs on a "
        "few dozen drawn images); score calibration or a deployment threshold for the adapted model; semantic or panoptic segmentation; "
        "keypoints; video tracking."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and PIL, and Colab or Jupyter familiarity; no prior experience with instance segmentation models or fine-tuning. The notebook explains region proposals, softmax scores after non-maximum suppression, masks, re-heading, AP50 and AP@[.50:.95] where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported runtime (Google Colab, Kaggle or Linux Jupyter — **Linux x86_64 only**; the notebook builds its own isolated Python 3.12.12 environment from manylinux wheels, so a Windows or macOS kernel is not supported). A CUDA GPU such as a Colab or Kaggle T4 is recommended for the fine-tune and is used automatically when present. Measured, each figure with its environment: a Kaggle Tesla T4 run of the earlier in-kernel-install revision (notebook blob `a8f04d11`, 2026-09-25, every field at its default) took 486.2 s in two passes — 377.1 s until the install cell stopped for a restart, then 109.1 s for every cell after the restart, fine-tune included; a local CPU run of this revision's learner cells at a **reduced** configuration (24-thread Windows workstation, 2026-10-06, install skipped, checkpoint pre-staged, `N_IMAGES = 8`, `EPOCHS = 1`) took 54.9 s of cell time, 5.7 s per training step of 2 images (the 2026-10-03 review measured about 30 s per step on the same workstation while it was shared with other jobs). **Estimate, not a measurement:** at that step time the default fine-tune (`EPOCHS = 10` over 30 images, 150 steps) would take about 15 minutes on such a CPU when it is otherwise idle, and over an hour at the slower step time, so use a GPU for the default path. This revision's isolated-environment build has not yet been timed on a hosted runtime; expect it to add several minutes (an estimate). The locked install (PyTorch 2.14.0 with its CUDA libraries) and the ~186 MB checkpoint are the largest downloads.",
        "- **Knowledge:** basic Python, PIL and NumPy; bounding boxes as xyxy pixel coordinates; binary masks as boolean arrays; intersection-over-union (IoU) for boxes and for masks; and how to read average precision (AP50 and AP@[.50:.95]).",
        "- **Data:** the default path generates everything in code with `samples.py` and downloads no dataset: one 640×480 COCO demonstration scene and a labelled sign dataset (`N_IMAGES`, default 40 images), each object with an exact mask. BYOD is optional and off by default. Expected BYOD input: one image, or a directory holding `annotations.json` — a list of `{'file': 'name.png', 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...], 'masks': ['name_mask0.png', ...]}` objects with boxes in **pixel** coordinates — and the image and mask files it names (one single-channel PNG per instance, non-zero pixels inside the instance). Through the upload dialog, upload `annotations.json`, the images and the masks as **flat files (no folders)**.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API. The default path uploads nothing.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the kernel's own "
        "packages are left alone, so no restart is needed and Run all completes in one pass), stages and digest-verifies the pinned checkpoint, "
        "runs COCO instance segmentation on a drawn scene, validates the sign dataset, splits it into training and held-out parts, measures the "
        "pre-adaptation baseline, **runs the bounded fine-tune**, re-evaluates on the held-out split, segments unseen images, exports the "
        "adapter, reloads it onto a fresh base model to verify the instances, and writes machine-readable outputs with provenance. Nothing is "
        "skipped behind a default-off flag, and no clone, DIMER worker, credential, upload dialog or configuration edit is required "
        "(NOTEBOOK_SPEC 2.2 §5, RUN7, FT2). CUDA is used when present; a GPU runtime is recommended. Measured times are listed under "
        "Prerequisites."
    ),
    "byod": (
        "Two optional BYOD branches are included, and both are off by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image through the same validation, segmentation and evaluation-report stages as the sample scene and "
        "writes its input manifest. `USE_BYOD_DATASET` takes your own labelled instance masks through the full adaptation workflow — validate, "
        "split, baseline, fine-tune, evaluate, export, reload and compare — under NOTEBOOK_SPEC 2.2 DAT14, and writes its own result JSON and "
        "detections CSV. Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_DIR` to read from a location without an upload dialog (EXE2)."
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
                "the verdict is `sample-sanity`.\n\n"
                "**Predict before running:** all four drawn objects are COCO classes. How many will the model find at 0.75 with mask IoU ≥ 0.5, "
                "and which one do you expect a model trained on photographs to miss in a flat drawing?"
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
                "**What to notice (Section 4):** the matched count, each instance's score, and the `box_iou` and `mask_iou` of each reference.\n\n"
                "<details><summary>Check your reasoning</summary>In the recorded runs (Kaggle T4, 2026-09-25, and a local CPU run, 2026-10-03, "
                "identical to the printed precision) the model returned `stop sign` 0.9995, `clock` 0.9969 and `traffic light` 0.9953, each with "
                "mask IoU above 0.97 against the drawn mask (stop sign `box_iou` 0.944 / `mask_iou` 0.984), so 3 of the 4 drawn objects matched. "
                "The **sports ball was not detected** and is recorded with IoU 0.0. A flat orange disc carries little of what a photograph of a "
                "ball does (shading, seams, context), while an octagon with a word on it, a box of coloured lamps and a clock face look much like "
                "their photographs. A miss on a drawing is a finding about renderings, recorded, not hidden.</details>\n\n"
                "## 5. Degenerate input probes: blank canvas and noise\n\n"
                "Ask a segmentation model about structure-free input before trusting it. A model that returns confident instances on a blank canvas "
                "or on uniform noise will return them on empty real frames too. The cell counts instances on both images at the default threshold "
                "and at the evaluation threshold `EVAL_DETECTION_THRESHOLD` (0.05) that average precision is computed at.\n\n"
                "**What to look for:** any instance above the default threshold on these two images is a false positive by construction. "
                "**Instances that appear only at 0.05 matter too.** Average precision (Section 9) ranks *every* instance scoring at least 0.05: a "
                "false positive ranked below every correct instance costs AP nothing, but one ranked above a correct instance lowers precision at "
                "that point of the ranking and so lowers AP. Low-scoring instances on nothing are harmless at 0.75 and are exactly what AP sees.\n\n"
                "**Predict before running:** will either image return an instance at 0.75? Will noise return any at 0.05, and would they cover a "
                "small patch or most of the frame?"
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
                "**What to notice (Section 5):** the counts at 0.75 against the counts at 0.05, and the mask areas in `top`.\n\n"
                "<details><summary>Check your reasoning</summary>The blank canvas returned nothing at either threshold. Noise returned nothing at "
                "0.75 but **2 instances at 0.05** — `umbrella` 0.26 and `bear` 0.13, each covering most of the 640×480 frame (Kaggle T4, "
                "2026-09-25, and a local CPU run alike). Neither would survive the default threshold, so the default output on noise is empty. "
                "On a labelled set, instances like these enter the AP ranking; scored at 0.13–0.26 they would sit below confident correct "
                "instances and cost little, but a model that scores clutter as high as its true objects loses AP for it.</details>\n\n"
                "## 6. Labelled adaptation dataset and validation\n\n"
                "Suppose your task needs sign classes that COCO does not have. `sign_dataset` draws a deterministic dataset of `N_IMAGES` images "
                "(default 40) over `SIGN_CLASSES`: `stop-sign`, `yield-sign` and `speed-limit-sign`. Every sign is drawn twice, once onto the image "
                "and once onto its own mask, so each reference mask is exact and each reference box is the tight box of its mask.\n\n"
                "**Keep the two vocabularies apart.** COCO has `stop sign` (with a space). The adaptation vocabulary uses `stop-sign` (hyphenated), "
                "`yield-sign` and `speed-limit-sign`. They are different class identities, and the adapted model answers in the new names only.\n\n"
                "`validate_dataset` checks every record before any model runs: the record keys, the image size ceilings, that every box is finite, "
                "non-empty and inside its image, that every mask has the image's shape, is boolean, is not empty and lies inside its box, and that "
                "every label is in the vocabulary. It returns a dataset manifest with the instance count per class and a finding for any class "
                "that has no instance. (The manifest also records the number of epochs; Section 8, where `EPOCHS` is set, validates again and "
                "keeps that manifest.)\n\n"
                "**Predict before running:** with the default `N_IMAGES = 40`, roughly how many instances will the dataset hold, and will the "
                "three classes be balanced?"
            ),
            "code": (
                'N_IMAGES = 40  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n\n'
                "records = sign_dataset(N_IMAGES, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SIGN_CLASSES)\n"
                "print(json.dumps({{k: v for k, v in dataset_manifest.items() if k not in ('schema', 'epochs')}}, indent=2))\n\n"
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
                "**What to notice (Section 6):** `n_records`, `n_instances` and `instances_per_class`, and in the preview each image above its mask.\n\n"
                "<details><summary>Check your reasoning</summary>At the defaults the recorded run validated 40 records with 68 instances: "
                "`stop-sign` 20, `yield-sign` 24, `speed-limit-sign` 24 — one to three signs per image, close to balanced. The manifest is built "
                "before any model runs, so a malformed record is refused before training, not discovered after it.</details>\n\n"
                "## 7. Split, re-head, and measure the pre-adaptation baseline\n\n"
                "The dataset is split at random into a training part (`1 - HOLDOUT`, default 75%) and a held-out part (`HOLDOUT`, default 25%); with "
                "the default 40 images that is 30 and 10. A random split is valid here because every image is drawn independently; records that "
                "share a photograph or a camera session must be split by that group instead. The held-out images are never shown to the "
                "optimizer, and no hyperparameter is selected on them.\n\n"
                "`from_pretrained(class_names=SIGN_CLASSES)` loads the verified checkpoint and replaces only the two class-specific predictors with "
                "freshly initialised layers, seeded by `SEED`: the box predictor (`roi_heads.box_predictor`, now 3 classes plus background) and the "
                "mask predictor (`roi_heads.mask_predictor`, one 28×28 mask channel per class). This is **re-heading**. The backbone, the FPN, the "
                "RPN and the shared ROI layers keep their COCO weights. The cell wraps that call in `reset_to_pretrained()`, which Section 8 uses "
                "to start every fine-tune from this same re-headed model.\n\n"
                "**The baseline is expected to be near zero.** Randomly initialised predictors have no information about the new classes, so "
                "this number is the floor the fine-tune has to beat, not a property of Mask R-CNN.\n\n"
                "**Predict before running:** the COCO model knew `stop sign`. Will the baseline score anything on `stop-sign`, and on the two "
                "new shapes?"
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
                "def reset_to_pretrained():\n"
                "    \"\"\"A fresh re-headed model: the verified checkpoint with new box and mask predictors initialised under SEED.\"\"\"\n"
                "    return MaskRcnnPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n\n"
                "adapter = reset_to_pretrained()\n"
                "print({{'class_names': list(adapter.class_names), 'device': adapter.device, 'reinitialised': list(adapter.reinitialised)}})\n\n"
                "baseline = adapter.evaluate(held_out)\n"
                "print(json.dumps({{key: round(baseline[key], 4) for key in ('ap', 'ap50', 'mask_ap', 'mask_ap50')}}\n"
                "                 | {{'per_class_ap50': {{k: round(v, 4) for k, v in baseline['per_class_ap50'].items()}}, 'n_references': baseline['n_references']}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 8. Bounded Fine-Tuning
        {
            "md": (
                "**What to notice (Section 7):** the split sizes, the two re-initialised predictors, and the baseline per class.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run split 30 / 10 images (51 / 17 instances) and measured a baseline "
                "box AP50 of 0.1782 and mask AP of 0.0: AP50 0.5347 on `stop-sign` and 0.0 on the two new shapes. The predictors are random, so "
                "whatever the baseline finds is chance passed through the shared COCO layers (why it is `stop-sign` that scores is a guess: the "
                "COCO model knew stop signs); no mask matched at all.Near zero is the expected floor.</details>\n\n"
                "## 8. Bounded fine-tuning\n\n"
                "This cell runs the real adaptation step in this runtime. It is gradient fine-tuning with Mask R-CNN's own training losses, which "
                "torchvision returns when the model is called with targets: RPN objectness and RPN box regression, ROI classification and ROI box "
                "regression, and a per-pixel binary cross-entropy on the 28×28 mask of each positive region. The optimised loss is their sum.\n\n"
                "- **Every training setting is in this cell:** `EPOCHS` (default 10), `LEARNING_RATE` (0.005, the value of torchvision's "
                "detection fine-tuning tutorial, with SGD momentum 0.9 and weight decay 0.0005), `BATCH_SIZE` (2) and `FREEZE_BACKBONE`; float32, "
                "seed `SEED`. The cell re-validates the dataset with `EPOCHS`, so the manifest records the schedule actually run.\n"
                "- **The ResNet-50 body is frozen.** With the default `FREEZE_BACKBONE = True`, parameters under `backbone.body.` keep their COCO values, and its "
                "BatchNorm layers are held in evaluation mode so their running statistics do not drift either. The FPN, the RPN and the ROI heads "
                "are trained. The cell prints the trainable and total parameter counts. (Unfreezing it is the Section 14 activity.)\n"
                "- **Adaptation always starts from the re-headed model.** `finetune` changes the model in place, and it refuses a model that is "
                "already adapted. If you re-run this cell after a fine-tune, it first calls `reset_to_pretrained()` — the same fresh predictors "
                "that Section 7's baseline measured — so a changed setting is compared from the same starting point, and the printed run record "
                "describes exactly the schedule that produced the model.\n\n"
                "**Read the loss as optimisation evidence only.** A falling loss says the optimizer is fitting the training images; the held-out "
                "average precision in the next section is the task evidence.\n\n"
                "**Predict before running:** will the first epoch's loss be above or below 1.0, and by what factor will it fall over the default "
                "10 epochs?"
            ),
            "code": (
                "import time\n\n"
                'EPOCHS = 10  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 0.005  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 2  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = True  # @param {{type:"boolean"}}\n\n'
                "dataset_manifest = validate_dataset(records, SIGN_CLASSES, epochs=EPOCHS)\n"
                "if adapter.adapted:\n"
                "    # A re-run: start again from the re-headed base (same SEED, same predictors as the Section 7 baseline).\n"
                "    adapter = reset_to_pretrained()\n"
                "    print({{'rebuilt_from_verified_snapshot': True, 'adapted': adapter.adapted}})\n"
                "train_started = time.perf_counter()\n"
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_backbone=FREEZE_BACKBONE,\n"
                "    progress=lambda row: print(f\"epoch {{row['epoch']}}/{{row['epochs']}}  loss {{row['loss']:.4f}}\"),\n"
                ")\n"
                "train_seconds = round(time.perf_counter() - train_started, 1)\n"
                "print(json.dumps({{**{{key: run[key] for key in ('freeze_backbone', 'trainable_parameters', 'total_parameters', 'epochs',\n"
                "                                             'batch_size', 'learning_rate', 'optimizer', 'precision', 'device')}},\n"
                "                  'started_from': 'the re-headed checkpoint (fresh predictors, seed ' + str(SEED) + ')',\n"
                "                  'first_epoch_loss': round(run['epoch_losses'][0], 4), 'final_loss': round(run['final_loss'], 4),\n"
                "                  'train_seconds': train_seconds}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate Held-Out
        {
            "md": (
                "**What to notice (Section 8):** the epoch-1 loss, how fast it falls, and the trainable share of the parameters.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run (Kaggle T4, defaults) started at 0.9822 — just below 1.0 — and "
                "ended at 0.0863 after 10 epochs (0.9822, 0.3973, 0.2616, 0.1543, 0.1247, 0.1093, 0.1026, 0.0903, 0.0854, 0.0863): about a "
                "tenfold fall, levelling off after epoch 7. With the ResNet-50 body frozen, 22,383,143 of 45,891,175 parameters were trainable. "
                "A loss this low says the training images are fitted; it does not yet say anything about images the optimizer never saw.</details>\n\n"
                "## 9. Evaluate on the held-out split, and what a 1.0 row means\n\n"
                "`evaluate` re-runs on the same held-out images with the same thresholds as the baseline, so the two rows are comparable. It scores "
                "every instance twice: once by box IoU (`ap`, `ap50`, `ap75`) and once by mask IoU (`mask_ap`, `mask_ap50`, `mask_ap75`). `ap50` is "
                "average precision at IoU 0.50; `ap` averages AP over the ten IoU thresholds 0.50–0.95, so it also rewards tight boxes and masks. A "
                "model can find every sign with a good box and still draw a poor mask, which only the mask rows show. AP ranks every instance "
                "scoring at least `EVAL_DETECTION_THRESHOLD` (0.05), so it measures ranking, not what a 0.75 threshold returns. These are tutorial "
                "metrics from one pass over the held-out images (10 at the defaults), with no dispersion estimate.\n\n"
                "**A 1.0 row is a ceiling.** AP cannot exceed 1.0: a row reading 1.0000 means every held-out instance was found, at every IoU "
                "threshold the row averages, and ranked above every false positive. On this drawn task the adapted model may reach that ceiling — "
                "the recorded run did on every mask row. A saturated row cannot show a further gain, so it cannot rank two settings: two runs that "
                "both read 1.0 may still differ in how much margin they have. It also says the task is easy for this model at this size, not that "
                "the model segments real signs perfectly. To compare settings, read the rows that are still below 1.0. The run-history table at "
                "the end of the cell keeps one row per evaluation, so a Section 14 re-run prints side by side with the default run.\n\n"
                "**Predict before running:** which rows will reach 1.0 after the default fine-tune, and which will stay below it? Why might box "
                "`ap` stay below mask `ap` on these drawn signs?"
            ),
            "code": (
                "adapted = adapter.evaluate(held_out)\n"
                "print(f\"{{'metric':<10s}} {{'baseline':>10s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in ('ap', 'ap50', 'ap75', 'mask_ap', 'mask_ap50', 'mask_ap75'):\n"
                "    print(f\"{{key:<10s}} {{baseline[key]:>10.4f}} {{adapted[key]:>10.4f}} {{adapted[key] - baseline[key]:>+10.4f}}\")\n"
                "print('per-class mask AP50:', {{k: round(v, 4) for k, v in adapted['per_class_mask_ap50'].items()}})\n"
                "saturated = [key for key in ('ap', 'ap50', 'ap75', 'mask_ap', 'mask_ap50', 'mask_ap75') if adapted[key] >= 1.0 - 1e-9]\n"
                "headroom = [key for key in ('ap', 'ap50', 'ap75', 'mask_ap', 'mask_ap50', 'mask_ap75') if key not in saturated]\n"
                "print({{'at_ceiling_1.0': saturated, 'below_ceiling': headroom, 'evaluation_threshold': EVAL_DETECTION_THRESHOLD}})\n\n"
                "RUN_HISTORY = globals().get('RUN_HISTORY') or []\n"
                "RUN_HISTORY.append({{'freeze_backbone': run['freeze_backbone'], 'epochs': run['epochs'], 'learning_rate': run['learning_rate'],\n"
                "                    'first_epoch_loss': round(run['epoch_losses'][0], 4), 'final_loss': round(run['final_loss'], 4),\n"
                "                    'ap': round(adapted['ap'], 4), 'ap75': round(adapted['ap75'], 4), 'mask_ap': round(adapted['mask_ap'], 4),\n"
                "                    'mask_ap75': round(adapted['mask_ap75'], 4), 'train_seconds': train_seconds}})\n"
                "print('run history (one row per evaluation; row 0 is the first run of this session):')\n"
                "for index, row in enumerate(RUN_HISTORY):\n"
                "    print(index, row)"
            ),
        },
        # ---------------------------------------------------------------- 10. New-data inference
        {
            "md": (
                "**What to notice (Section 9):** which rows reached 1.0, which did not, and the `below_ceiling` list.\n\n"
                "<details><summary>Check your reasoning</summary>In the recorded run (Kaggle T4, defaults) every held-out row moved from near zero "
                "to the ceiling except one: `ap50` 0.1782 → 1.0, `ap75` 0.0 → 1.0, `mask_ap`, `mask_ap50` and `mask_ap75` 0.0 → 1.0, and every "
                "per-class mask AP50 1.0; only box `ap` stayed below it, 0.0347 → 0.8693. Since box `ap50` and `ap75` are 1.0, the shortfall is at "
                "the strictest box IoU thresholds (above 0.75): some boxes are not tight to the last few pixels, while the masks are. Read the "
                "saturated rows as *the drawn task is solved at this size*, not as *perfect segmentation*: ten held-out images drawn by the same "
                "generator as the training images cannot show how far the model is from failing, nor anything about photographs. That is why "
                "Section 14 compares settings on box `ap`, the run history's one column with headroom here.</details>\n\n"
                "## 10. Inference on unseen images\n\n"
                "Three new images come from a seed the dataset never used (`NEW_DATA_SEED`, default 99). The adapted pipeline runs at the same "
                "`threshold` as the COCO model (0.75), and each reference instance is compared with the best same-label instance by box IoU and "
                "by mask IoU.\n\n"
                "**Predict before running:** how many of the reference signs will be found at 0.75, and will the mask IoUs be higher or lower than "
                "the box IoUs?"
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
                "    print(json.dumps(row))\n"
                "n_new = sum(len(row['truth']) for row in new_data_rows)\n"
                "found = sum(1 for row in new_data_rows for value in row['same_label_mask_iou'] if value >= 0.5)\n"
                "print(f'{{found}}/{{n_new}} reference instances found at threshold {{threshold}} (same label, mask IoU >= 0.5)')"
            ),
        },
        # ---------------------------------------------------------------- 11. Export, reload & verify
        {
            "md": (
                "**What to notice (Section 10):** the found count, each row's `detections`, and box IoU against mask IoU.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run (Kaggle T4, defaults) returned 6 instances for the 6 reference "
                "signs on the three unseen images, every label correct, with same-label `box_iou` 0.926–0.944 and `mask_iou` 0.964–0.980: the "
                "masks fit more tightly than the boxes, as the held-out `ap` against `mask_ap` suggested. These images come from the same "
                "drawing code as the training set, so this is a check that the adapted model generalises to new *drawings*, not to photographs.</details>\n\n"
                "## 11. Adapter export, fresh reload, and equivalence check\n\n"
                "`save_artifact` writes `outputs/maskrcnn_adapter.safetensors`: every tensor the fine-tune could change, plus a metadata header naming "
                "the base model, its pinned checkpoint SHA-256, the class names and the frozen prefixes. The frozen ResNet-50 body is left out because "
                "it equals the verified base checkpoint.\n\n"
                "`load_artifact` then builds a **fresh** pipeline from the verified base checkpoint, loads the adapter tensors onto it, and refuses "
                "an adapter whose format, base identity, base digest or tensor set does not fit. The cell compares the reloaded instances with the "
                "in-memory model's on an unseen image, with a stated tolerance: loading succeeding is not the check, reproducing the boxes, scores "
                "and masks is. The comparison runs at the evaluation threshold 0.05; if nothing passes it, it compares the 10 top-scoring "
                "instances, and if the model returns no instance at all it compares every weight tensor, so the check is never empty. The same `reload_equivalence` helper checks the BYOD adapter in Section 13.\n\n"
                "**Predict before running:** why compare instances after the reload instead of only checking that the file loads?"
            ),
            "code": (
                "artifact_path = OUTPUTS / 'maskrcnn_adapter.safetensors'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='Mask R-CNN ResNet-50 FPN v2 sign adaptation tutorial adapter')\n"
                "print(json.dumps(descriptor, indent=2))\n\n"
                "reloaded = MaskRcnnPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'class_names': list(reloaded.class_names)}})\n\n"
                "TOLERANCE = 1e-3\n"
                "MASK_AGREEMENT = 0.99\n\n"
                "def reload_equivalence(first, second, image, tolerance=TOLERANCE):\n"
                "    \"\"\"Both pipelines must return the same instances on `image` (label, box and score within `tolerance`, masks at IoU >= MASK_AGREEMENT).\"\"\"\n"
                "    at, limit = EVAL_DETECTION_THRESHOLD, None\n"
                "    if not first.detect(image, threshold=at)['detections']:  # nothing passes 0.05 (e.g. a short BYOD run)\n"
                "        at, limit = 0.0, 10  # compare the 10 top-scoring instances instead, so the check is never empty\n"
                "    out_orig = first.detect(image, threshold=at)\n"
                "    out_reloaded = second.detect(image, threshold=at)\n"
                "    pairs = list(zip(out_orig['detections'], out_reloaded['detections'], out_orig['masks'], out_reloaded['masks']))[:limit]\n"
                "    if len(out_orig['detections'][:limit]) != len(out_reloaded['detections'][:limit]):\n"
                "        raise RuntimeError(f\"the reloaded adapter returned {{len(out_reloaded['detections'])}} instances, the in-memory model {{len(out_orig['detections'])}}: the export is not faithful\")\n"
                "    for d1, d2, m1, m2 in pairs:\n"
                "        if d1['label'] != d2['label'] or not np.allclose(d1['box'], d2['box'], atol=tolerance) or abs(d1['score'] - d2['score']) > tolerance:\n"
                "            raise RuntimeError(f'reloaded instance {{d2}} differs from {{d1}} beyond tolerance {{tolerance}}: the export is not faithful')\n"
                "        if (m1.any() or m2.any()) and mask_iou(m1, m2) < MASK_AGREEMENT:\n"
                "            raise RuntimeError(f\"reloaded mask of {{d1['label']}} agrees with the original at IoU {{mask_iou(m1, m2):.4f}} < {{MASK_AGREEMENT}}\")\n"
                "    if not pairs:  # no instance at all, even unthresholded (e.g. a barely trained BYOD model): compare the weights instead\n"
                "        state_orig, state_reloaded = first.model.state_dict(), second.model.state_dict()\n"
                "        differing = [k for k in state_orig if not torch.allclose(state_orig[k].float().cpu(), state_reloaded[k].float().cpu(), atol=tolerance)]\n"
                "        if differing:\n"
                "            raise RuntimeError(f'the reloaded adapter differs from the in-memory model in {{len(differing)}} tensors, e.g. {{differing[:3]}}: the export is not faithful')\n"
                "        return {{'instances_compared': 0, 'tensors_compared': len(state_orig), 'tolerance': tolerance, 'equivalent': True}}\n"
                "    return {{'instances_compared': len(pairs), 'threshold': at, 'tolerance': tolerance, 'mask_iou_at_least': MASK_AGREEMENT, 'equivalent': True}}\n\n"
                "reload_check = reload_equivalence(adapter, reloaded, new_records[0]['image'])\n"
                "print(reload_check)"
            ),
        },
        # ---------------------------------------------------------------- 12. Outputs & provenance
        {
            "md": (
                "**What to notice (Section 11):** `instances_compared` (never 0) and `equivalent`.\n\n"
                "<details><summary>Check your reasoning</summary>A file can load and still hold the wrong weights — an earlier adapter, a different "
                "base checkpoint, a tensor saved before training finished. Comparing what the reloaded model *does* catches all of those. The "
                "recorded run wrote an 89,577,652-byte adapter of 114 tensors and found the instances equivalent at tolerance 0.001 and mask IoU ≥ "
                "0.99; any difference stops the notebook, because it is a contract failure, not a quality result.</details>\n\n"
                "## 12. Write machine-readable outputs and provenance\n\n"
                "The cell writes:\n"
                "- `outputs/maskrcnn_instance_segmentation_input_manifest.json`\n"
                "- `outputs/maskrcnn_instance_segmentation_evaluation_report.json`\n"
                "- `outputs/maskrcnn_instance_segmentation_result.json` (identity, runtime versions, device, dataset manifest, split, baseline and adapted metrics, "
                "the saturated and below-ceiling rows, the run history, fine-tuning configuration with its training time, new-data rows, adapter "
                "descriptor and reload check)\n"
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
                "    'evaluation_threshold': EVAL_DETECTION_THRESHOLD,\n"
                "    'coco_detections': coco_result['detections'],\n"
                "    'degenerate_probes': degenerate,\n"
                "    'adaptation': {{\n"
                "        'dataset': {{k: v for k, v in dataset_manifest.items() if k != 'schema'}},\n"
                "        'dataset_seed': DATASET_SEED,\n"
                "        'split': {{'train': len(train_records), 'held_out': len(held_out), 'seed': SEED, 'holdout': HOLDOUT}},\n"
                "        'finetune': {{**run, 'train_seconds': train_seconds}},\n"
                "        'baseline': baseline,\n"
                "        'adapted': adapted,\n"
                "        'at_ceiling': saturated,\n"
                "        'below_ceiling': headroom,\n"
                "        'run_history': RUN_HISTORY,\n"
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
                "- **Image branch** (`USE_BYOD_IMAGE`): one image file Pillow can open, each side between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` "
                "(4096) px. It runs through `validate_inputs`, `detect` and `evaluation_report` — the report is `not-measurable`, because no "
                "reference instances come with it — and writes `outputs/byod_maskrcnn_input_manifest.json`. Upload exactly one image; every "
                "upload replaces the previous one.\n"
                "- **Dataset branch** (`USE_BYOD_DATASET`): a directory with `annotations.json` (a list of `{{'file', 'boxes', 'labels', 'masks'}}` "
                "objects: **xyxy pixel boxes** — not coordinates normalised to 0..1 — and one mask PNG name per instance) and the image and mask "
                "files it names, at least 2 records and at most 5,000. File names must stay inside the directory. **Through the upload dialog, "
                "upload `annotations.json`, the images and the masks as flat files (no folders).** `BYOD_CLASS_NAMES` is a comma-separated "
                "vocabulary; leave it empty to use the sorted set of labels in the annotations. The branch runs the same validate → split → "
                "baseline → fine-tune → evaluate → export → reload stages as the sample with the Section 7 and 8 settings, compares the reloaded "
                "adapter's instances with the in-memory model's, and writes `outputs/byod_maskrcnn_result.json` (annotation digest, dataset "
                "manifest, split, fine-tune configuration, baseline and adapted metrics, adapter descriptor and reload check) and "
                "`outputs/byod_maskrcnn_detections.csv` (held-out instances at the evaluation threshold).\n\n"
                "Set `BYOD_IMAGE_PATH` (a file) or `BYOD_DATASET_DIR` (a directory) to read from a mounted or local location; leave them empty on "
                "Colab to get an upload dialog instead. Uploaded files are written under `outputs/byod/` in this runtime and are not sent anywhere "
                "else. **Every refusal names the file or the rule and what to do:** a path that is not a file or directory, a runtime without an "
                "upload dialog, a cancelled upload, a file Pillow cannot read, a mask file named in `annotations.json` but missing, and boxes "
                "that look normalised. The first lines of the cell show three of those refusals on purpose. After changing a field here, re-run "
                "this cell only."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'BYOD_IMAGE_PATH = ""  # @param {{type:"string"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_DATASET_DIR = ""  # @param {{type:"string"}}\n'
                'BYOD_CLASS_NAMES = ""  # @param {{type:"string"}}\n\n'
                "import shutil\n\n"
                "BYOD_DIR = OUTPUTS / 'byod'\n\n"
                "def _open_byod_image(path):\n"
                "    \"\"\"Open one BYOD image fully, or raise a ValueError that names the file and the fix.\"\"\"\n"
                "    try:\n"
                "        with Image.open(path) as handle:\n"
                "            return handle.convert('RGB')\n"
                "    except (OSError, SyntaxError, ValueError) as exc:  # PIL.UnidentifiedImageError is an OSError\n"
                "        raise ValueError(f'{{Path(path).name}} is not an image Pillow can read ({{type(exc).__name__}}): use a PNG or JPEG file, each side {{MIN_IMAGE_SIDE}}..{{MAX_IMAGE_SIDE}} px') from exc\n\n"
                "def _existing(path_text, field, kind):\n"
                "    \"\"\"The path in a location field, checked before anything is opened.\"\"\"\n"
                "    path = Path(path_text.strip()).expanduser()\n"
                "    if not (path.is_dir() if kind == 'directory' else path.is_file()):\n"
                "        expected = 'the directory holding annotations.json' if kind == 'directory' else 'one image file'\n"
                "        raise ValueError(f'{{field}} {{str(path)!r}} is not a {{kind}}: set it to the path of {{expected}} in this runtime')\n"
                "    return path\n\n"
                "def _upload_into(target, field):\n"
                "    \"\"\"Colab upload dialog into a fresh `target`: an earlier upload never mixes with this one.\"\"\"\n"
                "    try:\n"
                "        from google.colab import files  # type: ignore[import-not-found]\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'{{field}} is empty and this runtime has no Colab upload dialog: set {{field}} to a path in this runtime') from None\n"
                "    shutil.rmtree(target, ignore_errors=True)\n"
                "    target.mkdir(parents=True)\n"
                "    uploaded = files.upload() or {{}}\n"
                "    if not uploaded:\n"
                "        raise ValueError(f'the upload was cancelled or empty: run this cell again and choose the files, or set {{field}}')\n"
                "    for name, data in uploaded.items():\n"
                "        (target / Path(name).name).write_bytes(data)\n"
                "    return target\n\n"
                "_probe_dir = OUTPUTS / 'byod_refusal_probes'\n"
                "_probe_dir.mkdir(parents=True, exist_ok=True)\n"
                "(_probe_dir / 'notes.png').write_text('this is text, not an image', encoding='utf-8')\n"
                "_probe_image = blank_scene()\n"
                "_probe_mask = np.zeros((_probe_image.height, _probe_image.width), dtype=bool)\n"
                "_probe_mask[40:80, 40:80] = True\n"
                "for desc, probe in (\n"
                "    ('a file that is not an image', lambda: _open_byod_image(_probe_dir / 'notes.png')),\n"
                "    ('an empty mask', lambda: validate_dataset([{{'image': _probe_image, 'boxes': [[0, 0, 10, 10]], 'labels': [SIGN_CLASSES[0]],\n"
                "                                               'masks': [np.zeros_like(_probe_mask)]}}], SIGN_CLASSES)),\n"
                "    ('a normalised box', lambda: validate_dataset([{{'image': _probe_image, 'boxes': [[0.06, 0.08, 0.13, 0.17]], 'labels': [SIGN_CLASSES[0]],\n"
                "                                                  'masks': [_probe_mask]}}], SIGN_CLASSES)),\n"
                "):\n"
                "    try:\n"
                "        probe()\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print(f'refused as expected: {{desc}} -> {{type(exc).__name__}}: {{exc}}')\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    if BYOD_IMAGE_PATH.strip():\n"
                "        image_path = _existing(BYOD_IMAGE_PATH, 'BYOD_IMAGE_PATH', 'file')\n"
                "    else:\n"
                "        uploaded_images = sorted(p for p in _upload_into(BYOD_DIR / 'image', 'BYOD_IMAGE_PATH').iterdir() if p.is_file())\n"
                "        if len(uploaded_images) != 1:\n"
                "            raise ValueError(f'upload exactly one image for the image branch, got {{len(uploaded_images)}}: run this cell again')\n"
                "        image_path = uploaded_images[0]\n"
                "    byod_image = _open_byod_image(image_path)\n"
                "    byod_input_manifest = validate_inputs(byod_image, threshold=threshold, names=[image_path.name])\n"
                "    print(byod_input_manifest['verdict'])\n"
                "    byod_result = pipe.detect(byod_image, threshold=threshold)\n"
                "    for det in byod_result['detections'][:20]:\n"
                "        print(f\"{{det['label']:>16s}} {{det['score']:.3f}}  mask {{det['mask_area']:>7d}} px  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n"
                "    byod_input_manifest['n_detections'] = len(byod_result['detections'])\n"
                "    byod_input_manifest['report_verdict'] = evaluation_report(byod_result, None, sample_kind='byod')['verdict']\n"
                "    byod_input_manifest['sha256'] = hashlib.sha256(image_path.read_bytes()).hexdigest()\n"
                "    with open(OUTPUTS / 'byod_maskrcnn_input_manifest.json', 'w', encoding='utf-8') as f:\n"
                "        json.dump(byod_input_manifest, f, indent=2, default=str)\n"
                "    print(byod_input_manifest['report_verdict'], '-> wrote byod_maskrcnn_input_manifest.json')\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True to run segmentation on your own image.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    if BYOD_DATASET_DIR.strip():\n"
                "        dataset_dir = _existing(BYOD_DATASET_DIR, 'BYOD_DATASET_DIR', 'directory')\n"
                "    else:\n"
                "        dataset_dir = _upload_into(BYOD_DIR / 'dataset', 'BYOD_DATASET_DIR')\n"
                "    byod_records = read_detection_records(dataset_dir)\n"
                "    byod_names = [n.strip() for n in BYOD_CLASS_NAMES.split(',') if n.strip()] or sorted({{l for r in byod_records for l in r['labels']}})\n"
                "    byod_manifest = validate_dataset(byod_records, byod_names, epochs=EPOCHS)\n"
                "    print(json.dumps({{k: v for k, v in byod_manifest.items() if k != 'schema'}}, indent=2))\n"
                "    byod_train, byod_held = split_dataset(byod_records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "    byod_pipe = MaskRcnnPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_names, seed=SEED)\n"
                "    byod_baseline = byod_pipe.evaluate(byod_held)\n"
                "    byod_run = byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_backbone=FREEZE_BACKBONE)\n"
                "    byod_adapted = byod_pipe.evaluate(byod_held)\n"
                "    print({{key: (round(byod_baseline[key], 4), round(byod_adapted[key], 4)) for key in ('ap', 'ap50', 'mask_ap', 'mask_ap50')}})\n"
                "    byod_artifact = OUTPUTS / 'byod_maskrcnn_adapter.safetensors'\n"
                "    byod_descriptor = byod_pipe.save_artifact(byod_artifact, notes='BYOD adaptation adapter')\n"
                "    byod_reloaded = MaskRcnnPipeline.load_artifact(byod_artifact, weights_dir=WEIGHTS_DIR)\n"
                "    byod_reload_check = reload_equivalence(byod_pipe, byod_reloaded, byod_held[0]['image'])\n"
                "    print('BYOD adapter exported, reloaded and compared:', byod_descriptor['sha256'][:16], byod_reload_check)\n"
                "    with open(OUTPUTS / 'byod_maskrcnn_detections.csv', 'w', newline='', encoding='utf-8') as f:\n"
                "        writer = csv.writer(f)\n"
                "        writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1', 'mask_area'])\n"
                "        for record in byod_held:\n"
                "            for rank, d in enumerate(byod_pipe.detect(record['image'], threshold=EVAL_DETECTION_THRESHOLD)['detections']):\n"
                "                writer.writerow([record['id'], rank, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box']), d['mask_area']])\n"
                "    byod_export = {{\n"
                "        'notebook_source': NOTEBOOK_SOURCE, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,\n"
                "        'annotations_sha256': hashlib.sha256((dataset_dir / 'annotations.json').read_bytes()).hexdigest(),\n"
                "        'dataset': {{k: v for k, v in byod_manifest.items() if k != 'schema'}},\n"
                "        'split': {{'train': [r['id'] for r in byod_train], 'held_out': [r['id'] for r in byod_held], 'seed': SEED, 'holdout': HOLDOUT}},\n"
                "        'finetune': byod_run, 'baseline': byod_baseline, 'adapted': byod_adapted,\n"
                "        'evaluation_threshold': EVAL_DETECTION_THRESHOLD, 'artifact': byod_descriptor, 'reload_check': byod_reload_check,\n"
                "    }}\n"
                "    with open(OUTPUTS / 'byod_maskrcnn_result.json', 'w', encoding='utf-8') as f:\n"
                "        json.dump(byod_export, f, indent=2, default=str)\n"
                "    print('wrote', [p.name for p in sorted(OUTPUTS.glob('byod_maskrcnn_*'))])\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt Mask R-CNN on your own labelled images.')"
            ),
        },
    ],
    "closing": (
        "## 14. Your turn — change one thing: unfreeze the backbone\n\n"
        "Optional; **Predict → Change → Run → Observe → Explain**. It re-runs the fine-tune with the ResNet-50 body trainable, which takes "
        "longer than Section 8 did (about twice as many parameters get gradients). Compare on box `ap` (AP@[.50:.95]), the run history's "
        "column that was still below 1.0 in the recorded run, and on the losses and `train_seconds`; a row that reads 1.0 in both runs "
        "(Section 9) cannot tell them apart.\n\n"
        "1. **Predict:** with `FREEZE_BACKBONE = False`, will held-out box `ap` go up or down against the default run, and will training take "
        "longer? Write your guess next to row 0 of the Section 9 run history.\n"
        "2. **Change:** in Section 8 set `FREEZE_BACKBONE = False`. Change nothing else.\n"
        "3. **Run:** select the Section 8 cell and choose **Runtime → Run after**. Section 8 calls `reset_to_pretrained()` before training (it "
        "prints `rebuilt_from_verified_snapshot`), Section 9 adds a row to the run history, and Sections 10–12 re-run on the new adapter.\n"
        "4. **Observe:** Section 8 prints `rebuilt_from_verified_snapshot`, a trainable count of every parameter, and an epoch-1 loss close to "
        "the default run's (both runs start from the same re-headed model) — not the much lower loss that continuing from the adapted model "
        "would give. In Section 9, compare rows 0 and 1 of the run history: `ap`, `ap75`, `mask_ap`, `mask_ap75`, `final_loss` and "
        "`train_seconds`.\n"
        "5. **Explain:** in one sentence, why is a change in box `ap` informative here while an unchanged `mask_ap` of 1.0 is not?\n\n"
        "<details><summary>Check your reasoning</summary>In a local CPU check of this revision (2026-10-06, real weights, reduced to `N_IMAGES = 8` and `EPOCHS = 1` for time) the re-run after a default run printed `rebuilt_from_verified_snapshot`; its box and mask predictors on entry to `finetune` were equal, tensor for tensor, to a fresh `from_pretrained(class_names=SIGN_CLASSES, seed=0)`; all 45,891,175 parameters were trained; its first-epoch loss was 2.2127, a fresh start next to the frozen run's 2.1416; and the run record said `epochs: 1`, the schedule actually applied. Before this fix (the 2026-10-02 review) the same re-run continued training the adapted model while its record said one epoch. At that reduced size neither run is near the ceiling (box `ap` 0.1966 frozen and 0.4770 unfrozen, on 2 held-out images), which is far too small a test to rank the settings. The full 10-epoch unfrozen comparison on a GPU is not recorded "
        "for this revision, so your run history is the measurement. A `mask_ap` of 1.0 in both rows says both runs reached the ceiling on these "
        "ten drawn images; it cannot say which has more margin. Box `ap` below 1.0 can still move. One run on 10 held-out images carries no "
        "dispersion estimate, so read differences of a few hundredths as unresolved, not as a ranking of the two settings.</details>\n\n"
        "## Interpretation and limits\n\n"
        "Start from your own run: the Section 9 table and its `at_ceiling_1.0` list, the Section 10 found count and the Section 11 reload "
        "check. Then read the reference answer.\n\n"
        "<details><summary>Reference answer from the recorded run</summary>The pretrained COCO model found 3 of the 4 drawn COCO objects at "
        "0.75 (the sports ball was missed) and returned nothing on blank or noise images at 0.75, though noise produced 2 large instances at "
        "0.05. Re-heading onto three new sign classes started from a baseline box AP50 of 0.18 and mask AP of 0.0; a 10-epoch fine-tune "
        "with the body frozen took every held-out mask row and box AP50 and AP75 to the ceiling of 1.0 and box AP@[.50:.95] to 0.87 "
        "(Kaggle T4, 2026-09-25). The pretrained model already solves most of this drawn task once its predictors are re-headed, so the "
        "lesson is the workflow — validate, split, baseline, fine-tune, evaluate, export, reload and compare — not the size of the gain. The "
        "adapted model found all 6 signs on the three unseen drawings, and the adapter reloaded to the same instances within 1e-3.</details>\n\n"
        "**What this notebook established, in this runtime.** The pinned torchvision checkpoint for `torchvision/maskrcnn_resnet50_fpn_v2` was "
        "verified against a committed SHA-256 manifest before loading. The pretrained model was run on a drawn scene and on two structure-free "
        "probes, and its boxes and masks were compared with the drawn references by IoU. A three-class sign vocabulary that COCO does not contain "
        "was then adapted by replacing the box and mask predictors, training the FPN, RPN and ROI heads with the ResNet-50 body frozen, and "
        "scoring the held-out split with box and mask average precision before and after. The adapter was exported, reloaded onto a fresh base "
        "model and checked against the in-memory instances.\n\n"
        "**A task at its ceiling.** When the adapted rows read 1.0, the held-out images no longer separate good settings from better ones. "
        "That is a property of this drawn task and its size, stated here rather than hidden: the signs are drawn in fixed colours and shapes "
        "by the same code as the training images. A harder test needs harder data — more varied drawings, photographs, or your own labelled "
        "images through Section 13 — not a different metric chosen after the fact.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set and the "
        "pinned checkpoint together reproduce these stages in a fresh runtime, without the repository being cloned or installed and without "
        "any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the adapted model "
        "generalises beyond the synthetic images it was fitted to. The held-out AP is measured on a few drawn images (10 at the defaults) "
        "and carries no dispersion estimate. The scores are not calibrated probabilities.\n\n"
        "**Reproducibility.** Seeds are form fields (`DATASET_SEED`, `SEED`, `NEW_DATA_SEED`), the run is float32 with no data augmentation, "
        "and the new predictors are initialised under `SEED`. GPU kernels (ROI Align in particular) are not forced to be deterministic, so "
        "repeated GPU runs can differ in the last digits of the loss and the scores.\n\n"
        "**More experiments (optional).** Each one names the cell to change and where to re-run from; a re-run of Section 8 always starts "
        "from the re-headed model, and Section 9's run history keeps the earlier rows.\n\n"
        "- **Find where the ceiling starts:** set `EPOCHS = 1` (then 2, then 3) in Section 8, select Section 8 and choose **Runtime → Run "
        "after**. Watch at which epoch count the mask rows reach 1.0 and whether box `ap` is still climbing after that.\n"
        "- **A different learning rate:** set `LEARNING_RATE = 0.001` in Section 8 and **Run after** from Section 8.\n"
        "- **A smaller dataset:** set `N_IMAGES = 12` in Section 6, select Section 6 and choose **Run after**: fewer held-out images make "
        "every AP step coarser.\n"
        "- **Your own labelled images:** turn on `USE_BYOD_DATASET` in Section 13 and re-run that cell only.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux wheels; use Google Colab, "
        "Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails while downloading.** The `uv` wheel, the managed Python and the locked packages come from PyPI and "
        "python-build-standalone; run the cell again. A size or SHA-256 mismatch is refused on purpose — if it repeats, the download is being altered.\n"
        "- **A restart prompt.** This notebook never needs one: nothing is installed into the kernel. If the isolated process exits (usually out "
        "of memory), restart the session and choose **Run all**.\n"
        "- **Section 3 stops on a download or a digest mismatch.** The checkpoint is fetched from download.pytorch.org and re-hashed; a mismatch "
        "is never loaded. Run the cell again; if it repeats, delete `weights/` and run from Section 3.\n"
        "- **Out of memory in Section 8 or 14.** Lower `BATCH_SIZE` to 1 in Section 8, or keep `FREEZE_BACKBONE = True`, and **Run after** from "
        "Section 8. On a CPU runtime the fine-tune is slow, not broken (see the estimate under Prerequisites).\n"
        "- **\"Cannot fine-tune: this pipeline was already adapted\".** Something called `finetune` on a model that was already trained; "
        "Section 8 avoids this by calling `reset_to_pretrained()`. If you edited the cells, re-run from Section 7.\n"
        "- **Every adapted row reads 1.0.** That is the recorded result on this drawn task (Section 9 explains it), not an error; compare "
        "settings on the rows listed under `below_ceiling`.\n"
        "- **BYOD is refused.** Each refusal names the file or the rule and the fix: a path that is not a file or directory, no upload dialog "
        "outside Colab, a cancelled upload, a file Pillow cannot read, a mask file missing from the folder (usually uploaded in a sub-folder; "
        "upload flat files), boxes that look normalised to 0..1 (use pixel coordinates), a mask outside its box, an unknown label, or fewer "
        "than 2 records. Fix the data, then re-run Section 13.\n\n"
        "## Glossary\n\n"
        "- **Instance segmentation:** one box, one class and one pixel mask per object, so two touching objects of the same class stay separate.\n"
        "- **Region proposal network (RPN):** the first stage, which proposes candidate boxes; the ROI heads then classify, refine and mask each one.\n"
        "- **Score and threshold:** the softmax class probability after non-maximum suppression; not calibrated. The threshold is the caller's "
        "decision about what is returned, not a model property.\n"
        "- **NMS (non-maximum suppression):** drops overlapping boxes of the same class, keeping the highest-scoring one.\n"
        "- **IoU:** intersection over union, for boxes or for masks.\n"
        "- **AP50 / AP@[.50:.95]:** average precision with a match at IoU ≥ 0.50, and the mean over IoU thresholds 0.50–0.95; both rank every "
        "instance above the evaluation threshold by score.\n"
        "- **Ceiling (saturated metric):** a metric at its maximum (AP 1.0); it cannot show a further gain, so it cannot rank two settings.\n"
        "- **Re-heading:** replacing the class-specific box and mask predictors with freshly initialised ones for a new vocabulary.\n"
        "- **Frozen backbone:** the ResNet-50 body keeps its COCO weights and BatchNorm statistics during fine-tuning.\n"
        "- **Held-out split:** images never shown to the optimizer and never used to choose a setting; the task evidence.\n"
        "- **Adapter / reload equivalence:** the SafeTensors file of the tensors the fine-tune could change, and the check that a fresh base "
        "model with the adapter loaded returns the same instances.\n"
        "- **Isolated environment:** the separate hash-locked Python environment built in Section 1; every later cell runs there.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. The pretrained COCO model matched ___ of 4 drawn objects at 0.75; the noise probe returned ___ instances at 0.05.\n"
        "2. The fine-tune moved held-out mask AP from ___ to ___ and box `ap` from ___ to ___; the rows at the ceiling were ___.\n"
        "3. What a 1.0 row on this held-out split does not tell me: ___.\n"
        "4. In Section 14, unfreezing the backbone changed box `ap` by ___ and the training time by ___.\n"
        "5. What I would need before using this adapter on real images: ___.\n\n"
        "## References\n\n"
        "- He, K., Gkioxari, G., Dollár, P. and Girshick, R. (2017). *Mask R-CNN.* [arXiv:1703.06870](https://arxiv.org/abs/1703.06870).\n"
        "- Li, Y., Xie, S., Chen, X., Dollár, P., He, K. and Girshick, R. (2021). *Benchmarking Detection Transfer Learning with Vision Transformers.* [arXiv:2111.11429](https://arxiv.org/abs/2111.11429) — the paper torchvision cites for the v2 builder.\n"
        "- torchvision model documentation: [maskrcnn_resnet50_fpn_v2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html); upstream repository [pytorch/vision](https://github.com/pytorch/vision) — BSD-3-Clause.\n"
        "- Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312).\n"
        "- Repository model card: https://github.com/kurtvalcorza/maskrcnn-instance-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/maskrcnn-instance-segmentation-pipeline`](https://github.com/kurtvalcorza/maskrcnn-instance-segmentation-pipeline) — source repository for this pipeline."
    ),
}
