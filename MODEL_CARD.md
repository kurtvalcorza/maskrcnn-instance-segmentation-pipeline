---
license: bsd-3-clause
model_card_spec: "1.2"
pipeline_tag: image-segmentation
base_model: torchvision/maskrcnn_resnet50_fpn_v2
date_published: "2022-06"
date_published_source: "month of the torchvision 0.13 release, the first release that shipped MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 (the weights' recipe is recorded in their metadata as pytorch/vision pull request 5773)"
---

# Mask R-CNN ResNet-50 FPN v2, COCO — Instance Segmentation with Bounded Fine-Tuning

[![torchvision](https://img.shields.io/badge/torchvision-maskrcnn__resnet50__fpn__v2-ee4c2c?style=flat&logo=pytorch&logoColor=white)](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-pytorch%2Fvision-181717?style=flat&logo=github&logoColor=white)](https://github.com/pytorch/vision)
[![arXiv Paper](https://img.shields.io/badge/arXiv-1703.06870-b31b1b.svg)](https://arxiv.org/abs/1703.06870)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://github.com/pytorch/vision/blob/main/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> The upstream checkpoint is **not yet pinned**. `MODEL_REVISION` is the sentinel `"unpinned"` and the manifest records no SHA-256 digest and no byte size. Until `python tools/pin_snapshot.py` downloads the file, checks it, and records both, the package refuses to stage, verify or load the weights, and the tutorial cannot run.

---

## Interactive Colab Tutorials

- **End-to-end instance segmentation and adaptation tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/maskrcnn-instance-segmentation-pipeline/blob/main/tutorials/maskrcnn_instance_segmentation_colab.ipynb) [`maskrcnn_instance_segmentation_colab.ipynb`](https://github.com/kurtvalcorza/maskrcnn-instance-segmentation-pipeline/blob/main/tutorials/maskrcnn_instance_segmentation_colab.ipynb)
  *COCO instance segmentation on a drawn scene, degenerate-input probes, a bounded fine-tune that re-heads Mask R-CNN onto a three-class sign vocabulary, held-out box and mask average precision before and after, new-data inference, and SafeTensors adapter export and reload.*

---

#### Description

`torchvision/maskrcnn_resnet50_fpn_v2` names torchvision's `maskrcnn_resnet50_fpn_v2` model with its `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1` weights. The architecture is Mask R-CNN (He et al., arXiv:1703.06870) with a ResNet-50 backbone and a feature pyramid network (FPN). torchvision describes the v2 builder as the improved Mask R-CNN from "Benchmarking Detection Transfer Learning with Vision Transformers" (Li et al., arXiv:2111.11429), and the weights' metadata says they "were produced using an enhanced training recipe", recorded as pytorch/vision pull request 5773. torchvision's weight metadata lists 46,359,409 parameters, and a test in this repository counts the same number in the architecture it builds.

Mask R-CNN is a two-stage model. The ResNet-50 body and the FPN produce feature maps at five scales. A region proposal network (RPN) scores anchor boxes and proposes regions. For each region, ROI Align crops a fixed-size feature, and the ROI heads predict a class over 91 COCO category slots (one of them background), a refined box per class, and a 28×28 mask per class. At inference torchvision drops regions whose class probability is below 0.05, applies non-maximum suppression (NMS) per class at IoU 0.5, keeps at most 100 instances, and pastes each 28×28 mask back into the image. The v2 variant differs from the original torchvision Mask R-CNN in its heads: regular BatchNorm in the backbone, the FPN and the ROI heads, a two-convolution RPN head, and a four-convolution box head.

The pretrained weights come from supervised training on COCO 2017. This repository adds gradient fine-tuning on a caller's labelled instance masks. `from_pretrained(class_names=...)` replaces the box predictor and the mask predictor with new layers for the caller's vocabulary. `finetune` then trains with torchvision's own training losses: RPN objectness and box regression, ROI classification and box regression, and per-pixel binary cross-entropy on each positive region's mask. The ResNet-50 body is frozen by default, and its BatchNorm layers are held in evaluation mode so their running statistics stay at their COCO values.

What this repository adds to the upstream weights:

- `verify_snapshot` and `stage_missing_files`: manifest checks and staging of the one checkpoint file from its manifest URL, both refusing to run while the checkpoint is unpinned;
- `MaskRcnnPipeline.from_pretrained`: construction with `weights=None` (nothing downloaded by torchvision), then `torch.load(..., weights_only=True)` and `load_state_dict(strict=True)` from the verified file only;
- `detect`: input checks, threshold checks, and score-sorted pixel-space boxes with aligned boolean masks;
- `validate_inputs`, `validate_dataset`, `read_detection_records` and `evaluation_report`: the validation and single-image evaluation stages, with box and mask IoU;
- `finetune`, `evaluate` (package-local `average_precision` on boxes and on masks), `save_artifact`, `apply_artifact` and `load_artifact`: the bounded adaptation workflow and its SafeTensors adapter;
- `tools/pin_snapshot.py`, `tools/build_notebook.py` and `tools/validate_release_assets.py`: pinning, notebook generation and static release checks.

#### Intended Use and Limitations

The uses below are the ones the package was built to support. Everything else is out of scope (Out-of-scope use cases) or prohibited (Use cases).

###### Primary Intended Uses

The task is closed-vocabulary instance segmentation. `detect` takes one `PIL.Image.Image` and a threshold. It returns at most 100 instances, sorted by score. Each instance has an xyxy box in input pixels, a `label` from the model's class slots, a `score` and a `mask_area`, and a boolean mask of the image's height and width is returned alongside it.

The pretrained vocabulary fits everyday scenes: people and vehicles in street imagery, animals, furniture and household objects indoors, sports and food. The adaptation path fits a small labelled set in a new vocabulary, for example signage, parts or equipment that COCO does not name, where a per-pixel outline is needed rather than a box.

The intended role is a reference instance segmenter and a teaching baseline. Mask R-CNN is the two-stage model that later instance and panoptic segmenters are compared against. A reader's own application can embed `MaskRcnnPipeline` as the first stage of area measurement, cut-outs, counting or downstream classification.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, students and instructors. Settings envisioned are research prototypes, teaching, and self-hosted applications that run the code in this repository.

A user is expected to know the following before relying on the output:

- the pretrained vocabulary is the 90 non-background COCO category slots, of which 80 were trained with instances; anything else is mislabelled or missed;
- a `score` is a softmax probability under the model's own training distribution, not a calibrated probability on the user's images;
- the threshold trades recall against false instances and must be set per deployment;
- a mask is predicted at 28×28 and resized to the box, so thin structures and fine boundaries are coarse;
- drawn graphics, documents, aerial, medical and thermal imagery are distribution shifts from COCO photographs;
- average precision, precision and recall can only be measured on a labelled set the user supplies;
- a fine-tune on a few dozen images demonstrates the workflow and does not produce a deployable model.

###### Out-of-scope use cases

1. **Capability boundary:** no classes outside the loaded vocabulary and no text or point prompts; promptable and open-vocabulary segmentation are separate models. No semantic or panoptic segmentation of background regions ("stuff" such as sky or road). No keypoints, tracking or video. No calibrated confidence.
2. **Input boundary:** `validate_image` rejects anything that is not a `PIL.Image.Image` (`TypeError`), and sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px (`ValueError`). Thresholds outside `[0, 1]` are rejected. One image per `detect` call; at most `MAX_DETECTIONS = 100` instances.
3. **Input boundary:** the model's own transform resizes every image so its shorter side is 800 px and its longer side at most 1333 px. Objects only a few pixels tall after that resize, and structures thinner than a few pixels, are not a use this repository supports.
4. **Data boundary for adaptation:** `validate_dataset` accepts 1–5,000 records (`MAX_RECORDS`), each instance with a mask of the image's size, and `split_dataset` needs at least 2. The bounded tutorial fine-tune (10 epochs by default) is far shorter than the upstream multi-GPU recipe. It is not a way to train a production model.
5. **Decision boundary:** not for decisions that act on instances or masks without a person reviewing them. This covers vehicle control, security alerts, safety interlocks, robotic grasping, and medical or industrial inspection. It also requires precision and recall measured locally on the deployment's own labelled images at the chosen threshold.

#### Factors

###### Groups

The pipeline is human-centric in one respect: `person` is a COCO class, so the pretrained model outlines people in any image. Neither the upstream authors nor this repository evaluated person segmentation per group. Recall and mask quality may differ by skin tone, age, body size, clothing, hair, mobility aids and lighting; that difference is unknown, not known to be absent.

COCO's collection skews toward consumer photographs from a limited set of regions. Vehicles, foods, furniture and signage from under-represented regions may have lower recall; that is also unmeasured.

An adapted model inherits whatever group structure the caller's labelled data has. The operator who segments people, with the pretrained or an adapted model, owns a per-group audit on their own images before relying on the output.

###### Instrumentation

The upstream training data comes from consumer cameras: colour photographs at web resolution, with instance polygons drawn by crowd workers under COCO's annotation rules. Inference images arrive from whatever produced them: phones, CCTV or dashboard cameras, drones, or rendering engines.

Resolution, motion blur, compression, exposure, lens distortion and viewpoint all change the visual evidence. The resize to a shorter side of 800 px changes the pixel scale of every image. The pipeline checks only type and size. It cannot detect a night frame, a fisheye lens, a rendered scene or an empty frame.

The tutorial's sample data is itself an instrument: Pillow drawings with flat colours and a bundled font, far from any camera. Its masks are exact by construction, because each object is drawn once onto the image and once onto its mask. Masks a caller supplies carry whatever error their annotation process has: polygon simplification, missed holes, or a boundary drawn a few pixels wide. The fine-tune learns a systematic error of that kind as if it were correct.

###### Environment

**Operating environment.** Python 3.12 with the pins in `pyproject.toml`: `torch==2.14.0`, `torchvision==0.29.0`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`. Computation is float32. The code runs on CPU and uses CUDA automatically when available. torchvision supplies the architecture, the ROI Align operator and the training losses. No run with the pinned weights has been recorded yet, so no runtime, memory or throughput figure is given.

**Data environment.** The pretrained model assumes a photograph of an everyday scene containing COCO objects. An adapted model assumes inference images that resemble its training images in camera, scene and object appearance. The tutorial's adaptation data is synthetic, so a model adapted on it transfers to drawn signs of the same style and to nothing else. When these assumptions fail, the model still returns instances. The pipeline reports no signal that the distribution has shifted.

#### Metrics

###### Performance Measures

`evaluate(records)` reports average precision computed by the package-local `average_precision`, once on boxes and once on masks:

- `ap` and `mask_ap`: mean AP over the ten IoU thresholds 0.50, 0.55, …, 0.95 (COCO's AP@[.50:.95]), by box IoU and by mask IoU;
- `ap50`, `ap75`, `mask_ap50` and `mask_ap75`: AP at IoU 0.50 and 0.75;
- `per_class_ap50` and `per_class_mask_ap50`: AP at IoU 0.50 for each class that has at least one reference instance.

AP summarises the precision–recall trade-off over all score levels, so it does not depend on one threshold. `ap50` measures whether objects are found at all; `ap` also rewards tight localisation. The box and mask rows answer different questions: a model can find every object with a good box and still draw a poor outline, which only the mask rows show. The implementation uses greedy score-ordered matching and 101-point interpolation. It has no pycocotools area ranges or crowd handling, and it computes mask IoU on full-resolution boolean masks rather than run-length encodings, so its values are close to, but not identical with, the COCO evaluator.

`evaluation_report(result, references)` covers one image. It reports one `box_iou` and one `mask_iou` per supplied reference instance, against the best-overlapping instance **of the same label**, with the verdict `sample-sanity`. Without references it returns `not-measurable` and names the labelled data that would be needed.

torchvision's weight metadata reports box mAP 47.4 and mask mAP 41.8 on COCO val2017. Those values are upstream-reported, and this repository does not reproduce them. No value from this repository has been recorded yet.

###### Decision thresholds

`detect` applies two thresholds. An instance is kept when its class probability is at least `threshold`, and a pixel belongs to its mask when the predicted mask probability is at least `MASK_THRESHOLD = 0.5`. The default `DETECTION_THRESHOLD = 0.75` is the value torchvision's visualization-utilities gallery uses to keep Mask R-CNN instances, and 0.5 is the mask cut the same example uses. The gallery shows them with the v1 `maskrcnn_resnet50_fpn` weights; neither value was re-derived for the v2 weights, tuned or calibrated by this repository. Each instance carries one label, but torchvision scores every class separately above its internal floor of 0.05 and runs NMS per class, so one object can appear as two instances with different labels when both pass the threshold.

`evaluate` uses `EVAL_DETECTION_THRESHOLD = 0.05`, the same value as torchvision's internal floor, and keeps at most `MAX_EVAL_DETECTIONS = 100` instances per image. A low threshold is needed so that AP can see the full precision–recall curve. It is not a deployment setting.

No acceptance threshold on AP is set anywhere in the repository. The deployment owns choosing `threshold` on its own labelled images. Lower it when a missed object costs more than a false instance. Raise it when a false instance triggers downstream action. Re-tune it after any change of camera, scene, class mix or adapter. A deployment that measures areas from masks also owns checking `MASK_THRESHOLD` against its own boundaries.

###### Approaches to uncertainty and variability

Every AP value is one pass over one held-out split: no repeated runs, no cross-validation, no bootstrap, and no confidence interval. The tutorial's held-out split has 10 synthetic images. One image more or less found moves its AP visibly, so the value is tutorial evidence only.

Sources of run-to-run variability:

- the dataset draw, the split and the new-data draw, controlled by `DATASET_SEED`, `SEED` and `NEW_DATA_SEED`;
- the predictor initialisation and the batch order, controlled by the `seed` argument;
- the sampling of proposals and ROIs during training, which draws from torch's global generator after it is seeded;
- GPU kernel selection and the atomic additions in ROI Align's backward pass, which are not forced to be deterministic, so repeated GPU runs can differ slightly;
- the BatchNorm statistics of the FPN and the ROI heads, which update during fine-tuning from batches of 2 images.

A `score` is a softmax output, not a calibrated probability. A caller who needs calibrated confidence must fit a calibration map on labelled images from the deployment. A caller who needs an uncertainty estimate for AP must evaluate over many images, with repeated runs or bootstrap resampling.

#### Ethical considerations and biases

No external ethics board, red team, or population-specific review has examined this repository or, to our knowledge, the upstream checkpoint. Nothing below implies that one did.

###### Data

The weights are torchvision's `COCO_V1` weights for the COCO category list, and their metadata reports results on COCO val2017. The COCO 2017 detection split has 118k training and 5k validation images. The COCO paper describes images collected from Flickr and annotated by crowd workers. COCO contains identifiable people, licence plates, house numbers and private settings. Personal data is therefore present in the training corpus by construction; it was not audited here.

torchvision's README states that its pre-trained models may have their own licences or terms derived from the dataset used for training, and that the user must determine whether they have permission to use a model for their use case. The BSD-3-Clause licence above is torchvision's code licence; the COCO terms bear on the weights.

This repository distributes code, tests and documentation. It does not distribute the checkpoint, which is staged locally from download.pytorch.org and git-ignored; torchvision's weight metadata lists it as 177.219 MB. It ships no photographs: the tutorial scene and the adaptation dataset are drawn in code.

An exported adapter contains weights fitted to the caller's training images. It does not contain the images, but it can reflect them. The operator must audit the images they segment, or fine-tune on, for personal, proprietary or restricted content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. Neither this repository, torchvision's maintainers, nor any regulator has validated or certified it for any of them.

Some sensitive uses are foreseeable although not intended: pedestrian segmentation for vehicle control, person segmentation for surveillance, crowd counting for policing, body or wound measurement from masks, and PPE or hazard segmentation for safety interlocks. Any of them would be admissible only with human review of every acted-on instance. They would also need locally measured precision, recall and mask quality stratified by the groups named above, a documented threshold and re-validation policy, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** while `MODEL_REVISION` is `"unpinned"`, `verify_snapshot`, `stage_missing_files` and `from_pretrained` raise before any download or model import. Once pinned, `stage_missing_files` refuses a manifest whose `modelId` or `revision` differs from the package constants, and a manifest URL that differs from `WEIGHTS_URL`. It fetches only with `allow_download=True`, and `torch.hub.download_url_to_file` checks the file-name SHA-256 prefix on the way in. `verify_snapshot` then checks the file's byte size and full SHA-256 and refuses an entry with no recorded digest. `from_pretrained` builds the architecture with `weights=None`, so torchvision downloads nothing itself, and loads the verified file with `torch.load(..., weights_only=True)` and `load_state_dict(strict=True)`.
- **Tests of those refusals:** tests assert that an unpinned package, a missing checkpoint and a tampered digest are all refused before `torch` or `torchvision` is imported. Others assert that `COCO_CATEGORIES` and `WEIGHTS_URL` equal torchvision's own weight metadata, that the built architecture has the published parameter count, and that a frozen fine-tune leaves every weight and BatchNorm statistic of the ResNet-50 body unchanged.
- **Input integrity:** `validate_inputs` and `detect` share one checker for type, image size and threshold. `validate_dataset` rejects a record with missing keys, an out-of-range image, box, label and mask counts that differ, a non-finite, empty or out-of-bounds box, a mask of the wrong shape, a non-binary, empty or out-of-box mask, or an unknown label. It reports classes that have no instance as findings. `read_detection_records` refuses absolute file names and `..` segments for images and masks. `detect` raises on a malformed backend instance, an unknown label, a mask of the wrong shape or more than 100 instances.
- **Adapter integrity:** `save_artifact` writes SafeTensors, not pickle, with the base identity, base-checkpoint digest, class names and frozen prefixes in its header. `apply_artifact` refuses a different format, base identity or model key. It also refuses a different vocabulary, a different base digest, tensors the model does not have, and any missing trainable tensor.
- **Reproducibility:** exact `==` pins in `pyproject.toml`, carried into the notebook and checked by the parity tests. Seeds for data, split, predictor initialisation and batch order. Every result and adapter records `model_id` and `model_revision`.
- **Refusals:** no download without the explicit flag, no unrestricted pickle deserialisation, and no export of an unadapted pipeline.
- **Statistical mitigations:** none is implemented. There is no class balancing, re-sampling or augmentation; `validate_dataset` reports class coverage and does not change it.

###### Risks and harms

- **Instances on empty or unfamiliar input.** The model can return instances for images that contain no object of any trained class. The operator and any downstream consumer bear the harm of a fabricated count, area or alert. Likelihood on real empty frames is unmeasured; the tutorial probes a blank and a noise image and records what it finds.
- **Missed objects.** An object that is small, occluded or rendered unusually is simply absent from the output, and no field flags the miss. The harm falls on whoever relies on the output being complete.
- **Wrong outlines.** A correct box can carry a mask that leaks into the background or misses a thin part. Area measurements and cut-outs inherit the error, and a box-only check does not reveal it.
- **Mislabelling within a closed vocabulary.** An object outside the vocabulary that resembles a class is labelled as that class. Systems that act on labels inherit the error.
- **Overfitting in adaptation.** A fine-tune on a few dozen images can score well on a held-out split drawn from the same source and fail on anything else. The operator who deploys it bears the harm, which is realised whenever training and deployment images differ.
- **Person segmentation and surveillance.** `person` is a first-class output with unaudited per-group recall and mask quality. The people in the processed images bear the harm of misuse or unequal error.
- **Automation bias.** High softmax scores and crisp masks invite trust that an uncalibrated score has not earned. Operators who skip review turn a model error into a decision error.
- **Leakage through adaptation data.** A random split of records that share a source photograph or session puts near-duplicates on both sides. The resulting held-out AP overstates quality without any signal; `split_dataset` documents that grouped data must be split by group.
- **Resource use.** Inference at an 800 px shorter side on CPU is slow, and each returned mask is a full-resolution array. A video stream or a large batch can saturate a shared host's CPU and memory.

###### Use cases

The following uses are prohibited even where the model would work:

- segmenting people in order to surveil, track, profile or score them, or to support biometric or demographic profiling;
- unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement;
- processing images the operator has no right to process, or in breach of consent, privacy or data-protection obligations;
- deceptive uses that present instances or masks as verified facts or as evidence;
- autonomous physical control or safety interlocks driven by unreviewed instances;
- any use that violates the upstream BSD-3-Clause licence, the terms attached to the COCO training data, or the terms of the deployment running the pipeline.

## Immutable provenance

- Model: `torchvision/maskrcnn_resnet50_fpn_v2` (torchvision builder `maskrcnn_resnet50_fpn_v2`, weights `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`).
- Revision: **not yet pinned** (`MODEL_REVISION = "unpinned"`). A URL-hosted file has no commit, so the pinned revision is the SHA-256 of the checkpoint's bytes. `python tools/pin_snapshot.py` downloads the file, checks that its SHA-256 starts with `73cbd019` (the prefix in its file name), strict-loads it into the architecture, and records the full digest and the byte size.
- Checkpoint manifest: `weights/maskrcnn-resnet50-fpn-v2/dimer-base-manifest.json`, format `dimer_url_snapshot`, one file.
- Checkpoint: `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` at `https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth`; SHA-256 and byte size not yet recorded. torchvision's weight metadata lists the file as 177.219 MB.
- Loader: `maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91)`, then `load_state_dict(torch.load(<verified file>, map_location="cpu", weights_only=True), strict=True)`.

## Input/output contract

- `MaskRcnnPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, class_names=None, seed=20260924)`: stage (only with `allow_download=True`), verify, load; with `class_names`, replace the box and mask predictors deterministically under `seed`.
- `detect(image, *, threshold=0.75) -> dict`: keys `detections` (list of `{"box": [x0, y0, x1, y1], "label": str, "score": float, "mask_area": int}`, input pixels, descending score, at most 100), `masks` (list of boolean `(height, width)` arrays aligned with `detections`), `threshold`, `mask_threshold`, `width`, `height`, `class_names`, `adapted`, `model_id`, `model_revision`.
- `finetune(records, *, epochs=10, batch_size=2, learning_rate=0.005, seed=20260924, freeze_backbone=True, progress=None) -> dict`: SGD with momentum 0.9 and weight decay 0.0005, float32; returns the configuration, parameter counts and per-epoch losses.
- `evaluate(records, *, threshold=0.05, iou_thresholds=(0.50, …, 0.95), max_detections=100) -> dict`: `ap`, `ap50`, `ap75`, `per_class_ap50`, `mask_ap`, `mask_ap50`, `mask_ap75`, `per_class_mask_ap50`, `n_images`, `n_references`, `estimation`.
- `save_artifact(path, *, notes=None) -> dict`; `read_artifact_metadata(path) -> dict`; `apply_artifact(path)`; `load_artifact(path, *, weights_dir=None, device=None)`. The adapter format is `maskrcnn-adapter-v1`.
- Records: `{"image": PIL.Image.Image, "boxes": [[x0, y0, x1, y1], ...], "labels": [name, ...], "masks": [bool array (height, width), ...]}`; `read_detection_records(directory)` reads them from `annotations.json` plus image and mask PNG files.
- Constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_DETECTIONS = 100`, `MAX_RECORDS = 5000`, `MAX_CLASSES = 1000`, `COCO_CATEGORIES` (91 slots including background), `LABELS` (90 slots, 10 of them `"N/A"`), `DETECTION_THRESHOLD = 0.75`, `MASK_THRESHOLD = 0.5`, `EVAL_DETECTION_THRESHOLD = 0.05`.

## Verification records

No execution with the pinned weights has been recorded. The offline test suite runs the full `maskrcnn_resnet50_fpn_v2` architecture with random weights and a 64 px input size through fine-tuning, evaluation and adapter reload; that exercises the code path and is not a result about this model. `docs/release-verification.md` holds the release gate and the record table.

## References

- He, Gkioxari, Dollár and Girshick. Mask R-CNN. ICCV 2017. https://arxiv.org/abs/1703.06870
- Li, Xie, Chen, Dollár, He and Girshick. Benchmarking Detection Transfer Learning with Vision Transformers. 2021. https://arxiv.org/abs/2111.11429
- Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014. https://arxiv.org/abs/1405.0312
- torchvision model documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html
- Upstream code: https://github.com/pytorch/vision
