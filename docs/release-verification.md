# Release verification

`tutorials/maskrcnn_instance_segmentation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, the tiny-model test, JSON validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.1 (REL8). This file is the durable release-gate record for the notebook.

The upstream checkpoint is pinned to its SHA-256 `73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e` and the notebook is regenerated with that revision and manifest, so it can run; no execution is recorded yet.

## Automatic coverage (static and unit, every pull request)

CI installs the pinned CPU-only torch wheel and the other runtime pins, then runs:

- `ruff check src tests tools`;
- `pytest`: snapshot verification and staging against synthetic manifests, the unpinned refusals, the import boundary, input and dataset validation, BYOD directory reading, average precision, the evaluation report, the notebook parity checks, the weight-facts check, the pin tool against a fake download, and `tests/test_tiny_model.py` — the random-weight `maskrcnn_resnet50_fpn_v2` architecture at a 64 px input size taken through `finetune` (including the check that the frozen ResNet-50 body keeps its weights and BatchNorm statistics), `evaluate`, `save_artifact` and `apply_artifact`;
- `tools/validate_release_assets.py`: model card 1.2 structure and front matter, the pin state across the package, the manifest and the documents, identity consistency, the weight facts, the release-status tokens, and the notebook's structure, carried modules, parity, markers, BYOD gates and location fields;
- `tools/build_notebook.py --check`.

These are source, provenance and unit checks. None of them loads the pinned checkpoint, so none of them is execution evidence.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. confirm the checkpoint is pinned (`MODEL_REVISION` is the 64-hex SHA-256 of the checkpoint, equal to the manifest entry's `sha256`, and the entry records its byte size) and that static CI is green on the exact commit under review;
2. open that exact notebook revision in a new GPU runtime (Colab or Kaggle, Tesla T4 or similar) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells, with every form field at its default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`, `threshold = 0.75`, `EPOCHS = 10`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`, and that the installed versions equal the inline `PINS`;
5. verify that every default-path stage completes: the pinned install; the carried modules; staging of the checkpoint from download.pytorch.org and `verify_snapshot`; `validate_inputs` with the `threshold=1.5` rejection finding; COCO instance segmentation and the `sample-sanity` report with `box_iou` and `mask_iou`; the blank and noise probes; `validate_dataset`; the split with its leakage assertion; the baseline; `finetune`; the adapted evaluation; new-data inference; adapter export, reload and the tolerance and mask-agreement check; and the seven outputs in `outputs/`;
6. record the observed baseline and adapted `ap`, `ap50`, `ap75`, `mask_ap`, `mask_ap50` and `mask_ap75`, the per-epoch losses, the trainable and total parameter counts, the degenerate-probe counts and the wall time in the table below. No metric value is asserted in advance: a low adapted AP is a finding to record, not a failure by itself;
7. exercise the BYOD dataset branch once with a small labelled directory (`BYOD_DATASET_DIR`) and once with an incompatible `annotations.json`, and record both outcomes (REL12);
8. record the notebook Git blob id, commit, runtime (platform, GPU, Python, PyTorch, torchvision), model identifier and revision (the checkpoint SHA-256), whether the model cache was clean, and any warning judged harmless with the reason;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/maskrcnn_instance_segmentation_colab.ipynb` (verify with `git rev-parse <commit>:tutorials/maskrcnn_instance_segmentation_colab.ipynb`). Each record uses the fields Date, Subject, Runtime, Procedure, Observed result and Caveats.

| Date (UTC) | Subject (commit / notebook blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| — | — | — | — | No execution recorded yet. | — |
