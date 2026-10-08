# ruff: noqa: E501  -- assertion lines name the exported fields in full
"""Regression tests for the 2026-10-02 review of maskrcnn_instance_segmentation_colab (MRC-M1..M3, MRC-m1..m6).

The static tests read the committed notebook and the package without torch. The `notebook_run` tests execute the
notebook's own learner cells (Sections 4-13) from the committed .ipynb in a namespace built from the package, with
a random-weight Mask R-CNN at a 64 px input size standing in for the pinned checkpoint (the same architecture and
code paths as `tests/test_tiny_model.py`). The small model says nothing about segmentation quality; these tests
check control flow, the exported fields and the BYOD file handling. The real-weights checks are recorded in
docs/release-verification.md.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import types
from pathlib import Path

import pytest

with contextlib.suppress(ImportError):  # torch first when present (Windows DLL load order before NumPy linear algebra)
    import torch  # noqa: F401

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from maskrcnn_instance_segmentation_pipeline import pipeline as pipeline_module  # noqa: E402
from maskrcnn_instance_segmentation_pipeline import samples as samples_module  # noqa: E402
from maskrcnn_instance_segmentation_pipeline.pipeline import (  # noqa: E402
    DEFAULT_SEED,
    read_detection_records,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "maskrcnn_instance_segmentation_colab.ipynb"
SIDE = 64


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _markdown(nb: dict) -> str:
    return "\n".join(_source(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _section(nb: dict, number: int) -> tuple[str, str]:
    cells = nb["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and f"## {number}. " in _source(cell):
            return _source(cell), next(_source(c) for c in cells[index + 1 :] if c["cell_type"] == "code")
    raise KeyError(number)


def _set(source: str, name: str, value: str) -> str:
    new, n = re.subn(rf"^{name} = .*?(  # @param.*)$", lambda m: f"{name} = {value}{m.group(1)}", source, flags=re.M)
    assert n == 1, name
    return new


# ---------------------------------------------------------------- MRC-M1: uv isolated runtime, no in-kernel install


def test_only_the_uv_install_and_router_cells_run_in_the_kernel():
    nb = _notebook()
    code = [_source(c) for c in nb["cells"] if c["cell_type"] == "code"]
    kernel = [i for i, src in enumerate(code) if "# dimer: kernel cell" in src]
    assert kernel == [0, 1]
    install = code[0]
    assert '"--require-hashes", "--only-binary", ":all:"' in install and '"--managed-python"' in install
    assert 'platform.machine() != "x86_64"' in install
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.14.0", "torchvision==0.29.0", "safetensors==0.8.0", "numpy==2.5.3", "pillow==11.3.0"):
        assert f"\n{pin} \\" in lock
    markdown = _markdown(nb)
    assert "Restart the runtime" not in markdown and "the cell stops with a restart instruction" not in markdown
    assert "no restart is needed" in markdown and "**Linux x86_64 only**" in markdown


def test_restart_assisted_run_is_not_recorded_as_a_pass():
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "**PASSED (default path)**" not in record
    assert "Passed only after a manual restart" in record and "restarted: false" in record
    for name in ("STATUS.md", "README.md", "MODEL_CARD.md", "tutorials/README.md"):
        assert "not a one-pass Run all" in (ROOT / name).read_text(encoding="utf-8"), name


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raised on a spec-less stub."""
    import ast

    nb = _notebook()
    router = [_source(c) for c in nb["cells"] if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)  # raised ValueError before the fix
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


# ---------------------------------------------------------------- MRC-M2: guided layer


def test_guided_layer_is_present_and_infrastructure_is_labelled():
    nb = _notebook()
    markdown = _markdown(nb)
    for marker in ("**Who this is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**",
                   "## 14. Your turn — change one thing: unfreeze the backbone", "## Troubleshooting", "## Glossary",
                   "## Conclusion (your notes)", "> **Infrastructure.**"):
        assert marker in markdown, marker
    for number in range(4, 12):
        md, _code = _section(nb, number)
        assert "**Predict before running:**" in md.split(f"## {number}. ", 1)[1], number
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 9
    objectives = markdown.split("**Learning objectives:**", 1)[1].split("**This notebook does not demonstrate:**", 1)[0]
    for verb in ("describe", "explain", "read", "identify", "check", "predict"):
        assert verb in objectives
    for cell in nb["cells"]:
        if cell.get("metadata", {}).get("dimer", {}).get("embedded_module"):
            assert cell["metadata"]["jupyter"]["source_hidden"] is True


def test_environment_variables_read_by_code_are_documented():
    """MRC-m5: every environment variable a code cell reads is named in markdown (the worker's own flag aside)."""
    nb = _notebook()
    markdown = _markdown(nb)
    code = "\n".join(_source(c) for c in nb["cells"] if c["cell_type"] == "code")
    read = set(re.findall(r"""os\.environ\.get\(\s*["']([A-Z_]+)["']""", code))
    assert {"DIMER_NOTEBOOK_CI_PREINSTALLED", "DIMER_ISOLATED_ENV"} <= read
    for name in read - {"DIMER_KERNEL_IS_COLAB"}:  # set by the router for its own worker, internal
        assert f"`{name}" in markdown, name


# ---------------------------------------------------------------- MRC-m2 / MRC-m6 / MRC-m3: prose


def test_runtimes_are_measured_and_estimates_labelled():
    md = _markdown(_notebook())
    prereq = md.split("## Prerequisites", 1)[1].split("## 1.", 1)[0]
    assert "Runtimes are not measured" not in md
    assert "486.2 s" in prereq and "Kaggle Tesla T4" in prereq and "**Estimate, not a measurement:**" in prereq


def test_schedule_fields_live_in_section_8_and_counts_are_defaults():
    nb = _notebook()
    _md6, code6 = _section(nb, 6)
    _md8, code8 = _section(nb, 8)
    assert "EPOCHS" not in re.findall(r"^(\w+) = .*# @param", code6, re.M)
    assert {"EPOCHS", "LEARNING_RATE", "BATCH_SIZE", "FREEZE_BACKBONE"} <= set(re.findall(r"^(\w+) = .*# @param", code8, re.M))
    assert "dataset_manifest = validate_dataset(records, SIGN_CLASSES, epochs=EPOCHS)" in code8
    markdown = _markdown(nb)
    assert "a held-out part (25%, 10 images)" not in markdown and "`EPOCHS` epochs of SGD" not in markdown
    assert "with the default 40 images that is 30 and 10" in markdown


def test_saturated_rows_and_low_threshold_instances_are_explained():
    nb = _notebook()
    md9, code9 = _section(nb, 9)
    assert "**A 1.0 row is a ceiling.**" in md9
    assert "at_ceiling_1.0" in code9 and "below_ceiling" in code9
    md5, _ = _section(nb, 5)
    assert "**Instances that appear only at 0.05 matter too.**" in md5
    markdown = _markdown(nb)
    assert "**A task at its ceiling.**" in markdown
    assert "Change `FREEZE_BACKBONE` to `False` and compare held-out mask AP" not in markdown
    activity = markdown.split("## 14. Your turn", 1)[1].split("## Interpretation and limits", 1)[0]
    assert "box `ap`" in activity and "Runtime → Run after" in activity


# ---------------------------------------------------------------- MRC-m1: BYOD refusals name the file and the fix (no torch)


def _write_byod(folder: Path, n: int = 4) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, record in enumerate(samples_module.sign_dataset(n, seed=123)):
        name = f"img{k}.png"
        record["image"].save(folder / name)
        masks = []
        for j, mask in enumerate(record["masks"]):
            mask_name = f"img{k}_mask{j}.png"
            Image.fromarray((np.asarray(mask) * 255).astype(np.uint8)).save(folder / mask_name)
            masks.append(mask_name)
        entries.append({"file": name, "boxes": record["boxes"], "labels": record["labels"], "masks": masks})
    (folder / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")
    return entries


def test_byod_non_image_and_missing_mask_raise_named_value_errors(tmp_path):
    folder = tmp_path / "set"
    _write_byod(folder)
    (folder / "img1.png").write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match=r"entry 1: image file 'img1.png' is not an image Pillow can read .*save it as PNG"):
        read_detection_records(folder)
    _write_byod(folder)
    (folder / "img2_mask0.png").unlink()
    with pytest.raises(ValueError, match=r"entry 2: mask file 'img2_mask0.png' is missing .*correct its name"):
        read_detection_records(folder)


def test_normalised_boxes_get_a_pixel_coordinate_hint(tmp_path):
    folder = tmp_path / "set"
    entries = _write_byod(folder, n=2)
    for entry in entries:
        entry["boxes"] = [[v / 640 for v in box] for box in entry["boxes"]]
    (folder / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")
    records = read_detection_records(folder)
    with pytest.raises(ValueError, match="looks like normalised coordinates: boxes must be xyxy pixel coordinates"):
        validate_dataset(records, samples_module.SIGN_CLASSES)
    _write_byod(tmp_path / "ok", n=2)
    assert validate_dataset(read_detection_records(tmp_path / "ok"), samples_module.SIGN_CLASSES)["verdict"] == "accepted"


# ---------------------------------------------------------------- notebook_run: the learner cells with a small model


def _tiny_pipeline_class():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch

    MaskRcnnPipeline = pipeline_module.MaskRcnnPipeline

    class TinyPipeline(MaskRcnnPipeline):
        """Stand-in for the pinned checkpoint; `load_artifact` and `finetune` are inherited."""

        @classmethod
        def from_pretrained(cls, device=None, weights_dir=None, allow_download=False, class_names=None, seed=DEFAULT_SEED):
            torch.manual_seed(0)  # the "checkpoint": every tensor except the re-headed predictors is the same for every build
            model = pipeline_module.build_model(len(pipeline_module.COCO_CATEGORIES))
            model.transform.min_size = (SIDE,)
            model.transform.max_size = SIDE
            names = tuple(class_names) if class_names is not None else pipeline_module.LABELS
            reinitialised = ()
            if class_names is not None:
                torch.manual_seed(seed)
                reinitialised = pipeline_module._rehead(model, len(names))
            return cls(model=model.eval(), device="cpu", class_names=names, source="tiny",
                       base_state_digest=pipeline_module.MODEL_REVISION, reinitialised=reinitialised)

    return TinyPipeline


@pytest.fixture
def notebook_run(tmp_path, monkeypatch):
    """Run Sections 4-13 of the notebook at small sizes (8 images, 1 epoch); returns (namespace, run, TinyPipeline)."""
    tiny = _tiny_pipeline_class()
    import torch
    import torchvision

    monkeypatch.chdir(tmp_path)
    nb = _notebook()
    ns: dict = {"__name__": "__main__"}
    ns.update({k: v for k, v in vars(samples_module).items() if not k.startswith("__")})
    ns.update({k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")})
    ns.update(MaskRcnnPipeline=tiny, WEIGHTS_DIR=tmp_path / "weights", NOTEBOOK_SOURCE={"repository_revision": "test"},
              np=np, Image=Image, torch=torch, torchvision=torchvision, platform=__import__("platform"))
    ns["pipe"] = tiny.from_pretrained()

    def run(number: int, **fields: str) -> str:
        _md, source = _section(nb, number)
        for name, value in fields.items():
            source = _set(source, name, value)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, f"section{number}", "exec"), ns)
        return buffer.getvalue()

    for number in (4, 5):
        run(number)
    run(6, N_IMAGES="8")
    run(7)
    run(8, EPOCHS="1")
    for number in range(9, 14):
        run(number)
    return ns, run, tiny


def test_default_run_records_the_schedule_and_the_ceiling_view(notebook_run):
    ns, _run, _tiny = notebook_run
    assert ns["dataset_manifest"]["epochs"] == 1 == ns["run"]["epochs"]
    result = json.loads(Path("outputs/maskrcnn_instance_segmentation_result.json").read_text(encoding="utf-8"))
    adaptation = result["adaptation"]
    assert adaptation["finetune"]["epochs"] == 1 and adaptation["dataset"]["epochs"] == 1
    assert set(adaptation["at_ceiling"]) | set(adaptation["below_ceiling"]) == {"ap", "ap50", "ap75", "mask_ap", "mask_ap50", "mask_ap75"}
    assert len(adaptation["run_history"]) == 1 and "train_seconds" in adaptation["finetune"]
    check = adaptation["reload_check"]
    assert check["equivalent"] is True and (check["instances_compared"] > 0 or check["tensors_compared"] > 0)  # never vacuous


def test_rerunning_the_finetune_starts_from_the_reheaded_model(notebook_run, monkeypatch):
    """MRC-M3 acceptance: after a completed run, the Section 14 steps fine-tune from predictors equal to a fresh
    from_pretrained(class_names, seed), and the run record matches the schedule applied."""
    ns, run, tiny = notebook_run
    first = list(ns["run"]["epoch_losses"])
    assert ns["adapter"].adapted
    import torch

    fresh = tiny.from_pretrained(class_names=ns["SIGN_CLASSES"], seed=ns["SEED"]).model.state_dict()
    entry = {}
    original = tiny.finetune

    def spy(self, records, **kwargs):
        state = self.model.state_dict()
        entry["adapted"] = self.adapted
        entry["equal_fresh"] = all(torch.equal(state[k], v) for k, v in fresh.items())
        return original(self, records, **kwargs)

    monkeypatch.setattr(tiny, "finetune", spy)
    out = run(8, EPOCHS="1")  # same settings, re-run as a learner would
    assert "rebuilt_from_verified_snapshot" in out and entry == {"adapted": False, "equal_fresh": True}
    assert ns["run"]["epoch_losses"] == first  # identical start, not continued training
    run(9)
    out = run(8, EPOCHS="1", FREEZE_BACKBONE="False")
    assert "rebuilt_from_verified_snapshot" in out and entry["equal_fresh"] is True
    assert ns["run"]["freeze_backbone"] is False and ns["run"]["epochs"] == 1 and ns["run"]["frozen_prefixes"] == []
    run(9)
    assert [row["freeze_backbone"] for row in ns["RUN_HISTORY"]] == [True, True, False]
    assert all(row["epochs"] == 1 for row in ns["RUN_HISTORY"])


def test_finetune_refuses_an_adapted_pipeline(notebook_run, tmp_path):
    ns, _run, tiny = notebook_run
    with pytest.raises(RuntimeError, match="already adapted"):
        ns["adapter"].finetune(ns["train_records"], epochs=1)
    reloaded = tiny.load_artifact(Path("outputs/maskrcnn_adapter.safetensors"))
    with pytest.raises(RuntimeError, match="already adapted"):
        reloaded.finetune(ns["train_records"], epochs=1)


def test_byod_dataset_writes_results_and_compares_the_reload(notebook_run, tmp_path):
    _ns, run, _tiny = notebook_run
    folder = tmp_path / "mine"
    _write_byod(folder)
    out = run(13, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(folder)))
    assert "BYOD adapter exported, reloaded and compared" in out
    result = json.loads(Path("outputs/byod_maskrcnn_result.json").read_text(encoding="utf-8"))
    assert {"annotations_sha256", "dataset", "split", "finetune", "baseline", "adapted", "artifact", "reload_check"} <= set(result)
    check = result["reload_check"]
    assert result["dataset"]["n_records"] == 4 and (check["instances_compared"] > 0 or check["tensors_compared"] > 0)
    assert Path("outputs/byod_maskrcnn_detections.csv").read_text(encoding="utf-8").startswith("image,rank,label")


def test_byod_image_writes_a_manifest_and_refusals_are_named(notebook_run, tmp_path):
    _ns, run, _tiny = notebook_run
    folder = tmp_path / "img"
    _write_byod(folder, n=1)
    run(13, USE_BYOD_IMAGE="True", BYOD_IMAGE_PATH=repr(str(folder / "img0.png")))
    manifest = json.loads(Path("outputs/byod_maskrcnn_input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["report_verdict"] == "not-measurable" and len(manifest["sha256"]) == 64
    (folder / "notes.png").write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match="notes.png is not an image Pillow can read"):
        run(13, USE_BYOD_IMAGE="True", BYOD_IMAGE_PATH=repr(str(folder / "notes.png")))
    with pytest.raises(ValueError, match="BYOD_IMAGE_PATH .* is not a file"):
        run(13, USE_BYOD_IMAGE="True", BYOD_IMAGE_PATH=repr(str(folder / "absent.png")))
    with pytest.raises(RuntimeError, match="no Colab upload dialog"):
        run(13, USE_BYOD_DATASET="True")


def test_a_cancelled_upload_is_refused_and_a_second_upload_replaces_the_first(notebook_run, monkeypatch):
    ns, run, _tiny = notebook_run
    payloads = []
    for colour in ("red", "blue"):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), colour).save(buffer, format="PNG")
        payloads.append(buffer.getvalue())
    queue = iter([{}, {"a_first.png": payloads[0]}, {"z_second.png": payloads[1]}])
    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: next(queue)
    colab.files = files
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match="the upload was cancelled or empty"):
        run(13, USE_BYOD_IMAGE="True")
    run(13, USE_BYOD_IMAGE="True")
    assert ns["image_path"].name == "a_first.png"
    run(13, USE_BYOD_IMAGE="True")
    assert sorted(p.name for p in ns["image_path"].parent.iterdir()) == ["z_second.png"]


def test_in_cell_refusal_probes_use_real_failure_modes(notebook_run):
    _ns, run, _tiny = notebook_run
    out = run(13)
    assert "refused as expected: a file that is not an image -> ValueError" in out
    assert "refused as expected: a normalised box -> ValueError" in out and "pixel coordinates" in out
