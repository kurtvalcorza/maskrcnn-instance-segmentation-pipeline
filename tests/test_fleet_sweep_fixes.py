"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['maskrcnn_instance_segmentation_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'maskrcnn_instance_segmentation_colab': 7}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


NB = NOTEBOOKS[0]


def test_swp_a_only_contract_asserts_remain():
    code = "\n".join(c["source"] for c in _code_cells(_nb(NB)) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = [line.strip() for line in code.splitlines() if line.strip().startswith("assert ")]
    # The split-leak check and the reload-equivalence checks are contract integrity; none compares a metric with a baseline.
    assert asserts and all(("overlap" in a) or ("d1" in a) or ("m1" in a) or ("out_orig" in a) for a in asserts), asserts


def _byod_helpers(monkeypatch, tmp_path):
    source = _cell(_nb(NB), "BYOD_DATASET_DIR = ")
    block = source[source.index("def _upload_into(target, field):") : source.index("\n\n\nif USE_BYOD_IMAGE:")]
    ns = {"Path": Path}
    exec(block, ns)
    return ns["_upload_into"], ns["_existing"]


def test_swp_b_paths_are_checked_and_name_the_field(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    upload_into, existing = _byod_helpers(monkeypatch, tmp_path)
    image = tmp_path / "img.png"
    image.write_bytes(b"x")
    assert existing(str(image), "BYOD_IMAGE_PATH", "file") == image
    assert existing(str(tmp_path), "BYOD_DATASET_DIR", "directory") == tmp_path
    with pytest.raises(FileNotFoundError, match="BYOD_IMAGE_PATH .*missing.png.* is not a file: give the path of one image file"):
        existing(str(tmp_path / "missing.png"), "BYOD_IMAGE_PATH", "file")
    with pytest.raises(FileNotFoundError, match="BYOD_DATASET_DIR .* is not a directory: give the path of the directory holding annotations.json"):
        existing(str(image), "BYOD_DATASET_DIR", "directory")
    with pytest.raises(RuntimeError, match="BYOD_DATASET_DIR is empty and this runtime has no Colab upload dialog"):
        upload_into(tmp_path / "up", "BYOD_DATASET_DIR")


def test_swp_b_cancelled_upload_is_refused_and_a_real_one_is_written(monkeypatch, tmp_path):
    upload_into, _existing = _byod_helpers(monkeypatch, tmp_path)
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="upload was cancelled or empty: .*set BYOD_IMAGE_PATH"):
        upload_into(tmp_path / "up", "BYOD_IMAGE_PATH")
    _colab(monkeypatch, lambda: {"a.png": b"1"})
    target = upload_into(tmp_path / "up", "BYOD_IMAGE_PATH")
    assert (target / "a.png").read_bytes() == b"1"


def test_swp_g_checkpoint_answers_agree_with_the_recorded_run():
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    for fact in ("stop sign 0.9995, clock 0.9969, traffic light 0.9953", "**sports ball not detected**", "noise 0 at 0.75 and **2 at 0.05** (umbrella 0.26, bear 0.13)", "68 instances (stop-sign 20, yield-sign 24, speed-limit-sign 24)", "`ap` 0.0347 → 0.8693", "stop-sign 0.5347", "0.9822", "0.0863", "reload check 2 instances, tolerance 0.001"):
        assert fact in record, fact
    markdown = "\n".join(c["source"] for c in _nb(NB)["cells"] if c["cell_type"] == "markdown")
    for quoted in ("stop sign 0.9995, clock 0.9969, traffic light 0.9953", "umbrella 0.26, bear 0.13", "stop-sign 20, yield-sign 24, speed-limit-sign 24", "0.0347", "0.5347", "0.9822", "0.0863", "tolerance 0.001"):
        assert quoted in markdown, quoted
