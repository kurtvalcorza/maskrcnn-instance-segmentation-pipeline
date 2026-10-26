"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501, F811  -- one-line refusal messages; F811: the notebook_run fixture is imported from the review-fix tests
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from test_maskrcnn_instance_segmentation_colab_review_fixes import (  # noqa: F401  (fixture)
    _write_byod,
    notebook_run,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/maskrcnn_instance_segmentation_colab.ipynb"
BYOD_CELL = "maskrcnn_instance_segmentation-33"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


EARLIER_IMAGE = {"image_path": "old", "byod_image": "old", "byod_input_manifest": "old", "byod_result": "old"}
EARLIER_DATASET = {"dataset_dir": "old", "byod_records": "old", "byod_manifest": "old", "byod_pipe": "old", "byod_adapted": "old", "byod_export": "old"}
IMAGE_FILES = ["byod_maskrcnn_input_manifest.json"]
DATASET_FILES = ["byod_maskrcnn_adapter.safetensors", "byod_maskrcnn_detections.csv", "byod_maskrcnn_result.json"]


def _seed(ns: dict) -> None:
    """Results an earlier BYOD run would have left in the session and in outputs/."""
    ns.update(EARLIER_IMAGE, **EARLIER_DATASET)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    for name in IMAGE_FILES + DATASET_FILES:
        (out / name).write_text("{}", encoding="utf-8")


def _left(ns: dict, names: dict, files: list) -> tuple[set, list]:
    return set(names) & set(ns), [n for n in files if (Path("outputs") / n).exists()]


def _colab(monkeypatch, uploads):
    queue = iter(uploads)
    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: next(queue)
    colab.files = files
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)


def _image_cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    return {
        "text file (the scripted /etc/hostname run)": (repr(str(tmp_path / "hostname")), "hostname", "is not an image Pillow can read"),
        "folder": (repr(str(tmp_path / "folder")), "folder", "is not a file"),
        "missing path": (repr(str(tmp_path / "nope.png")), "nope.png", "is not a file"),
    }


@pytest.mark.parametrize("case", ["text file (the scripted /etc/hostname run)", "folder", "missing path"])
def test_image_problems_are_one_refused_line_and_remove_earlier_results(notebook_run, tmp_path, case):
    ns, run, _tiny = notebook_run
    value, file_name, rule = _image_cases(tmp_path)[case]
    _seed(ns)
    with pytest.raises(ValueError) as caught:
        run(13, USE_BYOD_IMAGE="True", BYOD_IMAGE_PATH=value)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "run this cell again" in line and "earlier BYOD results of this session and their outputs/ files were removed" in line
    assert _left(ns, EARLIER_IMAGE, IMAGE_FILES) == (set(), [])
    assert _left(ns, EARLIER_DATASET, DATASET_FILES) == (set(EARLIER_DATASET), DATASET_FILES)  # the other branch is untouched


@pytest.mark.parametrize("field", ["USE_BYOD_IMAGE", "USE_BYOD_DATASET"])
def test_upload_outside_colab_is_one_refused_line(notebook_run, monkeypatch, field):
    _ns, run, _tiny = notebook_run
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        run(13, **{field: "True"})
    assert _refusal_line(caught.value).startswith("Refused: (no file): BYOD_")
    assert "no Colab upload dialog" in str(caught.value)


@pytest.mark.parametrize(("uploads", "file_name", "rule"), [([{}], "(no file)", "the upload was cancelled or empty"),
                                                            ([{"a.png": b"1", "b.png": b"2"}], "a.png, b.png", "upload exactly one image")])
def test_cancelled_or_multiple_image_upload_is_one_refused_line(notebook_run, monkeypatch, uploads, file_name, rule):
    _ns, run, _tiny = notebook_run
    _colab(monkeypatch, uploads)
    with pytest.raises(ValueError) as caught:
        run(13, USE_BYOD_IMAGE="True")
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line


def _dataset_cases(tmp_path: Path) -> dict:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "hostname").write_text("runtime\n", encoding="utf-8")
    _write_byod(tmp_path / "broken_json")
    (tmp_path / "broken_json" / "annotations.json").write_text("{not json", encoding="utf-8")
    entries = _write_byod(tmp_path / "bad_box")
    entries[0]["boxes"] = [5]
    (tmp_path / "bad_box" / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")
    _write_byod(tmp_path / "one", n=1)
    return {
        "folder without annotations.json (the /etc shape)": ("etc", "annotations.json not found"),
        "annotations.json is not JSON": ("broken_json", "Expecting property name"),
        "a box that is not a list (TypeError)": ("bad_box", "annotations.json has a malformed record"),
        "too few records to split": ("one", "at least 2 records"),
        "missing folder": ("absent", "is not a directory"),
    }


@pytest.mark.parametrize("case", ["folder without annotations.json (the /etc shape)", "annotations.json is not JSON", "a box that is not a list (TypeError)",
                                  "too few records to split", "missing folder"])
def test_dataset_problems_are_one_refused_line_and_remove_earlier_results(notebook_run, tmp_path, case):
    ns, run, _tiny = notebook_run
    folder, rule = _dataset_cases(tmp_path)[case]
    _seed(ns)
    with pytest.raises(ValueError) as caught:
        run(13, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(tmp_path / folder)))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {folder}: ") and rule in line, line
    assert _left(ns, EARLIER_DATASET, DATASET_FILES) == (set(), [])
    assert _left(ns, EARLIER_IMAGE, IMAGE_FILES) == (set(EARLIER_IMAGE), IMAGE_FILES)


def test_dataset_genuine_internal_error_keeps_its_traceback(notebook_run, tmp_path):
    ns, run, _tiny = notebook_run
    _write_byod(tmp_path / "mine")
    _seed(ns)

    def broken(_directory):
        raise RuntimeError("internal bug")

    ns["read_detection_records"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        run(13, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(tmp_path / "mine")))
    assert not getattr(caught.value, "dimer_refusal", False)
    assert _left(ns, EARLIER_DATASET, DATASET_FILES) == (set(EARLIER_DATASET), DATASET_FILES)  # nothing removed


def test_fine_tune_failure_is_not_a_refusal(notebook_run, tmp_path):
    """Only acquisition, loading, validation and the split are user input; a training failure keeps its traceback."""
    ns, run, tiny = notebook_run
    _write_byod(tmp_path / "mine")

    class Broken(tiny):
        def finetune(self, *args, **kwargs):
            raise ValueError("internal training bug")

    ns["MaskRcnnPipeline"] = Broken
    with pytest.raises(ValueError, match="internal training bug") as caught:
        run(13, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(tmp_path / "mine")))
    assert not getattr(caught.value, "dimer_refusal", False)


def test_image_genuine_internal_error_keeps_its_traceback(notebook_run, tmp_path):
    ns, run, _tiny = notebook_run
    _write_byod(tmp_path / "img", n=1)

    def broken(*_args, **_kwargs):
        raise RuntimeError("internal bug")

    ns["validate_inputs"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        run(13, USE_BYOD_IMAGE="True", BYOD_IMAGE_PATH=repr(str(tmp_path / "img" / "img0.png")))
    assert not getattr(caught.value, "dimer_refusal", False)


def test_default_path_is_unchanged(notebook_run):
    _ns, run, _tiny = notebook_run
    out = run(13)
    assert "BYOD image branch is off" in out and "BYOD dataset branch is off" in out and "Refused" not in out


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(notebook_run):
    """The hosted refusal run sets USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH='/etc/hostname': a text file present in every fresh runtime."""
    _ns, run, _tiny = notebook_run
    with pytest.raises(ValueError) as caught:
        run(13, USE_BYOD_IMAGE="True", BYOD_IMAGE_PATH="'/etc/hostname'")
    assert _refusal_line(caught.value).startswith("Refused: hostname: hostname is not an image Pillow can read")
