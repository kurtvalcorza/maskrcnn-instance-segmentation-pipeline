"""Final prose pass (Kurt's ruling, 2026-10-26): the prose-rule items flagged on the confirming run are fixed.

Run-status sentences, timing promises and unlabelled expectations are removed or recast
as records of named runs.
Fails on PR #5 head 7799d9e.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "maskrcnn_instance_segmentation_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_final_prose_pass():
    md = _markdown(NB)
    for gone in (
        'The baseline is expected to be near zero',
        'Near zero is the expected floor',
    ):
        assert gone not in md, gone
    for kept in (
        '**The baseline is a floor.** Randomly initialised',
        'no mask matched at all.</details>',
    ):
        assert kept in md, kept
