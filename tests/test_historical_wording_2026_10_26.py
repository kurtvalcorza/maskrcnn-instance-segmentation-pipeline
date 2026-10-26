"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Kaggle T4 of blob a8f04d11 (2026-09-25); Colab T4 of blobs a5e05339 (2026-10-08),
7d8cfa26 (2026-10-24), c58649ed and ee86f619 (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "maskrcnn_instance_segmentation_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_expectations_and_run_status_sentences_are_gone():
    md = _markdown()
    for stale in ("took about 50 s (51 s to 55 s)", "so do not expect the final epoch to be the lowest",
                  "ended at 0.080 to 0.090", "elevenfold to twelvefold", "is not recorded",
                  "that is expected for the AP50 and AP75 rows", "0.87 to 0.94", "0.87–0.94"):
        assert stale not in md, stale


def test_confirming_run_of_ee86f619_is_recorded():
    md = _markdown()
    assert "`c58649ed` (2026-10-25) and `ee86f619` (2026-10-25)" in md
    assert "`ee86f619`: 0.9828, 0.4020, 0.2661" in md and "ended at 0.080 to 0.098" in md
    assert "0.8600 (Colab T4, `ee86f619`, 2026-10-25)" in md
    assert "between 47 s and 55 s for the isolated-environment build" in md
