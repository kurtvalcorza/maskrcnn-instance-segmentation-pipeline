"""Prose numbers stay true across every recorded hosted run (Kurt's ruling, 2026-10-25).

Runs: Kaggle T4 of blob a8f04d11 (2026-09-25); Colab T4 of blobs a5e05339 (2026-10-08),
7d8cfa26 (2026-10-24) and c58649ed (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "maskrcnn_instance_segmentation_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_single_run_claims_that_failed_a_later_run_are_gone():
    md = _markdown()
    for stale in ("AP@[.50:.95] read 0.92 for boxes and 0.99", "but not on the two AP@[.50:.95] rows",
                  "the two AP@[.50:.95] rows stayed below it", "`box_iou` 0.929–0.968",
                  "the default fine-tune 69.1 s", "about a twelvefold fall"):
        assert stale not in md, stale


def test_varying_quantities_are_stated_as_ranges_over_the_named_runs():
    md = _markdown()
    assert "AP@[.50:.95] read 0.86 to 0.94 for boxes and 0.98 to 1.0 for masks" in md
    assert "mask AP@[.50:.95] reached it in two of the five" in md
    assert "`box_iou` between 0.878 and 0.982 and `mask_iou` between 0.957 and 0.984" in md
    assert "in four of the five runs the last epoch's loss was slightly above the ninth's" in md
    assert "between 68.3 s and 73.5 s" in md
