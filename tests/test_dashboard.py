import json
import re
from pathlib import Path

import pytest

pytest.importorskip("plotly")

from dashboard.build import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_builds_from_committed_results(tmp_path):
    page = build(ROOT / "results", ROOT / "models", ROOT / "figures", tmp_path).read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script id="dash-data" type="application/json">(.*?)</script>', page, re.S)
                      .group(1).replace("<\\/", "</"))
    # every plot on the page has a figure for every combination of its selects
    for chart in set(re.findall(r'data-chart="([^"]+)"', page)):
        variants = data["charts"][chart]
        assert variants and all(v["data"] for v in variants.values()), chart
    assert (tmp_path / "assets" / "app.js").exists()
    assert (tmp_path / "figures" / "policy_bars.png").exists()
    assert "Simulated data only. Not for patient care." in page
    assert "—" not in page  # no em dashes
