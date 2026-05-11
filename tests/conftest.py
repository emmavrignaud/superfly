"""
Shared pytest fixtures.

These tests are intentionally hermetic: no Roboflow API, no real videos, no GPU.
Anything that needs the network, OpenCV GUI, or a video file lives outside the
test suite (and is exercised by the run_*.py scripts in real runs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `src` and `utils` importable from the repo root regardless of where
# pytest is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_yaml(repo_root: Path) -> dict:
    """Loaded config.yaml as a dict. Sessions-scoped so we read it once."""
    import yaml

    with open(repo_root / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mini_wide_df() -> pd.DataFrame:
    """A 4-frame, 3-track wide tracker DataFrame with one missing detection.

    Layout matches what export_tracks_xy_tuple_csv_one_config writes:
      frame | id1            | id2            | id3
      0     | "(10.0, 20.0)" | "(50.0, 60.0)" | NaN
      1     | "(11.0, 20.5)" | "(51.0, 60.0)" | "(90.0, 90.0)"
      ...
    """
    return pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "id1": ["(10.0, 20.0)", "(11.0, 20.5)", "(12.0, 21.0)", "(13.0, 21.5)"],
            "id2": ["(50.0, 60.0)", "(51.0, 60.0)", "(52.0, 60.0)", np.nan],
            "id3": [np.nan, "(90.0, 90.0)", "(91.0, 91.0)", "(92.0, 92.0)"],
        }
    )


@pytest.fixture
def mini_long_df_csv(tmp_path: Path, mini_wide_df: pd.DataFrame) -> Path:
    """Long-format CSV with a known left-to-right vial layout.

    Three flies, one in each vial-shaped strip:
      orig_id 1: x ~ 10  -> belongs in vial1 (left)
      orig_id 2: x ~ 50  -> belongs in vial2 (middle)
      orig_id 3: x ~ 90  -> belongs in vial3 (right)
    """
    rows = []
    for f in range(4):
        rows.append({"frame": f, "orig_id": "id1", "x": 10.0 + f, "y": 20.0})
        rows.append({"frame": f, "orig_id": "id2", "x": 50.0 + f, "y": 60.0})
        rows.append({"frame": f, "orig_id": "id3", "x": 90.0 + f, "y": 90.0})
    df = pd.DataFrame(rows)
    out = tmp_path / "mini_long.csv"
    df.to_csv(out, index=False)
    return out


@pytest.fixture
def mini_vial_rois_json(tmp_path: Path) -> Path:
    """Three side-by-side rectangular ROIs covering x in [0,30], [40,70], [80,110]."""
    rois = {
        "vial1": [0, 0, 30, 100],
        "vial2": [40, 0, 70, 100],
        "vial3": [80, 0, 110, 100],
    }
    out = tmp_path / "vial_rois.json"
    with open(out, "w") as f:
        json.dump(rois, f)
    return out
