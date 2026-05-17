"""
Unit-conversion correctness for src/features.py.

Locks down that the calibration block in config.yaml feeds through to the
kinematic columns. With px_per_cm=29 and (cm, s) units the velocity column
should be in cm/s, dt in s, area_covered in cm^2, and tortuosity unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as F


def _straight_path(n_frames: int = 4, step_px: float = 10.0) -> pd.DataFrame:
    """One fly moving step_px pixels to the right each frame."""
    return pd.DataFrame(
        {
            "frame": list(range(n_frames)),
            "ordered_id": [1] * n_frames,
            "x": [step_px * f for f in range(n_frames)],
            "y": [0.0] * n_frames,
        }
    )


def test_module_constants_are_consistent_with_config():
    """LENGTH_SCALE and TIME_SCALE must round-trip the unit choice."""
    assert F.LENGTH_UNIT in ("cm", "px")
    assert F.TIME_UNIT in ("s", "frame")
    assert F.LENGTH_SCALE > 0
    assert F.TIME_SCALE > 0

    if F.LENGTH_UNIT == "cm":
        assert np.isclose(F.LENGTH_SCALE * F.PX_PER_CM, 1.0)
    else:
        assert F.LENGTH_SCALE == 1.0

    if F.TIME_UNIT == "s":
        assert np.isclose(F.TIME_SCALE * F.fps, 1.0)
    else:
        assert F.TIME_SCALE == 1.0


def test_units_dict_advertises_correct_unit_strings():
    assert F.UNITS["length"] == F.LENGTH_UNIT
    assert F.UNITS["time"] == F.TIME_UNIT
    assert F.UNITS["velocity"] == f"{F.LENGTH_UNIT}/{F.TIME_UNIT}"
    assert F.UNITS["acceleration"] == f"{F.LENGTH_UNIT}/{F.TIME_UNIT}^2"
    assert F.UNITS["angular_velocity"] == f"rad/{F.TIME_UNIT}"
    assert F.UNITS["area"] == f"{F.LENGTH_UNIT}^2"
    assert F.UNITS["tortuosity"] == "dimensionless"


def test_velocity_in_configured_unit():
    """A fly moving step_px/frame should yield step_px*LENGTH_SCALE/TIME_SCALE."""
    step_px = 10.0
    out = F.extract_behavioral_features(_straight_path(n_frames=4, step_px=step_px))
    expected = step_px * F.LENGTH_SCALE / F.TIME_SCALE
    # First per-fly row is the baseline (dx=0). Skip it and check the steady-state.
    steady = out["velocity"].iloc[1:]
    assert np.allclose(steady, expected, rtol=1e-6), (
        f"velocity expected {expected} {F.UNITS['velocity']}, got {steady.tolist()}"
    )


def test_dt_in_configured_unit():
    out = F.extract_behavioral_features(_straight_path(n_frames=4))
    # frame.diff() == 1 between adjacent frames -> dt = 1 * TIME_SCALE.
    expected_dt = 1.0 * F.TIME_SCALE
    assert np.allclose(out["dt"], expected_dt, rtol=1e-6)


def test_step_distance_in_configured_length_unit():
    step_px = 10.0
    out = F.extract_behavioral_features(_straight_path(n_frames=4, step_px=step_px))
    expected = step_px * F.LENGTH_SCALE
    steady = out["step_distance"].iloc[1:]
    assert np.allclose(steady, expected, rtol=1e-6)


def test_distance_traveled_accumulates_in_configured_unit():
    step_px = 10.0
    n_frames = 4
    out = F.extract_behavioral_features(_straight_path(n_frames=n_frames, step_px=step_px))
    # After (n_frames - 1) moves of step_px the cumulative path is (n-1)*step.
    expected_total = (n_frames - 1) * step_px * F.LENGTH_SCALE
    assert np.isclose(out["distance_traveled"].iloc[-1], expected_total, rtol=1e-6)


def test_tortuosity_is_unit_invariant():
    """Tortuosity is a ratio of lengths -> must not depend on LENGTH_SCALE."""
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "ordered_id": [1] * 5,
            "x": [0.0, 10.0, 0.0, 10.0, 20.0],
            "y": [0.0, 10.0, 20.0, 30.0, 0.0],
        }
    )
    out = F.extract_behavioral_features(df)

    raw_x = df["x"].to_numpy()
    raw_y = df["y"].to_numpy()
    path_len = np.sqrt(np.diff(raw_x) ** 2 + np.diff(raw_y) ** 2).sum()
    net_disp = np.hypot(raw_x[-1] - raw_x[0], raw_y[-1] - raw_y[0])
    expected = path_len / net_disp

    assert np.isclose(out["tortuosity"].iloc[0], expected, rtol=1e-6)


def test_area_in_configured_length_unit_squared():
    """Right triangle (0,0)-(10,0)-(10,10) -> 50 px^2 raw, scaled by LENGTH_SCALE^2.

    Path is open (does not return to origin) so tortuosity is finite and
    extract_behavioral_features's dropna() keeps all rows.
    """
    df = pd.DataFrame(
        {
            "frame":      [0, 1, 2, 3],
            "ordered_id": [1] * 4,
            "x":          [0.0, 10.0, 10.0, 5.0],
            "y":          [0.0, 0.0, 10.0, 5.0],
        }
    )
    out = F.extract_behavioral_features(df)
    expected_area = 50.0 * (F.LENGTH_SCALE ** 2)
    assert np.isclose(out["area_covered"].iloc[0], expected_area, rtol=1e-6)
