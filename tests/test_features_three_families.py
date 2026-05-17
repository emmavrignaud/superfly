"""
Kinematic x / y / magnitude decomposition (``features.kinematic_three_families``).

Tests monkeypatch ``src.features.KINEMATIC_THREE_FAMILIES`` so they stay
independent of the checked-in config.yaml default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as F


def _straight_path(n_frames: int = 4, step_px: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": list(range(n_frames)),
            "ordered_id": [1] * n_frames,
            "x": [step_px * f for f in range(n_frames)],
            "y": [0.0] * n_frames,
        }
    )


def test_velocity_hypot_matches_speed_when_three_families(monkeypatch):
    monkeypatch.setattr(F, "KINEMATIC_THREE_FAMILIES", True)
    out = F.extract_behavioral_features(_straight_path())
    steady = out.iloc[1:]
    np.testing.assert_allclose(
        np.hypot(steady["velocity_x"], steady["velocity_y"]),
        steady["velocity"],
        rtol=1e-5,
        atol=1e-8,
    )


def test_axis_path_totals_match_on_axis_aligned_track(monkeypatch):
    monkeypatch.setattr(F, "KINEMATIC_THREE_FAMILIES", True)
    n_frames = 5
    step = 12.0
    out = F.extract_behavioral_features(_straight_path(n_frames=n_frames, step_px=step))
    last = out.iloc[-1]
    expected_path = (n_frames - 1) * step * F.LENGTH_SCALE
    assert np.isclose(last["distance_traveled"], expected_path, rtol=1e-6)
    assert np.isclose(last["distance_traveled_x"], expected_path, rtol=1e-6)
    assert np.isclose(last["distance_traveled_y"], 0.0, atol=1e-9)


def test_aggregate_includes_component_stats_when_enabled(monkeypatch):
    monkeypatch.setattr(F, "KINEMATIC_THREE_FAMILIES", True)
    feat = F.extract_behavioral_features(_straight_path())
    agg = F.aggregate_per_fly_features(feat)
    assert "mean_velocity_x" in agg.columns
    assert "mean_acceleration_x" in agg.columns
    assert "total_distance_x" in agg.columns
    assert len(F.classification_feature_columns()) == 26


def test_aggregate_mag_only_has_no_component_columns(monkeypatch):
    monkeypatch.setattr(F, "KINEMATIC_THREE_FAMILIES", False)
    feat = F.extract_behavioral_features(_straight_path())
    assert "velocity_x" not in feat.columns
    agg = F.aggregate_per_fly_features(feat)
    assert "mean_velocity_x" not in agg.columns
    assert len(F.classification_feature_columns()) == 9


def test_plot_titles_include_component_keys_when_enabled(monkeypatch):
    monkeypatch.setattr(F, "KINEMATIC_THREE_FAMILIES", True)
    titles = F.aggregate_feature_plot_titles()
    assert "mean_velocity_x" in titles
    assert "total_distance_y" in titles
