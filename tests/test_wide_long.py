"""
Unit tests for src.wide_long.wide_to_long.

Round-trips a synthetic wide-format DataFrame and validates the long-format
output matches what the rest of the pipeline expects (frame, orig_id, x, y).
"""

from __future__ import annotations

import pandas as pd

from src.wide_long import wide_to_long


def test_columns_and_dtypes(mini_wide_df: pd.DataFrame) -> None:
    long_df = wide_to_long(mini_wide_df)
    assert list(long_df.columns) == ["frame", "orig_id", "x", "y"]
    assert long_df["x"].dtype.kind == "f"
    assert long_df["y"].dtype.kind == "f"


def test_nan_cells_pass_through_with_nan_xy(mini_wide_df: pd.DataFrame) -> None:
    """Empty cells in the wide CSV become rows with NaN x/y in the long output.

    Downstream consumers (assign_vials_and_ordered_ids) filter these out via
    the vial-membership check, so passing them through is harmless. This test
    pins that contract: 4 frames x 3 ids = 12 rows, regardless of NaN.
    """
    long_df = wide_to_long(mini_wide_df)
    assert len(long_df) == 12
    assert long_df["x"].isna().sum() == 2
    assert long_df["y"].isna().sum() == 2


def test_per_track_counts(mini_wide_df: pd.DataFrame) -> None:
    long_df = wide_to_long(mini_wide_df)
    counts = long_df.groupby("orig_id").size().to_dict()
    assert counts == {"id1": 4, "id2": 4, "id3": 4}


def test_non_nan_observations_per_track(mini_wide_df: pd.DataFrame) -> None:
    """Among the rows with finite x/y, id1 has 4 valid points and id2/id3 have 3."""
    long_df = wide_to_long(mini_wide_df).dropna(subset=["x", "y"])
    counts = long_df.groupby("orig_id").size().to_dict()
    assert counts == {"id1": 4, "id2": 3, "id3": 3}


def test_coordinate_parsing(mini_wide_df: pd.DataFrame) -> None:
    long_df = wide_to_long(mini_wide_df)
    id1_frame_0 = long_df[(long_df["orig_id"] == "id1") & (long_df["frame"] == 0)].iloc[0]
    assert id1_frame_0["x"] == 10.0
    assert id1_frame_0["y"] == 20.0


def test_sorted_by_id_then_frame(mini_wide_df: pd.DataFrame) -> None:
    long_df = wide_to_long(mini_wide_df)
    for _, group in long_df.groupby("orig_id"):
        frames = group["frame"].tolist()
        assert frames == sorted(frames), f"frames within an id must be ascending: {frames}"


def test_writes_csv_when_path_given(tmp_path, mini_wide_df: pd.DataFrame) -> None:
    out = tmp_path / "long.csv"
    wide_to_long(mini_wide_df, out_csv=str(out))
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == ["frame", "orig_id", "x", "y"]


def test_empty_input_returns_empty_frame() -> None:
    empty = pd.DataFrame({"frame": [0, 1, 2]})
    long_df = wide_to_long(empty)
    assert long_df.empty
    assert list(long_df.columns) == ["frame", "orig_id", "x", "y"]
