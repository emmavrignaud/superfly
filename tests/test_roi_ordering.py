"""
Unit tests for src.roi.assign_vials_and_ordered_ids.

Validates the left-to-right invariant: within a vial, ordered_ids are assigned
in increasing order of median x. Across vials, ordered_ids are globally unique
and increase in vial order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.roi import assign_vials_and_ordered_ids


def test_vial_assignment_and_global_ids(
    tmp_path: Path,
    mini_long_df_csv: Path,
    mini_vial_rois_json: Path,
) -> None:
    out_csv = tmp_path / "ordered.csv"
    df = assign_vials_and_ordered_ids(
        ocsort_csv=str(mini_long_df_csv),
        roi_json=str(mini_vial_rois_json),
        out_csv=str(out_csv),
        fps=30.0,
    )

    assert "vial_id" in df.columns
    assert "ordered_id" in df.columns
    assert "fps" in df.columns
    assert (df["fps"] == 30.0).all()

    expected = {"id1": "vial1", "id2": "vial2", "id3": "vial3"}
    actual = df.groupby("orig_id")["vial_id"].agg(lambda s: s.iloc[0]).to_dict()
    assert actual == expected

    ids = df.groupby("orig_id")["ordered_id"].first().to_dict()
    assert sorted(ids.values()) == [1, 2, 3], f"global ordered_ids must be 1..N, got {ids}"
    assert ids["id1"] < ids["id2"] < ids["id3"], (
        "ordered_id must increase left-to-right across vials"
    )


def test_csv_is_persisted(
    tmp_path: Path,
    mini_long_df_csv: Path,
    mini_vial_rois_json: Path,
) -> None:
    out_csv = tmp_path / "ordered.csv"
    assign_vials_and_ordered_ids(
        ocsort_csv=str(mini_long_df_csv),
        roi_json=str(mini_vial_rois_json),
        out_csv=str(out_csv),
        fps=30.0,
    )
    reloaded = pd.read_csv(out_csv)
    assert {"frame", "orig_id", "x", "y", "vial_id", "ordered_id", "fps"}.issubset(reloaded.columns)


def test_points_outside_all_vials_are_dropped(
    tmp_path: Path,
    mini_vial_rois_json: Path,
) -> None:
    """A point at x=200 falls outside all three ROIs and must be filtered out."""
    rows = [
        {"frame": 0, "orig_id": "in_vial",  "x": 10.0,  "y": 20.0},
        {"frame": 0, "orig_id": "in_vial",  "x": 11.0,  "y": 20.0},
        {"frame": 0, "orig_id": "out",      "x": 200.0, "y": 200.0},
    ]
    csv = tmp_path / "long.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    df = assign_vials_and_ordered_ids(
        ocsort_csv=str(csv),
        roi_json=str(mini_vial_rois_json),
        out_csv=str(tmp_path / "ordered.csv"),
        fps=30.0,
    )
    assert "out" not in df["orig_id"].unique()
