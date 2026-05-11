"""
src.wide_long
=============

Wide -> long format conversion for OC-SORT tracker output.

The tracker writes a "wide" CSV: one row per frame, one column per tracked id,
each cell holds the string ``'(x, y)'`` or NaN. Downstream code (vial
assignment, kinematics, classification) wants a "long" frame: one row per
``(frame, orig_id)`` observation with numeric ``x`` and ``y`` columns.

This module exists so the active pipeline does not have to import
``src.stitching`` (which is otherwise full of deprecated post-hoc stitching
machinery now living in ``legacy/stitching.py``).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def wide_to_long(df_wide: pd.DataFrame, out_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Reshape a wide tracker DataFrame into long format.

    Input (wide):  one row per frame, one column per tracked ID,
                   each cell holds ``'(x, y)'`` or NaN.
    Output (long): columns ``[frame, orig_id, x, y]``,
                   one row per (frame, id) observation.

    Parameters
    ----------
    df_wide : pandas.DataFrame
        Tracker output as written by ``export_tracks_xy_tuple_csv_one_config``.
    out_csv : str, optional
        If given, the long DataFrame is also written to this path as CSV.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame, sorted by ``orig_id`` then ``frame``.
    """
    id_cols = [c for c in df_wide.columns if c != "frame"]

    if not id_cols:
        return pd.DataFrame(columns=["frame", "orig_id", "x", "y"])

    stacked = df_wide.set_index("frame")[id_cols].stack().reset_index()
    stacked.columns = ["frame", "orig_id", "xy_raw"]

    coords = (
        stacked["xy_raw"]
        .str.strip("()")
        .str.split(",", expand=True)
        .astype(float)
    )
    stacked["x"] = coords[0]
    stacked["y"] = coords[1]

    df_long = (
        stacked
        .drop(columns="xy_raw")
        .sort_values(["orig_id", "frame"])
        .reset_index(drop=True)
    )
    if out_csv is not None:
        df_long.to_csv(out_csv, index=False)
    return df_long
