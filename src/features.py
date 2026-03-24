"""
src/features.py

Behavioural feature extraction from compact_tracks CSV.

Pipeline
--------
add_kinematics()          — frame-level velocity, acceleration, turning angle
add_area_covered()        — convex-hull area of each fly's trajectory
add_path_tortuosity()     — total path length / net displacement
extract_behavioral_features()  — runs all three, drops NaN/Inf
aggregate_per_fly_features()   — collapses to one row per fly
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def add_kinematics(df: pd.DataFrame, group_col: str = "compact_id") -> pd.DataFrame:
    """Compute frame-level kinematics per group_col."""
    df = df.sort_values([group_col, "frame"]).copy()
    fps = float(df["fps"].iloc[0])

    df["dx"] = df.groupby(group_col)["x"].diff().fillna(0)
    df["dy"] = df.groupby(group_col)["y"].diff().fillna(0)
    df["dt"] = df.groupby(group_col)["frame"].diff().fillna(1) / fps

    df["step_distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["velocity"] = df["step_distance"] / df["dt"]
    df["acceleration"] = (
        df.groupby(group_col)["velocity"].diff().fillna(0) / df["dt"]
    )

    df["distance_traveled"] = df.groupby(group_col)["step_distance"].cumsum()
    df["heading"] = np.arctan2(df["dy"], df["dx"])
    dtheta = df.groupby(group_col)["heading"].diff().fillna(0)
    df["turning_angle"] = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    df["angular_velocity"] = df["turning_angle"] / df["dt"]

    return df


def add_area_covered(df: pd.DataFrame, group_col: str = "compact_id") -> pd.DataFrame:
    """Add convex-hull area of each fly's trajectory."""
    records = []
    for cid, g in df.groupby(group_col):
        pts = g[["x", "y"]].values
        if len(pts) < 3:
            area = 0.0
        else:
            try:
                area = ConvexHull(pts).volume
            except Exception:
                area = (g["x"].max() - g["x"].min()) * (g["y"].max() - g["y"].min())
        records.append((cid, area))

    area_df = pd.DataFrame(records, columns=[group_col, "area_covered"])
    return df.merge(area_df, on=group_col, how="left")


def add_path_tortuosity(df: pd.DataFrame, group_col: str = "compact_id") -> pd.DataFrame:
    """Add tortuosity = total path length / net displacement."""
    records = []
    for cid, g in df.groupby(group_col):
        total = g["step_distance"].sum()
        net = np.sqrt(
            (g["x"].iloc[-1] - g["x"].iloc[0]) ** 2
            + (g["y"].iloc[-1] - g["y"].iloc[0]) ** 2
        )
        value = total / net if net > 0 else np.nan
        records.append((cid, value))

    tort_df = pd.DataFrame(records, columns=[group_col, "tortuosity"])
    return df.merge(tort_df, on=group_col, how="left")


def extract_behavioral_features(
    df: pd.DataFrame,
    group_col: str = "compact_id",
) -> pd.DataFrame:
    """Run all feature extractors and drop NaN/Inf rows."""
    df = add_kinematics(df, group_col=group_col)
    df = add_area_covered(df, group_col=group_col)
    df = add_path_tortuosity(df, group_col=group_col)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.reset_index(drop=True)


def aggregate_per_fly_features(
    df: pd.DataFrame,
    group_col: str = "compact_id",
    pause_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Collapse frame-level features to one row per fly.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract_behavioral_features().
    group_col : str
        Column to group by (default: "compact_id").
    pause_threshold : float
        Velocity (px/s) below which a frame is counted as a pause.

    Returns
    -------
    pd.DataFrame with one row per group_col and summary statistics.
    """
    grouped = df.groupby(group_col)

    return grouped.apply(
        lambda g: pd.Series({
            "mean_velocity":             g["velocity"].mean(),
            "median_velocity":           g["velocity"].median(),
            "std_velocity":              g["velocity"].std(),
            "pause_fraction":            (g["velocity"] < pause_threshold).mean(),
            "mean_abs_turning_angle":    g["turning_angle"].abs().mean(),
            "mean_abs_angular_velocity": g["angular_velocity"].abs().mean(),
            "total_distance_traveled":   g["distance_traveled"].iloc[-1],
            "tortuosity":                g["tortuosity"].iloc[0],
            "area_covered":              g["area_covered"].iloc[0],
        })
    ).reset_index()