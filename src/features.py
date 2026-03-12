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

import warnings
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute frame-level kinematics per compact_id."""
    df = df.sort_values(["compact_id", "frame"]).copy()
    fps = float(df["fps"].iloc[0])

    df["dx"] = df.groupby("compact_id")["x"].diff().fillna(0)
    df["dy"] = df.groupby("compact_id")["y"].diff().fillna(0)
    df["dt"] = df.groupby("compact_id")["frame"].diff().fillna(1) / fps

    df["step_distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["velocity"] = df["step_distance"] / df["dt"]
    df["acceleration"] = (
        df.groupby("compact_id")["velocity"].diff().fillna(0) / df["dt"]
    )

    df["distance_traveled"] = df.groupby("compact_id")["step_distance"].cumsum()
    df["heading"] = np.arctan2(df["dy"], df["dx"])
    dtheta = df.groupby("compact_id")["heading"].diff().fillna(0)
    df["turning_angle"] = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    df["angular_velocity"] = df["turning_angle"] / df["dt"]

    return df


def add_area_covered(df: pd.DataFrame) -> pd.DataFrame:
    """Add convex-hull area of each fly's trajectory."""
    records = []
    for cid, g in df.groupby("compact_id"):
        pts = g[["x", "y"]].values
        if len(pts) < 3:
            area = 0.0
        else:
            try:
                area = ConvexHull(pts).volume
            except Exception:
                area = (g["x"].max() - g["x"].min()) * (g["y"].max() - g["y"].min())
        records.append((cid, area))

    area_df = pd.DataFrame(records, columns=["compact_id", "area_covered"])
    return df.merge(area_df, on="compact_id", how="left")


def add_path_tortuosity(df: pd.DataFrame) -> pd.DataFrame:
    """Add tortuosity = total path length / net displacement."""
    records = []
    for cid, g in df.groupby("compact_id"):
        total = g["step_distance"].sum()
        net = np.sqrt(
            (g["x"].iloc[-1] - g["x"].iloc[0]) ** 2
            + (g["y"].iloc[-1] - g["y"].iloc[0]) ** 2
        )
        value = total / net if net > 0 else np.nan
        records.append((cid, value))

    tort_df = pd.DataFrame(records, columns=["compact_id", "tortuosity"])
    return df.merge(tort_df, on="compact_id", how="left")


def extract_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature extractors and drop NaN/Inf rows."""
    df = add_kinematics(df)
    df = add_area_covered(df)
    df = add_path_tortuosity(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.reset_index(drop=True)


def aggregate_per_fly_features(
    df: pd.DataFrame,
    pause_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Collapse frame-level features to one row per fly.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract_behavioral_features().
    pause_threshold : float
        Velocity (px/s) below which a frame is counted as a pause.

    Returns
    -------
    pd.DataFrame with one row per compact_id and summary statistics.
    """
    grouped = df.groupby("compact_id")

    return grouped.apply(
        lambda g: pd.Series({
            "mean_velocity": g["velocity"].mean(),
            "median_velocity": g["velocity"].median(),
            "std_velocity": g["velocity"].std(),
            "pause_fraction": (g["velocity"] < pause_threshold).mean(),
            "mean_abs_turning_angle": g["turning_angle"].abs().mean(),
            "mean_abs_angular_velocity": g["angular_velocity"].abs().mean(),
            "total_distance_traveled": g["distance_traveled"].iloc[-1],
            "tortuosity": g["tortuosity"].iloc[0],
            "area_covered": g["area_covered"].iloc[0],
        })
    ).reset_index()
