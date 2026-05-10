"""
src/features.py

Behavioural feature extraction from ordered_tracks CSV.

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
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

fps = cfg['stitching']['fps']

def add_kinematics(df: pd.DataFrame, group_col: str = "ordered_id") -> pd.DataFrame:
    """
    Add frame-level kinematic columns to df, computed per fly.

    Each column is derived from the difference between consecutive frames.
    The first row of each fly group is a baseline — its diff is undefined,
    so it is filled with 0 (no movement assumed).

    Columns added
    -------------
    dx, dy             — displacement in x and y between consecutive frames (px)
    dt                 — time elapsed between consecutive frames (s)
    step_distance      — Euclidean distance moved in one step: sqrt(dx² + dy²)
    velocity           — speed in px/s: step_distance / dt
    acceleration       — change in velocity per second: velocity.diff() / dt
    distance_traveled  — cumulative path length from the start of the tracklet
    heading            — direction of movement in radians: arctan2(dy, dx)
    turning_angle      — change in heading between consecutive steps (rad),
                         wrapped to [-π, π] to handle reversals across the ±π boundary
    angular_velocity   — rate of turning in rad/s: turning_angle / dt
    """
    df = df.sort_values([group_col, "frame"]).copy()

    df["dx"] = df.groupby(group_col)["x"].diff().fillna(0)
    df["dy"] = df.groupby(group_col)["y"].diff().fillna(0)
    df["dt"] = df.groupby(group_col)["frame"].diff().fillna(1) / fps

    df["step_distance"]     = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["velocity"]          = df["step_distance"] / df["dt"]
    df["acceleration"]      = df.groupby(group_col)["velocity"].diff().fillna(0) / df["dt"]
    df["distance_traveled"] = df.groupby(group_col)["step_distance"].cumsum()

    df["heading"]         = np.arctan2(df["dy"], df["dx"])
    dtheta                = df.groupby(group_col)["heading"].diff().fillna(0)
    df["turning_angle"]   = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    df["angular_velocity"] = df["turning_angle"] / df["dt"]

    return df


def add_area_covered(df: pd.DataFrame, group_col: str = "ordered_id") -> pd.DataFrame:
    """
    Add an 'area_covered' column to df.

    Area covered is the convex hull area of all (x, y) positions visited by
    the fly — a measure of how much of the vial it explored. Falls back to
    bounding box area if the hull fails (e.g. all points collinear). Returns
    0.0 for tracklets with fewer than 3 points.
    """
    records = []
    for fly_id, fly_detections in df.groupby(group_col):
        positions = fly_detections[["x", "y"]].values
        if len(positions) < 3:
            area = 0.0
        else:
            try:
                area = ConvexHull(positions).volume
            except Exception:
                x_range = fly_detections["x"].max() - fly_detections["x"].min()
                y_range = fly_detections["y"].max() - fly_detections["y"].min()
                area    = x_range * y_range
        records.append((fly_id, area))

    area_df = pd.DataFrame(records, columns=[group_col, "area_covered"])
    return df.merge(area_df, on=group_col, how="left")


def add_path_tortuosity(df: pd.DataFrame, group_col: str = "ordered_id") -> pd.DataFrame:
    """
    Add a 'tortuosity' column to df.

    Tortuosity = total path length / net displacement (straight-line distance
    from start to end). 1.0 means perfectly straight; higher values mean a
    more winding path. Set to NaN when net displacement is zero.
    """
    records = []
    for fly_id, fly_detections in df.groupby(group_col):
        total_path_length = fly_detections["step_distance"].sum()
        net_displacement  = np.sqrt(
            (fly_detections["x"].iloc[-1] - fly_detections["x"].iloc[0]) ** 2
            + (fly_detections["y"].iloc[-1] - fly_detections["y"].iloc[0]) ** 2
        )
        tortuosity = total_path_length / net_displacement if net_displacement > 0 else np.nan
        records.append((fly_id, tortuosity))

    tortuosity_df = pd.DataFrame(records, columns=[group_col, "tortuosity"])
    return df.merge(tortuosity_df, on=group_col, how="left")


def extract_behavioral_features(
    df: pd.DataFrame,
    group_col: str = "ordered_id",
) -> pd.DataFrame:
    """Run all feature extractors and drop NaN/Inf rows."""
    df = add_kinematics(df, group_col=group_col)
    df = add_area_covered(df, group_col=group_col)
    df = add_path_tortuosity(df, group_col=group_col)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.reset_index(drop=True)


def aggregate_per_fly_features(
    df: pd.DataFrame,
    group_col: str = "ordered_id",
    pause_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Collapse frame-level features to one row per fly.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract_behavioral_features().
    group_col : str
        Column to group by (default: "ordered_id").
    pause_threshold : float
        Velocity (px/s) below which a frame is counted as a pause.

    Returns
    -------
    pd.DataFrame with one row per group_col and summary statistics.
    If the input has a ``vial_id`` column, each row also carries that fly's
    ``vial_id`` (for downstream plots that order genotypes by vial).
    """
    grouped = df.groupby(group_col)

    def _per_fly_row(g: pd.DataFrame) -> pd.Series:
        row = {
            "mean_velocity":             g["velocity"].mean(),
            "median_velocity":           g["velocity"].median(),
            "std_velocity":              g["velocity"].std(),
            "pause_fraction":            (g["velocity"] < pause_threshold).mean(),
            "mean_abs_turning_angle":    g["turning_angle"].abs().mean(),
            "mean_abs_angular_velocity": g["angular_velocity"].abs().mean(),
            "total_distance_traveled":   g["distance_traveled"].iloc[-1],
            "tortuosity":                g["tortuosity"].iloc[0],
            "area_covered":              g["area_covered"].iloc[0],
        }
        if "vial_id" in g.columns and len(g) > 0:
            row["vial_id"] = g["vial_id"].iloc[0]
        return pd.Series(row)

    return grouped.apply(_per_fly_row).reset_index()