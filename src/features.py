"""
src/features.py

Behavioural feature extraction from ordered_tracks CSV.

Output units
------------
Controlled by the ``calibration`` block in config.yaml::

    calibration:
      px_per_cm: 29.0
      length_unit: cm  | px
      time_unit:   s   | frame

With the default (cm, s) every length-bearing column is in cm and every
time-bearing column is in s::

    step_distance, distance_traveled, dx, dy   cm
    velocity                                    cm/s
    acceleration                                cm/s^2
    angular_velocity                            rad/s
    area_covered                                cm^2
    tortuosity                                  dimensionless

Optional ``features.kinematic_three_families`` (see config.yaml) adds signed
``velocity_x`` / ``velocity_y`` with ``velocity`` as speed magnitude,
component accelerations plus vector magnitude ``acceleration``, and
``distance_traveled_x`` / ``distance_traveled_y`` (cumulative sum of |dx|, |dy|)
alongside ``distance_traveled`` (cumulative Euclidean step length).

The module-level ``UNITS`` dict spells the unit of each output column at
runtime so plots and reports can label axes without re-deriving the
convention.

Pipeline
--------
add_kinematics()              frame-level velocity, acceleration, turning angle
add_area_covered()            convex-hull area of each fly's trajectory
add_path_tortuosity()         total path length / net displacement
extract_behavioral_features() runs all three on the scaled positions, drops NaN/Inf
aggregate_per_fly_features()  collapses to one row per fly
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from pathlib import Path

import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from utils import load_config as _load_config  # noqa: E402

_cfg        = _load_config(_REPO_ROOT / "config.yaml")
fps         = float(_cfg.video.fallback_fps)
PX_PER_CM   = float(_cfg.calibration.px_per_cm)
LENGTH_UNIT = str(_cfg.calibration.length_unit)
TIME_UNIT   = str(_cfg.calibration.time_unit)

KINEMATIC_THREE_FAMILIES = bool(_cfg.features.kinematic_three_families)

if LENGTH_UNIT not in ("cm", "px"):
    raise ValueError(
        f"calibration.length_unit must be 'cm' or 'px', got {LENGTH_UNIT!r}"
    )
if TIME_UNIT not in ("s", "frame"):
    raise ValueError(
        f"calibration.time_unit must be 's' or 'frame', got {TIME_UNIT!r}"
    )

LENGTH_SCALE = (1.0 / PX_PER_CM) if LENGTH_UNIT == "cm" else 1.0
TIME_SCALE   = (1.0 / fps)       if TIME_UNIT   == "s"  else 1.0

UNITS: dict = {
    "length":           LENGTH_UNIT,
    "time":             TIME_UNIT,
    "velocity":         f"{LENGTH_UNIT}/{TIME_UNIT}",
    "acceleration":     f"{LENGTH_UNIT}/{TIME_UNIT}^2",
    "angular_velocity": f"rad/{TIME_UNIT}",
    "area":             f"{LENGTH_UNIT}^2",
    "tortuosity":       "dimensionless",
}


def classification_feature_columns() -> list[str]:
    """Column names on ``aggregate_per_fly_features`` output for box plots / reports.

    When ``features.kinematic_three_families`` is true, includes x, y, and
    magnitude aggregates for velocity, acceleration, and distance travelled.
    """
    if not KINEMATIC_THREE_FAMILIES:
        return [
            "mean_velocity",
            "median_velocity",
            "std_velocity",
            "pause_fraction",
            "mean_abs_turning_angle",
            "mean_abs_angular_velocity",
            "total_distance_traveled",
            "tortuosity",
            "area_covered",
        ]
    return [
        "mean_velocity_x",
        "mean_velocity_y",
        "mean_velocity",
        "median_velocity_x",
        "median_velocity_y",
        "median_velocity",
        "std_velocity_x",
        "std_velocity_y",
        "std_velocity",
        "mean_acceleration_x",
        "mean_acceleration_y",
        "mean_acceleration",
        "median_acceleration_x",
        "median_acceleration_y",
        "median_acceleration",
        "std_acceleration_x",
        "std_acceleration_y",
        "std_acceleration",
        "total_distance_x",
        "total_distance_y",
        "total_distance_traveled",
        "pause_fraction",
        "mean_abs_turning_angle",
        "mean_abs_angular_velocity",
        "tortuosity",
        "area_covered",
    ]


def aggregate_feature_plot_titles(keys: list[str] | None = None) -> dict[str, str]:
    """
    Plain-language plot titles for per-fly aggregates.

    Written for readers who haven't built this pipeline (e.g. a PI reviewing
    the report). Units come from ``UNITS`` (driven by ``calibration`` in
    config.yaml). For axis-resolved features (v_x, v_y, a_x, a_y, distance_x/y),
    sign convention is explained once in the report's glossary card; the
    titles themselves stay free of math notation.

    Reload ``src.features`` or restart the kernel after changing
    ``calibration`` in config.yaml.
    """
    v = UNITS["velocity"]
    a = UNITS["acceleration"]
    L = UNITS["length"]
    ar = UNITS["area"]
    w = UNITS["angular_velocity"]
    full: dict[str, str] = {
        "mean_velocity": f"Average speed ({v})",
        "median_velocity": f"Median speed ({v})",
        "std_velocity": f"Variability of speed ({v})",
        "pause_fraction": "Fraction of time paused",
        "mean_abs_turning_angle": "Average turning angle per step (radians)",
        "mean_abs_angular_velocity": f"Average turning rate ({w})",
        "total_distance_traveled": f"Total distance walked ({L})",
        "tortuosity": "Path tortuosity (path length ÷ straight-line distance)",
        "area_covered": f"Area explored ({ar})",
    }
    if KINEMATIC_THREE_FAMILIES:
        # Per-axis variants: titles match the magnitude case but tag the axis.
        # Sign convention is documented once in the report glossary.
        full.update({
            "mean_velocity_x":   f"Average horizontal velocity ({v})",
            "mean_velocity_y":   f"Average vertical velocity ({v})",
            "median_velocity_x": f"Median horizontal velocity ({v})",
            "median_velocity_y": f"Median vertical velocity ({v})",
            "std_velocity_x":    f"Variability of horizontal velocity ({v})",
            "std_velocity_y":    f"Variability of vertical velocity ({v})",
            "mean_acceleration_x":   f"Average horizontal acceleration ({a})",
            "mean_acceleration_y":   f"Average vertical acceleration ({a})",
            "mean_acceleration":     f"Average acceleration magnitude ({a})",
            "median_acceleration_x": f"Median horizontal acceleration ({a})",
            "median_acceleration_y": f"Median vertical acceleration ({a})",
            "median_acceleration":   f"Median acceleration magnitude ({a})",
            "std_acceleration_x":    f"Variability of horizontal acceleration ({a})",
            "std_acceleration_y":    f"Variability of vertical acceleration ({a})",
            "std_acceleration":      f"Variability of acceleration magnitude ({a})",
            "total_distance_x":      f"Horizontal distance walked ({L})",
            "total_distance_y":      f"Vertical distance walked ({L})",
        })
    if keys is None:
        return full.copy()
    return {k: full[k] for k in keys if k in full}


def feature_families() -> list[tuple[str, list[str]]]:
    """
    Group axis-resolved features into 3-column (horizontal, vertical, magnitude)
    rows for the report layout.

    Returns ``[(row_title, [col_horizontal, col_vertical, col_magnitude]), ...]``.
    Empty when ``features.kinematic_three_families`` is false.

    The row title is unitless; per-column titles supplied by
    ``aggregate_feature_plot_titles`` carry the units.
    """
    if not KINEMATIC_THREE_FAMILIES:
        return []
    return [
        ("Average velocity",        ["mean_velocity_x",   "mean_velocity_y",   "mean_velocity"]),
        ("Median velocity",         ["median_velocity_x", "median_velocity_y", "median_velocity"]),
        ("Variability of velocity", ["std_velocity_x",    "std_velocity_y",    "std_velocity"]),
        ("Average acceleration",        ["mean_acceleration_x",   "mean_acceleration_y",   "mean_acceleration"]),
        ("Median acceleration",         ["median_acceleration_x", "median_acceleration_y", "median_acceleration"]),
        ("Variability of acceleration", ["std_acceleration_x",    "std_acceleration_y",    "std_acceleration"]),
        ("Distance walked",             ["total_distance_x",      "total_distance_y",      "total_distance_traveled"]),
    ]


def singleton_features() -> list[str]:
    """Features that are not part of an x/y/magnitude family (one figure each in the report)."""
    out = ["pause_fraction", "mean_abs_turning_angle", "mean_abs_angular_velocity",
           "tortuosity", "area_covered"]
    if not KINEMATIC_THREE_FAMILIES:
        # When families are off, the magnitude-only versions live as singletons.
        out = ["mean_velocity", "median_velocity", "std_velocity",
               "total_distance_traveled"] + out
    return out


# ---------------------------------------------------------------------------
# Geometric augmentation (rows, not columns)
# ---------------------------------------------------------------------------

GEOMETRIC_TRANSFORMS: dict = {
    "identity": lambda x, y: (x.copy(), y.copy()),
    "flip_x":   lambda x, y: (-x, y.copy()),
    "flip_y":   lambda x, y: (x.copy(), -y),
    "flip_xy":  lambda x, y: (-x, -y),
}

DEFAULT_GEOMETRIC_TRANSFORMS: list[str] = ["identity", "flip_x", "flip_y", "flip_xy"]

import re as _re

_ROTATION_NAME = _re.compile(r"^rot(-?\d+(?:\.\d+)?)$")


def _resolve_geometric_transform(name: str):
    """Return the (x, y) -> (x', y') callable for a transform name.

    Recognises the four named entries in ``GEOMETRIC_TRANSFORMS`` plus dynamic
    rotation entries of the form ``rot<angle_in_degrees>`` (e.g. ``rot15``,
    ``rot-30``, ``rot45.5``). Rotations are applied around each fly's mean
    (x, y) so the shape rotates in place; absolute translation is reapplied so
    feature extraction (which only uses diffs) is unaffected by the centering.
    """
    if name in GEOMETRIC_TRANSFORMS:
        return GEOMETRIC_TRANSFORMS[name]
    m = _ROTATION_NAME.match(name)
    if not m:
        raise ValueError(
            f"unknown geometric transform: {name!r}; "
            f"supported: {sorted(GEOMETRIC_TRANSFORMS)} or rot<angle> (e.g. rot15, rot-30)"
        )
    theta = np.deg2rad(float(m.group(1)))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def _rotate(x: np.ndarray, y: np.ndarray):
        cx, cy = float(np.mean(x)), float(np.mean(y))
        dx, dy = x - cx, y - cy
        return cos_t * dx - sin_t * dy + cx, sin_t * dx + cos_t * dy + cy

    return _rotate


def augment_trajectories_geometric(
    df_raw: pd.DataFrame,
    transforms: list[str] | None = None,
    group_col: str = "ordered_id",
) -> pd.DataFrame:
    """
    Apply label-preserving geometric transforms to raw (x, y) trajectories.

    Each transform produces a new copy of every trajectory with ordered_id
    relabelled ``{original}__aug:{name}`` and all other columns preserved.
    A new ``aug_group`` column carries the original ``ordered_id`` so callers
    can pass it as ``groups`` to ``GroupKFold`` and keep augmented copies of
    the same fly in a single CV fold (no leakage across folds).

    Supported transforms:
        identity  (x, y) -> (x, y)
        flip_x    (x, y) -> (-x, y)
        flip_y    (x, y) -> (x, -y)
        flip_xy   (x, y) -> (-x, -y)              # 180 deg rotation
        rot<deg>  arbitrary-angle rotation around each fly's mean (x, y);
                  examples: rot15, rot-30, rot45.5. The mean is restored so the
                  fly stays in its original neighbourhood; feature extraction
                  uses only diffs, so the per-fly centering choice doesn't
                  affect the resulting aggregates.

    Feature impact (with ``features.kinematic_three_families``):
    Sign-flips affect the per-axis means/medians of velocity/acceleration;
    magnitudes (std, totals, |v|, |a|, area, tortuosity, pause) are invariant.
    Non-trivial rotations DO mix x and y components, so signed axis aggregates
    change in a non-degenerate way.
    """
    if transforms is None:
        transforms = DEFAULT_GEOMETRIC_TRANSFORMS
    if not transforms:
        raise ValueError("transforms must be a non-empty list of transform names")
    # Validate up-front so a typo in the middle of a long list aborts cleanly.
    resolved = [(name, _resolve_geometric_transform(name)) for name in transforms]

    pieces: list[pd.DataFrame] = []
    for name, fn in resolved:
        sub = df_raw.copy()
        if name == "identity":
            new_x = sub["x"].to_numpy().copy()
            new_y = sub["y"].to_numpy().copy()
        else:
            # Per-fly application so rotations are around each fly's own mean.
            new_x = np.empty(len(sub), dtype=float)
            new_y = np.empty(len(sub), dtype=float)
            for fid, idx in sub.groupby(group_col).indices.items():
                xi = sub["x"].to_numpy()[idx]
                yi = sub["y"].to_numpy()[idx]
                rx, ry = fn(xi, yi)
                new_x[idx] = rx
                new_y[idx] = ry
        sub["x"] = new_x
        sub["y"] = new_y
        sub["aug_group"] = sub[group_col].astype(str)
        if name != "identity":
            sub[group_col] = sub[group_col].astype(str) + f"__aug:{name}"
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True)


def augment_trajectories_masking(
    df_raw: pd.DataFrame,
    mask_fraction: float = 0.10,
    group_col: str = "ordered_id",
) -> pd.DataFrame:
    """
    Block-mask each tracklet into ``N = round(1 / mask_fraction)`` copies.

    Each fly's frames (sorted by ``frame``) are split into N consecutive
    contiguous chunks of equal length. Copy ``k`` drops chunk ``k`` and keeps
    the rest, so the N copies together cover every block exactly once
    (temporal jackknife). With ``mask_fraction = 0.10`` you get 10 copies per
    fly, each missing a different 10% chunk of the run; with 0.50 you get 2
    copies (first half / second half dropped).

    Each masked copy gets a unique ordered_id ``{orig}__aug:maskblock<k>`` and
    carries ``aug_group`` = original ordered_id so callers can pass it as
    ``groups`` to GroupKFold and keep masked twins in a single CV fold.
    The original (unmasked) trajectory is **not** included; pair with
    ``augment_trajectories_geometric([identity, ...])`` to keep it.
    """
    if not (0.0 < mask_fraction < 1.0):
        raise ValueError(f"mask_fraction must be in (0, 1); got {mask_fraction}")
    n_copies = int(round(1.0 / mask_fraction))
    if n_copies < 2:
        raise ValueError(
            f"mask_fraction={mask_fraction} yields {n_copies} copies; "
            f"use a smaller fraction so 1/fraction >= 2"
        )

    base = df_raw.copy()
    base["aug_group"] = base[group_col].astype(str)
    # Sort within each fly by frame so the block split lines up with time.
    base = base.sort_values([group_col, "frame"], kind="stable").reset_index(drop=True)
    groups = list(base.groupby(group_col, sort=False).indices.items())

    pieces: list[pd.DataFrame] = []
    for k in range(n_copies):
        kept_idx_all: list[np.ndarray] = []
        for _fid, idx in groups:
            idx_arr = np.asarray(idx)
            n = len(idx_arr)
            if n <= 1:
                kept_idx_all.append(idx_arr)
                continue
            # Contiguous block [start, end) for copy k — np.array_split handles
            # uneven lengths so every frame belongs to exactly one block.
            block_starts = np.linspace(0, n, n_copies + 1, dtype=int)
            start, end = block_starts[k], block_starts[k + 1]
            if start >= end:
                kept_idx_all.append(idx_arr)
                continue
            keep = np.concatenate([np.arange(0, start), np.arange(end, n)])
            kept_idx_all.append(idx_arr[keep])
        kept_idx = np.concatenate(kept_idx_all) if kept_idx_all else np.array([], dtype=int)
        sub = base.iloc[kept_idx].copy()
        sub[group_col] = sub[group_col].astype(str) + f"__aug:maskblock{k}"
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True)


def add_kinematics(df: pd.DataFrame, group_col: str = "ordered_id") -> pd.DataFrame:
    """
    Add frame-level kinematic columns to df, computed per fly.

    Length-derived columns inherit the unit of df["x"], df["y"] (the function
    operates relatively); time is converted from frame index to the configured
    time unit using TIME_SCALE. The first row of each fly group has no
    predecessor so its diff is filled with 0.

    Columns added (L = length unit of df["x"]; T = configured time unit)
    --------------------------------------------------------------------
    dx, dy             displacement in x and y between consecutive frames (L)
    dt                 time elapsed between consecutive frames (T)
    step_distance      Euclidean distance moved in one step: sqrt(dx^2 + dy^2) (L)
    velocity           step_distance / dt = |v| (L/T)
    heading            direction of movement in radians: arctan2(dy, dx)
    turning_angle      change in heading between consecutive steps (rad),
                       wrapped to [-pi, pi]
    angular_velocity   turning_angle / dt (rad/T)

    When ``KINEMATIC_THREE_FAMILIES`` is true, also:
    velocity_x, velocity_y     signed (dx/dt, dy/dt); velocity = hypot(vx, vy)
    acceleration_x, acceleration_y  d(velocity_x)/dt, d(velocity_y)/dt
    acceleration           hypot(acceleration_x, acceleration_y) (vector-derivative magnitude)
    distance_traveled_x, distance_traveled_y  cumulative sum(|dx|), sum(|dy|)
    distance_traveled        cumulative sum(step_distance) unchanged

    When false, ``acceleration`` is d(velocity)/dt with velocity = |v| (scalar
    along-path derivative).
    """
    df = df.sort_values([group_col, "frame"]).copy()

    df["dx"] = df.groupby(group_col)["x"].diff().fillna(0)
    df["dy"] = df.groupby(group_col)["y"].diff().fillna(0)
    df["dt"] = df.groupby(group_col)["frame"].diff().fillna(1) * TIME_SCALE

    df["step_distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["velocity"] = df["step_distance"] / df["dt"]

    if KINEMATIC_THREE_FAMILIES:
        df["velocity_x"] = df["dx"] / df["dt"]
        df["velocity_y"] = df["dy"] / df["dt"]
        df["acceleration_x"] = df.groupby(group_col)["velocity_x"].diff().fillna(0) / df["dt"]
        df["acceleration_y"] = df.groupby(group_col)["velocity_y"].diff().fillna(0) / df["dt"]
        df["acceleration"] = np.hypot(df["acceleration_x"], df["acceleration_y"])
        df["distance_traveled_x"] = df.groupby(group_col)["dx"].transform(lambda s: s.abs().cumsum())
        df["distance_traveled_y"] = df.groupby(group_col)["dy"].transform(lambda s: s.abs().cumsum())
    else:
        df["acceleration"] = df.groupby(group_col)["velocity"].diff().fillna(0) / df["dt"]

    df["distance_traveled"] = df.groupby(group_col)["step_distance"].cumsum()

    df["heading"] = np.arctan2(df["dy"], df["dx"])
    dtheta = df.groupby(group_col)["heading"].diff().fillna(0)
    df["turning_angle"] = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    df["angular_velocity"] = df["turning_angle"] / df["dt"]

    return df


def add_area_covered(df: pd.DataFrame, group_col: str = "ordered_id") -> pd.DataFrame:
    """
    Add an ``area_covered`` column to df.

    Area is computed in (length unit of df["x"])^2. ``extract_behavioral_features``
    pre-scales x and y to the configured length unit before calling this, so the
    default cm/s setup produces area_covered in cm^2. Falls back to bounding-box
    area if the convex hull fails (e.g. all points collinear). Returns 0.0 for
    tracklets with fewer than 3 points.
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
                area = x_range * y_range
        records.append((fly_id, area))

    area_df = pd.DataFrame(records, columns=[group_col, "area_covered"])
    return df.merge(area_df, on=group_col, how="left")


def add_path_tortuosity(df: pd.DataFrame, group_col: str = "ordered_id") -> pd.DataFrame:
    """
    Add a ``tortuosity`` column to df.

    Dimensionless: total path length / straight-line displacement. As long as
    numerator and denominator are in the same length unit (they are: both use
    df["x"], df["y"]) the ratio is unit-invariant. Set to NaN when net
    displacement is zero.
    """
    records = []
    for fly_id, fly_detections in df.groupby(group_col):
        total_path_length = fly_detections["step_distance"].sum()
        net_displacement = np.sqrt(
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
    """
    Run all feature extractors and drop NaN/Inf rows.

    Scales df["x"], df["y"] to the configured length unit (LENGTH_SCALE) before
    running the per-frame and per-fly extractors, so every downstream length is
    consistent with the calibration block in config.yaml.
    """
    df = df.copy()
    df["x"] = df["x"].astype(float) * LENGTH_SCALE
    df["y"] = df["y"].astype(float) * LENGTH_SCALE
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
        Velocity below which a frame is counted as a pause, expressed in the
        configured velocity unit (UNITS["velocity"], default cm/s).
        With the default px_per_cm=29, pause_threshold=1.0 cm/s == ~29 px/s.
        Always uses speed magnitude ``velocity`` (|v|), not component speeds.

    Returns
    -------
    pd.DataFrame with one row per group_col and summary statistics.
    If the input has a ``vial_id`` column, each row also carries that fly's
    ``vial_id`` (for downstream plots that order genotypes by vial).
    """
    grouped = df.groupby(group_col)

    def _per_fly_row(g: pd.DataFrame) -> pd.Series:
        row = {
            "mean_velocity": g["velocity"].mean(),
            "median_velocity": g["velocity"].median(),
            "std_velocity": g["velocity"].std(),
            "pause_fraction": (g["velocity"] < pause_threshold).mean(),
            "mean_abs_turning_angle": g["turning_angle"].abs().mean(),
            "mean_abs_angular_velocity": g["angular_velocity"].abs().mean(),
            "total_distance_traveled": g["distance_traveled"].iloc[-1],
            "tortuosity": g["tortuosity"].iloc[0],
            "area_covered": g["area_covered"].iloc[0],
        }
        if KINEMATIC_THREE_FAMILIES:
            row.update({
                "mean_velocity_x": g["velocity_x"].mean(),
                "mean_velocity_y": g["velocity_y"].mean(),
                "median_velocity_x": g["velocity_x"].median(),
                "median_velocity_y": g["velocity_y"].median(),
                "std_velocity_x": g["velocity_x"].std(),
                "std_velocity_y": g["velocity_y"].std(),
                "mean_acceleration_x": g["acceleration_x"].mean(),
                "mean_acceleration_y": g["acceleration_y"].mean(),
                "mean_acceleration": g["acceleration"].mean(),
                "median_acceleration_x": g["acceleration_x"].median(),
                "median_acceleration_y": g["acceleration_y"].median(),
                "median_acceleration": g["acceleration"].median(),
                "std_acceleration_x": g["acceleration_x"].std(),
                "std_acceleration_y": g["acceleration_y"].std(),
                "std_acceleration": g["acceleration"].std(),
                "total_distance_x": g["distance_traveled_x"].iloc[-1],
                "total_distance_y": g["distance_traveled_y"].iloc[-1],
            })
        if "vial_id" in g.columns and len(g) > 0:
            row["vial_id"] = g["vial_id"].iloc[0]
        return pd.Series(row)

    return grouped.apply(_per_fly_row).reset_index()
