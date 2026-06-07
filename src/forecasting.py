"""
src/forecasting.py

Next-step displacement forecasting with genotype conditioning.

The model takes a sliding window of W frames of movement features
(dx, dy, speed, heading, angular_velocity) and predicts the next
(dx, dy) displacement.  Genotype is concatenated as a one-hot
conditioning vector.

Scientific question
-------------------
Does knowing a fly's genotype reduce prediction error?  If yes, the
mutations produce genuinely distinct movement dynamics — not just
different average speeds, but different sequential structure.

Architecture
------------
MLP: (W * n_feat + n_genotypes) → 256 → 128 → 2
Loss: MSE on (dx, dy)
Optimizer: Adam (pure numpy, same style as autoencoder.py)
Baseline: constant-velocity (repeat last (dx, dy) as prediction)

Ablation: train once with, once without genotype conditioning.
If |Δ MSE| > noise, genotype encodes something about movement dynamics.

Public functions
----------------
load_all_tracks()          load GT + auto-tracked, assign genotypes
extract_windows()          sliding-window dataset from track DataFrame
train_forecaster()         fit MLP, return model + training info
evaluate_forecaster()      per-genotype MSE vs baseline
make_forecast_report()     HTML report with Plotly figures
run_forecasting_analysis() end-to-end pipeline
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config():
    import sys
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from utils import load_config
    return load_config(_REPO_ROOT / "config.yaml")

_cfg        = _load_config()
_PX_PER_CM  = float(_cfg.calibration.px_per_cm)
_FPS        = float(_cfg.video.fallback_fps)
_LENGTH_SCALE = 1.0 / _PX_PER_CM  # px → cm
_TIME_SCALE   = 1.0 / _FPS        # frame → s

VIAL_TO_GENO = {
    "vial1": "WT",
    "vial2": "A90V",
    "vial3": "G287S",
    "vial4": "G294A",
    "vial5": "A315T",
    "vial6": "M337V",
}

# Default genotype order (consistent with classification.py)
GENOTYPE_ORDER = ["WT", "A90V", "G287S", "G294A", "A315T", "M337V"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _assign_vial(x: float, vial_rois: dict) -> str | None:
    """Return vial name whose x-span contains `x`, or None."""
    for vname, bbox in vial_rois.items():
        x1, _y1, x2, _y2 = bbox
        if x1 <= x <= x2:
            return vname
    return None


def _kinematics_from_xy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-fly DataFrame with columns [frame, x, y], compute
    dx, dy, speed, heading, angular_velocity per frame.

    The first frame of each track gets NaN kinematics and is dropped.
    Coordinates are converted to cm; time to seconds.
    """
    df = df.sort_values("frame").copy()
    df["x_cm"] = df["x"] * _LENGTH_SCALE
    df["y_cm"] = df["y"] * _LENGTH_SCALE

    df["dx"] = df["x_cm"].diff()
    df["dy"] = df["y_cm"].diff()
    df["frame_gap"] = df["frame"].diff()

    # Scale by actual frame gap (handles missing frames)
    valid = df["frame_gap"] > 0
    dt = df["frame_gap"] * _TIME_SCALE  # seconds

    df["dx_per_s"] = np.where(valid, df["dx"] / dt, np.nan)
    df["dy_per_s"] = np.where(valid, df["dy"] / dt, np.nan)
    df["speed"]    = np.sqrt(df["dx_per_s"] ** 2 + df["dy_per_s"] ** 2)
    df["heading"]  = np.arctan2(df["dy_per_s"], df["dx_per_s"])

    prev_heading = df["heading"].shift(1)
    raw_turn = df["heading"] - prev_heading
    # Wrap to [-π, π]
    df["angular_velocity"] = ((raw_turn + np.pi) % (2 * np.pi) - np.pi) / dt

    # Drop rows where frame_gap > 1 (track gaps — these are segment boundaries)
    # and the very first row of each track (diff = NaN)
    df = df[valid & (df["frame_gap"] == 1)].copy()
    df = df.dropna(subset=["dx_per_s", "dy_per_s", "speed", "heading", "angular_velocity"])
    return df


def load_gt_tracks(
    gt_csv: str,
    vial_rois_json: str,
    run_tag: str,
    source_tag: str = "gt",
) -> pd.DataFrame:
    """
    Load a GT annotation CSV (frame, ID, x, y) and return a frame-level
    DataFrame with columns:
        fly_id, frame, x, y, dx_per_s, dy_per_s, speed, heading,
        angular_velocity, genotype, run, source, frame_gap
    """
    gt = pd.read_csv(gt_csv)
    with open(vial_rois_json) as f:
        vial_rois = json.load(f)

    # Assign vial per fly (based on median x)
    vial_per_id = (
        gt.groupby("ID")["x"]
        .median()
        .apply(lambda x: _assign_vial(x, vial_rois))
        .rename("vial_id")
    )
    gt = gt.join(vial_per_id, on="ID")
    gt["genotype"] = gt["vial_id"].map(VIAL_TO_GENO)
    gt = gt.dropna(subset=["genotype"])

    parts = []
    for fly_id, grp in gt.groupby("ID"):
        kin = _kinematics_from_xy(grp)
        if kin.empty:
            continue
        kin["fly_id"] = f"{run_tag}::gt::{fly_id}"
        kin["genotype"] = grp["genotype"].iloc[0]
        kin["run"] = run_tag
        kin["source"] = source_tag
        parts.append(kin)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_auto_tracks(
    ordered_tracks_csv: str,
    vial_rois_json: str,
    run_tag: str,
    min_frames: int = 30,
) -> pd.DataFrame:
    """
    Load an auto-tracked ordered_tracks.csv and return the same schema as
    load_gt_tracks.  Flies with fewer than min_frames are excluded.
    """
    df = pd.read_csv(ordered_tracks_csv)
    # Normalise fly ID column to "ordered_id"
    for alt in ["compact_id", "stitched_id"]:
        if "ordered_id" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "ordered_id"})
            break
    with open(vial_rois_json) as f:
        vial_rois = json.load(f)

    # ordered_tracks may already have vial_id; if not, assign from x
    if "vial_id" not in df.columns:
        vial_per_id = (
            df.groupby("ordered_id")["x"]
            .median()
            .apply(lambda x: _assign_vial(x, vial_rois))
            .rename("vial_id")
        )
        df = df.join(vial_per_id, on="ordered_id")

    df["genotype"] = df["vial_id"].map(VIAL_TO_GENO)
    df = df.dropna(subset=["genotype"])

    # Filter short tracks
    track_len = df.groupby("ordered_id")["frame"].count()
    keep = track_len[track_len >= min_frames].index
    df = df[df["ordered_id"].isin(keep)]

    parts = []
    for fly_id, grp in df.groupby("ordered_id"):
        kin = _kinematics_from_xy(grp)
        if kin.empty:
            continue
        kin["fly_id"] = f"{run_tag}::auto::{fly_id}"
        kin["genotype"] = grp["genotype"].iloc[0]
        kin["run"] = run_tag
        kin["source"] = "auto"
        parts.append(kin)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_all_tracks(
    auto_run_dirs: list[str] | None = None,
    gt_pairs: list[tuple[str, str, str]] | None = None,
    min_auto_frames: int = 30,
) -> pd.DataFrame:
    """
    Load all available tracks (auto-tracked + GT) and return a combined
    frame-level DataFrame.

    Parameters
    ----------
    auto_run_dirs : list of paths to auto-tracked run output dirs
        Each must contain ordered_tracks.csv and vial_rois.json.
    gt_pairs : list of (gt_csv_path, vial_rois_json_path, run_tag)
    min_auto_frames : minimum track length to include from auto-tracked data

    Returns
    -------
    DataFrame with fly_id, frame, dx_per_s, dy_per_s, speed, heading,
    angular_velocity, genotype, run, source columns.
    """
    parts = []

    if auto_run_dirs:
        for rd in auto_run_dirs:
            rd = str(rd)
            tracks_csv = None
            for candidate in ["ordered_tracks.csv", "compact_tracks.csv", "tracks_long_format.csv"]:
                p = os.path.join(rd, candidate)
                if os.path.exists(p):
                    tracks_csv = p
                    break
            vial_json  = os.path.join(rd, "vial_rois.json")
            if tracks_csv is None or not os.path.exists(vial_json):
                print(f"  [SKIP auto] {rd} — missing tracks csv or vial_rois.json")
                continue
            run_tag = os.path.basename(rd.rstrip("/\\"))
            df = load_auto_tracks(tracks_csv, vial_json, run_tag, min_frames=min_auto_frames)
            if not df.empty:
                print(f"  [auto] {run_tag}: {df['fly_id'].nunique()} flies, "
                      f"{len(df)} frames")
                parts.append(df)

    if gt_pairs:
        for gt_csv, vial_json, run_tag in gt_pairs:
            if not os.path.exists(gt_csv) or not os.path.exists(vial_json):
                print(f"  [SKIP gt] {run_tag} — files missing")
                continue
            df = load_gt_tracks(gt_csv, vial_json, run_tag)
            if not df.empty:
                print(f"  [gt]   {run_tag}: {df['fly_id'].nunique()} flies, "
                      f"{len(df)} frames")
                parts.append(df)

    if not parts:
        raise ValueError("No tracks loaded — check paths.")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

WINDOW_FEATURES = ["dx_per_s", "dy_per_s", "speed", "heading", "angular_velocity"]
N_FEAT = len(WINDOW_FEATURES)  # 5


def extract_windows(
    df: pd.DataFrame,
    window_size: int = 20,
    stride: int = 1,
    genotype_order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Slide a window of `window_size` consecutive frames per fly track and
    produce (X, y_target, y_geno) arrays.

    Only uses continuous segments (frame_gap == 1).  The target is the
    next (dx_per_s, dy_per_s) after the window.

    Returns
    -------
    X        : (N, window_size * N_FEAT + n_genotypes)  — input with geno one-hot
    y        : (N, 2)                                   — target (dx, dy) in cm/s
    geno_idx : (N,) int                                 — genotype index
    geno_labels : list of str                           — genotype names by index
    """
    if genotype_order is None:
        present = sorted(df["genotype"].dropna().unique())
        genotype_order = [g for g in GENOTYPE_ORDER if g in present]
        genotype_order += [g for g in present if g not in genotype_order]

    geno_map = {g: i for i, g in enumerate(genotype_order)}
    n_geno = len(genotype_order)

    X_parts, y_parts, gi_parts = [], [], []

    for fly_id, grp in df.groupby("fly_id"):
        grp = grp.sort_values("frame").reset_index(drop=True)
        geno = grp["genotype"].iloc[0]
        gi = geno_map.get(geno, -1)
        if gi < 0:
            continue

        feat_vals = grp[WINDOW_FEATURES].to_numpy(dtype=float)
        geno_oh = np.zeros(n_geno, dtype=float)
        geno_oh[gi] = 1.0

        # The track only contains frame_gap==1 rows (gaps already dropped in _kinematics_from_xy)
        # so all consecutive rows are truly consecutive frames.
        n = len(feat_vals)
        for start in range(0, n - window_size, stride):
            window = feat_vals[start : start + window_size]   # (W, 5)
            target = feat_vals[start + window_size, :2]        # next (dx, dy)
            if np.any(np.isnan(window)) or np.any(np.isnan(target)):
                continue
            x_row = np.concatenate([window.ravel(), geno_oh])
            X_parts.append(x_row)
            y_parts.append(target)
            gi_parts.append(gi)

    if not X_parts:
        raise ValueError("No windows extracted — check track lengths vs window_size.")

    return (
        np.array(X_parts, dtype=np.float64),
        np.array(y_parts, dtype=np.float64),
        np.array(gi_parts, dtype=np.int64),
        genotype_order,
    )


def extract_windows_no_geno(X_with_geno: np.ndarray, n_geno: int) -> np.ndarray:
    """Strip the genotype one-hot from X (last n_geno columns → zeroed)."""
    X = X_with_geno.copy()
    X[:, -n_geno:] = 0.0
    return X


# ---------------------------------------------------------------------------
# Pure-numpy Adam MLP (same pattern as autoencoder.py)
# ---------------------------------------------------------------------------

def _relu(x):
    return np.maximum(0.0, x)

def _relu_back(g, pre):
    return g * (pre > 0)

def _he(rng, fan_in, fan_out):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


class _Adam:
    def __init__(self, shapes):
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads, lr, wd=0.0, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd > 0:
                g = g + wd * p
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g ** 2
            p -= lr * (self.m[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.v[i] / (1 - b2 ** self.t)) + eps
            )


class ForecastMLP:
    """
    Simple 3-layer MLP for displacement forecasting.
    Weights are stored as plain numpy arrays; trained via Adam.
    """

    def __init__(self, in_dim: int, hidden: tuple = (256, 128), out_dim: int = 2,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = _he(rng, in_dim, hidden[0]);  self.b1 = np.zeros(hidden[0])
        self.W2 = _he(rng, hidden[0], hidden[1]); self.b2 = np.zeros(hidden[1])
        self.W3 = _he(rng, hidden[1], out_dim);  self.b3 = np.zeros(out_dim)
        shapes = [p.shape for p in self._params()]
        self._adam = _Adam(shapes)

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        self._pre1 = x @ self.W1 + self.b1
        self._h1   = _relu(self._pre1)
        self._pre2 = self._h1 @ self.W2 + self.b2
        self._h2   = _relu(self._pre2)
        out        = self._h2 @ self.W3 + self.b3
        self._x    = x
        return out

    def predict(self, x):
        h1  = _relu(x @ self.W1 + self.b1)
        h2  = _relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def backward_and_step(self, y_pred, y_true, lr, wd=0.0):
        n   = len(y_pred)
        d3  = 2 * (y_pred - y_true) / n                     # MSE grad
        dW3 = self._h2.T @ d3;    db3 = d3.sum(0)
        d2  = _relu_back(d3 @ self.W3.T, self._pre2)
        dW2 = self._h1.T @ d2;    db2 = d2.sum(0)
        d1  = _relu_back(d2 @ self.W2.T, self._pre1)
        dW1 = self._x.T  @ d1;    db1 = d1.sum(0)
        self._adam.step(self._params(), [dW1, db1, dW2, db2, dW3, db3],
                        lr=lr, wd=wd)
        return float(np.mean((y_pred - y_true) ** 2))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_forecaster(
    X: np.ndarray,
    y: np.ndarray,
    hidden: tuple = (256, 128),
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.2,
    patience: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[ForecastMLP, dict]:
    """
    Train the forecasting MLP.  Returns (model, info_dict).
    X already includes the genotype one-hot in the last columns.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * val_fraction)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = ForecastMLP(X.shape[1], hidden=hidden, seed=seed)
    best_val = float("inf")
    best_W = [p.copy() for p in model._params()]
    stale, history = 0, []

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(X_tr))
        tr_losses = []
        for start in range(0, len(X_tr), batch_size):
            bi = perm[start : start + batch_size]
            xb, yb = X_tr[bi], y_tr[bi]
            pred = model.forward(xb)
            loss = model.backward_and_step(pred, yb, lr=lr, wd=weight_decay)
            tr_losses.append(loss)

        val_pred = model.predict(X_val)
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        history.append((ep, float(np.mean(tr_losses)), val_loss))

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_W = [p.copy() for p in model._params()]
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Restore best weights
    for p, bw in zip(model._params(), best_W):
        p[:] = bw

    if verbose:
        print(f"[forecaster] epochs={len(history)}  best_val_mse={best_val:.6g}")

    return model, {
        "epochs_trained": len(history),
        "best_val_mse": best_val,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _baseline_mse(X: np.ndarray, y: np.ndarray, window_size: int) -> float:
    """Constant-velocity baseline: predict last (dx, dy) in the window."""
    last_dx = X[:, (window_size - 1) * N_FEAT + 0]   # last dx in window
    last_dy = X[:, (window_size - 1) * N_FEAT + 1]   # last dy in window
    pred = np.stack([last_dx, last_dy], axis=1)
    return float(np.mean((pred - y) ** 2))


def evaluate_forecaster(
    model_with_geno: ForecastMLP,
    model_no_geno: ForecastMLP,
    X: np.ndarray,
    y: np.ndarray,
    geno_idx: np.ndarray,
    geno_labels: list[str],
    window_size: int = 20,
    n_geno: int | None = None,
) -> pd.DataFrame:
    """
    Compute per-genotype MSE for:
      - constant-velocity baseline
      - MLP without genotype conditioning
      - MLP with genotype conditioning

    Returns a DataFrame with one row per genotype.
    """
    if n_geno is None:
        n_geno = len(geno_labels)

    X_no_geno = extract_windows_no_geno(X, n_geno)

    rows = []
    for gi, gname in enumerate(geno_labels):
        mask = geno_idx == gi
        if mask.sum() < 5:
            continue
        Xg, yg, Xg_ng = X[mask], y[mask], X_no_geno[mask]

        baseline  = _baseline_mse(Xg, yg, window_size)
        pred_ng   = model_no_geno.predict(Xg_ng)
        mse_ng    = float(np.mean((pred_ng - yg) ** 2))
        pred_g    = model_with_geno.predict(Xg)
        mse_g     = float(np.mean((pred_g - yg) ** 2))

        rows.append({
            "genotype": gname,
            "n_windows": int(mask.sum()),
            "baseline_mse": baseline,
            "mlp_no_geno_mse": mse_ng,
            "mlp_with_geno_mse": mse_g,
            "improvement_vs_baseline_pct": 100 * (baseline - mse_g) / baseline,
            "geno_conditioning_gain_pct":  100 * (mse_ng - mse_g) / mse_ng,
        })

    return pd.DataFrame(rows).sort_values("genotype").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_forecast_figures(
    results: pd.DataFrame,
    train_info_geno: dict,
    train_info_no_geno: dict,
) -> dict[str, go.Figure]:
    """Produce Plotly figures summarising forecasting results."""

    figs = {}

    # 1. Per-genotype MSE comparison (3 bars per genotype)
    long = pd.melt(
        results,
        id_vars=["genotype", "n_windows"],
        value_vars=["baseline_mse", "mlp_no_geno_mse", "mlp_with_geno_mse"],
        var_name="model",
        value_name="mse",
    )
    long["model"] = long["model"].map({
        "baseline_mse":       "Constant velocity",
        "mlp_no_geno_mse":    "MLP (no genotype)",
        "mlp_with_geno_mse":  "MLP + genotype",
    })
    figs["mse_comparison"] = px.bar(
        long,
        x="genotype", y="mse", color="model", barmode="group",
        title="Per-genotype forecast MSE — baseline vs MLP",
        labels={"mse": "MSE (cm/s)²", "genotype": "Genotype", "model": ""},
        color_discrete_map={
            "Constant velocity": "#adb5bd",
            "MLP (no genotype)": "#4895ef",
            "MLP + genotype":    "#e63946",
        },
    )
    figs["mse_comparison"].update_layout(height=460)

    # 2. Genotype conditioning gain (% MSE reduction from adding genotype info)
    figs["conditioning_gain"] = px.bar(
        results.sort_values("geno_conditioning_gain_pct", ascending=False),
        x="genotype", y="geno_conditioning_gain_pct",
        title="Genotype conditioning gain (% MSE reduction from adding genotype)",
        labels={
            "geno_conditioning_gain_pct": "% MSE reduction",
            "genotype": "Genotype",
        },
        color="geno_conditioning_gain_pct",
        color_continuous_scale="RdYlGn",
    )
    figs["conditioning_gain"].add_hline(
        y=0, line_dash="dash", line_color="gray",
        annotation_text="no gain", annotation_position="bottom right",
    )
    figs["conditioning_gain"].update_layout(height=420, coloraxis_showscale=False)

    # 3. Training loss curves (with geno)
    hist = train_info_geno.get("history", [])
    if hist:
        ep, tr, val = zip(*hist)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=list(ep), y=list(tr),  name="train", line=dict(color="#4895ef")))
        fig_loss.add_trace(go.Scatter(x=list(ep), y=list(val), name="val",   line=dict(color="#e63946")))
        fig_loss.update_layout(
            title="Training curve (MLP + genotype)",
            xaxis_title="Epoch", yaxis_title="MSE",
            height=380,
        )
        figs["training_curve"] = fig_loss

    return figs


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def make_forecast_report(
    results: pd.DataFrame,
    figs: dict[str, go.Figure],
    out_dir: str,
) -> str:
    """Write individual HTML files and a summary report. Returns report path."""
    os.makedirs(out_dir, exist_ok=True)

    for name, fig in figs.items():
        fig.write_html(os.path.join(out_dir, f"forecast_{name}.html"))

    def _iframe(fname):
        return (
            f'<iframe src="{fname}" width="100%" height="480" '
            f'frameborder="0" scrolling="no"></iframe>'
        )

    table_html = results.round(4).to_html(index=False, border=0,
                                           classes="results-table")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fly forecasting report</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; }}
  h1 {{ color: #222; }}
  h2 {{ color: #444; margin-top: 2em; }}
  .results-table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
  .results-table th {{ background: #f5f5f5; text-align: center; }}
  p.note {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Next-step displacement forecasting</h1>
<p class="note">
  Model: MLP (256→128→2) trained on W=20-frame windows of (dx, dy, speed, heading, angular_velocity).
  Genotype is concatenated as a one-hot vector. Target: next (dx, dy) in cm/s.
  Baseline: constant-velocity (repeat last step).
</p>

<h2>Per-genotype MSE comparison</h2>
{_iframe("forecast_mse_comparison.html")}

<h2>Genotype conditioning gain</h2>
<p class="note">
  % reduction in MSE when the model knows the fly's genotype vs not knowing it.
  Positive = genotype encodes information about movement dynamics.
</p>
{_iframe("forecast_conditioning_gain.html")}

<h2>Training curve</h2>
{_iframe("forecast_training_curve.html")}

<h2>Numeric results</h2>
{table_html}
</body>
</html>"""

    report_path = os.path.join(out_dir, "forecast_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_forecasting_analysis(
    auto_run_dirs: list[str] | None = None,
    gt_pairs: list[tuple[str, str, str]] | None = None,
    out_dir: str = "outputs/classification/forecasting",
    window_size: int = 20,
    stride: int = 3,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline:
      1. Load all tracks (auto + GT)
      2. Extract sliding windows
      3. Normalise features
      4. Train MLP with genotype conditioning
      5. Train MLP without genotype conditioning (ablation)
      6. Evaluate per-genotype MSE vs baseline
      7. Write HTML report

    Returns dict with keys: tracks_df, results_df, model_with_geno,
    model_no_geno, figures, report_path.
    """
    print("\n[forecasting] Loading tracks...")
    df = load_all_tracks(auto_run_dirs=auto_run_dirs, gt_pairs=gt_pairs)
    n_flies = df["fly_id"].nunique()
    n_frames = len(df)
    print(f"[forecasting] {n_flies} flies, {n_frames} frames total")
    print(f"[forecasting] Sources: {df.groupby('source')['fly_id'].nunique().to_dict()}")
    print(f"[forecasting] Genotypes: {df.groupby('genotype')['fly_id'].nunique().to_dict()}")

    print(f"\n[forecasting] Extracting windows (W={window_size}, stride={stride})...")
    X, y, geno_idx, geno_labels = extract_windows(df, window_size=window_size, stride=stride)
    n_geno = len(geno_labels)
    print(f"[forecasting] {len(X)} windows, {X.shape[1]} input dims, {n_geno} genotypes")

    # Normalise features (but not the genotype one-hot)
    feat_dim = window_size * N_FEAT
    scaler = StandardScaler()
    X[:, :feat_dim] = scaler.fit_transform(X[:, :feat_dim])
    y_scaler = StandardScaler()
    y_norm = y_scaler.fit_transform(y)

    print("\n[forecasting] Training MLP with genotype conditioning...")
    model_geno, info_geno = train_forecaster(
        X, y_norm, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, verbose=verbose,
    )

    print("\n[forecasting] Training MLP without genotype conditioning (ablation)...")
    X_no_geno = extract_windows_no_geno(X, n_geno)
    model_no_geno, info_no_geno = train_forecaster(
        X_no_geno, y_norm, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed+1, verbose=verbose,
    )

    print("\n[forecasting] Evaluating...")
    results = evaluate_forecaster(
        model_geno, model_no_geno, X, y_norm, geno_idx, geno_labels,
        window_size=window_size, n_geno=n_geno,
    )
    print(results[["genotype", "n_windows", "baseline_mse",
                    "mlp_with_geno_mse", "geno_conditioning_gain_pct"]].to_string(index=False))

    figs = make_forecast_figures(results, info_geno, info_no_geno)
    report_path = make_forecast_report(results, figs, out_dir)
    print(f"\n[forecasting] Report: {report_path}")

    return {
        "tracks_df": df,
        "results_df": results,
        "model_with_geno": model_geno,
        "model_no_geno": model_no_geno,
        "geno_labels": geno_labels,
        "figures": figs,
        "report_path": report_path,
        "train_info_geno": info_geno,
        "train_info_no_geno": info_no_geno,
    }
