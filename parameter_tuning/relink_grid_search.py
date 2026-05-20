"""relink_grid_search.py — Grid search over post-hoc relink parameters.

Loads pre-saved observation logs (_obs_logs.json) so the tracker never
re-runs.  Each combo applies relink() to the in-memory tracker state and
evaluates HOTA — much faster than the full tracker grid search.

Usage (from repo root):
    python parameter_tuning/relink_grid_search.py

    # Cluster: split across N jobs
    python parameter_tuning/relink_grid_search.py --job-id 3 --n-jobs 100

    # After all jobs finish:
    python parameter_tuning/relink_grid_search.py --merge

Observation logs are written by tracking.py alongside each ocsort_tracks.csv
as  <run_dir>/ocsort_tracks_obs_logs.json.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

# ── Sequences ──────────────────────────────────────────────────────────────────
OUTPUTS = ROOT.parent / "outputs"

SEQUENCES: dict[str, dict] = {
    "13d_002": {
        "obs_logs": OUTPUTS / "run_112" / "ocsort_tracks_obs_logs.json",
        "det_csv":  OUTPUTS / "run_112" / "detections_raw.csv",
        "fps":      29.88,
    },
    "31d_005": {
        "obs_logs": OUTPUTS / "run_114_31DPE_n005" / "ocsort_tracks_obs_logs.json",
        "det_csv":  OUTPUTS / "run_114_31DPE_n005" / "detections_raw.csv",
        "fps":      29.893,
    },
}

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "relink_grid_search_results.csv"

# ── Behavioral weight presets ──────────────────────────────────────────────────
# Same repartition philosophy as the main grid search — normalized weights that
# sum to 1 (or 0 for "off"). Keys match the feature names used by relink_tracklets.
RELINK_BW_PRESETS: list[dict] = [
    # name            speed   turning  pause   accel   tortuosity
    {"name": "off",          "median_speed": 0.00, "mean_turning_angle": 0.00, "pause_fraction": 0.00, "mean_acceleration": 0.00, "tortuosity": 0.00},
    {"name": "speed_only",   "median_speed": 1.00, "mean_turning_angle": 0.00, "pause_fraction": 0.00, "mean_acceleration": 0.00, "tortuosity": 0.00},
    {"name": "speed_turn",   "median_speed": 0.50, "mean_turning_angle": 0.50, "pause_fraction": 0.00, "mean_acceleration": 0.00, "tortuosity": 0.00},
    {"name": "speed_tort",   "median_speed": 0.50, "mean_turning_angle": 0.00, "pause_fraction": 0.00, "mean_acceleration": 0.00, "tortuosity": 0.50},
    {"name": "speed_pause",  "median_speed": 0.50, "mean_turning_angle": 0.00, "pause_fraction": 0.50, "mean_acceleration": 0.00, "tortuosity": 0.00},
    {"name": "kinematics",   "median_speed": 0.50, "mean_turning_angle": 0.00, "pause_fraction": 0.00, "mean_acceleration": 0.50, "tortuosity": 0.00},
    {"name": "path_shape",   "median_speed": 0.00, "mean_turning_angle": 0.50, "pause_fraction": 0.00, "mean_acceleration": 0.00, "tortuosity": 0.50},
    {"name": "speed_heavy",  "median_speed": 0.60, "mean_turning_angle": 0.10, "pause_fraction": 0.10, "mean_acceleration": 0.10, "tortuosity": 0.10},
    {"name": "equal_all",    "median_speed": 0.20, "mean_turning_angle": 0.20, "pause_fraction": 0.20, "mean_acceleration": 0.20, "tortuosity": 0.20},
    {"name": "no_speed",     "median_speed": 0.00, "mean_turning_angle": 0.25, "pause_fraction": 0.25, "mean_acceleration": 0.25, "tortuosity": 0.25},
    {"name": "pause_heavy",  "median_speed": 0.20, "mean_turning_angle": 0.10, "pause_fraction": 0.50, "mean_acceleration": 0.10, "tortuosity": 0.10},
    {"name": "tort_heavy",   "median_speed": 0.20, "mean_turning_angle": 0.10, "pause_fraction": 0.10, "mean_acceleration": 0.10, "tortuosity": 0.50},
]

# ── Parameter grid ─────────────────────────────────────────────────────────────
# Total combos = 3×4×4×12 = 576
PARAM_GRID: dict[str, list] = {
    "min_length":        [5,    10,   20              ],
    "swap_threshold":    [0.02, 0.05, 0.10, 0.20      ],
    "confidence_weight": [0.0,  0.5,  1.0,  2.0       ],
    "bw_preset":         list(range(len(RELINK_BW_PRESETS))),
}

# ── MOT helpers ────────────────────────────────────────────────────────────────

def _load_gt_mot(seq: str) -> list[str]:
    df = pd.read_csv(DATA_DIR / f"ground_truth_{seq}_cleaned.csv")
    df = df.rename(columns={"ID": "id"})
    return _mot_lines(df, c7=1)


def _mot_lines(df: pd.DataFrame, *, c7: float) -> list[str]:
    w = (df["x2"] - df["x1"]).astype(float)
    h = (df["y2"] - df["y1"]).astype(float)
    lines = []
    for frame, tid, bbl, bbt, bbw, bbh in zip(
        df["frame"].astype(int) + 1,
        df["id"].astype(int),
        df["x1"].astype(float), df["y1"].astype(float),
        w, h,
    ):
        lines.append(f"{frame},{tid},{bbl:.3f},{bbt:.3f},{bbw:.3f},{bbh:.3f},{c7},1,1.0")
    return lines


def _apply_relink_to_logs(
    obs_records: list[dict],
    weights: dict,
    min_length: int,
    swap_threshold: float,
    confidence_weight: float,
    fps: float,
) -> dict[int, int]:
    """Run relink_tracklets on stub tracker objects built from saved obs logs.

    Returns a {old_id: new_id} remapping dict (identity entries included).
    """
    from src.association import relink_tracklets

    # Build minimal stub objects that relink_tracklets expects:
    # needs .id (0-based int) and .observation_log list of (frame, bbox, score)
    class _StubTracker:
        def __init__(self, record):
            self.id = record["id"] - 1   # relink_tracklets uses 0-based internally, emits 1-based
            self.observation_log = [
                (entry[0], np.array(entry[1], dtype=float), entry[2])
                for entry in record["log"]
            ]

    stubs = [_StubTracker(r) for r in obs_records]

    swaps = relink_tracklets(
        trackers=stubs,
        weights=weights,
        min_length=min_length,
        swap_threshold=swap_threshold,
        confidence_weight=confidence_weight,
        fps=fps,
    )

    # Build id remapping: default = identity
    all_ids = {r["id"] for r in obs_records}
    remap = {i: i for i in all_ids}
    for id_a, id_b, _split_frame in swaps:
        remap[id_a], remap[id_b] = remap[id_b], remap[id_a]
    return remap


def _relink_mot_lines(wide_csv: Path, det_csv: Path, remap: dict[int, int]) -> list[str]:
    """Apply remap to wide CSV and return MOT lines."""
    wide_df = pd.read_csv(wide_csv)
    id_cols = [c for c in wide_df.columns if c != "frame"]

    dets = pd.read_csv(det_csv)
    dets["cx"] = ((dets["x1"] + dets["x2"]) / 2.0).round(2)
    dets["cy"] = ((dets["y1"] + dets["y2"]) / 2.0).round(2)
    dets = dets.sort_values("conf", ascending=False).drop_duplicates(
        subset=["frame", "cx", "cy"], keep="first"
    )

    long = wide_df.melt(id_vars="frame", value_vars=id_cols,
                        var_name="raw_id", value_name="pos")
    long = long.dropna(subset=["pos"])
    long["orig_id"] = long["raw_id"].str.extract(r"(\d+)").astype(int)
    long["track_id"] = long["orig_id"].map(remap).fillna(long["orig_id"]).astype(int)
    long["x"] = long["pos"].str.extract(r"\(([^,]+),").astype(float)
    long["y"] = long["pos"].str.extract(r",\s*([^)]+)\)").astype(float)
    long["x_r"] = long["x"].round(2)
    long["y_r"] = long["y"].round(2)

    merged = long.merge(
        dets[["frame", "cx", "cy", "x1", "y1", "x2", "y2"]],
        left_on=["frame", "x_r", "y_r"],
        right_on=["frame", "cx", "cy"],
        how="inner",
    )
    merged = merged.rename(columns={"track_id": "id"})
    return _mot_lines(merged[["frame", "id", "x1", "y1", "x2", "y2"]], c7=1.0)


# ── Per-combo evaluation ────────────────────────────────────────────────────────

def _run_combo(params: dict, gt_txt: Path, tr_txt: Path) -> dict[str, Any]:
    from trackers.eval.evaluate import evaluate_mot_sequence

    preset = RELINK_BW_PRESETS[params["bw_preset"]]
    weights = {k: v for k, v in preset.items() if k != "name"}
    min_length        = params["min_length"]
    swap_threshold    = params["swap_threshold"]
    confidence_weight = params["confidence_weight"]

    seq_results: dict[str, dict] = {}

    for seq, scfg in SEQUENCES.items():
        obs_logs_path = scfg["obs_logs"]
        wide_csv = obs_logs_path.parent / "ocsort_tracks.csv"

        if not obs_logs_path.exists():
            raise FileNotFoundError(
                f"Observation logs not found: {obs_logs_path}\n"
                f"Run the tracker once to generate them."
            )

        obs_records = json.loads(obs_logs_path.read_text())
        remap = _apply_relink_to_logs(
            obs_records, weights, min_length, swap_threshold, confidence_weight, scfg["fps"]
        )

        tr_lines = _relink_mot_lines(wide_csv, scfg["det_csv"], remap)
        if not tr_lines:
            seq_results[seq] = {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "IDSW": 9999, "MOTA": 0.0}
            continue

        gt_txt.write_text("\n".join(_load_gt_mot(seq)) + "\n", encoding="utf-8")
        tr_txt.write_text("\n".join(tr_lines) + "\n", encoding="utf-8")

        r = evaluate_mot_sequence(
            gt_path=gt_txt, tracker_path=tr_txt,
            metrics=["HOTA", "CLEAR"],
        )
        seq_results[seq] = {
            "HOTA": float(r.HOTA.HOTA),
            "DetA": float(r.HOTA.DetA),
            "AssA": float(r.HOTA.AssA),
            "IDSW": int(r.CLEAR.IDSW),
            "MOTA": float(r.CLEAR.MOTA),
        }

    row: dict[str, Any] = {k: v for k, v in params.items() if k != "bw_preset"}
    row["bw_preset"] = params["bw_preset"]
    row["bw_name"] = preset["name"]
    for feat, val in weights.items():
        row[f"bw_{feat}"] = val
    for seq, m in seq_results.items():
        s = seq.replace("d_", "d")
        for k, v in m.items():
            row[f"{k}_{s}"] = v
    seqs = list(SEQUENCES.keys())
    row["HOTA_combined"] = float(np.mean([seq_results[s]["HOTA"] for s in seqs]))
    row["AssA_combined"] = float(np.mean([seq_results[s]["AssA"] for s in seqs]))
    row["IDSW_total"]    = int(sum(seq_results[s]["IDSW"] for s in seqs))
    return row


# ── Main ────────────────────────────────────────────────────────────────────────

def _combo_key(params: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(params.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-job CSVs into results file and print top 20")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        parts = sorted(RESULTS_DIR.glob("relink_grid_job*.csv"))
        if not parts:
            print("No per-job CSV files found.")
            return
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.sort_values("HOTA_combined", ascending=False)
        df.to_csv(RESULTS_CSV, index=False)
        keys = list(PARAM_GRID.keys())
        seq_cols = []
        for seq in SEQUENCES:
            s = seq.replace("d_", "d")
            seq_cols += [f"HOTA_{s}", f"AssA_{s}", f"IDSW_{s}"]
        show = keys + ["HOTA_combined", "AssA_combined", "IDSW_total"] + seq_cols
        print(f"Merged {len(parts)} files → {RESULTS_CSV}  ({len(df)} rows)")
        print(df.head(20)[show].to_string(index=False, float_format="{:.3f}".format))
        return

    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[PARAM_GRID[k] for k in keys])]
    n_total = len(combos)
    chunk   = [c for i, c in enumerate(combos) if i % args.n_jobs == args.job_id]
    n       = len(chunk)

    n_vals = " × ".join(str(len(PARAM_GRID[k])) for k in keys)
    print(f"Relink grid: {n_vals} = {n_total} total combinations")
    print(f"This job   : {args.job_id + 1}/{args.n_jobs}  ({n} combos)")

    job_csv = RESULTS_DIR / f"relink_grid_job{args.job_id:05d}.csv"

    done_keys: set[str] = set()
    all_rows: list[dict] = []
    if job_csv.exists():
        prev = pd.read_csv(job_csv)
        for _, row in prev.iterrows():
            p = {k: row[k] for k in keys}
            done_keys.add(_combo_key(p))
            all_rows.append(row.to_dict())
        print(f"Resuming — {len(done_keys)} done, {n - len(done_keys)} remaining.\n")
    else:
        print()

    with tempfile.TemporaryDirectory() as tmp:
        gt_txt = Path(tmp) / "gt.txt"
        tr_txt = Path(tmp) / "tr.txt"

        t0 = time.time()
        done_this_run = 0

        for i, params in enumerate(chunk):
            ck = _combo_key(params)
            if ck in done_keys:
                continue
            try:
                row = _run_combo(params, gt_txt, tr_txt)
            except Exception as e:
                print(f"  [{i+1:4d}/{n}] ERROR: {e}  params={params}")
                continue

            all_rows.append(row)
            done_keys.add(ck)
            done_this_run += 1
            pd.DataFrame(all_rows).to_csv(job_csv, index=False)

            elapsed = time.time() - t0
            rate = done_this_run / elapsed
            eta  = (n - len(done_keys)) / rate if rate > 0 else float("inf")
            print(
                f"[{i+1:4d}/{n}]  HOTA={row['HOTA_combined']:.3f}  "
                f"AssA={row['AssA_combined']:.3f}  IDSW={row['IDSW_total']:4d}  "
                f"| {' '.join(f'{k}={v}' for k, v in params.items())}  "
                f"(ETA {eta/60:.1f}m)"
            )

    print(f"\nDone. Results: {job_csv}")
    print("Merge with:  python parameter_tuning/relink_grid_search.py --merge")


if __name__ == "__main__":
    main()
