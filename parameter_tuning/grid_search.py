"""grid_search.py — Exhaustive grid search over tracker association parameters.

Uses cached detections (detections_raw.csv) — RF-DETR is never called.
Watershed is disabled so only tracker parameters vary; the relative ranking
is still valid for choosing the best association settings.

Edit PARAM_GRID to change the search space.
Results are written to results/grid_search_results.csv (incrementally — safe
to interrupt and resume; already-scored combos are skipped).

Usage (from repo root):
    python parameter_tuning/grid_search.py

Baseline (current config, for comparison):
    video        HOTA    DetA    AssA    MOTA    MOTP   IDSW    IDF1
    13d_002     0.683   0.727   0.642   0.939   0.784     34   0.887
    31d_005     0.522   0.640   0.428   0.850   0.736    195   0.713
    COMBINED    0.600   0.684   0.530   0.894   0.760    232   0.800
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from trackers.eval.evaluate import evaluate_mot_sequence

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from src.tracking import export_tracks_xy_tuple_csv_one_config

# ── Sequences ──────────────────────────────────────────────────────────────────
OUTPUTS = ROOT.parent / "outputs"

SEQUENCES: dict[str, dict] = {
    "13d_002": {
        "video":          OUTPUTS / "run_112" / "2024-02-12_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m_13d_002-converted_raw_cropped.mp4",
        "det_csv":        OUTPUTS / "run_112" / "detections_raw.csv",
        "roi_json":       OUTPUTS / "run_112" / "vial_rois.json",
        "fps":            29.88,
        "expected_count": 38,
    },
    "31d_005": {
        "video":          OUTPUTS / "run_114_31DPE_n005" / "2024-03-01_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m_31d_005-converted_raw_cropped.mp4",
        "det_csv":        OUTPUTS / "run_114_31DPE_n005" / "detections_raw.csv",
        "roi_json":       OUTPUTS / "run_114_31DPE_n005" / "vial_rois.json",
        "fps":            29.893,
        "expected_count": 38,
    },
}

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "grid_search_results.csv"

# ── Behavioral weight presets ──────────────────────────────────────────────────
# Instead of sweeping 5 bw_* weights independently (which produces thousands of
# near-identical combos once normalised), we define a small set of meaningful
# repartitions. bw_preset in PARAM_GRID is an index into this list.
# scale is excluded — bbox size has near-zero discriminative power between flies.
BW_PRESETS: list[dict] = [
    # name            speed  turning  pause  accel  tortuosity
    {"name": "off",          "speed": 0.00, "turning_angle": 0.00, "pause": 0.00, "acceleration": 0.00, "tortuosity": 0.00},
    {"name": "speed_only",   "speed": 1.00, "turning_angle": 0.00, "pause": 0.00, "acceleration": 0.00, "tortuosity": 0.00},
    {"name": "speed_turn",   "speed": 0.50, "turning_angle": 0.50, "pause": 0.00, "acceleration": 0.00, "tortuosity": 0.00},
    {"name": "speed_tort",   "speed": 0.50, "turning_angle": 0.00, "pause": 0.00, "acceleration": 0.00, "tortuosity": 0.50},
    {"name": "speed_pause",  "speed": 0.50, "turning_angle": 0.00, "pause": 0.50, "acceleration": 0.00, "tortuosity": 0.00},
    {"name": "kinematics",   "speed": 0.50, "turning_angle": 0.00, "pause": 0.00, "acceleration": 0.50, "tortuosity": 0.00},
    {"name": "path_shape",   "speed": 0.00, "turning_angle": 0.50, "pause": 0.00, "acceleration": 0.00, "tortuosity": 0.50},
    {"name": "speed_heavy",  "speed": 0.60, "turning_angle": 0.10, "pause": 0.10, "acceleration": 0.10, "tortuosity": 0.10},
    {"name": "equal_all",    "speed": 0.20, "turning_angle": 0.20, "pause": 0.20, "acceleration": 0.20, "tortuosity": 0.20},
    {"name": "no_speed",     "speed": 0.00, "turning_angle": 0.25, "pause": 0.25, "acceleration": 0.25, "tortuosity": 0.25},
    {"name": "pause_heavy",  "speed": 0.20, "turning_angle": 0.10, "pause": 0.50, "acceleration": 0.10, "tortuosity": 0.10},
    {"name": "tort_heavy",   "speed": 0.20, "turning_angle": 0.10, "pause": 0.10, "acceleration": 0.10, "tortuosity": 0.50},
]

# ── Parameter grid ─────────────────────────────────────────────────────────────
# Total combos = 4×5×4×4×4×4×3×3×4×3×12×3×2 = 39,813,120
PARAM_GRID: dict[str, list] = {
    # Core association
    "confidence":                 [0.0,  0.10, 0.25, 0.55],
    "minimum_matching_threshold": [0.01, 0.05, 0.10, 0.25, 0.50],
    "inertia":                    [0.0,  0.10, 0.25, 0.50],
    # Kalman noise
    "brownian_pos_noise":         [5,    15,   30,   50  ],
    # OCM direction lookback (-1 = full frame history)
    "delta_t":                    [3,    10,   30,   -1  ],
    # Jump round (all three swept together — they interact)
    "jump_factor":                [1.0,  2.0,  3.0,  4.0 ],
    "jump_iou_threshold":         [0.01, 0.05, 0.20       ],
    "jump_inertia":               [0.0,  0.05, 0.20       ],
    # Overlap handling
    "overlap_weight_scale":       [1.0,  3.0,  6.0,  10.0],
    "overlap_iou_scale":          [0.05, 0.10, 0.20       ],
    # Behavioral fingerprint preset (index into BW_PRESETS)
    "bw_preset":                  list(range(len(BW_PRESETS))),
    # Count-aware spawning penalties
    "w_under":                    [5.0,  15.0, 30.0],
    "w_over":                     [1.0,  2.0       ],
}

# Everything else held at current config values.
FIXED_PARAMS: dict[str, Any] = {
    "detection_confidence_rfdetr": 0.4,
    "lost_track_buffer":           400,
    "minimum_consecutive_frames":  1,
    "min_area":                    20,
    "asso_func":                   "diou",
    "aspect_weight":               0.05,
    "edge_fraction":               0.1,
    "watershed_cfg":               None,   # disabled — tuning tracker params, not detector
}


# ── Per-sequence cache ────────────────────────────────────────────────────────
# Built lazily on first access in each worker process. Avoids re-parsing the
# same detections / GT / ROI files millions of times across the combo loop.
_SEQ_CACHE: dict[str, dict] = {}


def _get_seq_cache(seq: str) -> dict:
    if seq in _SEQ_CACHE:
        return _SEQ_CACHE[seq]
    scfg = SEQUENCES[seq]
    with open(scfg["roi_json"]) as f:
        vial_rois = {k: tuple(v) for k, v in json.load(f).items()}
    gt_lines = _load_gt_mot(seq)
    dets = pd.read_csv(scfg["det_csv"])
    dets["cx"] = ((dets["x1"] + dets["x2"]) / 2.0).round(2)
    dets["cy"] = ((dets["y1"] + dets["y2"]) / 2.0).round(2)
    dets = dets.sort_values("conf", ascending=False).drop_duplicates(
        subset=["frame", "cx", "cy"], keep="first"
    )
    dets = dets[["frame", "cx", "cy", "x1", "y1", "x2", "y2"]].reset_index(drop=True)
    _SEQ_CACHE[seq] = {"vial_rois": vial_rois, "gt_lines": gt_lines, "dets": dets, "scfg": scfg}
    return _SEQ_CACHE[seq]


# ── MOT helpers ────────────────────────────────────────────────────────────────

def _load_gt_mot(seq: str) -> list[str]:
    """Build GT MOT lines from ground_truth_{seq}_cleaned.csv."""
    df = pd.read_csv(DATA_DIR / f"ground_truth_{seq}_cleaned.csv")
    df = df.rename(columns={"ID": "id"})
    return _mot_lines(df, c7=1)


def _mot_lines(df: pd.DataFrame, *, c7: float) -> list[str]:
    w = (df["x2"] - df["x1"]).astype(float)
    h = (df["y2"] - df["y1"]).astype(float)
    lines = []
    for frame, tid, bbl, bbt, bbw, bbh in zip(
        df["frame"].astype(int) + 1,   # 1-indexed
        df["id"].astype(int),
        df["x1"].astype(float), df["y1"].astype(float),
        w, h,
    ):
        lines.append(f"{frame},{tid},{bbl:.3f},{bbt:.3f},{bbw:.3f},{bbh:.3f},{c7},1,1.0")
    return lines


def _tracker_mot_lines(wide_df: pd.DataFrame, dets: pd.DataFrame) -> list[str]:
    """
    Convert wide tracker output to MOT lines.

    1. Melt wide → long (frame, track_id, x, y)
    2. Join to cached detections on (frame, round(cx,2), round(cy,2)) to recover bboxes
    3. Return MOT-format lines
    """
    id_cols = [c for c in wide_df.columns if c != "frame" and pd.notna(wide_df[c]).any()]
    long = wide_df.melt(id_vars="frame", value_vars=id_cols,
                        var_name="raw_id", value_name="pos")
    long = long.dropna(subset=["pos"])
    long["track_id"] = long["raw_id"].str.extract(r"(\d+)").astype(int)
    long["x"] = long["pos"].str.extract(r"\(([^,]+),").astype(float)
    long["y"] = long["pos"].str.extract(r",\s*([^)]+)\)").astype(float)
    long["x_r"] = long["x"].round(2)
    long["y_r"] = long["y"].round(2)

    merged = long.merge(
        dets,
        left_on=["frame", "x_r", "y_r"],
        right_on=["frame", "cx", "cy"],
        how="inner",   # drop rows where no detection found (Kalman-only frames)
    )
    merged = merged.rename(columns={"track_id": "id"})
    return _mot_lines(merged[["frame", "id", "x1", "y1", "x2", "y2"]], c7=1.0)


# ── Per-combo evaluation ────────────────────────────────────────────────────────

def _run_combo(params: dict) -> dict[str, float]:
    """Run one parameter combination; return dict of metric values per sequence.

    Self-contained: builds its own temp files so this function is safe to call
    concurrently from a process pool.
    """
    # Look up behavioral weight preset; drop bw_preset from tracker kwargs.
    preset = BW_PRESETS[params["bw_preset"]]
    behavioral_weights = {k: v for k, v in preset.items() if k != "name"}
    flat_params = {k: v for k, v in params.items() if k != "bw_preset"}
    all_params = {**FIXED_PARAMS, **flat_params, "behavioral_weights": behavioral_weights}
    seq_results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        wide_csv = str(tmp_dir / "wide.csv")
        gt_txt = tmp_dir / "gt.txt"
        tr_txt = tmp_dir / "tr.txt"

        for seq in SEQUENCES:
            cache = _get_seq_cache(seq)
            scfg = cache["scfg"]
            all_params["expected_count"] = scfg["expected_count"]

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                wide_df, _ = export_tracks_xy_tuple_csv_one_config(
                    video_path=str(scfg["video"]),
                    output_csv=wide_csv,
                    api_key="",
                    model_id="",
                    det_log_csv=str(scfg["det_csv"]),
                    vial_rois=cache["vial_rois"],
                    fps_assumed=scfg["fps"],
                    **all_params,
                )

            tr_lines = _tracker_mot_lines(wide_df, cache["dets"])
            if not tr_lines:
                seq_results[seq] = {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "IDSW": 9999}
                continue

            gt_txt.write_text("\n".join(cache["gt_lines"]) + "\n", encoding="utf-8")
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

    row: dict[str, Any] = dict(params)
    row["bw_name"] = preset["name"]
    for feat, val in behavioral_weights.items():
        row[f"bw_{feat}"] = val
    for seq, m in seq_results.items():
        seq_short = seq.replace("d_", "d")   # "13d_002" → "13d002"
        for k, v in m.items():
            row[f"{k}_{seq_short}"] = v
    seqs = list(SEQUENCES.keys())
    row["HOTA_combined"] = np.mean([seq_results[s]["HOTA"] for s in seqs])
    row["AssA_combined"] = np.mean([seq_results[s]["AssA"] for s in seqs])
    row["IDSW_total"]    = sum(seq_results[s]["IDSW"] for s in seqs)
    return row


# ── Main ────────────────────────────────────────────────────────────────────────

def _safe_run(params: dict):
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    try:
        return _run_combo(params)
    except Exception as e:
        print(f"  ERROR in combo {params}: {e}", flush=True)
        return None


def _combo_key(params: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(params.items()))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, default=0,
                        help="0-based index of this job (cluster array index)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Total number of parallel jobs")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all per-job CSVs into a single sorted results file and exit")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)),
                        help="Worker processes within this job (default: os.cpu_count())")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        parts = sorted(RESULTS_DIR.glob("grid_search_job*.csv"))
        if not parts:
            print("No per-job CSV files found.")
            return
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.sort_values("HOTA_combined", ascending=False)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"Merged {len(parts)} files → {RESULTS_CSV}  ({len(df)} rows)")
        keys = list(PARAM_GRID.keys())
        seq_cols = []
        for seq in SEQUENCES:
            s = seq.replace("d_", "d")
            seq_cols += [f"HOTA_{s}", f"AssA_{s}", f"IDSW_{s}"]
        show_cols = keys + ["HOTA_combined", "AssA_combined", "IDSW_total"] + seq_cols
        print(df.head(20)[show_cols].to_string(index=False, float_format="{:.3f}".format))
        return

    # ── Normal / cluster mode ─────────────────────────────────────────────────
    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[PARAM_GRID[k] for k in keys])]
    n_total = len(combos)

    # Slice for this job
    chunk = [c for i, c in enumerate(combos) if i % args.n_jobs == args.job_id]
    n = len(chunk)

    n_vals = " × ".join(str(len(PARAM_GRID[k])) for k in keys)
    print(f"Grid search: {n_vals} = {n_total} total combinations")
    print(f"This job   : {args.job_id + 1}/{args.n_jobs}  ({n} combos)")
    print(f"Parameters : {', '.join(keys)}")

    # Per-job output file — avoids write conflicts on shared filesystems
    job_csv = RESULTS_DIR / f"grid_search_job{args.job_id:05d}.csv"

    # Resume support: skip already-scored combos for this job
    done_keys: set[str] = set()
    existing_rows: list[dict] = []
    if job_csv.exists():
        prev = pd.read_csv(job_csv)
        for _, row in prev.iterrows():
            p = {k: row[k] for k in keys}
            done_keys.add(_combo_key(p))
            existing_rows.append(row.to_dict())
        print(f"Resuming — {len(done_keys)} combos already scored, {n - len(done_keys)} remaining.\n")
    else:
        print()

    all_rows = list(existing_rows)
    pending = [p for p in chunk if _combo_key(p) not in done_keys]
    n_pending = len(pending)

    print(f"Workers    : {args.workers}\n")

    t0 = time.time()
    done_this_run = 0

    def _flush() -> None:
        pd.DataFrame(all_rows).to_csv(job_csv, index=False)

    if args.workers <= 1:
        results_iter = ((p, _safe_run(p)) for p in pending)
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers)
        futures = {ex.submit(_safe_run, p): p for p in pending}
        results_iter = ((futures[f], f.result()) for f in as_completed(futures))

    try:
        for i, (params, row) in enumerate(results_iter, start=1):
            if row is None:
                print(f"  [{i:4d}/{n_pending}] ERROR: params={params}")
                continue
            all_rows.append(row)
            done_keys.add(_combo_key(params))
            done_this_run += 1
            _flush()

            elapsed = time.time() - t0
            rate    = done_this_run / elapsed
            remaining = n_pending - done_this_run
            eta     = remaining / rate if rate > 0 else float("inf")
            print(
                f"[{i:4d}/{n_pending}]  HOTA={row['HOTA_combined']:.3f}  "
                f"AssA={row['AssA_combined']:.3f}  IDSW={row['IDSW_total']:4d}  "
                f"| {' '.join(f'{k}={v}' for k, v in params.items())}  "
                f"(ETA {eta/60:.1f}m)"
            )
    finally:
        if args.workers > 1:
            ex.shutdown(wait=True)

    print(f"\nJob {args.job_id} done. Results: {job_csv}")
    print("Once all jobs finish, run:  python parameter_tuning/grid_search.py --merge")


if __name__ == "__main__":
    main()
