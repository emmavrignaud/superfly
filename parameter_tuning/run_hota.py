"""Score a prepared mot_inputs directory with TrackEval's HOTA.

Expected on-disk layout (produced by `build_mot_files.build`):
    <mot_inputs_dir>/gt/<video>/gt/gt.txt
    <mot_inputs_dir>/trackers/<tracker_name>/data/<video>.txt

HOTA averages over IoU thresholds alpha in {0.05, 0.10, ..., 0.95} (19 levels)
and reports a single number plus DetA, AssA, LocA decomposition.

TrackEval bypass tricks used:
    SKIP_SPLIT_FOL=True   -> no MOT17-train nesting between folder and video
    SEQ_INFO={...}        -> no seqmap.txt, no seqinfo.ini required
    DO_PREPROC=False      -> no MOT distractor/occlusion preprocessing
                             (we have class=1 and zero_marked=1 everywhere)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# TrackEval (last updated pre-numpy-2) uses np.float / np.int / np.bool, which
# numpy removed in 1.24. Restore the aliases before importing trackeval. No
# behaviour change, just survives the attribute lookup.
for _name, _alias in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

__all__ = ["score"]


# ── public API ──────────────────────────────────────────────────────────────

def score(
    *,
    mot_inputs_dir: str | Path,
    tracker_name: str,
    sequences: dict[str, int],
    output_dir: str | Path | None = None,
    print_results: bool = True,
) -> dict:
    """Run TrackEval HOTA on the prepared MOT layout.

    Args:
        mot_inputs_dir: root containing gt/ and trackers/ subfolders.
        tracker_name:   which tracker subfolder to score.
        sequences:      {video_name: num_frames}. Drives TrackEval's SEQ_INFO,
                        bypassing the seqmap.txt / seqinfo.ini machinery.
        output_dir:     where TrackEval writes its per-alpha CSVs. If None,
                        defaults to <mot_inputs_dir>/../hota_scores.
        print_results:  whether TrackEval prints its formatted summary.

    Returns:
        Nested dict keyed by [dataset_name][tracker_name][video][class].
        Caller usually pulls [...][video]["pedestrian"]["HOTA"] out of this.
    """
    import trackeval
    from trackeval.datasets import MotChallenge2DBox
    from trackeval.metrics import HOTA

    mot_inputs_dir = Path(mot_inputs_dir)
    if output_dir is None:
        output_dir = mot_inputs_dir.parent / "hota_scores"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = {
        "GT_FOLDER":         str(mot_inputs_dir / "gt"),
        "TRACKERS_FOLDER":   str(mot_inputs_dir / "trackers"),
        "OUTPUT_FOLDER":     str(output_dir),
        "TRACKERS_TO_EVAL":  [tracker_name],
        "CLASSES_TO_EVAL":   ["pedestrian"],   # only class the loader accepts
        "BENCHMARK":         "fly",
        "SPLIT_TO_EVAL":     "all",
        "INPUT_AS_ZIP":      False,
        "PRINT_CONFIG":      False,
        "DO_PREPROC":        False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "SEQ_INFO":          dict(sequences),
        "GT_LOC_FORMAT":     "{gt_folder}/{seq}/gt/gt.txt",
        "SKIP_SPLIT_FOL":    True,
    }
    eval_config = {
        "USE_PARALLEL":         False,
        "NUM_PARALLEL_CORES":   1,
        "BREAK_ON_ERROR":       True,
        "RETURN_ON_ERROR":      False,
        "LOG_ON_ERROR":         str(output_dir / "error_log.txt"),
        "PRINT_RESULTS":        print_results,
        "PRINT_ONLY_COMBINED":  False,
        "PRINT_CONFIG":         False,
        "TIME_PROGRESS":        False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY":       True,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED":      True,
        "PLOT_CURVES":          False,
    }
    dataset = MotChallenge2DBox(dataset_config)
    evaluator = trackeval.Evaluator(eval_config)
    output_res, _ = evaluator.evaluate([dataset], [HOTA()])
    return output_res


def summary_table(results: dict, tracker_name: str, videos: list[str]) -> list[dict]:
    """Flatten the TrackEval output into one dict per video (+ COMBINED_SEQ).

    Each entry: {video, HOTA, DetA, AssA, LocA} — values are floats (means
    over the 19 alpha thresholds).
    """
    ds_name = next(iter(results.keys()))
    per_video = results[ds_name][tracker_name]
    rows: list[dict] = []
    for v in videos + ["COMBINED_SEQ"]:
        if v not in per_video:
            continue
        h = per_video[v]["pedestrian"]["HOTA"]
        rows.append({
            "video": v,
            "HOTA": float(h["HOTA"].mean()),
            "DetA": float(h["DetA"].mean()),
            "AssA": float(h["AssA"].mean()),
            "LocA": float(h["LocA"].mean()),
        })
    return rows


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(description="Score a prepared mot_inputs dir with HOTA.")
    p.add_argument("--mot-inputs-dir", required=True, type=Path)
    p.add_argument("--tracker-name", required=True)
    p.add_argument("--seq", action="append", metavar="NAME=N", required=True,
                   help="repeat for each sequence: --seq 13d_002=324 --seq 31d_005=363")
    p.add_argument("--output-dir", default=None, type=Path)
    args = p.parse_args()

    seqs: dict[str, int] = {}
    for item in args.seq:
        if "=" not in item:
            p.error(f"--seq expects NAME=N (got {item!r})")
        name, n = item.split("=", 1)
        seqs[name.strip()] = int(n)

    results = score(
        mot_inputs_dir=args.mot_inputs_dir,
        tracker_name=args.tracker_name,
        sequences=seqs,
        output_dir=args.output_dir,
    )
    rows = summary_table(results, args.tracker_name, list(seqs.keys()))
    print(f"\n=== HOTA summary ({args.tracker_name}) ===")
    print(f"{'video':<22}  {'HOTA':>6}  {'DetA':>6}  {'AssA':>6}  {'LocA':>6}")
    for r in rows:
        print(f"{r['video']:<22}  {r['HOTA']:>6.3f}  {r['DetA']:>6.3f}  "
              f"{r['AssA']:>6.3f}  {r['LocA']:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
