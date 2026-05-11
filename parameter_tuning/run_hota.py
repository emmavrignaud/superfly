"""Score baseline tracker output against ground truth using TrackEval's HOTA.

HOTA averages over IoU thresholds alpha in {0.05, 0.10, ..., 0.95} (19 levels)
and reports a single number plus DetA, AssA, LocA decomposition.

Inputs (produced by build_mot_files.py):
    parameter_tuning/results/mot_inputs/gt/<video>/gt/gt.txt
    parameter_tuning/results/mot_inputs/trackers/baseline/data/<video>.txt

Outputs:
    parameter_tuning/results/hota_scores/baseline/<video>_*.csv  (per-video)
    parameter_tuning/results/hota_scores/baseline/pedestrian_*.csv (summary)
    plus a brief console printout.

TrackEval bypass tricks used:
    SKIP_SPLIT_FOL=True   -> no MOT17-train nesting between folder and video
    SEQ_INFO={...}        -> no seqmap.txt, no seqinfo.ini required
    DO_PREPROC=False      -> no MOT distractor/occlusion preprocessing
                             (we have class=1 and zero_marked=1 everywhere)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# TrackEval (last updated pre-numpy-2) uses np.float / np.int / np.bool, which
# numpy removed in 1.24. Restore the aliases before importing trackeval. The
# aliases just point to the builtins — no behavior change, just survives the
# attribute lookup. Cleaner than editing third-party source.
for _name, _alias in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

import trackeval
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MOT_INPUTS = RESULTS_DIR / "mot_inputs"
HOTA_OUT = RESULTS_DIR / "hota_scores"
TRACKER_NAME = "baseline"
VIDEOS = ["13d_002", "31d_005"]


def _video_length(video: str) -> int:
    """Number of frames in the video. Per user A: full-video GT, so we use
    detections_raw max-frame + 1 (every frame YOLO ran on).
    """
    df = pd.read_csv(DATA_DIR / f"detections_raw_{video}.csv")
    return int(df["frame"].max()) + 1


def main() -> None:
    seq_info = {v: _video_length(v) for v in VIDEOS}
    print("Video lengths (frames):", seq_info)

    HOTA_OUT.mkdir(parents=True, exist_ok=True)

    dataset_config = {
        "GT_FOLDER":         str(MOT_INPUTS / "gt"),
        "TRACKERS_FOLDER":   str(MOT_INPUTS / "trackers"),
        "OUTPUT_FOLDER":     str(HOTA_OUT),
        "TRACKERS_TO_EVAL":  [TRACKER_NAME],
        "CLASSES_TO_EVAL":   ["pedestrian"],   # only class the loader accepts
        "BENCHMARK":         "fly",            # string is inert when SKIP_SPLIT_FOL=True
        "SPLIT_TO_EVAL":     "all",
        "INPUT_AS_ZIP":      False,
        "PRINT_CONFIG":      False,
        "DO_PREPROC":        False,            # no MOT-specific GT filtering
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "SEQ_INFO":          seq_info,         # bypass seqmap/seqinfo.ini
        "GT_LOC_FORMAT":     "{gt_folder}/{seq}/gt/gt.txt",
        "SKIP_SPLIT_FOL":    True,             # bypass BENCHMARK-SPLIT nesting
    }

    eval_config = {
        "USE_PARALLEL":         False,
        "NUM_PARALLEL_CORES":   1,
        "BREAK_ON_ERROR":       True,
        "RETURN_ON_ERROR":      False,
        "LOG_ON_ERROR":         str(HOTA_OUT / "error_log.txt"),
        "PRINT_RESULTS":        True,
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
    metrics = [HOTA()]

    output_res, output_msg = evaluator.evaluate([dataset], metrics)

    # Pull the per-video HOTA numbers out of the nested results dict for a
    # tidy summary at the end. Schema:
    # output_res[dataset_name][tracker_name][video_name][class]['HOTA'] -> dict
    ds_name = dataset.get_name()
    per_video = output_res[ds_name][TRACKER_NAME]
    print("\n=== HOTA summary (baseline) ===")
    print(f"{'video':<10}  {'HOTA':>6}  {'DetA':>6}  {'AssA':>6}  {'LocA':>6}")
    for video in VIDEOS + ["COMBINED_SEQ"]:
        if video not in per_video:
            continue
        h = per_video[video]["pedestrian"]["HOTA"]
        # Each is an array over the 19 alpha thresholds; report the mean.
        def m(k: str) -> float:
            return float(h[k].mean())
        print(f"{video:<10}  {m('HOTA'):>6.3f}  {m('DetA'):>6.3f}  "
              f"{m('AssA'):>6.3f}  {m('LocA'):>6.3f}")


if __name__ == "__main__":
    main()
