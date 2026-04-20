"""
Build notebooks/run_tracking.ipynb from 01_tracking_pipeline.ipynb.
Identical pipeline but with no stitching step:
  - stitched_id == orig_id (passthrough)
  - tracking call wired with vial_rois, aspect_weight, behavioral_weight
  - stitching imports removed
"""
import json, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "notebooks" / "01_tracking_pipeline.ipynb"
DST  = ROOT / "notebooks" / "run_tracking.ipynb"

with open(SRC) as f:
    nb = json.load(f)

new_nb = copy.deepcopy(nb)

# ── Cell 0: markdown title ───────────────────────────────────────────────────
new_nb["cells"][0]["source"] = (
    "# Notebook — Tracking Pipeline (no stitching)\n\n"
    "Identical to `01_tracking_pipeline.ipynb` but with no Hungarian stitching.\n"
    "`stitched_id` equals `orig_id` — OC-SORT track IDs are used directly.\n\n"
    "New tracker features active:\n"
    "- OCM direction term (velocity consistency in association)\n"
    "- Vial-aware hard constraint (no cross-vial matches)\n"
    "- Behavioral consistency bonus (speed + scale plausibility)\n\n"
    "## Stages\n"
    "1. Setup & configuration\n"
    "2. (Optional) Background subtraction\n"
    "3. Draw vial ROIs\n"
    "4. RF-DETR + OC-SORT tracking → wide CSV\n"
    "5. Passthrough (no stitching) → stitched long CSV with stitched_id == orig_id\n"
    "6. Vial assignment + compact IDs → compact_tracks.csv\n"
    "7. Overlay video rendering\n\n"
    "**Replace all `PLACEHOLDER` paths with your actual file paths.**"
)

# ── Cell 1: imports — drop build_tracklets and stitch ───────────────────────
new_nb["cells"][1]["source"] = """\
import sys
sys.path.insert(0, '..')

import json
import os
import re
import cv2
import yaml
import pandas as pd
from pathlib import Path
from IPython.display import Video

from src.preprocessing import preprocess_bgsub_gui
from src.metrics import run_diagnostics, compute_stitching_objectives, print_stitching_objectives
from src.tracking import export_tracks_xy_tuple_csv_one_config
from src.stitching import wide_to_long
from src.roi import draw_and_save_vial_rois, assign_vials_and_compact_ids
from src.visualization import render_vial_overlay_video, render_raw_overlay_video, render_detections_video
from utils import save_run_params"""

# ── Cell 3: config — add aspect_weight + behavioral_weight ──────────────────
new_nb["cells"][3]["source"] = """\
# ---- EDIT THESE ----
RAW_VIDEO = r"../2024-02-05_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m\\41 DPE\\004\\2024-03-11_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m_41d_004-converted.mp4"
MODEL_ID  = "flies-123/1"   # e.g. "flies-123/1"

# Load API key from creds_config.yaml (not committed to git)
with open("../creds_config.yaml", "r") as f:
    creds_config = yaml.safe_load(f)
API_KEY = creds_config["API_KEY"]

# Load defaults from config.yaml (override below if needed)
with open("../config.yaml") as _f:
    _cfg = yaml.safe_load(_f)
_t = _cfg.get("tracker", {})
_s = _cfg.get("stitching", {})
_p = _cfg.get("preprocessing", {})

detection_confidence_rfdetr = _t.get("detection_confidence_rfdetr", 0.4)
confidence              = _t.get("confidence", 0.1)
lost_track_buffer       = _t.get("lost_track_buffer", 90)
min_matching_threshold  = _t.get("minimum_matching_threshold", 0.2)
min_consecutive_frames  = _t.get("minimum_consecutive_frames", 3)
asso_func               = _t.get("asso_func", "diou")
brownian_pos_noise      = _t.get("brownian_pos_noise", 1.0)
aspect_weight           = _t.get("aspect_weight", 0.05)
behavioral_weight       = _t.get("behavioral_weight", 0.05)
jump_factor             = _t.get("jump_factor", 2.0)
jump_iou_threshold      = _t.get("jump_iou_threshold", 0.05)
jump_inertia            = _t.get("jump_inertia", 0.05)
bg_gain                 = _p.get("bg_gain", 1.2)
bg_white_level          = _p.get("bg_white_level", 245)
bg_percentile           = _p.get("bg_percentile", 85.0)
bg_sample_stride        = _p.get("bg_sample_stride", 1)
default_end             = _p.get("default_end", 700)

# Extract short label from the "N DPE/NNN" directory convention in the video path.
_m = re.search(r'(\\d+)\\s+DPE[/\\\\](\\d+)', RAW_VIDEO)
short_name = f"{_m.group(1)}DPE_n{_m.group(2).zfill(3)}" if _m else Path(RAW_VIDEO).stem[:20]

# Auto-increment output directory
_outputs_root = Path("../outputs")
_outputs_root.mkdir(parents=True, exist_ok=True)
_existing = [d for d in _outputs_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
_next_n = max((int(d.name.split("_")[1]) for d in _existing if d.name.split("_")[1].isdigit()), default=0) + 1
_dir_name = f"run_{_next_n}_{_m.group(1)}DPE_n{_m.group(2).zfill(3)}" if _m else f"run_{_next_n}"
OUTPUT_PATH = str(_outputs_root / _dir_name)

os.makedirs(OUTPUT_PATH, exist_ok=True)

import shutil
_dest_video = os.path.join(OUTPUT_PATH, Path(RAW_VIDEO).name)
if not os.path.exists(_dest_video):
    try:
        os.link(RAW_VIDEO, _dest_video)
    except OSError:
        shutil.copy2(RAW_VIDEO, _dest_video)
PATH_TO_VID = RAW_VIDEO

print("Output dir:", OUTPUT_PATH)
print("Short name:", short_name)
print(f"detection_confidence_rfdetr={detection_confidence_rfdetr}, asso_func={asso_func}")
print(f"aspect_weight={aspect_weight}, behavioral_weight={behavioral_weight}")
print(f"jump_factor={jump_factor}, jump_iou_threshold={jump_iou_threshold}, jump_inertia={jump_inertia}")
_cap = cv2.VideoCapture(RAW_VIDEO)
save_run_params(OUTPUT_PATH, "config", {
    "video": RAW_VIDEO, "output_dir": OUTPUT_PATH, "short_name": short_name,
    "video_fps": _cap.get(cv2.CAP_PROP_FPS),
    "video_width": int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "video_height": int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "video_frames": int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    "tracker": {"detection_confidence_rfdetr": detection_confidence_rfdetr,
                 "confidence": confidence, "lost_track_buffer": lost_track_buffer,
                 "min_matching_threshold": min_matching_threshold,
                 "min_consecutive_frames": min_consecutive_frames, "asso_func": asso_func,
                 "brownian_pos_noise": brownian_pos_noise,
                 "aspect_weight": aspect_weight, "behavioral_weight": behavioral_weight,
                 "jump_factor": jump_factor, "jump_iou_threshold": jump_iou_threshold,
                 "jump_inertia": jump_inertia},
    "preprocessing": {"bg_gain": bg_gain, "bg_white_level": bg_white_level,
                       "bg_percentile": bg_percentile, "bg_sample_stride": bg_sample_stride},
})
_cap.release()"""

# ── Cell 9: tracking — add vial_rois, aspect_weight, behavioral_weight ───────
new_nb["cells"][9]["source"] = """\
WIDE_CSV = os.path.join(OUTPUT_PATH, "tracks_wide_format.csv")
DET_LOG_CSV = os.path.join(OUTPUT_PATH, "detections_raw.csv")

# ── Detection cache ───────────────────────────────────────────────────────────
# Set CACHED_DETS to a previous run's detections_raw.csv to skip RF-DETR.
# Leave as None to run inference and save fresh detections to DET_LOG_CSV.
CACHED_DETS = None  # e.g. "../outputs/run_40_41DPE_n004/detections_raw.csv"

_det_source = CACHED_DETS if (CACHED_DETS and os.path.exists(CACHED_DETS)) else DET_LOG_CSV
if CACHED_DETS and os.path.exists(CACHED_DETS):
    print(f"Using cached detections: {CACHED_DETS}")
else:
    print("No cache found — running RF-DETR inference")

df_wide, tracker = export_tracks_xy_tuple_csv_one_config(
    video_path=str(PATH_TO_VID),
    output_csv=WIDE_CSV,
    api_key=API_KEY,
    model_id=MODEL_ID,
    detection_confidence_rfdetr=detection_confidence_rfdetr,
    confidence=confidence,
    lost_track_buffer=lost_track_buffer,
    minimum_matching_threshold=min_matching_threshold,
    minimum_consecutive_frames=min_consecutive_frames,
    asso_func=asso_func,
    brownian_pos_noise=brownian_pos_noise,
    det_log_csv=_det_source,
    vial_rois=_vials,
    aspect_weight=aspect_weight,
    behavioral_weight=behavioral_weight,
    jump_factor=jump_factor,
    jump_iou_threshold=jump_iou_threshold,
    jump_inertia=jump_inertia,
    max_frames=None,
)

print(df_wide.shape)
save_run_params(OUTPUT_PATH, "tracker_output", {
    "wide_csv": WIDE_CSV, "frames": int(df_wide.shape[0]), "track_count": int(df_wide.shape[1] - 1),
})
df_wide.head()

with open(os.path.join(OUTPUT_PATH, "tracker_log.json"), "w") as _f:
    json.dump({
        "detection_log":     tracker.detection_log,
        "suppressed_tracks": tracker.suppressed_tracks,
        "min_hits":          tracker.min_hits,
        "max_age":           tracker.max_age,
    }, _f)

render_detections_video(
    video_path=str(PATH_TO_VID),
    det_log_csv=_det_source,
    out_mp4=os.path.join(OUTPUT_PATH, f"{short_name}_detections_RF-DETR.mp4"),
)"""

# ── Cell 11: markdown — rename step 5 ───────────────────────────────────────
new_nb["cells"][11]["source"] = (
    "## 5 — No stitching (passthrough)\n\n"
    "OC-SORT track IDs are used directly. `stitched_id` is set equal to `orig_id`.\n"
    "`wide_to_long` converts the wide CSV to long format as usual."
)

# ── Cell 12: replace stitching with passthrough ──────────────────────────────
new_nb["cells"][12]["source"] = """\
STITCHED_CSV = os.path.join(OUTPUT_PATH, "tracks_xy_stitched_long.csv")
LONG_CSV     = os.path.join(OUTPUT_PATH, "tracks_long_format.csv")

with open(ROI_JSON) as f:
    vial_rois = {k: tuple(v) for k, v in json.load(f).items()}

long_df = wide_to_long(pd.read_csv(WIDE_CSV), out_csv=LONG_CSV)

# No stitching: stitched_id == orig_id
stitched_df = long_df.copy()
stitched_df["stitched_id"] = stitched_df["orig_id"]
stitched_df.to_csv(STITCHED_CSV, index=False)

print(f"Track IDs (no stitching): {stitched_df['orig_id'].nunique()}")
save_run_params(OUTPUT_PATH, "stitching_output", {
    "stitched_csv": STITCHED_CSV,
    "stitched_ids": int(stitched_df["stitched_id"].nunique()),
    "original_ids": int(stitched_df["orig_id"].nunique()),
})"""

# ── Cell 15: diagnostics — same but works without separate stitching step ────
new_nb["cells"][15]["source"] = """\
df_wide = pd.read_csv(WIDE_CSV)
num_frames = int(df_wide["frame"].max()) + 1

stitching_objectives = compute_stitching_objectives(
    df_stitched       = stitched_df,
    vial_rois         = vial_rois,
    num_frames        = num_frames,
    expected_per_vial = _s.get("expected_per_vial", 7),
    short_frac        = _s.get("short_track_frac", 0.10),
)
print_stitching_objectives(stitching_objectives)
save_run_params(OUTPUT_PATH, "stitching_objectives", {k: float(v) for k, v in stitching_objectives.items()})

run_diagnostics(
    tracker              = tracker,
    df_wide              = df_wide,
    df_stitched          = stitched_df,
    df_compact           = df_compact,
    n_expected           = _s.get("expected_per_vial", 7) * len(vial_rois),
    fps                  = _s.get("fps", 30),
    vial_rois            = vial_rois,
    config               = _cfg,
    output_dir           = OUTPUT_PATH,
    stitching_objectives = stitching_objectives,
)"""

with open(DST, "w") as f:
    json.dump(new_nb, f, indent=1)

print(f"Written: {DST}")
