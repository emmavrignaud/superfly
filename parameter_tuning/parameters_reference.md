# Tracking Parameters — Full Reference

Generated 2026-05-12. Current values reflect `config.yaml` + hardcoded defaults in `tracking.py` / `ocsort.py`.

---

## Detection

| Parameter | Config key | Current | Where used | Grid search range |
|---|---|---|---|---|
| RF-DETR confidence | `detection_confidence_rfdetr` | 0.4 | Filters RF-DETR output before anything enters the tracker | 0.3–0.6 |
| Min bbox area | `min_area` | 20 px² | Drops detections smaller than this after RF-DETR | 10–50 |

---

## Core Tracker (Round 1 Association)

| Parameter | Config key | Current | Where used | Grid search range |
|---|---|---|---|---|
| Tracker match threshold | `confidence` → `det_thresh` | 0.55 | Min score for a detection to be considered at all in round 1 | 0.3–0.7 |
| Min IoU to keep match | `minimum_matching_threshold` → `iou_threshold` | 0.2 | Hungarian assignment — pairs below this are rejected | 0.05–0.4 |
| Direction weight (inertia) | **not in config** → `inertia` | **0.2** (hardcoded in tracking.py) | Scales `angle_diff_cost` — how strongly "is tracker heading toward this detection?" influences assignment | 0.05–0.5 |
| Velocity lookback | **not in config** → `delta_t` | **3** (hardcoded) | How many frames back to compute tracker velocity direction | 1–5 |
| IoU variant | `asso_func` | diou | The base similarity metric (iou / giou / diou / ciou / hmiou) | categorical |
| Kalman position noise | `brownian_pos_noise` | 15 | Kalman Q matrix — higher = more tolerant of fast erratic movement | 5–30 |
| Aspect ratio bonus | `aspect_weight` | 0.05 | Bonus for matching same aspect-ratio boxes — near-useless for flies | 0.0–0.1 |
| Behavioral bonus | `behavioral_weight` | 0.05 | Speed plausibility + scale bonus on normal frames | 0.0–0.3 |
| Behavioral bonus (overlap) | `behavioral_weight_overlap` | 0.30 | Same bonus but only active when detections overlap (validated by overlap_analysis.py) | 0.1–0.5 |
| Overlap IoU scale | **not in config** → `overlap_iou_scale` | **0.1** | IoU is multiplied by this during overlaps — suppresses positional signal so behavioral bonus can dominate | 0.05–0.3 |
| Edge exclusion | **not in config** → `edge_fraction` | **0.1** | Detections within 10% of vial wall excluded from overlap handling | 0.05–0.2 |

---

## Track Lifecycle

| Parameter | Config key | Current | Where used | Grid search range |
|---|---|---|---|---|
| Ghost track lifetime | `lost_track_buffer` → `max_age` | 400 | Frames a track survives without a detection before deletion — very high for short clips | 50–500 |
| Min hits to appear | `minimum_consecutive_frames` → `min_hits` | 1 | Consecutive matched frames before a track enters the output CSV | 1–5 |

---

## Jump Round (Round 2 Association)

| Parameter | Config key | Current | Where used | Grid search range |
|---|---|---|---|---|
| Jump factor | `jump_factor` | 2.0 | Velocity scale + bbox inflation for searching lost tracks | 1.0–4.0 |
| Jump IoU threshold | `jump_iou_threshold` | 0.05 | Looser matching threshold in round 2 | 0.01–0.15 |
| Jump direction weight | `jump_inertia` | 0.05 | Direction consistency weight in round 2 (lower = direction less reliable after gap) | 0.0–0.2 |

---

## Count-Aware Spawning

All three are wired through `tracking.py → ocsort.py` but have no config entries — using hardcoded defaults.

| Parameter | Config key | Current | Where used | Grid search range |
|---|---|---|---|---|
| Expected fly count | **not in config** → `expected_count` | **None** (disabled) | Steers spawner toward this many active tracks total | 42 (fixed per experiment) |
| Under-count penalty | **not in config** → `w_under` | **15.0** | Cost per tracker below `expected_count` — heavily penalises missing flies | 5.0–30.0 |
| Over-count penalty | **not in config** → `w_over` | **2.0** | Cost per tracker above `expected_count` — lightly penalises ghost tracks | 0.5–5.0 |

---

## Relink (Second-Pass ID Swap Correction)

| Parameter | Config key | Current | Where used | Grid search range |
|---|---|---|---|---|
| Min track length | `relink: min_length` | 10 | Tracks shorter than this are ignored by relink | 5–20 |
| Swap threshold | `relink: swap_threshold` | 0.05 | Min behavioral improvement required to accept a swap (lowered from 0.2 on 2026-05-12) | 0.01–0.2 |
| Confidence weight | `relink: confidence_weight` | 0.0 | How much the per-frame match score penalises swapping — set to 0 because high IoU at overlap is a false signal | 0.0 (fixed) |
| Speed weight | `relink: behavioral_weights: median_speed` | 0.3 | Weight on speed difference in behavioral profile comparison (down-weighted — weak signal for slow old flies) | 0.0–1.0 |
| Turning angle weight | `relink: behavioral_weights: mean_turning_angle` | 2.0 | Weight on turning angle — overlap often causes path deflection | 0.5–3.0 |
| Angular velocity weight | `relink: behavioral_weights: mean_angular_velocity` | 1.5 | Weight on angular velocity difference | 0.5–2.5 |
| Tortuosity weight | `relink: behavioral_weights: tortuosity` | 2.0 | Weight on path tortuosity — overlap deflects trajectory | 0.5–3.0 |
| Pause fraction weight | `relink: behavioral_weights: pause_fraction` | 1.0 | Weight on fraction of time spent paused | 0.5–2.0 |
| Acceleration weight | `relink: behavioral_weights: mean_acceleration` | 1.0 | Weight on mean acceleration difference | 0.5–2.0 |
| Large displacements weight | `relink: behavioral_weights: n_large_displacements` | 1.0 | Weight on count of large jumps | 0.5–2.0 |

---

## Current HOTA Scores (baseline for grid search)

Runs: `run_112` (13d_002) and `run_114_31DPE_n005` (31d_005), evaluated 2026-05-12.

| video | HOTA | DetA | AssA | MOTA | MOTP | IDSW | IDF1 |
|---|---|---|---|---|---|---|---|
| 13d_002 | 0.683 | 0.727 | 0.642 | 0.939 | 0.784 | 34 | 0.887 |
| 31d_005 | 0.474 | 0.607 | 0.372 | 0.704 | 0.736 | 1349 | 0.647 |
| COMBINED | 0.578 | 0.667 | 0.507 | 0.821 | 0.760 | 1383 | 0.767 |

Previous baseline (before parameter changes on 2026-05-11/12):

| video | HOTA | DetA | AssA | MOTA | MOTP | IDSW | IDF1 |
|---|---|---|---|---|---|---|---|
| 13d_002 | 0.565 | 0.602 | 0.531 | 0.834 | 0.684 | 24 | 0.843 |
| 31d_005 | 0.438 | 0.555 | 0.347 | 0.629 | 0.702 | 1257 | 0.610 |

---

## What Needs Wiring Before Grid Search

The following parameters are fully implemented in `ocsort.py` and `tracking.py` but have **no `config.yaml` entry** — they use hardcoded defaults. Adding a config entry + passing them in the notebook tracking cell is all that's needed:

- `inertia` (round-1 direction weight, default 0.2)
- `delta_t` (velocity lookback, default 3)
- `overlap_iou_scale` (default 0.1)
- `edge_fraction` (default 0.1)
- `expected_count` (default None / disabled)
- `w_under` (default 15.0)
- `w_over` (default 2.0)
