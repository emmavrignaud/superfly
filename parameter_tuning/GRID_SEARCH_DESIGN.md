# Grid Search Design

This document explains the parameter grid used in `grid_search.py` — what each parameter does, why it was included or excluded from the sweep, and the reasoning behind the chosen values.

---

## Overview

The grid search exhaustively evaluates tracker association parameters across two annotated videos. RF-DETR detection is bypassed entirely — each run uses cached `detections_raw.csv` files so only tracker logic varies. Watershed splitting is also disabled for the same reason.

**Sequences evaluated:**

| Sequence | Run | Baseline HOTA | Baseline AssA | Baseline IDSW |
|----------|-----|---------------|---------------|---------------|
| 13d_002  | run_112 | 0.683 | 0.642 | 34 |
| 31d_005  | run_114_31DPE_n005 | 0.522 | 0.428 | 195 |
| Combined | — | 0.600 | 0.530 | 232 |

These two videos were chosen because they bracket the difficulty range: 13d_002 is well-tracked at baseline, 31d_005 has many identity switches and represents the harder case (more occlusion, more flies per vial on average).

**Ground truth:** `parameter_tuning/data/ground_truth_{seq}_cleaned.csv` — cleaned copies of the manually annotated GT, filtered to remove stray few-frame tracks. See `project_gt_files.md` for provenance.

**Scale:** ~40M combinations. Designed for cluster with `--job-id N --n-jobs TOTAL` (round-robin slice). Results written incrementally to `results/grid_search_results.csv` — safe to interrupt and resume.

---

## Swept parameters

### Core association

**`confidence` — [0.0, 0.10, 0.25, 0.55]**
Detection confidence threshold applied before association. Low values (0.0) include noisy detections and risk false tracks; high values (0.55) may drop real flies with partial occlusion. The spread covers the full range from no filtering to aggressive filtering.

**`minimum_matching_threshold` — [0.01, 0.05, 0.10, 0.25, 0.50]**
Minimum IoU/dIoU score for a detection–track match to be accepted. Higher values force the tracker to only link detections that substantially overlap with predictions, reducing false associations across vials. Lower values are more permissive and help when the Kalman prediction drifts.

**`inertia` — [0.0, 0.10, 0.25, 0.50]**
How much the tracker blends the previous velocity into the current Kalman prediction. Higher inertia makes tracks smoother but slower to adapt to sudden direction changes. Flies can stop and reverse quickly, so we want to find the balance.

### Kalman noise

**`brownian_pos_noise` — [5, 15, 30, 50]**
Process noise added to the Kalman filter's position estimate (in pixels). Higher values make the filter less confident in its own prediction and more willing to accept detections that are further away. Flies in vials move unpredictably, so this needs to be large enough to handle sudden jumps without being so large that cross-vial associations become possible.

### Direction lookback

**`delta_t` — [3, 10, 30, -1]**
Number of previous frames used to estimate a track's direction for the OCM (Observation-Centric Momentum) velocity consistency check. `-1` means use the full frame history. Short windows (3) are sensitive to recent motion; long windows (30, -1) capture persistent behavioral trends but can mislead after a fly stops or reverses.

### Jump round

These three parameters control the second-chance "jump" association pass, which links detections that were missed in the primary pass (typically occluded flies that reappear).

**`jump_factor` — [1.0, 2.0, 3.0, 4.0]**
Multiplier applied to the search radius for jump associations. Higher values allow linking across larger gaps — useful when a fly was occluded for several frames and has drifted from its predicted position.

**`jump_iou_threshold` — [0.01, 0.05, 0.20]**
Minimum overlap score required for a jump match to be accepted. Lower values allow looser spatial matches during the jump pass. Previously fixed at 0.05 — added to the sweep because it interacts with `jump_factor`.

**`jump_inertia` — [0.0, 0.05, 0.20]**
Inertia weight applied specifically during the jump pass. Decoupled from the main `inertia` so the jump round can apply different smoothing. Previously fixed at 0.05.

### Overlap handling

Flies in adjacent vials occasionally produce bounding boxes that overlap. These parameters control how the tracker down-weights associations across overlapping detections.

**`overlap_weight_scale` — [1.0, 3.0, 6.0, 10.0]**
How strongly the overlap penalty is applied. Higher values more aggressively suppress associations involving overlapping detections.

**`overlap_iou_scale` — [0.05, 0.10, 0.20]**
The IoU threshold at which the overlap penalty starts to kick in. Lower values apply the penalty to even slightly overlapping boxes.

### Behavioral fingerprint

Instead of sweeping 5 behavioral weight values independently (which would produce tens of thousands of near-redundant combinations once normalized), we define 12 named presets covering the meaningful repartitions. `bw_preset` is an index into `BW_PRESETS`.

**`bw_preset` — 12 presets**

| Name | speed | turning | pause | accel | tortuosity |
|------|-------|---------|-------|-------|------------|
| off | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| speed_only | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| speed_turn | 0.50 | 0.50 | 0.00 | 0.00 | 0.00 |
| speed_tort | 0.50 | 0.00 | 0.00 | 0.00 | 0.50 |
| speed_pause | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 |
| kinematics | 0.50 | 0.00 | 0.00 | 0.50 | 0.00 |
| path_shape | 0.00 | 0.50 | 0.00 | 0.00 | 0.50 |
| speed_heavy | 0.60 | 0.10 | 0.10 | 0.10 | 0.10 |
| equal_all | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| no_speed | 0.00 | 0.25 | 0.25 | 0.25 | 0.25 |
| pause_heavy | 0.20 | 0.10 | 0.50 | 0.10 | 0.10 |
| tort_heavy | 0.20 | 0.10 | 0.10 | 0.10 | 0.50 |

Weights are normalized to sum to 1 (or 0 for `off`). `bw_scale` (bbox size similarity) was excluded — fly bboxes are nearly identical in size so it has no discriminative power.

The `off` preset serves as a clean ablation baseline: if behavioral fingerprinting hurts or is neutral, we know not to use it.

### Count-aware spawning

**`w_under` — [5.0, 15.0, 30.0]**
Penalty weight applied when the number of active tracks falls below `expected_count`. Higher values more aggressively spawn new tracks to reach the expected count. Under-counting is more costly than over-counting because a missing track means lost data for the whole duration.

**`w_over` — [1.0, 2.0]**
Penalty weight for having more active tracks than `expected_count`. Kept to a small range (1–2) because mild over-counting is preferable to aggressively suppressing real flies.

`expected_count` is **not swept** — it is fixed to the known true fly count per sequence (13d_002 = 38, 31d_005 = 38, both verified against per-vial counts in `roi_library_working.json` and against the maximum simultaneous tracks in the annotated GT).

---

## Fixed parameters

| Parameter | Value | Reason fixed |
|-----------|-------|-------------|
| `detection_confidence_rfdetr` | 0.4 | Detector threshold — not a tracker param |
| `lost_track_buffer` | 400 | Generous buffer so tracks survive long occlusions; not the bottleneck |
| `minimum_consecutive_frames` | 1 | All detections considered; filtering handled by `confidence` |
| `min_area` | 20 | Removes sub-pixel noise; not worth sweeping |
| `asso_func` | diou | Committed to dIoU; switching association function is a separate experiment |
| `aspect_weight` | 0.05 | Fly bboxes are all similar aspect ratio — near-zero discriminative power |
| `edge_fraction` | 0.1 | Geometric constant; not a tuning parameter |
| `watershed_cfg` | None | Watershed disabled — only tuning tracker, not detector post-processing |

---

## Two-stage strategy

This grid search only covers the tracker association parameters. A separate, smaller grid search (`relink_grid_search.py`) tunes the **re-linking** post-processing pass, which stitches broken tracks back together using behavioral similarity. Re-linking is run after the tracker and depends on the `ocsort_tracks_obs_logs.json` output, so it is tuned independently once good tracker parameters are found here.
