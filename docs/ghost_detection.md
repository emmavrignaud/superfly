# Ghost Detection for Occluded Flies

## Background

When two flies overlap in a vial one disappears from the detector.
The tracker marks it as unmatched, Kalman-predicts its position for
however long it's hidden, then typically assigns a new ID when it
reappears — breaking the track. This is the most common source of
ID switches in dense vials.

## Approach

Because we track offline (fixed video + detection cache), we can use
look-ahead: run the tracker once to find gaps, then run it again with
synthetic detections injected during those gaps.

**Pass 1** — normal tracking, output used for analysis only

**Gap analysis:**
- Per-track gaps in the CSV are found by scanning consecutive frames where a track disappears then reappears
- Gap must be short (≤ `occlusion_max_gap`, default 90 frames ≈ 3s)
- Vial count must have been full just before the gap and dropped during it — rules out ordinary detection misses
- Top-exit guard: if the track's centroid was within `top_exit_px` of the vial ROI top edge at disappearance, it exited the vial; no ghost is injected and the event is logged as a `top_exit`

**Ghost injection:**
- One synthetic detection per gap frame, confidence < `det_thresh` so it goes through the jump-low pass and can only UPDATE an existing tracker, never spawn a new one
- Position = occluder centroid + (`offset_fraction` × bbox_w) in the direction from the occluder toward the missing fly's last known position — the ghost sits in the occluder's shadow, not on top of it
- Occluder = the closest active track in the same vial to the missing fly's last known position
- Falls back to frozen last-known position when no other track is visible in the vial

**Pass 2** — tracker re-run with ghost detections in `det_by_frame`; original track ID stays alive through the gap, preventing the ID switch on reappearance

## Top Exits and Reentries

Tracks that hit the vial ROI top edge are flagged as `top_exit_events`.
Tracks that first appear near the top are flagged as `top_reentry_events`.
Both are saved to `tracker_log.json` — no change to the CSV or track IDs,
just metadata so downstream analysis can choose to include or exclude them.

## Configuration

All parameters live under `tracker.ghost_detection` in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Off by default — opt in per run |
| `offset_fraction` | `0.5` | Ghost placed N bbox-widths from occluder toward missing fly's last position |
| `confidence` | `0.45` | Keep below `tracker.confidence` so ghosts update tracks but never spawn new ones |
| `occlusion_max_gap` | `90` | Max frames of gap before we stop treating it as an occlusion (≈3s at 30fps) |
| `top_exit_px` | `2` | Centroid within N px of vial ROI top edge = top exit; no ghost fires |

To enable, set in `config.yaml`:
```yaml
tracker:
  ghost_detection:
    enabled: true
```

Requires `n_flies` to be set per vial in the ROI JSON (the GUI writes this automatically).

## Debugging

`tracker_log.json` gains three new fields when ghost detection runs:

```json
{
  "ghost_log": [
    {"frame": 155, "missing_track_id": 7, "occluder_track_id": 3,
     "ghost_cx": 236.1, "ghost_cy": 158.4, "vial": "vial1"}
  ],
  "top_exit_events": [
    {"track_id": 12, "frame": 430, "vial": "vial2"}
  ],
  "top_reentry_events": [
    {"track_id": 15, "frame": 445, "vial": "vial2"}
  ]
}
```

Use `ghost_log` to verify the ghost fired at the right frames, shadowed the right occluder, and was placed at a sensible position. If ghosts are firing incorrectly (e.g. on flies that exited normally), increase `top_exit_px` or decrease `occlusion_max_gap`.
