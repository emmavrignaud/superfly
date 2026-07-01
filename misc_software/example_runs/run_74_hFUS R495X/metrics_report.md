# Metrics Report

> Interactive version: [metrics_report.html](metrics_report.html)

## Configuration

```json
{
  "tracker": {
    "detection_confidence_rfdetr": 0.4,
    "confidence": 0.55,
    "track_activation_threshold": 0.1,
    "lost_track_buffer": 180,
    "minimum_matching_threshold": 0.2,
    "minimum_consecutive_frames": 3,
    "min_area": 20,
    "asso_func": "diou",
    "brownian_pos_noise": 15,
    "aspect_weight": 0.05
  },
  "stitching": {
    "stitching_mode": "per_vial",
    "stop_mode": "converge",
    "max_rounds": 10,
    "w_under": 15,
    "w_over": 2.0,
    "vial_count_cap": 7,
    "general_count_cap": 56,
    "fps": 30,
    "pause_threshold": 1.0,
    "edge_fraction": 0.1,
    "min_points_for_scale": 10,
    "expected_per_vial": 7,
    "short_track_frac": 0.1,
    "link_score_weights": {
      "extrap": 0.4,
      "direction": 0.3,
      "behavioral": 0.3
    },
    "direction_weights": {
      "heading_vs_gap": 0.7,
      "overall_vs_overall": 0.3
    },
    "behavioral_weights": {
      "median_velocity": 1.0,
      "pause_fraction": 1.0,
      "mean_turning_angle": 1.0,
      "mean_angular_velocity": 1.0,
      "mean_acceleration": 1.0,
      "n_large_displacements": 1.0,
      "tortuosity": 1.0
    }
  },
  "preprocessing": {
    "bg_gain": 1.2,
    "bg_white_level": 245,
    "bg_percentile": 85.0,
    "bg_sample_stride": 1,
    "default_end": 700,
    "codec": "mp4v"
  },
  "visualization": {
    "overlay_source": "raw_cropped",
    "fps_out": 30,
    "show_ids": true,
    "tick_len": 2,
    "tick_thick": 0.3,
    "chip_font_scale": 0.32,
    "label_offset_x": 10,
    "label_offset_y": -10,
    "chip_pad": 1,
    "leader_thick": 1,
    "show_border": true,
    "shadow_text": true,
    "anchor_radius": 0,
    "show_unmatched_detections": true,
    "unmatched_tolerance_k": 2.0
  },
  "roi": {
    "use_saved_roi": false,
    "snap_threshold_pct": 0.005,
    "snap_enabled": true
  }
}
```

## Summary

```
=======================================================
  TRACKER DIAGNOSTICS
=======================================================
  Frames processed        : 748
  Mean detections / frame : 12.55
  Mean emitted / frame    : 12.09
  Unique IDs in CSV       : 50
  Fragmentation ratio     : 1.4x  (50 ids / 35 expected)
  Mean ID coverage        : 23.8% of frames
  Suppressed tracks       : 21  (died before min_hits=3)
  Mean hits (suppressed)  : 1.1
  Near threshold (>=70%)  : 0.0% of suppressed

  Interpretation:
    ⚠ Detector is missing flies frequently → check preprocessing or lower confidence threshold
    ⚠ Low mean coverage per ID → tracks are very short and broken → check Q matrix (Brownian noise) or IoU threshold
=======================================================
```

## Stitching Duplicate Check

**0 duplicate (frame, stitched_id) pairs — clean.**

## Stitching Quality Objectives

| Objective | Value |
|---|---|
| vial_count_error | 12.0 |
| per_id_coverage_loss | 0.0 frames/fly |
| short_track_count | 0 |
| per_frame_id_variance | 4.639 |

## XY Trajectories

![XY Trajectories](metrics_xy_trajectories.png)

## Pipeline Diagnostics

![Pipeline Diagnostics](metrics_pipeline.png)
