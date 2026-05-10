#!/usr/bin/env python
"""
scripts/run_stitching.py  —  DEPRECATED

Post-hoc stitching is no longer part of the pipeline. All association logic
now runs live inside OC-SORT (jump round, count-aware spawning, behavioral
cost matrix).

Use scripts/run_tracking.py instead — it covers the full pipeline:
  RF-DETR + OC-SORT  →  ocsort_tracks.csv
  Vial assignment    →  ordered_tracks.csv
  Overlay videos     →  overlay_raw_ocsort.mp4 + overlay_ordered.mp4
"""

raise SystemExit(
    "\nrun_stitching.py is deprecated — use run_tracking.py instead.\n"
    "See the module docstring for details.\n"
)
