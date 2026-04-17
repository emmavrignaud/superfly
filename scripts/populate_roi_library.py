#!/usr/bin/env python
"""
scripts/populate_roi_library.py

Walk the experiment folder, open the bg-subtraction GUI and vial ROI GUI for
each new video, and populate outputs/roi_dictionary.json.

Videos already present in the dictionary (both 'preprocessing' and 'vial_rois'
keys) are skipped — safe to re-run after a crash.

Dictionary key: relative path from root without extension, e.g.
    "31 DPE/003/2024-03-01_..._31d_003-converted"
This avoids stem collisions where two subfolders hold the same filename.

Output layout
-------------
  data/raw/<N> DPE/<num>/<video>.mp4          copy of raw video
  data/processed/<N> DPE/<num>/<video>_pp.mp4 bg-subtracted
  outputs/roi_dictionary.json                 dictionary (rename to roi_library.json when done)

Usage
-----
  python scripts\\populate_roi_library.py
  python scripts\\populate_roi_library.py --root path\\to\\experiment --dictionary outputs\\my_dict.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import preprocess_bgsub_gui
from src.roi import draw_and_save_vial_rois

_REPO_ROOT    = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT / "2024-02-05_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m"
_DEFAULT_DICT = _REPO_ROOT / "outputs" / "roi_dictionary.json"


def _load_dictionary(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_dictionary(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Populate roi_dictionary.json by running bg-sub + vial ROI GUIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=str(_DEFAULT_ROOT),
        help="Experiment folder to scan recursively for *-converted.mp4 files",
    )
    parser.add_argument(
        "--dictionary",
        default=str(_DEFAULT_DICT),
        help="Path to roi_dictionary.json (created fresh if missing)",
    )
    args = parser.parse_args()

    root      = Path(args.root)
    dict_path = Path(args.dictionary)

    if not root.exists():
        print(f"ERROR: root folder not found: {root}")
        sys.exit(1)

    # Load config
    cfg_path = _REPO_ROOT / "config.yaml"
    with open(cfg_path) as f:
        _cfg = yaml.safe_load(f)
    _p = _cfg.get("preprocessing", {})
    bg_gain          = _p.get("bg_gain", 1.2)
    bg_white_level   = _p.get("bg_white_level", 245)
    bg_percentile    = _p.get("bg_percentile", 85.0)
    bg_sample_stride = _p.get("bg_sample_stride", 1)
    default_end      = _p.get("default_end", 700)
    codec            = _p.get("codec", "mp4v")

    data_raw       = _REPO_ROOT / "data" / "raw"
    data_processed = _REPO_ROOT / "data" / "processed"

    # Glob: *-converted.mp4 naturally excludes *-converted-video.mp4 and *_pp.mp4
    videos = sorted(root.rglob("*-converted.mp4"))
    n      = len(videos)
    print(f"Found {n} video(s) under {root}\n")

    dictionary = _load_dictionary(dict_path)

    for i, video in enumerate(videos, 1):
        key     = str(video.relative_to(root).with_suffix(""))
        rel_dir = video.relative_to(root).parent
        raw_dst = data_raw       / rel_dir / video.name
        pp_dst  = data_processed / rel_dir / (video.stem + "_pp.mp4")

        entry = dictionary.get(key, {})
        if "preprocessing" in entry and "vial_rois" in entry:
            print(f"[{i}/{n}] SKIP  {key}")
            continue

        print(f"\n[{i}/{n}] {key}")

        # Step 1 — bg subtraction GUI
        if "preprocessing" not in entry:
            raw_dst.parent.mkdir(parents=True, exist_ok=True)
            pp_dst.parent.mkdir(parents=True, exist_ok=True)

            if not raw_dst.exists():
                try:
                    raw_dst.hardlink_to(video)
                except (OSError, NotImplementedError):
                    shutil.copy2(video, raw_dst)

            stored_crop = entry.get("preprocessing")
            if stored_crop:
                print("  Found existing crop params — re-running bg sub with stored crop.")
            else:
                print("  Opening bg subtraction GUI...")

            pp_path, crop_params = preprocess_bgsub_gui(
                video_path       = str(video),
                out_mp4          = str(pp_dst),
                default_end      = default_end,
                gain             = bg_gain,
                white_level      = bg_white_level,
                codec            = codec,
                bg_sample_stride = bg_sample_stride,
                bg_percentile    = bg_percentile,
                crop_params      = stored_crop,
            )

            dictionary.setdefault(key, {})
            dictionary[key]["preprocessing"] = crop_params
            dictionary[key]["video_path"]    = str(video)
            _save_dictionary(dictionary, dict_path)
            print(f"  Saved bg-sub → {pp_dst.name}")

        # Step 2 — vial ROI GUI
        if "vial_rois" not in dictionary.get(key, {}):
            print("  Opening vial ROI GUI...")
            _tmp_roi_json = pp_dst.parent / "_vial_rois_tmp.json"
            vials = draw_and_save_vial_rois(
                video_path    = str(pp_dst),
                roi_json_path = str(_tmp_roi_json),
            )
            if _tmp_roi_json.exists():
                _tmp_roi_json.unlink()

            dictionary[key]["vial_rois"] = {k: list(v) for k, v in vials.items()}
            _save_dictionary(dictionary, dict_path)
            print(f"  Saved {len(vials)} vial ROIs.")

    print(f"\nDone. Dictionary written to: {dict_path}")
    print("When all videos are processed, rename roi_dictionary.json → roi_library.json")
    print("and delete the old roi_library.json.")


if __name__ == "__main__":
    main()
