#!/usr/bin/env python
"""
scripts/populate_roi_library.py

Walk the experiment folder, open the bg-subtraction GUI and vial ROI GUI for
each new video, and populate roi_library_working.json.

Videos already present in the dictionary (both 'preprocessing' and 'vial_rois'
keys) are skipped and can be safely revisited after a crash.

Dictionary key: relative path from root without extension, e.g.
    "31 DPE/003/2024-03-01_..._31d_003-converted"
This avoids stem collisions where two subfolders hold the same filename.

Output layout
-------------
  data/raw/<N> DPE/<num>/<video>.mp4          copy of raw video
  data/processed/<N> DPE/<num>/<video>_pp.mp4 bg-subtracted
  roi_library_working.json                    collaborative working library
  roi_library.json                            finalized shared library at repo root

Usage
-----
  python scripts\\populate_roi_library.py
  python scripts\\populate_roi_library.py --root path\\to\\experiment --dictionary roi_library_working.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import preprocess_bgsub_gui
from src.roi import draw_and_save_vial_rois
from src.ui_context import WorkflowCompanion, parse_video_context
from utils import load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT / "2024-02-05_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m"
_DEFAULT_DICT = _REPO_ROOT / "roi_library_working.json"


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
        description="Populate roi_library_working.json by running bg-sub and vial ROI GUIs.",
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
        help="Path to roi_library_working.json (created fresh if missing)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    dict_path = Path(args.dictionary)

    if not root.exists():
        print(f"ERROR: root folder not found: {root}")
        sys.exit(1)

    cfg = load_config(_REPO_ROOT / "config.yaml")
    p = cfg.preprocessing

    data_raw = _REPO_ROOT / "data" / "raw"
    data_processed = _REPO_ROOT / "data" / "processed"

    videos = sorted(root.rglob("*-converted.mp4"))
    n = len(videos)
    print(f"Found {n} video(s) under {root}\n")

    app = QApplication.instance() or QApplication(sys.argv)
    companion = WorkflowCompanion(n)
    companion.show()
    companion.update_status(
        video_context=None,
        stage="Queued",
        detail="Ready to scan experiment videos",
        completed=0,
        tone="queued",
    )
    app.processEvents()

    dictionary = _load_dictionary(dict_path)

    for i, video in enumerate(videos, 1):
        key = str(video.relative_to(root).with_suffix(""))
        rel_dir = video.relative_to(root).parent
        raw_dst = data_raw / rel_dir / video.name
        pp_dst = data_processed / rel_dir / (video.stem + "_pp.mp4")
        video_context = parse_video_context(str(video))

        companion.update_status(
            video_context=video_context,
            stage="Queued",
            detail="Checking ROI library entry",
            completed=i - 1,
            tone="queued",
        )
        app.processEvents()

        entry = dictionary.get(key, {})
        if "preprocessing" in entry and "vial_rois" in entry:
            print(f"[{i}/{n}] SKIP  {key}")
            companion.update_status(
                video_context=video_context,
                stage="Skipped",
                detail="Skipped complete",
                completed=i,
                tone="skipped",
            )
            app.processEvents()
            continue

        print(f"\n[{i}/{n}] {key}")

        if "preprocessing" not in entry:
            raw_dst.parent.mkdir(parents=True, exist_ok=True)
            pp_dst.parent.mkdir(parents=True, exist_ok=True)

            if not raw_dst.exists():
                try:
                    raw_dst.hardlink_to(video)
                except (OSError, NotImplementedError):
                    shutil.copy2(video, raw_dst)

            stored_crop = entry.get("preprocessing")
            companion.update_status(
                video_context=video_context,
                stage="Preprocessing",
                detail="Manual crop selection",
                completed=i - 1,
                tone="preprocessing",
            )
            app.processEvents()
            if stored_crop:
                print("  Found existing crop params; re-running bg subtraction with stored crop.")
            else:
                print("  Opening bg subtraction GUI...")

            pp_path, crop_params = preprocess_bgsub_gui(
                video_path=str(video),
                out_mp4=str(pp_dst),
                gain=p.bg_gain,
                white_level=p.bg_white_level,
                codec=p.codec,
                bg_sample_stride=p.bg_sample_stride,
                bg_percentile=p.bg_percentile,
                crop_params=stored_crop,
                video_context=video_context,
            )

            dictionary.setdefault(key, {})
            dictionary[key]["preprocessing"] = crop_params
            dictionary[key]["video_path"] = str(video)
            _save_dictionary(dictionary, dict_path)
            print(f"  Saved bg-sub -> {pp_dst.name}")

        if "vial_rois" not in dictionary.get(key, {}):
            companion.update_status(
                video_context=video_context,
                stage="ROI",
                detail="Manual vial annotation",
                completed=i - 1,
                tone="roi",
            )
            app.processEvents()
            print("  Opening vial ROI GUI...")
            tmp_roi_json = pp_dst.parent / "_vial_rois_tmp.json"
            draw_and_save_vial_rois(
                video_path=str(video),
                roi_json_path=str(tmp_roi_json),
                video_context=video_context,
            )
            # Read the full saved JSON (includes n_flies) before deleting
            with open(tmp_roi_json) as _f:
                dictionary[key]["vial_rois"] = json.load(_f)
            tmp_roi_json.unlink()
            _save_dictionary(dictionary, dict_path)
            print(f"  Saved {len(vials)} vial ROIs.")

        companion.update_status(
            video_context=video_context,
            stage="Done",
            detail="Saved ROI library entry",
            completed=i,
            tone="done",
        )
        app.processEvents()

    companion.update_status(
        video_context=None,
        stage="Done",
        detail=f"Finished processing {n} video(s)",
        completed=n,
        tone="done",
    )
    app.processEvents()

    print(f"\nDone. Working library written to: {dict_path}")
    print("Runtime code still reads roi_library.json.")
    print("Promote roi_library_working.json into roi_library.json when the sweep is ready.")


if __name__ == "__main__":
    main()
