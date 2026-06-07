"""
draw_rois.py -- Draw preprocessing crop + vial ROIs for all videos missing them.

Usage:
    python scripts/draw_rois.py [--data-root data/raw] [--library roi_library.json]

For each *-converted.mp4 that is missing preprocessing crop or vial_rois in the
library, opens the relevant GUI(s) one at a time. Saves after every step so a
crash loses at most one entry.

GUI controls (vial ROIs):
    Drag mouse  -- draw ROI
    U           -- undo last ROI
    R           -- reset all ROIs
    Enter       -- finish (need at least 1 ROI)
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _n_frames(video_path: Path) -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Draw preprocessing crop and vial ROIs for all videos missing them"
    )
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--library", default="roi_library.json")
    parser.add_argument("--force-preprocessing", action="store_true",
                        help="Re-draw preprocessing crop even if already stored")
    parser.add_argument("--force-rois", action="store_true",
                        help="Re-draw vial ROIs even if already stored")
    parser.add_argument("--video", default=None,
                        help="Process only this video stem (e.g. 2024-02-08_..._9d_001-converted)")
    args = parser.parse_args()

    data_root = REPO_ROOT / args.data_root
    lib_path  = REPO_ROOT / args.library

    from src.preprocessing import preprocess_bgsub_gui
    from src.roi import draw_and_save_vial_rois

    library: dict = {}
    if lib_path.exists():
        with open(lib_path) as f:
            library = json.load(f)
    print(f"Library: {lib_path} ({len(library)} entries)")

    videos = sorted(data_root.rglob("*-converted.mp4"))
    if args.video:
        videos = [v for v in videos if v.stem == args.video]
        if not videos:
            print(f"No video found with stem: {args.video}")
            return
    print(f"Found {len(videos)} *-converted.mp4 in {data_root}\n")

    needs_work = []
    for v in videos:
        key = v.stem
        entry = library.get(key, {})
        need_pre = args.force_preprocessing or "preprocessing" not in entry
        need_roi = args.force_rois or "vial_rois" not in entry
        if need_pre or need_roi:
            needs_work.append((v, need_pre, need_roi))

    print(f"{len(needs_work)} videos need attention.\n")
    if not needs_work:
        print("Nothing to do.")
        return

    for i, (video, need_pre, need_roi) in enumerate(needs_work, 1):
        key = video.stem
        print(f"[{i}/{len(needs_work)}] {video.name}")

        if key not in library:
            library[key] = {}

        # --- Step 1: Preprocessing crop ---
        if need_pre:
            print("  Opening preprocessing crop GUI...")
            pp_out      = video.parent / (video.stem + "_pp.mp4")
            raw_crop_out = video.parent / (video.stem + "_raw_cropped.mp4")
            try:
                _, crop_params = preprocess_bgsub_gui(
                    str(video),
                    str(pp_out),
                    str(raw_crop_out),
                )
                library[key]["preprocessing"] = crop_params
                with open(lib_path, "w") as f:
                    json.dump(library, f, indent=2)
                print(f"  Crop saved: {crop_params}")
            except Exception as e:
                print(f"  [SKIP preprocessing] {e}")
                continue
        else:
            # Use stored crop to produce raw_cropped for the vial GUI
            crop_params = library[key].get("preprocessing")
            raw_crop_out = video.parent / (video.stem + "_raw_cropped.mp4")
            if not raw_crop_out.exists() and crop_params:
                print("  Re-generating raw cropped clip for ROI GUI...")
                pp_out = video.parent / (video.stem + "_pp.mp4")
                try:
                    preprocess_bgsub_gui(
                        str(video),
                        str(pp_out),
                        str(raw_crop_out),
                        crop_params=crop_params,
                    )
                except Exception as e:
                    print(f"  [warn] Could not re-generate raw crop: {e}")
                    raw_crop_out = video  # fall back to original

        # --- Step 2: Vial ROIs ---
        if need_roi:
            print("  Opening vial ROI GUI...")
            overlay = raw_crop_out if raw_crop_out.exists() else video
            roi_json_tmp = str(video.parent / (video.stem + "_rois_tmp.json"))
            try:
                bbox_dict = draw_and_save_vial_rois(str(overlay), roi_json_tmp)
            except Exception as e:
                print(f"  [SKIP vial ROIs] {e}")
                continue

            if not bbox_dict:
                print("  [SKIP] No ROIs drawn.")
                continue

            library[key]["vial_rois"] = bbox_dict
            with open(lib_path, "w") as f:
                json.dump(library, f, indent=2)
            print(f"  {len(bbox_dict)} vial ROI(s) saved.")

            try:
                Path(roi_json_tmp).unlink()
            except Exception:
                pass

        print(f"  Done. Library: {len(library)} entries total.")

    print(f"\nFinished. Library now has {len(library)} entries.")


if __name__ == "__main__":
    main()
