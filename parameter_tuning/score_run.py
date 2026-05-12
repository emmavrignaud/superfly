"""End-to-end HOTA scoring for one tracker run folder.

Reads a tracker run directory (e.g. outputs/run_104), figures out which video
the run was on, locates the matching ground truth in the labeler session
folder, builds MOT files inside the run dir, scores with TrackEval, writes a
hota.json next to the run's other artifacts.

Returns None (no error) when no GT is available for the run's video.

Usage:
    from parameter_tuning import score_run
    result = score_run("outputs/run_104")
    # result is None if no GT, else dict with summary + raw results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .build_mot_files import build
from .run_hota import score, summary_table

__all__ = ["score_run", "find_gt", "video_stem_from_run"]

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELER_ROOT = REPO_ROOT / "data" / "manual_labelling"


def video_stem_from_run(run_dir: Path) -> str:
    """Return the cropped-video stem the labeler uses for this run.

    Newer runs expose preprocessing.video_raw_cropped directly. Older runs
    feed an already-cropped video as input; in that case the input video
    stem is itself the labeler stem.
    """
    params = json.loads((run_dir / "run_params.json").read_text(encoding="utf-8"))
    cropped = params.get("preprocessing", {}).get("video_raw_cropped")
    if cropped:
        return Path(cropped).stem
    return Path(params["config"]["video"]).stem


def find_gt(video_stem: str) -> tuple[Path, str] | tuple[None, None]:
    """Locate the labeler-exported GT for a video.

    Tries the stem as-is, then ``<stem>_raw_cropped`` (handles pipelines that
    crop internally and don't record the cropped path in run_params).

    Returns ``(gt_path, resolved_stem)`` on success, ``(None, None)`` otherwise.
    """
    candidates = [video_stem]
    if not video_stem.endswith("_raw_cropped"):
        candidates.append(f"{video_stem}_raw_cropped")
    for stem in candidates:
        p = LABELER_ROOT / stem / f"{stem}.gt.csv"
        if p.exists():
            return p, stem
    return None, None


def _check_video_alignment(run_dir: Path, video_stem: str) -> None:
    """Warn if the run's video dimensions/frame-count don't match the
    annotated video's. A mismatch means GT coordinates were drawn on a
    different frame than the tracker produced predictions on, so HOTA
    scores are meaningless — but we only warn (not error) so the caller
    can choose to inspect the numbers anyway.
    """
    meta_path = LABELER_ROOT / video_stem / "metadata.json"
    params_path = run_dir / "run_params.json"
    if not (meta_path.exists() and params_path.exists()):
        return  # nothing to compare against; skip silently

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        params = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return  # malformed files; don't block scoring

    gt_v = meta.get("video", {})
    run_v = params.get("config", {})
    gt_dims = (gt_v.get("width"), gt_v.get("height"), gt_v.get("frame_count"))
    run_dims = (run_v.get("video_width"), run_v.get("video_height"),
                run_v.get("video_frames"))
    if None in gt_dims or None in run_dims:
        return
    if gt_dims != run_dims:
        gt_w, gt_h, gt_f = gt_dims
        rw, rh, rf = run_dims
        print(
            f"  WARNING: run video ({rw}x{rh}, {rf} frames) does not match "
            f"GT-annotated video ({gt_w}x{gt_h}, {gt_f} frames). HOTA scores "
            f"may be invalid — coordinates likely don't share a frame.")


def _video_length(dets_csv: Path) -> int:
    """max(frame)+1 from detections_raw — every frame YOLO ran on.
    Used as the SEQ_INFO length passed to TrackEval.
    """
    return int(pd.read_csv(dets_csv)["frame"].max()) + 1


def score_run(run_dir: str | Path, *, print_results: bool = True) -> dict | None:
    """Score one tracker run against its (labeler-resident) ground truth.

    Returns:
        None: no GT found for this run's video — caller can ignore.
        dict: {"video": str, "tracker_name": str, "summary": [...],
               "raw": <TrackEval nested dict>}
              Also written to <run_dir>/hota.json.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "run_params.json").exists():
        raise FileNotFoundError(f"{run_dir} has no run_params.json — not a tracker run dir")

    candidate_stem = video_stem_from_run(run_dir)
    gt_csv, video_stem = find_gt(candidate_stem)
    if gt_csv is None:
        msg = f"no GT found for video {candidate_stem!r}; skipping HOTA"
        if print_results:
            print(msg)
        return None

    _check_video_alignment(run_dir, video_stem)

    tracks_csv = run_dir / "ordered_tracks.csv"
    dets_csv = run_dir / "detections_raw.csv"
    for required in (tracks_csv, dets_csv):
        if not required.exists():
            raise FileNotFoundError(f"{run_dir} missing required artifact: {required.name}")

    # Build MOT files inside the run dir; tracker_name = run folder name so
    # multiple scored runs never collide on disk.
    mot_dir = run_dir / "mot_inputs"
    tracker_name = run_dir.name
    n_gt, n_tr = build(
        gt_csv=gt_csv, tracks_csv=tracks_csv, dets_csv=dets_csv,
        video_name=video_stem, out_dir=mot_dir, tracker_name=tracker_name,
    )
    if print_results:
        print(f"  built {n_gt} GT + {n_tr} tracker MOT rows for {video_stem}")

    results = score(
        mot_inputs_dir=mot_dir,
        tracker_name=tracker_name,
        sequences={video_stem: _video_length(dets_csv)},
        output_dir=run_dir / "hota_scores",
        print_results=print_results,
    )
    rows = summary_table(results, tracker_name, [video_stem])

    payload = {
        "video": video_stem,
        "tracker_name": tracker_name,
        "gt_csv": str(gt_csv.as_posix()),
        "summary": rows,
    }
    (run_dir / "hota.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if print_results:
        print(f"  wrote {run_dir / 'hota.json'}")
    return payload


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(description="Score one tracker run dir with HOTA.")
    p.add_argument("run_dir", type=Path, help="path to outputs/run_XXX")
    args = p.parse_args()
    result = score_run(args.run_dir)
    if result is None:
        return 0
    # Pretty print
    print(f"\n=== HOTA summary ({result['tracker_name']}) ===")
    print(f"{'video':<48}  {'HOTA':>6}  {'DetA':>6}  {'AssA':>6}  {'LocA':>6}")
    for r in result["summary"]:
        print(f"{r['video']:<48}  {r['HOTA']:>6.3f}  {r['DetA']:>6.3f}  "
              f"{r['AssA']:>6.3f}  {r['LocA']:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
