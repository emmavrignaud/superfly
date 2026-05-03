"""Entry point for the Superfly labeling tool.

Usage (from repo root):
    python -m labeler.main --video PATH --raw RAW_CSV [--tracks TRACKS_CSV]
                           [--out-dir DIR] [--fresh] [--resume PATH]
                           [--resume-autosave]

    args explained:
    --video    path to the video file to label (use the same one you ran the tracker on)
    --raw      path to the raw detections CSV (frame,x1,y1,x2,y2,conf) from RF-DETR
    --tracks   tracks CSV for seeded suggestions (compact_tracks.csv long-format
               or legacy tracks_wide_format.csv — auto-detected)
    --out-dir  override the default per-video labeling folder
    --fresh    ignore any existing session and start clean
    --resume   explicit session/autosave file to resume from
    --resume-autosave  prefer .labeler.autosave.json over .labeler.json
    --ocsort   (deprecated alias for --tracks)

    example: (run from repo root! hhhhhhh don't be like me and cd into labeler/ )
            python -m labeler.main --video outputs/run_84_31DPE_n005/2024-03-01_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m_31d_005-converted_raw_cropped.mp4 --raw outputs/run_84_31DPE_n005/detections_raw.csv --tracks outputs/run_84_31DPE_n005/compact_tracks.csv --fresh

The --tracks CSV may be either:
  - long-format  (e.g. compact_tracks.csv: frame, x, y, compact_id, ...)
  - wide-format  (legacy tracks_wide_format.csv: frame, id1, id2, ...)
The format is auto-detected from the column header.

`--ocsort` is kept as a deprecated alias for `--tracks`.

Default out-dir: <repo>/data/manual_labelling/<videostem>/  (auto-created).
Override with --out-dir for one-off runs outside this layout.

On startup the labeler populates the folder with:
    <video filename>            copied from --video (idempotent)
    detections_raw.csv          copied from --raw (idempotent)
    tracks_long.csv             long-format melt of --tracks (or copy if already long)
    metadata.json               session/source/git provenance + live counts

Working files written by the labeler:
    <videostem>.labeler.json           session (Ctrl+S writes here)
    <videostem>.labeler.autosave.json  autosave (every 60 s if dirty)
    <videostem>.gt.csv                 ground-truth export (Ctrl+E writes here)
    <videostem>.gt_summary.txt         QC summary (written alongside the CSV)

By default, the labeler resumes `.labeler.json` if it exists, otherwise
falls back to `.labeler.autosave.json`. Pass `--resume-autosave` to
explicitly prefer autosave, or `--resume PATH` to choose a file yourself.
Pass --fresh to start fresh (useful when you want to re-seed from a
freshly-rerun tracker).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from .backend import LabelerBackend
from .session import choose_resume_path
from .video_provider import VideoFrameProvider


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Superfly ground-truth labeler")
    p.add_argument("--video", required=True, help="path to video file")
    p.add_argument("--raw", required=True, help="raw detection bbox CSV (frame,x1,y1,x2,y2,conf)")
    p.add_argument("--tracks", default=None,
                   help="optional tracks CSV for OC-SORT-style suggestions. "
                        "Long-format (compact_tracks.csv) or legacy wide-format "
                        "(tracks_wide_format.csv) — auto-detected.")
    p.add_argument("--ocsort", default=None,
                   help="(deprecated alias for --tracks)")
    p.add_argument("--out-dir", default=None,
                   help="where to read/write session + ground-truth CSV "
                        "(default: <repo>/data/manual_labelling/<videostem>/)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing .labeler.json/.autosave.json and start fresh "
                        "(default: resume .labeler.json if present, else autosave)")
    p.add_argument("--resume", default=None,
                   help="explicit session/autosave file to resume from")
    p.add_argument("--resume-autosave", action="store_true",
                   help="prefer .labeler.autosave.json over .labeler.json when both exist")
    return p.parse_args(argv)


def _repo_root() -> Path:
    # labeler/main.py → labeler/ → repo root (the `superfly` directory)
    return Path(__file__).resolve().parent.parent


def derive_paths(args: argparse.Namespace) -> dict:
    """Compute the per-video labeling folder and all derived file paths."""
    stem = Path(args.video).stem
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = _repo_root() / "data" / "manual_labelling" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    session_path  = out_dir / f"{stem}.labeler.json"
    autosave_path = out_dir / f"{stem}.labeler.autosave.json"
    export_path   = out_dir / f"{stem}.gt.csv"
    summary_path  = out_dir / f"{stem}.gt_summary.txt"

    resume_from = ""
    if not args.fresh:
        if args.resume:
            explicit = Path(args.resume)
            if not explicit.exists():
                raise FileNotFoundError(f"resume file not found: {explicit}")
            resume_from = str(explicit.as_posix())
        else:
            resume_from = choose_resume_path(
                str(session_path),
                str(autosave_path),
                prefer_autosave=bool(args.resume_autosave),
            )

    return {
        "out_dir": str(out_dir.as_posix()),
        "session_path": str(session_path.as_posix()),
        "autosave_path": str(autosave_path.as_posix()),
        "export_path": str(export_path.as_posix()),
        "summary_path": str(summary_path.as_posix()),
        "resume_from": resume_from,
    }


def main(argv: list[str] | None = None) -> int:
    from .assets import init_metadata, populate_folder
    import cv2

    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Resolve --ocsort alias into args.tracks. --tracks wins if both given.
    if args.tracks is None and args.ocsort is not None:
        print("warning: --ocsort is deprecated; use --tracks instead.", file=sys.stderr)
        args.tracks = args.ocsort

    paths = derive_paths(args)

    # Populate the per-video folder with copied assets + melted long CSV.
    out_dir = Path(paths["out_dir"])
    copy_actions = populate_folder(
        out_dir,
        video_path=Path(args.video),
        raw_csv=Path(args.raw),
        tracks_csv=Path(args.tracks) if args.tracks else None,
    )

    # Snapshot video properties for metadata.json (cheap; same call as backend.load).
    cap = cv2.VideoCapture(args.video)
    if cap.isOpened():
        video_props = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    else:
        video_props = {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0}
    cap.release()

    init_metadata(
        out_dir,
        repo_root=_repo_root(),
        video_path=Path(args.video),
        raw_csv=Path(args.raw),
        tracks_csv=Path(args.tracks) if args.tracks else None,
        video_props=video_props,
        copy_actions=copy_actions,
    )

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Superfly Labeler")
    app.setOrganizationName("EPFL")

    backend = LabelerBackend()
    backend.load(
        args.video, args.raw, args.tracks,
        session_path=paths["session_path"],
        autosave_path=paths["autosave_path"],
        export_path=paths["export_path"],
        resume_from=paths["resume_from"],
        out_dir=paths["out_dir"],
        summary_path=paths["summary_path"],
    )
    if paths["resume_from"]:
        print(f"resumed from {paths['resume_from']}")
    else:
        if args.fresh:
            print("starting fresh (--fresh given)")
        else:
            print("starting fresh (no existing session found)")

    provider = VideoFrameProvider(backend)

    engine = QQmlApplicationEngine()
    engine.addImageProvider("videoframes", provider)
    engine.rootContext().setContextProperty("backend", backend)

    qml_path = Path(__file__).parent / "qml" / "main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        print("ERROR: failed to load QML", file=sys.stderr)
        return 1

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
