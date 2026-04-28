"""Entry point for the Superfly labeling tool.

Usage (from repo root):
    python -m labeler.main --video PATH --raw RAW_CSV [--ocsort OCSORT_CSV]
                           [--overlay-video PATH] [--out-dir DIR] [--fresh]
                           
    example: 
        python -m labeler.main --video r"..\..\outputs\run_88_31DPE_n005\2024-03-01_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m_31d_005-converted_raw_cropped.mp4" --raw r"..\..\outputs\run_88_31DPE_n005\detections_raw.csv" --ocsort r"..\..\outputs\run_88_31DPE_n005\tracks_wide_format.csv" --fresh

Default out-dir: <repo>/data/manual_labelling/<videostem>/  (auto-created).
Override with --out-dir for one-off runs outside this layout.

On startup the labeler populates the folder with:
    <video filename>            copied from --video (idempotent)
    detections_raw.csv          copied from --raw (idempotent)
    tracks_long.csv             melted long-format from --ocsort wide
    metadata.json               session/source/git provenance + live counts
    [overlay video]             only if --overlay-video given

Working files written by the labeler:
    <videostem>.labeler.json           session (Ctrl+S writes here)
    <videostem>.labeler.autosave.json  autosave (every 60 s if dirty)
    <videostem>.gt.csv                 ground-truth export (Ctrl+E writes here)
    <videostem>.gt_summary.txt         QC summary (written alongside the CSV)

By default, the labeler auto-resumes the newer of .labeler.json /
.labeler.autosave.json if either exists. Pass --fresh to start fresh
(useful when you want to re-seed from a freshly-rerun OC-SORT).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from .backend import LabelerBackend
from .video_provider import VideoFrameProvider


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Superfly ground-truth labeler")
    p.add_argument("--video", required=True, help="path to video file")
    p.add_argument("--raw", required=True, help="raw detection bbox CSV (frame,x1,y1,x2,y2,conf)")
    p.add_argument("--ocsort", default=None,
                   help="optional OC-SORT wide-format CSV (frame,id1,id2,...) for suggestions")
    p.add_argument("--overlay-video", default=None,
                   help="optional reference video (e.g. OC-SORT overlay) to copy into the folder")
    p.add_argument("--out-dir", default=None,
                   help="where to read/write session + ground-truth CSV "
                        "(default: <repo>/data/manual_labelling/<videostem>/)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing .labeler.json/.autosave.json and start fresh "
                        "(default: auto-resume the newer of the two if either exists)")
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

    # Auto-resume by default. Pass --fresh for a clean start.
    resume_from = ""
    if not args.fresh:
        candidates = [p for p in (session_path, autosave_path) if p.exists()]
        if candidates:
            resume_from = str(max(candidates, key=lambda p: p.stat().st_mtime))

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
    paths = derive_paths(args)

    # Populate the per-video folder with copied assets + melted long CSV.
    out_dir = Path(paths["out_dir"])
    overlay = Path(args.overlay_video) if args.overlay_video else None
    copy_actions = populate_folder(
        out_dir,
        video_path=Path(args.video),
        raw_csv=Path(args.raw),
        ocsort_wide_csv=Path(args.ocsort) if args.ocsort else None,
        overlay_video=overlay,
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
        ocsort_wide_csv=Path(args.ocsort) if args.ocsort else None,
        overlay_video=overlay,
        video_props=video_props,
        copy_actions=copy_actions,
    )

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Superfly Labeler")
    app.setOrganizationName("EPFL")

    backend = LabelerBackend()
    backend.load(
        args.video, args.raw, args.ocsort,
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
