"""Entry point for the Superfly labeling tool.

Usage (from repo root):
    python -m labeler.main --video PATH --raw RAW_CSV [--ocsort OCSORT_CSV]
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Superfly Labeler")
    app.setOrganizationName("EPFL")

    backend = LabelerBackend()
    backend.load(args.video, args.raw, args.ocsort)

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
