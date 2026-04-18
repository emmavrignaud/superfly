"""
Shared UI helpers for video-aware PyQt workflows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class VideoContext:
    label: str
    dpe: str
    sample: str
    date: str
    source_name: str


_PALETTES = {
    "preprocessing": {"bg": "#89b4fa", "fg": "#11111b"},
    "roi": {"bg": "#a6e3a1", "fg": "#11111b"},
    "queued": {"bg": "#f9e2af", "fg": "#11111b"},
    "skipped": {"bg": "#6c7086", "fg": "#f5e0dc"},
    "done": {"bg": "#94e2d5", "fg": "#11111b"},
    "idle": {"bg": "#45475a", "fg": "#cdd6f4"},
    "dpe": {"bg": "#89b4fa", "fg": "#11111b"},
    "sample": {"bg": "#94e2d5", "fg": "#11111b"},
    "date": {"bg": "#f9e2af", "fg": "#11111b"},
}


def parse_video_context(path_or_key: str) -> VideoContext:
    """
    Build a display-friendly context from a video path or relative dataset key.
    """
    raw = str(path_or_key or "")
    normalized = raw.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]

    source_name = Path(parts[-1] if parts else raw).stem or "Unknown video"
    dpe_num = None
    sample = None

    for idx, part in enumerate(parts):
        match = re.fullmatch(r"(\d+)\s+DPE", part, flags=re.IGNORECASE)
        if match:
            dpe_num = match.group(1)
            if idx + 1 < len(parts):
                sample = parts[idx + 1]
            break

    date_match = re.match(r"(\d{4}-\d{2}-\d{2})", source_name)
    date = date_match.group(1).replace("-", "/") if date_match else "Unknown date"
    dpe = f"{dpe_num} DPE" if dpe_num else "Unknown DPE"
    sample_text = sample or "?"
    label = f"{dpe} / {sample_text}" if dpe_num else source_name

    return VideoContext(
        label=label,
        dpe=dpe,
        sample=sample_text,
        date=date,
        source_name=source_name,
    )


def build_window_title(base_title: str, video_context: VideoContext | None) -> str:
    if video_context is None:
        return base_title
    return f"{base_title} | {video_context.label}"


def build_context_chips(video_context: VideoContext) -> QWidget:
    widget = QWidget()
    root = QHBoxLayout(widget)
    root.setContentsMargins(0, 0, 0, 2)
    root.setSpacing(6)
    root.addWidget(_make_badge(video_context.dpe, "dpe"))
    root.addWidget(_make_badge(f"Sample {video_context.sample}", "sample"))
    root.addWidget(_make_badge(video_context.date, "date"))
    root.addStretch()
    return widget


class WorkflowCompanion(QWidget):
    """
    Small always-on-top status panel for batch ROI-library population.
    """

    def __init__(self, total_items: int, parent=None):
        super().__init__(parent)
        self.total_items = max(0, int(total_items))
        self._build()

    def _build(self) -> None:
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowStaysOnTopHint
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
        )
        self.setWindowTitle("Workflow Companion")
        self.resize(400, 190)
        self.setStyleSheet(
            """
            QWidget {
                background: #1e1e2e;
                color: #cdd6f4;
                font-family: "Segoe UI", sans-serif;
            }
            QFrame#panel {
                background: #25273a;
                border: 1px solid #313244;
                border-radius: 14px;
            }
            QLabel#heading {
                color: #89b4fa;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#title {
                color: #f5e0dc;
                font-size: 15px;
                font-weight: 600;
            }
            QLabel#meta {
                color: #a6adc8;
                font-size: 11px;
            }
            QLabel#detail {
                color: #bac2de;
                font-size: 12px;
            }
            QLabel#counter {
                color: #bac2de;
                font-size: 11px;
            }
            QProgressBar {
                border: none;
                background: #313244;
                border-radius: 5px;
                height: 10px;
            }
            QProgressBar::chunk {
                background: #89b4fa;
                border-radius: 5px;
            }
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        panel = QFrame()
        panel.setObjectName("panel")
        outer.addWidget(panel)

        root = QVBoxLayout(panel)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)

        heading = QLabel("ROI LIBRARY WORKFLOW")
        heading.setObjectName("heading")
        root.addWidget(heading)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self.title_lbl = QLabel("Waiting for batch")
        self.title_lbl.setObjectName("title")
        self.title_lbl.setWordWrap(True)
        top.addWidget(self.title_lbl, 1)

        self.stage_badge = _make_badge("Queued", "queued")
        top.addWidget(self.stage_badge, 0, Qt.AlignTop)
        root.addLayout(top)

        self.meta_lbl = QLabel("No video loaded yet")
        self.meta_lbl.setObjectName("meta")
        root.addWidget(self.meta_lbl)

        self.detail_lbl = QLabel("Preparing run")
        self.detail_lbl.setObjectName("detail")
        root.addWidget(self.detail_lbl)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)

        self.counter_lbl = QLabel(self._counter_text(0))
        self.counter_lbl.setObjectName("counter")
        footer.addWidget(self.counter_lbl)
        footer.addStretch()
        root.addLayout(footer)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

    def update_status(
        self,
        *,
        video_context: VideoContext | None,
        stage: str,
        detail: str,
        completed: int,
        tone: str,
    ) -> None:
        if video_context is None:
            if stage.lower() == "done":
                self.title_lbl.setText("Batch complete")
                self.meta_lbl.setText(f"{self.total_items} video(s) in batch")
            else:
                self.title_lbl.setText("ROI library batch")
                self.meta_lbl.setText("No video loaded yet")
        else:
            self.title_lbl.setText(video_context.label)
            self.meta_lbl.setText(
                f"{video_context.dpe} | Sample {video_context.sample} | {video_context.date}"
            )

        self.detail_lbl.setText(detail)
        self.counter_lbl.setText(self._counter_text(completed))
        self.progress.setValue(self._progress_value(completed))

        self.stage_badge.setText(stage)
        _apply_badge_palette(self.stage_badge, tone)
        self.progress.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                background: #313244;
                border-radius: 5px;
                height: 10px;
            }}
            QProgressBar::chunk {{
                background: {_PALETTES.get(tone, _PALETTES["idle"])["bg"]};
                border-radius: 5px;
            }}
            """
        )

    def _counter_text(self, completed: int) -> str:
        if self.total_items <= 0:
            return "0 / 0 completed"
        return f"{max(0, min(completed, self.total_items))} / {self.total_items} completed"

    def _progress_value(self, completed: int) -> int:
        if self.total_items <= 0:
            return 0
        return int(100 * max(0, min(completed, self.total_items)) / self.total_items)


def _make_badge(text: str, tone: str) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumHeight(24)
    _apply_badge_palette(label, tone)
    return label


def _apply_badge_palette(label: QLabel, tone: str) -> None:
    palette = _PALETTES.get(tone, _PALETTES["idle"])
    label.setStyleSheet(
        f"""
        QLabel {{
            background: {palette["bg"]};
            color: {palette["fg"]};
            border-radius: 10px;
            padding: 3px 10px;
            font-size: 13px;
            font-weight: 700;
        }}
        """
    )
