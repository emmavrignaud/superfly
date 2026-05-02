"""Glasbey palette lookup for track IDs.

Uses `colorcet.glasbey_dark` — 256 perceptually distinct DARK colors,
designed for categorical data on a light background. Picked because the
labeler videos are bright (flies on near-white illumination) and the
previous `glasbey_bw_minc_20` palette emitted some near-white entries
that vanished against the video.
"""
from __future__ import annotations

from colorcet import glasbey_dark as _PALETTE


def track_color_rgb(track_id: int) -> tuple[int, int, int]:
    entry = _PALETTE[track_id % len(_PALETTE)]
    if isinstance(entry, str):
        h = entry.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return tuple(int(round(c * 255)) for c in entry[:3])


def track_color_hex(track_id: int) -> str:
    r, g, b = track_color_rgb(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"
