"""Glasbey palette lookup for track IDs.

Uses `colorcet.glasbey_bw_minc_20` — 256 perceptually distinct colors,
already used elsewhere in the project for track visualization.
"""
from __future__ import annotations

from colorcet import glasbey_bw_minc_20


def track_color_rgb(track_id: int) -> tuple[int, int, int]:
    entry = glasbey_bw_minc_20[track_id % len(glasbey_bw_minc_20)]
    if isinstance(entry, str):
        h = entry.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return tuple(int(round(c * 255)) for c in entry[:3])


def track_color_hex(track_id: int) -> str:
    r, g, b = track_color_rgb(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"
