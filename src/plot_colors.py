"""
Stable Plotly discrete colors for genotype (and WT vs pooled mutant).

All genotype-coloured figures should use ``genotype_color_map_for_dataframe``
with the same ``preferred_genotype_order`` (typically
``genotype_category_order(df)`` from ``src.classification``) so a given
genotype string always maps to the same hex across box plots, embeddings, etc.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Canonical genotype -> colour overrides written by the setup window
# (scripts/app.py). When a user picks a vial colour, it is recorded here against
# that vial's genotype, and every genotype-coloured figure follows it. Last pick
# wins. Absent file -> no overrides (default palette everywhere).
_GENOTYPE_COLORS_PATH = Path(__file__).resolve().parent.parent / "genotype_colors.json"


def load_genotype_color_overrides() -> dict[str, str]:
    """Return the canonical {genotype: hex} overrides, or {} if none/unreadable."""
    try:
        with open(_GENOTYPE_COLORS_PATH) as f:
            raw = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def write_genotype_color_overrides(genotype_to_color: dict[str, str]) -> None:
    """Merge {genotype: hex} into the canonical file (last pick wins).

    Called whenever a user picks vial colours in any GUI (the setup window or the
    vial-ROI dialog), so classification / embedding figures follow the newest
    choice. Genotypes not passed keep their previous colour.
    """
    data = load_genotype_color_overrides()
    for g, c in genotype_to_color.items():
        if c:
            data[str(g)] = str(c)
    _GENOTYPE_COLORS_PATH.write_text(json.dumps(data, indent=2))


def parse_vial_genotypes(video_path) -> list[str] | None:
    """Genotype per vial (left->right) from the video filename, or None.

    Mirrors ``classification.map_vial_to_genotype`` parsing without a run dir:
    ``<date>_<..>_hTDP43_<GT1>-<GT2>-...>_<rest>``. None when the filename is not
    this convention, so genotype colours are only written when the mapping is known.
    """
    parts = Path(video_path).name.split("_")
    if len(parts) > 3 and parts[2] == "hTDP43":
        return parts[3].split("-")
    return None


# Plotly default qualitative sequence (stable identifiers)
QUALITATIVE_PALETTE: list[str] = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

# WT vs pooled mutants: single lane for all non-WT flies (not a genotype label)
MUTANT_POOL_COLOR: str = "#9E9E9E"


def genotype_color_discrete_map(ordered_genotypes: list[str]) -> dict[str, str]:
    """Map each genotype string to a color by its index in ``ordered_genotypes``.

    A genotype present in the canonical overrides (genotype_colors.json, written
    by the setup window) takes that colour; the rest fall back to the palette by
    index. This is the single funnel both ``genotype_color_map_for_dataframe``
    and the classification/embedding plots use, so a GUI colour pick flows to all
    genotype-coloured figures from here.
    """
    overrides = load_genotype_color_overrides()
    return {
        g: overrides.get(g, QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)])
        for i, g in enumerate(ordered_genotypes)
    }


def genotype_color_map_for_dataframe(
    df: pd.DataFrame,
    preferred_genotype_order: list[str] | None,
    genotype_col: str = "genotype",
) -> dict[str, str]:
    """
    Build a color map for all genotypes present in ``df``.

    Genotypes appear in ``preferred_genotype_order`` first (when present), then
    any remaining genotypes in sorted order.
    """
    if df.empty or genotype_col not in df.columns:
        return {}
    present = set(df[genotype_col].astype(str).unique())
    base = preferred_genotype_order or []
    ordered: list[str] = []
    seen: set[str] = set()
    for g in base:
        if g in present and g not in seen:
            ordered.append(g)
            seen.add(g)
    for g in sorted(present - seen):
        ordered.append(g)
    return genotype_color_discrete_map(ordered)


def wt_vs_mutant_color_map(genotype_color_map: dict[str, str]) -> dict[str, str]:
    """Colors for the synthetic ``WT`` / ``Mutant`` lanes (mutant pool is neutral)."""
    wt = genotype_color_map.get("WT", QUALITATIVE_PALETTE[0])
    return {"WT": wt, "Mutant": MUTANT_POOL_COLOR}
