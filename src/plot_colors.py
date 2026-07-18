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

# Single source of truth for vial colours: a fixed ``vial_colors`` block at the
# top of roi_library.json ({"vial1": "#rrggbb", ...}), edited via the setup GUI.
# A vial always carries the same genotype (vial i -> genotype i), so this one
# palette drives both the overlay (keyed by vial) and analysis figures (keyed by
# genotype).
DEFAULT_VIAL_COLORS: list[str] = [
    "#E69F00",  # vial1 orange
    "#56B4E9",  # vial2 sky blue
    "#009E73",  # vial3 green
    "#F0E442",  # vial4 yellow
    "#0072B2",  # vial5 blue
    "#CC79A7",  # vial6 reddish purple
]
_ROI_LIBRARY_PATH = Path(__file__).resolve().parent.parent / "roi_library.json"


def load_vial_palette() -> dict[str, str]:
    """Return {vial_id: hex} from roi_library.json's top-level ``vial_colors``.

    Missing or unreadable entries fall back to ``DEFAULT_VIAL_COLORS`` by index,
    so vial1..vial6 always resolve to a colour.
    """
    palette = {f"vial{i + 1}": c for i, c in enumerate(DEFAULT_VIAL_COLORS)}
    try:
        with open(_ROI_LIBRARY_PATH) as f:
            stored = json.load(f).get("vial_colors", {})
    except (FileNotFoundError, ValueError, OSError):
        stored = {}
    for k, v in stored.items():
        if isinstance(v, str):
            palette[k] = v
    return palette


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
    """Map genotypes to colours by vial position (genotype ``i`` -> vial ``i+1``).

    ``ordered_genotypes`` is assumed to be in vial order (vial1 first). This is
    the single funnel every genotype-coloured figure uses, so the overlay and the
    analysis plots share the one ``vial_colors`` palette. Past the palette length
    it cycles the defaults.
    """
    palette = load_vial_palette()
    return {
        g: palette.get(f"vial{i + 1}", DEFAULT_VIAL_COLORS[i % len(DEFAULT_VIAL_COLORS)])
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

    # When the frame carries each genotype's vial, colour it by that vial (robust
    # to ordering). Otherwise fall back to positional vial order.
    if "vial_id" in df.columns:
        palette = load_vial_palette()
        pairs = (df[[genotype_col, "vial_id"]].dropna().astype(str)
                 .drop_duplicates(subset=[genotype_col]))
        by_vial = {g: palette[v]
                   for g, v in zip(pairs[genotype_col], pairs["vial_id"])
                   if v in palette}
        positional = genotype_color_discrete_map(ordered)
        return {g: by_vial.get(g, positional[g]) for g in ordered}
    return genotype_color_discrete_map(ordered)


def wt_vs_mutant_color_map(genotype_color_map: dict[str, str]) -> dict[str, str]:
    """Colors for the synthetic ``WT`` / ``Mutant`` lanes (mutant pool is neutral)."""
    wt = genotype_color_map.get("WT", QUALITATIVE_PALETTE[0])
    return {"WT": wt, "Mutant": MUTANT_POOL_COLOR}
