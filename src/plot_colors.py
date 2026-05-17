"""
Stable Plotly discrete colors for genotype (and WT vs pooled mutant).

All genotype-coloured figures should use ``genotype_color_map_for_dataframe``
with the same ``preferred_genotype_order`` (typically
``genotype_category_order(df)`` from ``src.classification``) so a given
genotype string always maps to the same hex across box plots, embeddings, etc.
"""

from __future__ import annotations

import pandas as pd

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
    """Map each genotype string to a color by its index in ``ordered_genotypes``."""
    return {
        g: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
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
