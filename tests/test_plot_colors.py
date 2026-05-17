import pandas as pd

from src.plot_colors import (
    QUALITATIVE_PALETTE,
    genotype_color_discrete_map,
    genotype_color_map_for_dataframe,
    wt_vs_mutant_color_map,
)


def test_genotype_color_map_respects_preferred_order():
    df = pd.DataFrame({"genotype": ["B", "A", "B"]})
    m = genotype_color_map_for_dataframe(df, ["A", "B"])
    assert m["A"] == QUALITATIVE_PALETTE[0]
    assert m["B"] == QUALITATIVE_PALETTE[1]


def test_unknown_genotypes_sorted_after_preferred():
    df = pd.DataFrame({"genotype": ["Z", "A"]})
    m = genotype_color_map_for_dataframe(df, ["A"])
    assert list(m.keys()) == ["A", "Z"]


def test_wt_vs_mutant_uses_wt_from_map():
    g = genotype_color_discrete_map(["WT", "X"])
    w = wt_vs_mutant_color_map(g)
    assert w["WT"] == g["WT"]
    assert "Mutant" in w
