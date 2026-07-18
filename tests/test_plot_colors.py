import pandas as pd

from src.plot_colors import (
    DEFAULT_VIAL_COLORS,
    genotype_color_discrete_map,
    genotype_color_map_for_dataframe,
    wt_vs_mutant_color_map,
)


def test_genotype_color_map_respects_preferred_order():
    df = pd.DataFrame({"genotype": ["B", "A", "B"]})
    m = genotype_color_map_for_dataframe(df, ["A", "B"])
    # No vial_id column -> colours fall back to positional vial order.
    assert m["A"] == DEFAULT_VIAL_COLORS[0]
    assert m["B"] == DEFAULT_VIAL_COLORS[1]


def test_genotype_inherits_its_vial_colour():
    palette = {f"vial{i + 1}": c for i, c in enumerate(DEFAULT_VIAL_COLORS)}
    df = pd.DataFrame({"genotype": ["A", "B"], "vial_id": ["vial2", "vial1"]})
    m = genotype_color_map_for_dataframe(df, ["A", "B"])
    assert m["A"] == palette["vial2"]
    assert m["B"] == palette["vial1"]


def test_unknown_genotypes_sorted_after_preferred():
    df = pd.DataFrame({"genotype": ["Z", "A"]})
    m = genotype_color_map_for_dataframe(df, ["A"])
    assert list(m.keys()) == ["A", "Z"]


def test_wt_vs_mutant_uses_wt_from_map():
    g = genotype_color_discrete_map(["WT", "X"])
    w = wt_vs_mutant_color_map(g)
    assert w["WT"] == g["WT"]
    assert "Mutant" in w
