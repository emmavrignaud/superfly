"""
src/statistics.py

Per-feature significance testing and effect-size visualisation.

Functions
---------
feature_significance_report   Kruskal-Wallis + pairwise Cliff's delta per feature,
                               with BH-FDR correction.  Returns a ranked table,
                               a volcano plot, and an effect-size bar chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import kruskal, mannwhitneyu


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.  Input/output: same-length float array."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, p * n / rank)
    # Enforce monotonicity from the smallest p upward
    adj_mono = np.minimum.accumulate(adj[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = adj_mono
    return result


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1].

    Positive values mean group *a* tends to be larger than *b*.
    Does not assume normality or equal variance.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return np.nan
    dom = int(np.sum(a[:, None] > b)) - int(np.sum(a[:, None] < b))
    return dom / (m * n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def feature_significance_report(
    df: pd.DataFrame,
    features: list[str],
    group_col: str = "genotype",
    wt_label: str = "WT",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, go.Figure, go.Figure]:
    """
    Run per-feature significance tests across groups and produce visualisations.

    Steps
    -----
    1. Kruskal-Wallis test across all groups (tests any group difference).
    2. Pairwise Mann-Whitney U: WT vs. each non-WT group.
    3. Cliff's delta effect size for each WT-vs-mutant pair.
    4. BH-FDR correction on Kruskal-Wallis p-values.

    Parameters
    ----------
    df : DataFrame
        One row per fly.  Must contain ``group_col`` and all columns in
        ``features``.
    features : list of str
        Feature column names to test.
    group_col : str
        Column whose unique values define groups (default: "genotype").
    wt_label : str
        Label of the reference group (default: "WT").
    alpha : float
        FDR threshold for significance colouring (default: 0.05).

    Returns
    -------
    results_df : DataFrame
        One row per feature, sorted by KW p-value.  Columns include
        ``kw_stat``, ``kw_p``, ``kw_p_adj``, ``max_abs_delta``, and per-group
        ``delta_vs_<g>`` / ``p_vs_<g>`` columns.
    volcano_fig : go.Figure
        Scatter: max |Cliff's δ| (x) vs −log₁₀(KW p) (y).
    bar_fig : go.Figure
        Bar chart of max |Cliff's δ| ranked by effect size.
    """
    groups = sorted(df[group_col].dropna().astype(str).unique())
    non_wt = [g for g in groups if g != str(wt_label)]

    rows: list[dict] = []
    for feat in features:
        row: dict = {"feature": feat}

        # Kruskal-Wallis across all groups
        vals_by_group = [
            df.loc[df[group_col].astype(str) == g, feat].dropna().values
            for g in groups
        ]
        valid_vals = [v for v in vals_by_group if len(v) >= 2]
        if len(valid_vals) >= 2:
            try:
                kw_stat, kw_p = kruskal(*valid_vals)
            except Exception:
                kw_stat, kw_p = np.nan, np.nan
        else:
            kw_stat, kw_p = np.nan, np.nan
        row["kw_stat"] = kw_stat
        row["kw_p"] = kw_p

        # Pairwise WT vs. each mutant
        wt_vals = df.loc[df[group_col].astype(str) == str(wt_label), feat].dropna().values
        max_delta = 0.0
        for g in non_wt:
            g_vals = df.loc[df[group_col].astype(str) == g, feat].dropna().values
            if len(wt_vals) >= 2 and len(g_vals) >= 2:
                try:
                    _, mw_p = mannwhitneyu(wt_vals, g_vals, alternative="two-sided")
                except Exception:
                    mw_p = np.nan
                delta = cliffs_delta(wt_vals, g_vals)
            else:
                mw_p, delta = np.nan, np.nan
            row[f"delta_vs_{g}"] = delta
            row[f"p_vs_{g}"] = mw_p
            if not np.isnan(delta) and abs(delta) > abs(max_delta):
                max_delta = delta

        row["max_abs_delta"] = abs(max_delta)
        rows.append(row)

    results = pd.DataFrame(rows)

    # BH-FDR on KW p-values
    valid_mask = results["kw_p"].notna()
    if valid_mask.any():
        results.loc[valid_mask, "kw_p_adj"] = _bh_correction(
            results.loc[valid_mask, "kw_p"].values
        )
    else:
        results["kw_p_adj"] = np.nan

    results = results.sort_values("kw_p", na_position="last").reset_index(drop=True)

    # Volcano plot: effect size vs. significance
    results["neg_log10_p"] = -np.log10(results["kw_p"].clip(lower=1e-10))
    sig_col = results["kw_p_adj"].fillna(1.0).lt(alpha).map(
        {True: "significant (FDR)", False: "n.s."}
    )
    volcano_fig = px.scatter(
        results,
        x="max_abs_delta",
        y="neg_log10_p",
        text="feature",
        color=sig_col,
        color_discrete_map={"significant (FDR)": "#e63946", "n.s.": "#adb5bd"},
        title=f"Feature significance — Kruskal-Wallis across {group_col}s",
        labels={
            "max_abs_delta": "Max |Cliff's δ| (WT vs. mutant)",
            "neg_log10_p": "−log₁₀(KW p-value)",
            "color": "",
        },
    )
    volcano_fig.add_hline(
        y=-np.log10(alpha),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"p = {alpha}",
        annotation_position="bottom right",
    )
    volcano_fig.update_traces(textposition="top center", marker=dict(size=10))
    volcano_fig.update_layout(height=520)

    # Bar chart: max |Cliff's δ| per feature, sorted descending
    bar_df = results.sort_values("max_abs_delta", ascending=False).copy()
    bar_sig = bar_df["kw_p_adj"].fillna(1.0).lt(alpha).map(
        {True: "significant (FDR)", False: "n.s."}
    )
    bar_fig = px.bar(
        bar_df,
        x="feature",
        y="max_abs_delta",
        color=bar_sig,
        color_discrete_map={"significant (FDR)": "#e63946", "n.s.": "#adb5bd"},
        title="Effect size: max |Cliff's δ| per feature (WT vs. mutants)",
        labels={"max_abs_delta": "Max |Cliff's δ|", "feature": "Feature", "color": ""},
    )
    bar_fig.update_layout(height=460, xaxis_tickangle=-40)

    return results, volcano_fig, bar_fig
