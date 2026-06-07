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


# ---------------------------------------------------------------------------
# Per-feature box plots
# ---------------------------------------------------------------------------

GENOTYPE_ORDER = ["WT", "A90V", "G287S", "G294A", "A315T", "M337V"]

GENOTYPE_COLORS = {
    "WT":    "#4361ee",
    "A90V":  "#f72585",
    "G287S": "#7209b7",
    "G294A": "#3a0ca3",
    "A315T": "#4cc9f0",
    "M337V": "#f77f00",
}

FEATURE_LABELS = {
    "mean_velocity":              "Mean speed (cm/s)",
    "median_velocity":            "Median speed (cm/s)",
    "std_velocity":               "Speed variability (cm/s)",
    "pause_fraction":             "Pause fraction",
    "pause_count":                "Pause count",
    "mean_pause_duration":        "Mean pause duration (s)",
    "max_pause_duration":         "Max pause duration (s)",
    "burst_count":                "Burst count",
    "mean_burst_duration":        "Mean burst duration (s)",
    "mean_burst_speed":           "Mean burst speed (cm/s)",
    "latency_to_first_movement":  "Latency to first movement (s)",
    "mean_speed_early":           "Mean speed — first 10 s (cm/s)",
    "reversal_rate":              "Reversal rate (180° turns/s)",
    "mean_abs_turning_angle":     "Mean |turning angle| (rad)",
    "mean_abs_angular_velocity":  "Mean |angular velocity| (rad/s)",
    "total_distance_traveled":    "Total distance (cm)",
    "tortuosity":                 "Tortuosity",
    "area_covered":               "Area covered (cm²)",
}


def make_feature_boxplots(
    df: pd.DataFrame,
    features: list[str],
    group_col: str = "genotype",
    alpha: float = 0.05,
    results_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """
    Return a dict of {feature_name: Plotly box-plot Figure}.
    Boxes are coloured by genotype; title shows p-value and Cliff's delta
    when results_df is provided.
    """
    present_genos = [g for g in GENOTYPE_ORDER if g in df[group_col].values]
    color_map = {g: GENOTYPE_COLORS.get(g, "#888") for g in present_genos}

    figs = {}
    for feat in features:
        if feat not in df.columns:
            continue

        title = FEATURE_LABELS.get(feat, feat)
        if results_df is not None and not results_df.empty:
            row = results_df[results_df["feature"] == feat]
            if not row.empty:
                p    = row.iloc[0]["kw_p_adj"]
                d    = row.iloc[0]["max_abs_delta"]
                sig  = "★ " if p < alpha else ""
                title = f"{sig}{title}<br><sup>FDR p={p:.3g}  |δ|={d:.2f}</sup>"

        fig = px.box(
            df,
            x=group_col,
            y=feat,
            color=group_col,
            color_discrete_map=color_map,
            category_orders={group_col: present_genos},
            points="all",
            title=title,
            labels={feat: FEATURE_LABELS.get(feat, feat), group_col: ""},
        )
        fig.update_traces(marker_size=4, jitter=0.35)
        fig.update_layout(
            height=380,
            showlegend=False,
            margin=dict(t=80, b=40, l=60, r=20),
        )
        figs[feat] = fig
    return figs


# ---------------------------------------------------------------------------
# Full HTML significance report
# ---------------------------------------------------------------------------

def write_significance_report(
    results: pd.DataFrame,
    volcano_fig: go.Figure,
    bar_fig: go.Figure,
    df: pd.DataFrame,
    features: list[str],
    out_dir: str,
    group_col: str = "genotype",
    alpha: float = 0.05,
    n_runs: int | None = None,
) -> str:
    """
    Write per-feature HTML files + a self-contained summary report.
    Returns the path to the report HTML.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Save overview figures
    volcano_fig.write_html(os.path.join(out_dir, "volcano.html"))
    bar_fig.write_html(os.path.join(out_dir, "effect_sizes.html"))

    # Box plots for every feature
    box_figs = make_feature_boxplots(df, features, group_col=group_col,
                                     alpha=alpha, results_df=results)
    for feat, fig in box_figs.items():
        fig.write_html(os.path.join(out_dir, f"box_{feat}.html"))

    # Split features into significant / notable / rest
    sig_feats = results[results["kw_p_adj"] < alpha]["feature"].tolist()
    notable_feats = results[
        (results["kw_p_adj"] >= alpha) &
        (results["max_abs_delta"] >= 0.45)
    ]["feature"].tolist()

    n_flies  = df[group_col].notna().sum()
    n_genos  = df[group_col].nunique()
    n_sig    = len(sig_feats)
    n_tested = len(features)

    def _iframe(src, height=400):
        return (f'<iframe src="{src}" width="100%" height="{height}" '
                f'frameborder="0" scrolling="no"></iframe>')

    def _box_grid(feat_list, cols=3):
        if not feat_list:
            return "<p><em>None.</em></p>"
        rows_html = []
        for i in range(0, len(feat_list), cols):
            chunk = feat_list[i:i+cols]
            cells = "".join(
                f'<td style="width:{100//cols}%;padding:4px">'
                f'{_iframe(f"box_{f}.html", 400)}</td>'
                for f in chunk
            )
            # pad incomplete row
            for _ in range(cols - len(chunk)):
                cells += f'<td style="width:{100//cols}%"></td>'
            rows_html.append(f"<tr>{cells}</tr>")
        return f'<table style="width:100%;border-collapse:collapse">{"".join(rows_html)}</table>'

    # Results table HTML — highlight significant rows
    def _results_table(df_res):
        cols_show = ["feature", "kw_p", "kw_p_adj", "max_abs_delta"]
        # Add per-genotype delta columns if present
        delta_cols = [c for c in df_res.columns if c.startswith("delta_vs_")]
        cols_show += delta_cols
        sub = df_res[cols_show].copy()
        sub = sub.rename(columns={
            "feature": "Feature",
            "kw_p": "KW p",
            "kw_p_adj": "KW p (FDR)",
            "max_abs_delta": "Max |δ|",
        })
        for dc in delta_cols:
            sub = sub.rename(columns={dc: dc.replace("delta_vs_", "δ vs ")})

        rows = []
        for _, r in sub.iterrows():
            is_sig = float(r["KW p (FDR)"]) < alpha
            style  = ' style="background:#fff0f0"' if is_sig else ""
            cells  = "".join(
                f"<td>{v:.4g}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                for v in r
            )
            rows.append(f"<tr{style}>{cells}</tr>")
        headers = "".join(f"<th>{c}</th>" for c in sub.columns)
        return (
            f'<table class="results-table"><thead><tr>{headers}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )

    run_note = f"{n_runs} runs × " if n_runs else ""
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Feature significance report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1300px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1   {{ font-size: 1.7em; margin-bottom: 0.2em; }}
  h2   {{ font-size: 1.2em; color: #444; margin-top: 2.2em; border-bottom: 2px solid #eee;
          padding-bottom: 0.3em; }}
  .stat-row {{ display: flex; gap: 2em; margin: 1em 0 1.5em; flex-wrap: wrap; }}
  .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 0.8em 1.4em;
                text-align: center; min-width: 110px; }}
  .stat-card .num {{ font-size: 2em; font-weight: 700; color: #e63946; }}
  .stat-card .lbl {{ font-size: 0.8em; color: #666; margin-top: 2px; }}
  .note {{ color: #666; font-size: 0.88em; margin: 0.4em 0 1em; }}
  .results-table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; margin-top:1em; }}
  .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 5px 9px; text-align: right; }}
  .results-table th {{ background: #f0f0f0; text-align: center; font-weight: 600; }}
  .results-table td:first-child {{ text-align: left; font-weight: 500; }}
  .legend {{ font-size:0.82em; color:#666; margin-top:0.4em; }}
</style>
</head>
<body>

<h1>Behavioural feature significance report</h1>
<p class="note">
  Kruskal-Wallis test across {n_genos} genotypes · pairwise Mann-Whitney U (WT vs. each mutant) ·
  Cliff's δ effect size · Benjamini-Hochberg FDR correction · α = {alpha}<br>
  Dataset: {run_note}{n_flies} flies · {n_tested} features tested
</p>

<div class="stat-row">
  <div class="stat-card"><div class="num">{n_flies}</div><div class="lbl">flies</div></div>
  <div class="stat-card"><div class="num">{n_tested}</div><div class="lbl">features tested</div></div>
  <div class="stat-card"><div class="num">{n_sig}</div><div class="lbl">significant (FDR &lt; {alpha})</div></div>
  <div class="stat-card"><div class="num">{len(notable_feats)}</div><div class="lbl">notable (|δ| ≥ 0.45)</div></div>
</div>

<h2>Volcano plot — effect size vs. significance</h2>
<p class="note">Right = larger effect. Up = more significant. Red = survives FDR correction.</p>
{_iframe("volcano.html", 540)}

<h2>Effect sizes ranked</h2>
<p class="note">Max |Cliff's δ| across all WT vs. mutant pairs. Red = FDR significant.</p>
{_iframe("effect_sizes.html", 480)}

<h2>★ Significant features (FDR p &lt; {alpha})</h2>
<p class="note">Box plots show per-fly distributions. ★ = FDR significant. Points are individual flies.</p>
{_box_grid(sig_feats)}

<h2>Notable features (|δ| ≥ 0.45, not FDR significant)</h2>
<p class="note">Large effect size but didn't survive multiple-testing correction — worth watching with more data.</p>
{_box_grid(notable_feats)}

<h2>All results</h2>
<p class="legend">Highlighted rows = FDR significant. δ = Cliff's delta (WT vs. that mutant).</p>
{_results_table(results)}

</body>
</html>"""

    report_path = os.path.join(out_dir, "significance_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path
