"""
src/classification.py

Genotype classification and statistical visualisation.

Classifiers: LDA, Logistic Regression, SVC (linear kernel)
Plots: cross-validation accuracy, feature importance, per-genotype boxes
       (Kruskal-Wallis + Dunn/Holm significance brackets), WT-vs-mutant
       comparison with Cliff's delta; optional pooled + per-trial HTML report.
"""

import html as html_module
import json
import logging
import os
import re
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import mannwhitneyu, kruskal

import scikit_posthocs as sp

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_PNG_EXPORT_WARNING_SHOWN = False

# Silence verbose Kaleido/Choreographer INFO logs during Plotly PNG export.
for _logger_name in ("kaleido", "choreographer", "logistro", "Kaleido", "Choreographer"):
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.WARNING)
    _logger.propagate = False


# ---------------------------------------------------------------------------
# Genotype mapping
# ---------------------------------------------------------------------------

def map_vial_to_genotype(run_dir: str) -> pd.DataFrame:
    """
    Load compact_tracks.csv from a run directory and add a "genotype" column.

    The source video filename is read from run_params.json (config.video).
    Expected filename format: <date>_<something>_hTDP43_<GT1>-<GT2>-..._<rest>.mp4
    """
    params_path = os.path.join(run_dir, "run_params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    video_path = params["config"]["video"]
    # Handle both forward and backward slashes from Windows-recorded paths
    video_name = os.path.basename(video_path.replace("\\", "/"))
    parts = video_name.split("_")
    assert len(parts) > 3 and parts[2] == "hTDP43", (
        f"Unexpected video filename format: {video_name}"
    )

    # Token order matches vial1, vial2, … left-to-right ROI order only if the
    # experiment filename follows the same convention as ROI drawing.
    genotypes = parts[3].split("-")
    vial_to_genotype = {f"vial{i + 1}": genotypes[i] for i in range(len(genotypes))}

    csv_path = os.path.join(run_dir, "compact_tracks.csv")
    df = pd.read_csv(csv_path)
    df["genotype"] = df["vial_id"].map(vial_to_genotype)
    return df


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def make_classifier(model_name: str):
    """Return an unfitted classifier by name (lda | logistic | svc)."""
    if model_name == "lda":
        return LinearDiscriminantAnalysis()
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000)
    if model_name == "svc":
        return SVC(kernel="linear")
    raise ValueError("model_name must be lda, logistic, or svc")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_xy(df: pd.DataFrame):
    """Return (X array, feature_names list) dropping non-numeric and compact_id."""
    X = df.select_dtypes(include=[np.number]).drop(columns=["compact_id"], errors="ignore")
    return X.values, X.columns.tolist()


def prepare_target(df: pd.DataFrame, mode: str = "multiclass"):
    """
    Return target labels.

    mode: "multiclass" -> raw genotype string
          "binary"     -> "WT" or "Mutant"
    """
    if mode == "multiclass":
        return df["genotype"].values
    if mode == "binary":
        return np.where(df["genotype"] == "WT", "WT", "Mutant")
    raise ValueError("mode must be multiclass or binary")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_cross_validation(
    model,
    model_name: str,
    X,
    y,
    classification_mode: str,
    cv: int = 5,
    outdir: str = "report_figures",
    groups: np.ndarray | None = None,
    save: bool = True,
):
    """
    Run k-fold CV, plot bar chart, save figure, return scores array.

    If ``groups`` is None (default), uses sklearn's default splitter with integer
    ``cv`` (StratifiedKFold for classifiers).

    If ``groups`` is provided, uses ``GroupKFold`` so entire groups (e.g. one
    video / run) stay in train or test — no leakage across videos. Number of
    splits is ``min(cv, n_unique_groups)``.

    Returns
    -------
    scores : ndarray
        Per-fold accuracy scores.
    fig : plotly.graph_objects.Figure
        Bar chart of fold scores.
    """
    if groups is not None:
        n_unique = len(np.unique(groups))
        n_splits = min(int(cv), n_unique)
        if n_splits < 2:
            raise ValueError(
                f"GroupKFold needs at least 2 groups; got {n_unique} unique group(s)."
            )
        splitter = GroupKFold(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=splitter, groups=groups)
        scheme = "group"
    else:
        scores = cross_val_score(model, X, y, cv=cv)
        scheme = "stratified"

    fig = go.Figure()
    fig.add_bar(x=list(range(1, len(scores) + 1)), y=scores)
    fig.add_hline(y=scores.mean(), line_dash="dash")
    fig.update_layout(
        title=(
            f"Cross-validation accuracy: {model_name.upper()} ({classification_mode}, {scheme}) "
            f"- mean={scores.mean():.3f}"
        ),
        xaxis_title="CV fold",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
    )

    if save:
        save_plotly_figure(fig, outdir, f"{model_name}_{classification_mode}")
    return scores, fig


def plot_feature_importance(
    model,
    X,
    y,
    feature_names,
    model_name: str,
    classification_mode: str,
):
    """Fit model and plot feature importance as a horizontal bar chart."""
    model.fit(X, y)

    if model_name == "logistic":
        values = np.mean(np.abs(model.named_steps["clf"].coef_), axis=0)
        xlabel = "Mean |coefficient|"
    elif model_name == "lda":
        values = np.mean(np.abs(model.named_steps["clf"].scalings_), axis=1)
        xlabel = "Mean |loading|"
    elif model_name == "svc":
        # OvR: coef_ is (n_classes, n_features); collapse to one weight per feature like logistic.
        coef = model.named_steps["clf"].coef_
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        xlabel = "|weight|"
    else:
        return None

    idx = np.argsort(values)
    fig = go.Figure()
    fig.add_bar(x=values[idx], y=[feature_names[i] for i in idx], orientation="h")
    fig.update_layout(
        title=f"Feature importance: {model_name.upper()} ({classification_mode})",
        xaxis_title=xlabel,
        yaxis_title="Feature",
    )
    return fig


def run_classifier(
    df: pd.DataFrame,
    outdir: str = "report_figures",
    model_name: str = "lda",
    classification_mode: str = "multiclass",
    cv: int = 5,
    plot_importance: bool = True,
    groups: np.ndarray | None = None,
    save_files: bool = True,
    return_figures: bool = False,
):
    """
    Full classification run: prepare data, CV, optional feature-importance plot.

    Parameters
    ----------
    df : pd.DataFrame
        Output of aggregate_per_fly_features() with "genotype" column.
    outdir : str
        Directory for saving figures.
    model_name : str
        One of "lda", "logistic", "svc".
    classification_mode : str
        "multiclass" or "binary".
    cv : int
        Number of CV folds (or max folds for GroupKFold when ``groups`` is set).
    plot_importance : bool
        Whether to produce and save a feature-importance figure.
    groups : np.ndarray, optional
        One group id per row (e.g. video / run name). If given, CV uses
        GroupKFold so no fly from a held-out video appears in training.
    save_files : bool
        If False, figures are not written to ``outdir`` (for HTML bundling).
    return_figures : bool
        If True, returns a list of ``(figure_id, go.Figure)`` (figures are still
        saved when ``save_files`` is True).
    """
    X, feature_names = prepare_xy(df)
    y = prepare_target(df, classification_mode)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", make_classifier(model_name)),
    ])

    scores, fig_cv = run_cross_validation(
        pipeline, model_name, X, y, cv=cv,
        classification_mode=classification_mode, outdir=outdir,
        groups=groups,
        save=save_files,
    )

    artifacts: list[tuple[str, go.Figure]] = []
    if return_figures:
        artifacts.append((f"{model_name}_{classification_mode}_cv", fig_cv))

    if plot_importance:
        fig_imp = plot_feature_importance(
            pipeline, X, y, feature_names, model_name, classification_mode
        )
        if fig_imp is not None:
            if save_files:
                save_plotly_figure(
                    fig_imp, outdir, f"{model_name}_{classification_mode}_importance"
                )
            if return_figures:
                artifacts.append((f"{model_name}_{classification_mode}_importance", fig_imp))

    if return_figures:
        return artifacts
    return None


# ---------------------------------------------------------------------------
# Statistical visualisation
# ---------------------------------------------------------------------------

def cliffs_delta(x, y):
    """
    Cliff's delta: dominance measure between two samples (non-parametric effect size).

    For every pair (one value from ``x``, one from ``y``), count whether x > y
    or x < y. Delta = (wins − losses) / (n_x · n_y), in [-1, 1]. 0 means no
    tendency for either sample to be larger; ±1 means strict separation.
    Unlike Cohen's d, it does not assume normality or homoscedasticity.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return (np.sum(x[:, None] > y) - np.sum(x[:, None] < y)) / (len(x) * len(y))


def save_plotly_figure(fig, outdir: str, name: str, show: bool = True):
    """Save a Plotly figure as HTML + PNG and optionally display it."""
    os.makedirs(outdir, exist_ok=True)
    fig.write_html(os.path.join(outdir, f"{name}.html"))
    png_path = os.path.join(outdir, f"{name}.png")
    try:
        fig.write_image(png_path, width=1200, height=800, scale=2)
    except Exception as exc:
        # Avoid noisy repeated Kaleido warnings; HTML is still saved and usable.
        global _PNG_EXPORT_WARNING_SHOWN
        if not _PNG_EXPORT_WARNING_SHOWN:
            print(f"[classification] PNG export disabled (Kaleido unavailable/misconfigured): {exc}")
            _PNG_EXPORT_WARNING_SHOWN = True
    if show:
        fig.show()


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _vial_sort_index(vial_id) -> int:
    """Parse vial1, vial2, … for numeric ordering (vial2 before vial10)."""
    s = str(vial_id).strip()
    m = re.match(r"(?i)vial\s*(\d+)\s*$", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(?i)vial\s*(\d+)", s)
    if m2:
        return int(m2.group(1))
    return 10**9


def genotype_category_order(df: pd.DataFrame) -> list[str]:
    """
    Left-to-right x-axis order for genotype box plots.

    Uses ``vial_id`` (vial1, vial2, …) within each ``run``, in first-seen run
    order — consistent with ``map_vial_to_genotype`` when the filename token
    order matches ROI vial numbering. If ``vial_id`` is missing, falls back to
    first-seen genotype order in ``df``.
    """
    if df.empty or "genotype" not in df.columns:
        return []

    d = df.copy()
    d["genotype"] = d["genotype"].astype(str)

    if "vial_id" not in d.columns:
        order: list[str] = []
        for g in d["genotype"].values:
            if g not in order:
                order.append(g)
        return order

    ordered: list[str] = []
    seen: set[str] = set()

    if "run" in d.columns:
        blocks = (g for _, g in d.groupby("run", sort=False))
    else:
        blocks = (d,)

    for dfx in blocks:
        if dfx.empty:
            continue
        pairs = dfx[["vial_id", "genotype"]].dropna()
        if pairs.empty:
            continue
        pairs = pairs.drop_duplicates(subset=["vial_id"], keep="first")
        pairs = pairs.assign(_vi=pairs["vial_id"].map(_vial_sort_index))
        pairs = pairs.sort_values("_vi", kind="stable")
        for g in pairs["genotype"].astype(str).values:
            if g not in seen:
                ordered.append(g)
                seen.add(g)

    for g in sorted(d["genotype"].unique()):
        if g not in seen:
            ordered.append(g)
            seen.add(g)
    return ordered


def pairwise_dunn_holm(
    df: pd.DataFrame,
    feat: str,
    genotype_order: list[str],
    alpha: float = 0.05,
) -> tuple[list[tuple[str, str, float]], list[str]]:
    """
    Dunn's post-hoc with Holm correction on ``feat``, grouped by ``genotype``.

    Parameters
    ----------
    genotype_order
        Category order (same as x-axis / ``genotype_category_order``).

    Returns
    -------
    sig_pairs
        (genotype_a, genotype_b, p_adj) with p_adj < alpha, sorted by
        increasing index distance so brackets stack bottom-to-top.
    labels
        Genotype labels used for the matrix (subset of ``genotype_order`` plus
        any genotypes present in data but missing from that list).
    """
    sub = df[["genotype", feat]].dropna().copy()
    sub["genotype"] = sub["genotype"].astype(str)
    present = set(sub["genotype"].unique())
    labels = [g for g in genotype_order if g in present]
    for g in sorted(present):
        if g not in labels:
            labels.append(g)
    if len(labels) < 2:
        return [], labels

    p_mat = sp.posthoc_dunn(
        sub, val_col=feat, group_col="genotype", p_adjust="holm"
    )
    p_mat = p_mat.reindex(index=labels, columns=labels)

    sig_pairs: list[tuple[str, str, float]] = []
    for i, gi in enumerate(labels):
        for j in range(i + 1, len(labels)):
            gj = labels[j]
            p_adj = float(p_mat.loc[gi, gj])
            if p_adj < alpha:
                sig_pairs.append((gi, gj, p_adj))

    def span(pair: tuple[str, str, float]) -> int:
        ia = labels.index(pair[0])
        ib = labels.index(pair[1])
        return abs(ib - ia)

    sig_pairs.sort(key=lambda p: (span(p), labels.index(p[0]), labels.index(p[1])))
    return sig_pairs, labels


def _add_significance_brackets(
    fig: go.Figure,
    df: pd.DataFrame,
    feat: str,
    sig_pairs: list[tuple[str, str, float]],
    genotype_order: list[str],
) -> None:
    """Overlay horizontal significance lines + stars between genotype pairs."""
    if not sig_pairs:
        return

    yvals = df[feat].dropna().to_numpy()
    if yvals.size == 0:
        return
    y_min = float(np.min(yvals))
    y_max = float(np.max(yvals))
    span_y = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
    tick = 0.04 * span_y
    step = 0.08 * span_y
    y0 = y_max + 0.05 * span_y

    n_brackets = len(sig_pairs)
    y_top = y0 + (n_brackets - 1) * step + tick
    fig.update_layout(yaxis_range=[None, y_top + 0.06 * span_y + 0.5 * tick])

    line_kw = dict(
        line=dict(color="black", width=1),
        xref="x",
        yref="y",
        layer="above",
    )

    for k, (ga, gb, p_adj) in enumerate(sig_pairs):
        y = y0 + k * step
        fig.add_shape(type="line", x0=ga, x1=gb, y0=y, y1=y, **line_kw)
        fig.add_shape(type="line", x0=ga, x1=ga, y0=y - tick, y1=y, **line_kw)
        fig.add_shape(type="line", x0=gb, x1=gb, y0=y - tick, y1=y, **line_kw)

        ia = genotype_order.index(ga)
        ib = genotype_order.index(gb)
        x_mid = (ia + ib) / 2
        star_lbl = _significance_stars(p_adj)
        if star_lbl:
            # Bracket shapes use layer="above"; annotations render above shapes.
            # Offset uses tick (same scale as bracket arms), not a tiny fraction of span_y.
            fig.add_annotation(
                x=x_mid,
                y=y + 0.15 * tick,
                xref="x",
                yref="y",
                text=star_lbl,
                showarrow=False,
                font=dict(size=10, color="black"),
                xanchor="center",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.93)",
                borderpad=1,
            )


def make_figure_by_genotype(
    df: pd.DataFrame,
    feat: str,
    feature_titles: dict[str, str],
    hover_data: list[str],
    genotype_order: list[str] | None = None,
) -> go.Figure:
    """One Kruskal-Wallis + Dunn/Holm genotype box plot (no file I/O)."""
    if genotype_order is None:
        genotype_order = genotype_category_order(df)
    groups = [g[feat].values for _, g in df.groupby("genotype")]
    _, p_kw = kruskal(*groups)
    sig_pairs, labels = pairwise_dunn_holm(df, feat, genotype_order)

    fig = px.box(
        df,
        x="genotype",
        y=feat,
        color="genotype",
        points="all",
        hover_data=hover_data,
        category_orders={"genotype": labels},
        title=f"{feature_titles[feat]} (Kruskal-Wallis p={p_kw:.3g})",
    )
    fig.update_traces(jitter=0.30, marker=dict(size=5, opacity=0.70))
    fig.update_layout(showlegend=False)
    _add_significance_brackets(fig, df, feat, sig_pairs, labels)
    return fig


def make_figure_wt_vs_mutant(
    df: pd.DataFrame,
    feat: str,
    feature_titles: dict[str, str],
    hover_data: list[str],
) -> go.Figure:
    """One WT vs pooled-mutant box plot with MWU + Cliff's delta (no file I/O)."""
    d = df.copy()
    d["WT_vs_mutant"] = np.where(d["genotype"] == "WT", "WT", "Mutant")
    wt = d[d["WT_vs_mutant"] == "WT"][feat]
    mut = d[d["WT_vs_mutant"] == "Mutant"][feat]
    _, p_u = mannwhitneyu(wt, mut, alternative="two-sided")
    delta = cliffs_delta(wt.values, mut.values)

    fig = px.box(
        d,
        x="WT_vs_mutant",
        y=feat,
        color="WT_vs_mutant",
        points="all",
        hover_data=hover_data,
        title=(
            f"{feature_titles[feat]} — WT vs Mutant "
            f"(MWU p={p_u:.3g}, Cliff's delta={delta:.2f})"
        ),
    )
    fig.update_traces(jitter=0.30, marker=dict(size=5, opacity=0.70))
    fig.update_layout(showlegend=False)
    return fig


def plot_by_genotype(df, features, feature_titles, hover_data, outdir="report_figures"):
    """Box plots per feature, grouped by genotype with Kruskal-Wallis + Dunn/Holm brackets."""
    genotype_order = genotype_category_order(df)
    for feat in features:
        fig = make_figure_by_genotype(
            df, feat, feature_titles, hover_data, genotype_order=genotype_order
        )
        save_plotly_figure(fig, outdir, f"{feat}_by_genotype")


def plot_wt_vs_mutant(df, features, feature_titles, hover_data, outdir="report_figures"):
    """Box plots per feature, WT vs all mutants, with MWU p-value and Cliff's delta."""
    for feat in features:
        fig = make_figure_wt_vs_mutant(df, feat, feature_titles, hover_data)
        save_plotly_figure(fig, outdir, f"{feat}_WT_vs_mutant")


def _stratified_cv_cap(df: pd.DataFrame, cv: int) -> int:
    """Upper bound on n_splits for StratifiedKFold given class counts."""
    if df.empty or "genotype" not in df.columns:
        return 2
    vc = df["genotype"].astype(str).value_counts()
    if len(vc) < 2:
        return 2
    m = int(vc.min())
    n = len(df)
    return max(2, min(int(cv), m, max(2, n // 2)))


def _collect_all_classifier_figures(
    df: pd.DataFrame,
    cv: int = 5,
    groups: np.ndarray | None = None,
) -> list[tuple[str, go.Figure]]:
    """Run all classifier modes; return (figure_id, figure) for HTML embedding."""
    if len(df) < 4:
        return []
    if groups is not None and len(np.unique(groups)) < 2:
        groups = None
    cv_eff = _stratified_cv_cap(df, cv) if groups is None else cv

    out: list[tuple[str, go.Figure]] = []
    for model_name in ("lda", "logistic", "svc"):
        for mode in ("multiclass", "binary"):
            try:
                batch = run_classifier(
                    df,
                    outdir=".",
                    model_name=model_name,
                    classification_mode=mode,
                    cv=cv_eff,
                    plot_importance=True,
                    groups=groups,
                    save_files=False,
                    return_figures=True,
                )
                if batch:
                    out.extend(batch)
            except Exception:
                continue
    return out


def _trial_sort_key(run_label) -> tuple:
    s = str(run_label)
    m = re.search(r"n(\d{3})", s, re.I)
    if m:
        return (0, int(m.group(1)))
    return (1, s)


def _figure_to_embed_html(fig: go.Figure, div_id: str, include_plotlyjs) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        div_id=div_id,
        config={"displayModeBar": True},
    )


def _html_cv_blurb_pooled(groups: np.ndarray | None) -> str:
    if groups is not None:
        return (
            "<p class=\"plot-blurb\"><strong>Cross-validation design:</strong> "
            "<em>GroupKFold</em> by run. Each fold holds out one complete recording, "
            "so no fly from that recording appears in training.</p>"
        )
    return (
        "<p class=\"plot-blurb\"><strong>Cross-validation design:</strong> "
        "<em>Stratified K-fold</em> on pooled flies. Folds may include flies from "
        "multiple recordings.</p>"
    )


def _html_cv_blurb_per_trial() -> str:
    return (
        "<p class=\"plot-blurb\"><strong>Cross-validation design:</strong> "
        "<em>Stratified K-fold</em> within a single run. Group-based splitting is "
        "not applicable when only one run is analyzed.</p>"
    )


def _pretty_classifier_caption(figure_id: str) -> str:
    """
    Convert internal figure IDs into reader-facing captions.
    Example: lda_multiclass_cv -> LDA multiclass cross-validation accuracy
    """
    tok = figure_id.split("_")
    if len(tok) < 3:
        return figure_id.replace("_", " ")
    model, mode, suffix = tok[0], tok[1], tok[-1]
    model_map = {"lda": "LDA", "logistic": "Logistic regression", "svc": "Linear SVM"}
    suffix_map = {"cv": "cross-validation accuracy", "importance": "feature importance"}
    model_txt = model_map.get(model, model.upper())
    suffix_txt = suffix_map.get(suffix, suffix)
    return f"{model_txt} {mode} {suffix_txt}"


def write_classification_html_report(
    df: pd.DataFrame,
    features: list[str],
    feature_titles: dict[str, str],
    hover_data: list[str],
    out_html: str,
    *,
    trial_column: str = "run",
    report_title: str = "24DPE — pooled and per-trial",
    pooled_cv: int = 5,
    pooled_cv_groups: np.ndarray | None = None,
    per_trial_cv: int = 5,
    include_pooled: bool = True,
    include_per_trial: bool = True,
) -> None:
    """
    Write a single standalone HTML file: pooled exploratory + classifiers, then
    each trial (``trial_column``) separately with the same plots and classifiers.

    Plotly.js is loaded from CDN once; figures are embedded as divs.
    """
    abs_out = os.path.abspath(out_html)
    out_dir = os.path.dirname(abs_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_f = len(df)
    n_r = int(df[trial_column].nunique()) if trial_column in df.columns else 0

    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html lang=\"en\"><head><meta charset=\"utf-8\"/>",
        f"<title>{html_module.escape(report_title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:1rem 2rem;max-width:1280px;line-height:1.45;}",
        "h1{border-bottom:1px solid #ccc;padding-bottom:0.25rem;}",
        "h2{margin-top:2.5rem;color:#222;}",
        "h3{margin-top:1.5rem;}",
        "h4{margin:0.75rem 0 0.25rem;font-size:0.95rem;color:#333;}",
        ".plot-wrap{margin-bottom:1.25rem;}",
        ".report-lead{background:#f6f8fa;border-radius:6px;padding:0.75rem 1rem;margin:1rem 0;}",
        ".section-intro{color:#333;margin:0.5rem 0 1rem;font-size:0.95rem;}",
        ".plot-blurb{font-size:0.88rem;color:#444;margin:0 0 0.35rem;padding:0.35rem 0.5rem;"
        "border-left:3px solid #467;font-style:normal;background:#fafafa;}",
        "</style></head><body>",
        f"<h1>{html_module.escape(report_title)}</h1>",
        "<div class=\"report-lead\">",
        "<p><strong>How to read this report</strong></p>",
        "<ul>",
        "<li><strong>Pooled cohort</strong> - all flies from all runs are analyzed together.</li>",
        "<li><strong>Per-run analysis</strong> - each run is analyzed independently.</li>",
        "</ul>",
        "<p>Titles include key statistical results (for example Kruskal-Wallis p-values, "
        "Mann-Whitney p-values, and cross-validation accuracy). Significance brackets "
        "show Dunn post-hoc comparisons with Holm-adjusted p-values.</p>",
        "</div>",
    ]

    plotly_cdn_used = False
    div_i = 0

    def emit(blurb_html: str, caption: str, fig: go.Figure) -> None:
        nonlocal plotly_cdn_used, div_i
        if blurb_html:
            parts.append(blurb_html)
        parts.append(f"<h4>{html_module.escape(caption)}</h4>")
        parts.append("<div class=\"plot-wrap\">")
        js = "cdn" if not plotly_cdn_used else False
        plotly_cdn_used = True
        parts.append(_figure_to_embed_html(fig, f"fig_{div_i}", js))
        div_i += 1
        parts.append("</div>")

    if include_pooled:
        # --- Pooled ---
        parts.append("<section>")
        parts.append("<h2>Pooled cohort analysis</h2>")
        parts.append(
            f"<p class=\"section-intro\">This section analyzes all "
            f"<strong>{n_f}</strong> flies across <strong>{n_r}</strong> runs together. "
            f"Genotype ordering follows vial order within each run, then run order.</p>"
        )
        g_order_pool = genotype_category_order(df)
        for feat in features:
            ft = html_module.escape(feature_titles[feat])
            blurb = (
                f"<p class=\"plot-blurb\">Each point represents one fly. "
                f"The title reports the Kruskal-Wallis omnibus test for <em>{ft}</em>; "
                f"horizontal brackets indicate Dunn post-hoc comparisons with Holm adjustment.</p>"
            )
            emit(
                blurb,
                f"{feature_titles[feat]} — by genotype (pooled)",
                make_figure_by_genotype(
                    df, feat, feature_titles, hover_data, genotype_order=g_order_pool
                ),
            )
        for feat in features:
            ft = html_module.escape(feature_titles[feat])
            blurb = (
                f"<p class=\"plot-blurb\">This panel compares WT against all mutant genotypes "
                f"for <em>{ft}</em>. The title reports Mann-Whitney p-value and Cliff's delta.</p>"
            )
            emit(
                blurb,
                f"{feature_titles[feat]} — WT vs mutant (pooled)",
                make_figure_wt_vs_mutant(df, feat, feature_titles, hover_data),
            )
        parts.append("<h3>Genotype classification (pooled cohort)</h3>")
        parts.append(_html_cv_blurb_pooled(pooled_cv_groups))
        for cap, fig in _collect_all_classifier_figures(
            df, cv=pooled_cv, groups=pooled_cv_groups
        ):
            pretty = _pretty_classifier_caption(cap)
            cid = html_module.escape(pretty)
            emit(
                f"<p class=\"plot-blurb\">Classification results for the pooled cohort "
                f"using <code>{cid}</code>.</p>",
                pretty,
                fig,
            )
        parts.append("</section>")

    # --- Per trial (skip if only one run — same as pooled) ---
    if include_per_trial and trial_column in df.columns and df[trial_column].nunique() > 1:
        trials = sorted(df[trial_column].unique(), key=_trial_sort_key)
    else:
        trials = []

    for trial in trials:
        sub = df[df[trial_column] == trial].copy()
        if len(sub) < 2:
            continue
        parts.append("<section>")
        parts.append(f"<h2>Per-run analysis - {html_module.escape(str(trial))}</h2>")
        m_trial = re.search(r"n(\d{3})", str(trial), re.I)
        trial_idx = f"n{m_trial.group(1)}" if m_trial else None
        idx_line = (
            f"Trial index <strong>{html_module.escape(trial_idx)}</strong> — "
            if trial_idx
            else ""
        )
        parts.append(
            f"<p class=\"section-intro\">{idx_line}"
            f"This section includes only flies from run "
            f"<code>{html_module.escape(str(trial))}</code> "
            f"(<strong>{len(sub)}</strong> flies). No other runs contribute to this block.</p>"
        )
        g_order = genotype_category_order(sub)
        for feat in features:
            ft = html_module.escape(feature_titles[feat])
            tr = html_module.escape(str(trial))
            blurb = (
                f"<p class=\"plot-blurb\">Within-run distribution of <em>{ft}</em> by genotype "
                f"for run <code>{tr}</code>. Statistical tests match the pooled analysis.</p>"
            )
            emit(
                blurb,
                f"{feature_titles[feat]} — by genotype",
                make_figure_by_genotype(
                    sub, feat, feature_titles, hover_data, genotype_order=g_order
                ),
            )
        for feat in features:
            ft = html_module.escape(feature_titles[feat])
            tr = html_module.escape(str(trial))
            blurb = (
                f"<p class=\"plot-blurb\">Within-run WT versus mutant comparison for "
                f"<em>{ft}</em> in run <code>{tr}</code>.</p>"
            )
            emit(
                blurb,
                f"{feature_titles[feat]} — WT vs mutant",
                make_figure_wt_vs_mutant(sub, feat, feature_titles, hover_data),
            )
        parts.append("<h3>Genotype classification (single run)</h3>")
        parts.append(_html_cv_blurb_per_trial())
        for cap, fig in _collect_all_classifier_figures(
            sub, cv=per_trial_cv, groups=None
        ):
            pretty = _pretty_classifier_caption(cap)
            cid = html_module.escape(pretty)
            tr = html_module.escape(str(trial))
            emit(
                f"<p class=\"plot-blurb\">Classification results for run "
                f"<code>{tr}</code> using <code>{cid}</code>.</p>",
                pretty,
                fig,
            )
        parts.append("</section>")

    parts.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_classification_report_site(
    df: pd.DataFrame,
    features: list[str],
    feature_titles: dict[str, str],
    hover_data: list[str],
    out_dir: str,
    *,
    trial_column: str = "run",
    report_title: str = "Classification report",
    pooled_cv: int = 5,
    pooled_cv_groups: np.ndarray | None = None,
    per_trial_cv: int = 5,
    entry_filename: str = "classification_report.html",
    latent_page_filename: str | None = None,
) -> str:
    """
    Write a navigable multi-page report site and return entry HTML path.

    Layout:
    - entry page (left nav + overview)
    - pooled page
    - per-trial page(s)
    - optional latent-space page link
    """
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    sections_dir = os.path.join(out_dir_abs, "report_sections")
    os.makedirs(sections_dir, exist_ok=True)

    pooled_html = os.path.join(sections_dir, "pooled.html")
    write_classification_html_report(
        df=df,
        features=features,
        feature_titles=feature_titles,
        hover_data=hover_data,
        out_html=pooled_html,
        trial_column=trial_column,
        report_title=f"{report_title} - pooled and per-trial",
        pooled_cv=pooled_cv,
        pooled_cv_groups=pooled_cv_groups,
        per_trial_cv=per_trial_cv,
        include_pooled=True,
        include_per_trial=False,
    )

    # Keep per-trial pages separate so very large cohorts stay responsive.
    trial_links: list[tuple[str, str]] = []
    if trial_column in df.columns and df[trial_column].nunique() > 1:
        for trial in sorted(df[trial_column].unique(), key=_trial_sort_key):
            sub = df[df[trial_column] == trial].copy()
            if len(sub) < 2:
                continue
            safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(trial))
            fn = f"trial_{safe}.html"
            trial_html = os.path.join(sections_dir, fn)
            write_classification_html_report(
                df=sub,
                features=features,
                feature_titles=feature_titles,
                hover_data=hover_data,
                out_html=trial_html,
                trial_column=trial_column,
                report_title=f"{report_title} - {trial}",
                pooled_cv=per_trial_cv,
                pooled_cv_groups=None,
                per_trial_cv=per_trial_cv,
                include_pooled=True,
                include_per_trial=False,
            )
            trial_links.append((str(trial), os.path.join("report_sections", fn).replace("\\", "/")))

    nav_items = [
        ("Overview", os.path.join("report_sections", os.path.basename(pooled_html)).replace("\\", "/")),
        ("Pooled + classifiers", os.path.join("report_sections", os.path.basename(pooled_html)).replace("\\", "/")),
    ]
    if latent_page_filename:
        nav_items.append(("Latent-space", latent_page_filename.replace("\\", "/")))

    entry_path = os.path.join(out_dir_abs, entry_filename)
    nav_html = "\n".join(
        f"<li><a href=\"{html_module.escape(href)}\" target=\"viewer\">{html_module.escape(label)}</a></li>"
        for label, href in nav_items
    )
    pooled_src = os.path.join("report_sections", os.path.basename(pooled_html)).replace("\\", "/")
    trials_html = "\n".join(
        f"<li><a href=\"{html_module.escape(href)}\" target=\"viewer\">{html_module.escape(label)}</a></li>"
        for label, href in trial_links
    )
    with open(entry_path, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "<!DOCTYPE html>",
                    "<html lang=\"en\"><head><meta charset=\"utf-8\"/>",
                    f"<title>{html_module.escape(report_title)}</title>",
                    "<style>",
                    "body{margin:0;font-family:system-ui,sans-serif;display:grid;grid-template-columns:320px 1fr;min-height:100vh;}",
                    "aside{border-right:1px solid #ddd;padding:1rem;overflow:auto;background:#fafafa;}",
                    "main{padding:0;margin:0;}",
                    "h1{font-size:1.05rem;margin:0 0 0.5rem;}",
                    "h2{font-size:0.95rem;margin:1rem 0 0.4rem;color:#333;}",
                    "ul{list-style:none;padding:0;margin:0;}",
                    "li{margin:0.2rem 0;}",
                    "a{text-decoration:none;color:#1a4c8b;}",
                    "a:hover{text-decoration:underline;}",
                    "iframe{border:0;width:100%;height:100vh;}",
                    ".hint{font-size:0.86rem;color:#555;line-height:1.35;}",
                    "</style></head><body>",
                    f"<aside><h1>{html_module.escape(report_title)}</h1>",
                    "<p class=\"hint\">Navigation hub for pooled, per-trial, and optional latent-space outputs.</p>",
                    "<h2>Main sections</h2>",
                    f"<ul>{nav_html}</ul>",
                    "<h2>Per-trial pages</h2>",
                    f"<ul>{trials_html or '<li><em>Single-trial dataset: none generated.</em></li>'}</ul>",
                    "</aside>",
                    f"<main><iframe name=\"viewer\" src=\"{html_module.escape(pooled_src)}\"></iframe></main>",
                    "</body></html>",
                ]
            )
        )
    return entry_path
