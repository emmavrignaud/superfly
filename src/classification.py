"""
src/classification.py

Genotype classification and statistical visualisation.

One classifier method is active per run, selected by
``classification.method`` in config.yaml. Defaults to SVC with an RBF kernel;
LDA and Logistic Regression are also available. Each method reads its own
hyperparameters from its sub-block in config (e.g. ``classification.svc``).

Plots: cross-validation accuracy, feature importance (linear backends only,
non-linear kernels skip gracefully), per-genotype boxes
       (Kruskal-Wallis + Dunn/Holm significance brackets), WT-vs-mutant
       comparison with Cliff's delta; optional pooled + per-trial HTML report.
"""

import html as html_module
import json
import logging
import os
import re
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import mannwhitneyu, kruskal

import scikit_posthocs as sp

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from utils import load_config as _load_config  # noqa: E402

from src.plot_colors import (  # noqa: E402
    genotype_color_map_for_dataframe,
    wt_vs_mutant_color_map,
)

_CFG = _load_config(_REPO_ROOT / "config.yaml").classification

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
    Load ordered_tracks.csv from a run directory and add a "genotype" column.

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

    csv_path = os.path.join(run_dir, "ordered_tracks.csv")
    if not os.path.exists(csv_path):
        # Older runs used compact_tracks.csv with compact_id instead of ordered_id
        csv_path = os.path.join(run_dir, "compact_tracks.csv")
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"compact_id": "ordered_id"})
    else:
        df = pd.read_csv(csv_path)
    df["genotype"] = df["vial_id"].map(vial_to_genotype)
    return df


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

_LINEAR_KERNELS = {"linear"}


def _resolve_method(model_name: str | None) -> str:
    """Pick the active method: explicit override first, otherwise config."""
    name = (model_name or _CFG.method).strip().lower()
    if name not in {"svc", "lda", "logistic"}:
        raise ValueError(f"classification.method must be svc, lda, or logistic; got {name!r}")
    return name


def _gamma_value(raw):
    """Accept "scale", "auto", or a numeric string/float for SVC.gamma."""
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"scale", "auto"}:
            return s
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"svc.gamma must be 'scale', 'auto', or a number; got {raw!r}")
    return float(raw)


def make_classifier(model_name: str | None = None):
    """Return an unfitted classifier.

    With ``model_name`` left as ``None``, reads ``classification.method`` from
    config.yaml. Each backend's hyperparameters are sourced from its own
    sub-block (``classification.svc``, ``classification.lda``,
    ``classification.logistic``).
    """
    name = _resolve_method(model_name)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver=str(_CFG.lda.solver))
    if name == "logistic":
        return LogisticRegression(
            C=float(_CFG.logistic.C),
            max_iter=int(_CFG.logistic.max_iter),
            solver=str(_CFG.logistic.solver),
        )
    return SVC(
        kernel=str(_CFG.svc.kernel),
        C=float(_CFG.svc.C),
        gamma=_gamma_value(_CFG.svc.gamma),
    )


def _svc_grid_search_enabled(model_name: str) -> bool:
    """True when SVC + grid_search.enabled is set in config.yaml."""
    if _resolve_method(model_name) != "svc":
        return False
    gs = getattr(_CFG.svc, "grid_search", None)
    return bool(getattr(gs, "enabled", False))


def _polynomial_cfg():
    """Return the polynomial-features config block, or None if absent/disabled."""
    poly = getattr(_CFG, "polynomial", None)
    if poly is None or not bool(getattr(poly, "enabled", False)):
        return None
    return poly


def _polynomial_step():
    """sklearn step for the polynomial-features stage, or None if disabled.

    The pipeline always carries the step when polynomial.enabled is true so
    grid search can vary ``degree`` and ``interaction_only`` without
    re-building the pipeline. ``degree=1`` means no expansion (identity).
    """
    poly = _polynomial_cfg()
    if poly is None:
        return None
    deg = getattr(poly, "degree", 2)
    if isinstance(deg, (list, tuple)):
        deg = int(deg[0])
    return PolynomialFeatures(
        degree=int(deg),
        interaction_only=bool(getattr(poly, "interaction_only", False)),
        include_bias=bool(getattr(poly, "include_bias", False)),
    )


def _polynomial_grid_params() -> dict:
    """Pipeline param-grid entries for polynomial.degree / interaction_only.

    Only adds keys whose config value is a list (i.e. explicit grid). Single
    scalars set at pipeline-construction time stay fixed.
    """
    poly = _polynomial_cfg()
    if poly is None:
        return {}
    grid: dict = {}
    deg = getattr(poly, "degree", None)
    if isinstance(deg, (list, tuple)):
        grid["poly__degree"] = [int(d) for d in deg]
    inter = getattr(poly, "interaction_only", None)
    if isinstance(inter, (list, tuple)):
        grid["poly__interaction_only"] = [bool(v) for v in inter]
    return grid


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_xy(df: pd.DataFrame):
    """Return (X array, feature_names list) dropping non-numeric and ordered_id."""
    X = df.select_dtypes(include=[np.number]).drop(columns=["ordered_id"], errors="ignore")
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

def _resolve_splitter(cv: int, groups: np.ndarray | None):
    """Pick a splitter consistent with the calling context.

    Returns (splitter, scheme_label, fit_kwargs). ``fit_kwargs`` is the dict to
    pass to ``cross_val_score`` / ``GridSearchCV.fit`` so the same call site
    can serve both grouped and stratified runs.
    """
    if groups is not None:
        n_unique = len(np.unique(groups))
        n_splits = min(int(cv), n_unique)
        if n_splits < 2:
            raise ValueError(
                f"GroupKFold needs at least 2 groups; got {n_unique} unique group(s)."
            )
        return GroupKFold(n_splits=n_splits), "group", {"groups": groups}
    return StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=42), "stratified", {}


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
    Run k-fold CV, plot bar chart, save figure, return (scores, fig, info).

    If ``groups`` is None (default), uses StratifiedKFold with integer ``cv``.
    If ``groups`` is provided, uses ``GroupKFold`` so entire groups (e.g. one
    video / run) stay in train or test - no leakage across videos. Number of
    splits is ``min(cv, n_unique_groups)``.

    When the active method is SVC and ``classification.svc.grid_search.enabled``
    is true, ``cross_val_score`` is replaced by ``GridSearchCV`` over
    (C, gamma). The best params and best CV score are printed and returned in
    ``info``; the returned model is the refit estimator.

    Returns
    -------
    scores : ndarray
        Per-fold accuracy scores (best-estimator scores when grid searching).
    fig : plotly.graph_objects.Figure
        Bar chart of fold scores.
    info : dict
        Run summary: ``{"scheme", "mean_accuracy", "best_estimator", ...}``.
        When grid searching: also ``"best_params"``, ``"best_score"``,
        ``"cv_results"`` (top 5 ranked rows).
    """
    splitter, scheme, fit_kwargs = _resolve_splitter(cv, groups)

    info: dict = {"scheme": scheme}

    use_grid = _svc_grid_search_enabled(model_name)
    title_suffix = ""

    if use_grid:
        gs_cfg = _CFG.svc.grid_search
        param_grid = {
            "clf__C": [float(c) for c in gs_cfg.C],
            "clf__gamma": [_gamma_value(g) for g in gs_cfg.gamma],
            **_polynomial_grid_params(),
        }
        combos = 1
        for v in param_grid.values():
            combos *= len(v)
        print(
            f"[classification] grid search enabled: {combos} combos over "
            f"{list(param_grid)} x {splitter.get_n_splits(X, y, **fit_kwargs)} folds ({scheme})"
        )
        search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=splitter,
            scoring="accuracy",
            refit=True,
            n_jobs=-1,
        )
        search.fit(X, y, **fit_kwargs)

        best_idx = search.best_index_
        scores = np.array([
            search.cv_results_[f"split{i}_test_score"][best_idx]
            for i in range(search.n_splits_)
        ])
        best_params = {k.replace("clf__", "").replace("poly__", "poly."): v
                       for k, v in search.best_params_.items()}
        print(
            f"[classification] grid search best (svc): "
            f"C={best_params.get('C')} gamma={best_params.get('gamma')} "
            f"-> mean CV accuracy = {search.best_score_:.4f}"
        )

        ranked = sorted(
            zip(
                search.cv_results_["mean_test_score"],
                search.cv_results_["params"],
            ),
            key=lambda r: r[0],
            reverse=True,
        )[:5]
        info["best_estimator"] = search.best_estimator_
        info["best_params"] = best_params
        info["best_score"] = float(search.best_score_)
        info["cv_results_top5"] = [
            {
                "mean_test_score": float(s),
                "params": {k.replace("clf__", "").replace("poly__", "poly."): v
                           for k, v in p.items()},
            }
            for s, p in ranked
        ]
        title_suffix = ""  # hyperparameter search details printed to stdout, not the title
    else:
        scores = cross_val_score(model, X, y, cv=splitter, **fit_kwargs)
        info["best_estimator"] = model

    info["mean_accuracy"] = float(scores.mean())
    info["std_accuracy"] = float(scores.std())

    pretty_mode = {"multiclass": "all genotypes", "binary": "WT vs mutant"}.get(
        classification_mode, classification_mode
    )

    fig = go.Figure()
    fig.add_bar(x=list(range(1, len(scores) + 1)), y=scores)
    fig.add_hline(y=scores.mean(), line_dash="dash")
    fig.update_layout(
        title=(
            f"Genotype prediction accuracy, {pretty_mode}<br>"
            f"<sup>mean = {scores.mean() * 100:.1f}%</sup>"
        ),
        xaxis_title="Fold",
        yaxis_title="Accuracy (fraction correct)",
        yaxis_range=[0, 1],
    )

    if save:
        save_plotly_figure(fig, outdir, f"{model_name}_{classification_mode}")
    return scores, fig, info


def plot_feature_importance(
    model,
    X,
    y,
    feature_names,
    model_name: str,
    classification_mode: str,
    refit: bool = True,
):
    """Fit (optionally) and plot feature importance as a horizontal bar chart.

    Returns ``None`` (and prints a one-line notice) when the active backend
    does not expose linear weights, e.g. SVC with a non-linear kernel
    (rbf/poly/sigmoid). Use ``refit=False`` to plot importance for an already
    fitted estimator (e.g. the GridSearchCV best_estimator_).
    """
    name = (model_name or "").strip().lower()

    if name == "svc":
        kernel = str(_CFG.svc.kernel).strip().lower()
        if kernel not in _LINEAR_KERNELS:
            print(
                f"[classification] feature importance skipped: svc kernel={kernel!r} "
                "is non-linear (no coef_)."
            )
            return None

    if refit:
        model.fit(X, y)

    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    # When a PolynomialFeatures step is in the pipeline, the classifier's
    # coefficients map to the expanded feature space, not the original columns.
    if hasattr(model, "named_steps") and "poly" in model.named_steps:
        poly = model.named_steps["poly"]
        try:
            feature_names = list(poly.get_feature_names_out(feature_names))
        except Exception:
            n = clf.coef_.shape[-1] if hasattr(clf, "coef_") else len(feature_names)
            feature_names = [f"poly_{i}" for i in range(n)]

    if name == "logistic":
        values = np.mean(np.abs(clf.coef_), axis=0)
        xlabel = "Mean |coefficient|"
    elif name == "lda":
        values = np.mean(np.abs(clf.scalings_), axis=1)
        xlabel = "Mean |loading|"
    elif name == "svc":
        coef = clf.coef_
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        xlabel = "|weight|"
    else:
        return None

    idx = np.argsort(values)
    fig = go.Figure()
    fig.add_bar(x=values[idx], y=[feature_names[i] for i in idx], orientation="h")
    fig.update_layout(
        title=f"Feature importance: {name.upper()} ({classification_mode})",
        xaxis_title=xlabel,
        yaxis_title="Feature",
    )
    return fig


def run_classifier(
    df: pd.DataFrame,
    outdir: str = "report_figures",
    model_name: str | None = None,
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
    model_name : str, optional
        Override for the active method (``"lda" | "logistic" | "svc"``). When
        left as ``None`` (recommended) the method comes from
        ``classification.method`` in config.yaml.
    classification_mode : str
        "multiclass" or "binary".
    cv : int
        Number of CV folds (or max folds for GroupKFold when ``groups`` is set).
    plot_importance : bool
        Whether to produce and save a feature-importance figure. Skipped
        gracefully when the active SVC kernel is non-linear.
    groups : np.ndarray, optional
        One group id per row (e.g. video / run name). If given, CV uses
        GroupKFold so no fly from a held-out video appears in training.
    save_files : bool
        If False, figures are not written to ``outdir`` (for HTML bundling).
    return_figures : bool
        If True, returns a list of ``(figure_id, go.Figure)`` (figures are still
        saved when ``save_files`` is True).
    """
    name = _resolve_method(model_name)

    X, feature_names = prepare_xy(df)
    y = prepare_target(df, classification_mode)
    print(
        f"[classification] dataset: n_samples={X.shape[0]}, n_features={X.shape[1]} "
        f"({classification_mode}, {name})"
    )

    poly_step = _polynomial_step()
    steps = [("scaler", StandardScaler())]
    if poly_step is not None:
        steps.append(("poly", poly_step))
        # PolynomialFeatures works best when its inputs are already centred;
        # a second standardisation after expansion keeps SVC's RBF gamma sensible.
        steps.append(("rescale", StandardScaler()))
    steps.append(("clf", make_classifier(name)))
    pipeline = Pipeline(steps)

    scores, fig_cv, info = run_cross_validation(
        pipeline, name, X, y, cv=cv,
        classification_mode=classification_mode, outdir=outdir,
        groups=groups,
        save=save_files,
    )

    artifacts: list[tuple[str, go.Figure]] = []
    if return_figures:
        artifacts.append((f"{name}_{classification_mode}_cv", fig_cv))

    if plot_importance:
        # When GridSearchCV ran, ``info["best_estimator"]`` is already fitted;
        # avoid refitting and use it as-is. Otherwise refit the fresh pipeline.
        importance_model = info.get("best_estimator", pipeline)
        do_refit = importance_model is pipeline
        fig_imp = plot_feature_importance(
            importance_model, X, y, feature_names, name, classification_mode, refit=do_refit
        )
        if fig_imp is not None:
            if save_files:
                save_plotly_figure(
                    fig_imp, outdir, f"{name}_{classification_mode}_importance"
                )
            if return_figures:
                artifacts.append((f"{name}_{classification_mode}_importance", fig_imp))

    if return_figures:
        return artifacts
    return None


def run_permutation_test(
    df: pd.DataFrame,
    outdir: str = "report_figures",
    model_name: str | None = None,
    classification_mode: str = "multiclass",
    cv: int = 5,
    groups: np.ndarray | None = None,
    n_permutations: int = 500,
    seed: int = 42,
    save_files: bool = True,
) -> dict:
    """
    Permutation test for classifier significance.

    Runs CV on the real labels, then repeats ``n_permutations`` times with
    shuffled labels to build a null distribution.  Reports where the real
    balanced accuracy falls in that null.

    Returns a dict with keys:
        real_score   float — mean balanced-accuracy across CV folds
        null_scores  ndarray(n_permutations,)
        p_value      float — fraction of null ≥ real (one-tailed)
        fig          go.Figure — histogram of null + vertical line for real score
    """
    name = _resolve_method(model_name)
    X, _ = prepare_xy(df)
    y = prepare_target(df, classification_mode)

    poly_step = _polynomial_step()
    steps = [("scaler", StandardScaler())]
    if poly_step is not None:
        steps.append(("poly", poly_step))
        steps.append(("rescale", StandardScaler()))
    steps.append(("clf", make_classifier(name)))
    pipeline = Pipeline(steps)

    splitter, scheme, fit_kwargs = _resolve_splitter(cv, groups)

    real_scores = cross_val_score(
        pipeline, X, y, cv=splitter, scoring="balanced_accuracy", **fit_kwargs
    )
    real_score = float(real_scores.mean())

    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        s = cross_val_score(
            pipeline, X, y_perm, cv=splitter, scoring="balanced_accuracy", **fit_kwargs
        )
        null_scores[i] = s.mean()

    p_value = float((null_scores >= real_score).sum() + 1) / (n_permutations + 1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=null_scores, name="Null distribution",
        nbinsx=30, marker_color="#adb5bd", opacity=0.85,
    ))
    fig.add_vline(
        x=real_score,
        line_color="#e63946", line_width=2.5,
        annotation_text=f"Real: {real_score:.3f}  p={p_value:.4g}",
        annotation_position="top right",
        annotation_font_color="#e63946",
    )
    pretty_mode = {"multiclass": "all genotypes", "binary": "WT vs mutant"}.get(
        classification_mode, classification_mode
    )
    fig.update_layout(
        title=(
            f"Permutation test [{pretty_mode}] — balanced accuracy "
            f"(n={n_permutations}, {scheme} CV)"
        ),
        xaxis_title="Balanced accuracy (mean over folds)",
        yaxis_title="Count",
        height=420,
        showlegend=False,
    )

    if save_files:
        save_plotly_figure(fig, outdir, f"permutation_test_{classification_mode}", show=False)

    print(
        f"[permutation_test] {classification_mode}: "
        f"real={real_score:.3f}  p={p_value:.4g}  (n={n_permutations})"
    )
    return {
        "real_score": real_score,
        "null_scores": null_scores,
        "p_value": p_value,
        "fig": fig,
    }


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


def pretty_run_label(run) -> str:
    """Display-only rename for a run identifier.

    Strips the ``run_<n>_`` prefix and the ``n`` before the trial index so
    ``run_122_6DPE_n001`` reads as ``6DPE_001`` in titles, section headers,
    and per-run boxplot x-axis labels. Internal dataframe values keep the
    full name (callers map at display time).
    """
    s = str(run)
    m = re.match(r"(?:run_\d+_)?(\d+DPE)_n?(\d+)", s, re.I)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return s


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


def pooled_genotype_color_map(df: pd.DataFrame) -> dict[str, str]:
    """Stable genotype → hex map for every Plotly chart built from this pooled frame."""
    return genotype_color_map_for_dataframe(df, genotype_category_order(df))


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


def _boxplot_trim_settings() -> tuple[bool, float, int]:
    """Read classification.boxplot_trim from config (defaults if missing)."""
    try:
        t = _CFG.boxplot_trim
    except (AttributeError, KeyError):
        return False, 4.0, 1
    enabled = bool(t.enabled)
    n_std = float(t.n_std)
    ddof = int(getattr(t, "ddof", 1))
    return enabled, n_std, ddof


def _boxplot_trim_suffix() -> str:
    _, n_std, _ = _boxplot_trim_settings()
    return f" (values beyond ±{n_std:g} SD of cohort mean excluded)"


def boxplot_trim_title_suffix() -> str:
    """Non-empty note for figure titles when ``classification.boxplot_trim.enabled``."""
    enabled, _, _ = _boxplot_trim_settings()
    return _boxplot_trim_suffix() if enabled else ""


def apply_boxplot_trim_for_feature(df: pd.DataFrame, feat: str) -> pd.DataFrame:
    """
    Return a copy of ``df``. If ``classification.boxplot_trim.enabled``, keep only
    rows whose ``feat`` lies in [mean − n_std·σ, mean + n_std·σ] computed on that
    column (cohort-wide). Caller dataframe is never modified.

    Use this for notebook-only ``px.box`` loops so they match ``plot_by_genotype``.
    """
    out = df.copy()
    enabled, n_std, ddof = _boxplot_trim_settings()
    if not enabled or n_std <= 0 or feat not in out.columns:
        return out
    s = out[feat].dropna()
    if s.empty:
        return out
    mu = float(s.mean())
    sig = float(s.std(ddof=ddof))
    if not np.isfinite(sig) or sig <= 0:
        return out
    lo, hi = mu - n_std * sig, mu + n_std * sig
    return out.loc[out[feat].between(lo, hi)].copy()


def _local_df_for_boxplot(
    df: pd.DataFrame,
    feat: str,
    hover_data: list[str],
) -> tuple[pd.DataFrame, str]:
    """
    Local copy for one feature's boxplot + univariate tests. Original ``df`` unchanged.

    When ``classification.boxplot_trim.enabled`` is true, rows on the copy whose
    ``feat`` lies outside mean ± n_std·σ (cohort-wide on that copy) are dropped
    from the copy only; Kruskal-Wallis, Dunn, MWU, Cliff, and px.box use this copy.
    """
    cols = ["genotype", feat] + [c for c in hover_data if c in df.columns]
    work = df.loc[:, cols].copy()
    enabled, n_std, _ = _boxplot_trim_settings()
    if not enabled or n_std <= 0:
        return work, ""
    trimmed = apply_boxplot_trim_for_feature(work, feat)
    if len(trimmed) == len(work):
        return work, ""
    return trimmed, _boxplot_trim_suffix()


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
    work, trim_suffix = _local_df_for_boxplot(df, feat, hover_data)
    groups = [g[feat].values for _, g in work.groupby("genotype") if len(g[feat].dropna()) > 0]
    if len(groups) < 2:
        p_kw = float("nan")
    else:
        _, p_kw = kruskal(*groups)
    sig_pairs, labels = pairwise_dunn_holm(work, feat, genotype_order)
    hover_cols = [c for c in hover_data if c in work.columns]
    pref = genotype_order if genotype_order is not None else genotype_category_order(df)
    color_discrete_map = genotype_color_map_for_dataframe(work, pref)

    fig = px.box(
        work,
        x="genotype",
        y=feat,
        color="genotype",
        points="all",
        hover_data=hover_cols,
        category_orders={"genotype": labels},
        color_discrete_map=color_discrete_map,
        title=f"{feature_titles[feat]}<br><sup>p = {p_kw:.3g}</sup>",
    )
    fig.update_traces(jitter=0.30, marker=dict(size=5, opacity=0.70))
    fig.update_layout(showlegend=False, yaxis_title=feature_titles[feat])
    _add_significance_brackets(fig, work, feat, sig_pairs, labels)
    return fig


def make_figure_wt_vs_mutant(
    df: pd.DataFrame,
    feat: str,
    feature_titles: dict[str, str],
    hover_data: list[str],
) -> go.Figure:
    """One WT vs pooled-mutant box plot with MWU + Cliff's delta (no file I/O)."""
    work, trim_suffix = _local_df_for_boxplot(df, feat, hover_data)
    d = work.copy()
    d["WT_vs_mutant"] = np.where(d["genotype"] == "WT", "WT", "Mutant")
    wt = d[d["WT_vs_mutant"] == "WT"][feat]
    mut = d[d["WT_vs_mutant"] == "Mutant"][feat]
    if len(wt) < 1 or len(mut) < 1:
        p_u = float("nan")
        delta = float("nan")
    else:
        _, p_u = mannwhitneyu(wt, mut, alternative="two-sided")
        delta = cliffs_delta(wt.values, mut.values)

    hover_cols = [c for c in hover_data if c in d.columns]
    gmap = genotype_color_map_for_dataframe(d, genotype_category_order(d))
    fig = px.box(
        d,
        x="WT_vs_mutant",
        y=feat,
        color="WT_vs_mutant",
        points="all",
        hover_data=hover_cols,
        category_orders={"WT_vs_mutant": ["WT", "Mutant"]},
        color_discrete_map=wt_vs_mutant_color_map(gmap),
        title=(
            f"{feature_titles[feat]}, WT vs mutant<br>"
            f"<sup>p = {p_u:.3g}, effect size = {delta:.2f}</sup>"
        ),
    )
    fig.update_traces(jitter=0.30, marker=dict(size=5, opacity=0.70))
    fig.update_layout(showlegend=False, yaxis_title=feature_titles[feat])
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
    """Run the configured classifier only; return (figure_id, figure) for HTML embedding."""
    if len(df) < 4:
        return []
    if groups is not None and len(np.unique(groups)) < 2:
        groups = None
    cv_eff = _stratified_cv_cap(df, cv) if groups is None else cv

    out: list[tuple[str, go.Figure]] = []
    model_name = _resolve_method(None)
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
            "<p class=\"plot-blurb\"><strong>How the classifier is tested:</strong> "
            "leave-one-video-out. Each fold holds out one complete recording and asks the "
            "classifier to predict the genotype of those flies, having been trained only on "
            "flies from other recordings. This is the strictest test: no fly from a held-out "
            "video ever appears in training.</p>"
        )
    return (
        "<p class=\"plot-blurb\"><strong>How the classifier is tested:</strong> "
        "stratified k-fold. Flies from all recordings are pooled, then split into folds "
        "that preserve the genotype balance. Folds can mix flies from different recordings.</p>"
    )


def _html_cv_blurb_per_trial() -> str:
    return (
        "<p class=\"plot-blurb\"><strong>How the classifier is tested:</strong> "
        "stratified k-fold within this single recording. Leave-one-video-out is not applicable "
        "because only one video is in scope here.</p>"
    )


def _pretty_classifier_caption(figure_id: str) -> str:
    """Convert internal figure ids into reader-facing captions (no acronyms in the lede)."""
    tok = figure_id.split("_")
    if len(tok) < 3:
        return figure_id.replace("_", " ")
    model, mode, suffix = tok[0], tok[1], tok[-1]
    model_map = {
        "lda": "Linear Discriminant Analysis",
        "logistic": "Logistic Regression",
        "svc": "Support Vector Machine",
    }
    mode_map = {"multiclass": "all genotypes", "binary": "WT vs mutant"}
    suffix_map = {
        "cv": "Prediction accuracy",
        "importance": "Which features matter most",
    }
    return (
        f"{suffix_map.get(suffix, suffix)}, "
        f"{mode_map.get(mode, mode)} ({model_map.get(model, model.upper())})"
    )


def _glossary_card_html() -> str:
    """One-time glossary explaining the stats + sign conventions used everywhere in the report."""
    from src.features import UNITS as _U  # local import: avoid module-load order issue
    v = _U["velocity"]
    return (
        "<div class=\"glossary-card\">"
        "<h3>How to read this report</h3>"
        "<ul>"
        "<li><strong>p-value</strong>: smaller = stronger evidence that the groups differ. "
        "Conventional thresholds: <code>* p&lt;0.05</code>, <code>** p&lt;0.01</code>, "
        "<code>*** p&lt;0.001</code>.</li>"
        "<li><strong>Per-genotype box plots</strong>: a Kruskal-Wallis test asks "
        "<em>is there any difference between genotypes?</em> Significant pairs are then "
        "highlighted with brackets (Dunn post-hoc with Holm correction).</li>"
        "<li><strong>WT vs mutant box plots</strong>: a Mann-Whitney test compares wild-type "
        "against the pooled mutants. The <em>effect size</em> (Cliff's delta) runs from "
        "−1 to 1. 0 means no tendency for either group to be larger; ±1 means strict "
        "separation.</li>"
        "<li><strong>Classifier accuracy</strong>: each fold trains on most of the cohort "
        "and predicts the held-out flies' genotype. The reported number is the mean fraction "
        "of correct predictions across folds. <em>Leave-one-video-out</em> means each fold "
        "holds out an entire recording, so the classifier is never tested on a fly whose "
        "video it has seen.</li>"
        f"<li><strong>Sign convention for axis-resolved features</strong> (v_x, v_y, a_x, a_y, "
        f"horizontal and vertical distance): positive = motion to the right or downward in the "
        f"video frame. Negative = left or upward. Velocity averages near zero "
        f"(typical units {html_module.escape(v)}) indicate <em>no preferred direction</em>, "
        f"not absence of motion. Variability and magnitude columns capture how much the "
        f"fly actually moves.</li>"
        "<li><strong>Three-column rows</strong> (e.g. <em>Average velocity</em>): the same "
        "quantity broken into its horizontal component, vertical component, and combined "
        "magnitude.</li>"
        "</ul>"
        "</div>"
    )


def _families_and_singletons(features: list[str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Split a feature list into (families, singletons) using src.features groupings.

    Features not present in ``features`` are dropped from a family; a family
    with fewer than 2 surviving columns degrades to singletons.
    """
    from src.features import feature_families, singleton_features
    feat_set = set(features)
    fams: list[tuple[str, list[str]]] = []
    used: set[str] = set()
    for row_title, cols in feature_families():
        kept = [c for c in cols if c in feat_set]
        if len(kept) >= 2:
            fams.append((row_title, kept))
            used.update(kept)
        else:
            used.update(kept)
    singletons = [f for f in singleton_features() if f in feat_set]
    # Anything in ``features`` not handled above ends up as a singleton.
    for f in features:
        if f not in used and f not in singletons:
            singletons.append(f)
    return fams, singletons


def write_classification_html_report(
    df: pd.DataFrame,
    features: list[str],
    feature_titles: dict[str, str],
    hover_data: list[str],
    out_html: str,
    *,
    trial_column: str = "run",
    report_title: str = "Classification report",
    pooled_cv: int = 5,
    pooled_cv_groups: np.ndarray | None = None,
    per_trial_cv: int = 5,
    include_pooled: bool = True,
    include_per_trial: bool = True,
) -> None:
    """
    Write a single standalone HTML file: pooled exploratory + classifiers, then
    each trial (``trial_column``) separately with the same plots and the same
    configured classifier (``classification.method``).

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
        ".section-intro{color:#333;margin:0.5rem 0 1rem;font-size:0.95rem;}",
        ".plot-blurb{font-size:0.88rem;color:#444;margin:0 0 0.35rem;padding:0.35rem 0.5rem;"
        "border-left:3px solid #467;font-style:normal;background:#fafafa;}",
        ".glossary-card{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;"
        "padding:0.9rem 1.1rem;margin:1.2rem 0;}",
        ".glossary-card h3{margin-top:0;}",
        ".glossary-card ul{margin:0.4rem 0 0 1.2rem;}",
        ".glossary-card li{margin:0.35rem 0;}",
        ".family-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0.75rem;"
        "margin-bottom:1rem;}",
        ".family-row .plot-wrap{margin-bottom:0;}",
        "</style></head><body>",
        f"<h1>{html_module.escape(report_title)}</h1>",
        _glossary_card_html(),
    ]

    plotly_cdn_used = False
    div_i = 0

    def _embed(fig: go.Figure) -> str:
        nonlocal plotly_cdn_used, div_i
        js = "cdn" if not plotly_cdn_used else False
        plotly_cdn_used = True
        html = _figure_to_embed_html(fig, f"fig_{div_i}", js)
        div_i += 1
        return html

    def emit(blurb_html: str, caption: str, fig: go.Figure) -> None:
        if blurb_html:
            parts.append(blurb_html)
        parts.append(f"<h4>{html_module.escape(caption)}</h4>")
        parts.append("<div class=\"plot-wrap\">")
        parts.append(_embed(fig))
        parts.append("</div>")

    def emit_row(blurb_html: str, caption: str, figs: list[go.Figure]) -> None:
        """Side-by-side row of independent px.box figures (CSS grid)."""
        if blurb_html:
            parts.append(blurb_html)
        parts.append(f"<h4>{html_module.escape(caption)}</h4>")
        parts.append("<div class=\"family-row\">")
        for fig in figs:
            parts.append("<div class=\"plot-wrap\">")
            parts.append(_embed(fig))
            parts.append("</div>")
        parts.append("</div>")

    families, singletons = _families_and_singletons(features)

    def _emit_genotype_for(d: pd.DataFrame, g_order: list[str]) -> None:
        for row_title, cols in families:
            figs = [
                make_figure_by_genotype(d, c, feature_titles, hover_data, genotype_order=g_order)
                for c in cols
            ]
            emit_row("", f"{row_title}, by genotype", figs)
        for feat in singletons:
            emit(
                "",
                f"{feature_titles[feat]}, by genotype",
                make_figure_by_genotype(d, feat, feature_titles, hover_data, genotype_order=g_order),
            )

    def _emit_wtmut_for(d: pd.DataFrame) -> None:
        for row_title, cols in families:
            figs = [
                make_figure_wt_vs_mutant(d, c, feature_titles, hover_data)
                for c in cols
            ]
            emit_row("", f"{row_title}, WT vs mutant", figs)
        for feat in singletons:
            emit(
                "",
                f"{feature_titles[feat]}, WT vs mutant",
                make_figure_wt_vs_mutant(d, feat, feature_titles, hover_data),
            )

    if include_pooled:
        parts.append("<section>")
        parts.append("<h2>Pooled cohort analysis</h2>")
        parts.append(
            f"<p class=\"section-intro\">This section analyzes all "
            f"<strong>{n_f}</strong> flies across <strong>{n_r}</strong> runs together. "
            f"Genotype ordering follows vial order within each run, then run order.</p>"
        )
        g_order_pool = genotype_category_order(df)
        parts.append("<h3>By genotype</h3>")
        _emit_genotype_for(df, g_order_pool)
        parts.append("<h3>WT versus mutants (pooled)</h3>")
        _emit_wtmut_for(df)
        parts.append("<h3>Genotype prediction accuracy</h3>")
        parts.append(_html_cv_blurb_pooled(pooled_cv_groups))
        for cap, fig in _collect_all_classifier_figures(
            df, cv=pooled_cv, groups=pooled_cv_groups
        ):
            pretty = _pretty_classifier_caption(cap)
            emit(
                f"<p class=\"plot-blurb\">Each bar is the fraction of held-out flies whose "
                f"genotype was predicted correctly in that fold. Higher is better; the dashed "
                f"line is the mean across folds.</p>",
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
        trial_label = pretty_run_label(trial)
        parts.append(f"<h2>Per-run analysis, {html_module.escape(trial_label)}</h2>")
        parts.append(
            f"<p class=\"section-intro\">"
            f"This section includes only flies from run "
            f"<code>{html_module.escape(trial_label)}</code> "
            f"(<strong>{len(sub)}</strong> flies). No other runs contribute to this block.</p>"
        )
        g_order = genotype_category_order(sub)
        parts.append("<h3>By genotype</h3>")
        _emit_genotype_for(sub, g_order)
        parts.append("<h3>WT versus mutants</h3>")
        _emit_wtmut_for(sub)
        parts.append("<h3>Genotype prediction accuracy (this run only)</h3>")
        parts.append(_html_cv_blurb_per_trial())
        for cap, fig in _collect_all_classifier_figures(
            sub, cv=per_trial_cv, groups=None
        ):
            pretty = _pretty_classifier_caption(cap)
            emit(
                f"<p class=\"plot-blurb\">Each bar = fraction correct in that fold for this "
                f"recording only. The dashed line is the mean across folds.</p>",
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
                report_title=f"{report_title} - {pretty_run_label(trial)}",
                pooled_cv=per_trial_cv,
                pooled_cv_groups=None,
                per_trial_cv=per_trial_cv,
                include_pooled=True,
                include_per_trial=False,
            )
            trial_links.append((pretty_run_label(trial), os.path.join("report_sections", fn).replace("\\", "/")))

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
                    "body{margin:0;font-family:system-ui,sans-serif;display:grid;"
                    "grid-template-columns:320px 1fr;min-height:100vh;transition:grid-template-columns 0.18s ease;}",
                    "body.nav-collapsed{grid-template-columns:0 1fr;}",
                    "aside{border-right:1px solid #ddd;padding:1rem;overflow:auto;background:#fafafa;}",
                    "body.nav-collapsed aside{display:none;}",
                    "main{position:relative;padding:0;margin:0;}",
                    "h1{font-size:1.05rem;margin:0 0 0.5rem;}",
                    "h2{font-size:0.95rem;margin:1rem 0 0.4rem;color:#333;}",
                    "ul{list-style:none;padding:0;margin:0;}",
                    "li{margin:0.2rem 0;}",
                    "a{text-decoration:none;color:#1a4c8b;}",
                    "a:hover{text-decoration:underline;}",
                    "iframe{border:0;width:100%;height:100vh;display:block;}",
                    ".hint{font-size:0.86rem;color:#555;line-height:1.35;}",
                    ".nav-toggle{position:fixed;top:0.6rem;left:0.6rem;z-index:10;"
                    "width:2rem;height:2rem;border:1px solid #ccc;border-radius:4px;"
                    "background:#fff;color:#333;font-size:1.1rem;line-height:1;cursor:pointer;"
                    "box-shadow:0 1px 3px rgba(0,0,0,0.08);}",
                    ".nav-toggle:hover{background:#f0f0f0;}",
                    "</style></head><body>",
                    "<button class=\"nav-toggle\" type=\"button\" "
                    "title=\"Toggle navigation panel\" "
                    "onclick=\"document.body.classList.toggle('nav-collapsed');\">&#9776;</button>",
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
