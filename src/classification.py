"""
src/classification.py

Genotype classification and statistical visualisation.

Classifiers: LDA, Logistic Regression, SVC (linear kernel)
Plots: cross-validation accuracy, feature importance, per-genotype boxes,
       WT-vs-mutant comparison with Cliff's delta.
"""

import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import mannwhitneyu, kruskal

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Genotype mapping
# ---------------------------------------------------------------------------

def map_vial_to_genotype(df_path: str) -> pd.DataFrame:
    """
    Parse genotypes from the filename and add a "genotype" column.

    Expected filename format: <date>_<something>_hTDP43_<GT1>-<GT2>-..._<rest>.csv
    """
    filename = os.path.basename(df_path)
    parts = filename.split("_")
    assert parts[2] == "hTDP43", "Unexpected filename format"

    genotypes = parts[3].split("-")
    vial_to_genotype = {f"vial{i + 1}": genotypes[i] for i in range(len(genotypes))}

    df = pd.read_csv(df_path)
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
):
    """Run k-fold CV, plot bar chart, save figure, return scores array."""
    scores = cross_val_score(model, X, y, cv=cv)

    fig = go.Figure()
    fig.add_bar(x=list(range(1, cv + 1)), y=scores)
    fig.add_hline(y=scores.mean(), line_dash="dash")
    fig.update_layout(
        title=f"{classification_mode.upper()} Cross-validation accuracy "
              f"(mean={scores.mean():.3f})",
        xaxis_title="CV fold",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
    )

    save_plotly_figure(fig, outdir, f"{model_name}_{classification_mode}")
    return scores


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
        values = np.abs(model.named_steps["clf"].coef_).ravel()
        xlabel = "|weight|"
    else:
        return None

    idx = np.argsort(values)
    fig = go.Figure()
    fig.add_bar(x=values[idx], y=[feature_names[i] for i in idx], orientation="h")
    fig.update_layout(
        title=f"{classification_mode.upper()} Feature importance ({model_name.upper()})",
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
        Number of CV folds.
    plot_importance : bool
        Whether to produce and save a feature-importance figure.
    """
    X, feature_names = prepare_xy(df)
    y = prepare_target(df, classification_mode)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", make_classifier(model_name)),
    ])

    run_cross_validation(pipeline, model_name, X, y, cv=cv,
                         classification_mode=classification_mode, outdir=outdir)

    if plot_importance:
        fig = plot_feature_importance(
            pipeline, X, y, feature_names, model_name, classification_mode
        )
        if fig is not None:
            save_plotly_figure(fig, outdir, f"{model_name}_{classification_mode}_importance")


# ---------------------------------------------------------------------------
# Statistical visualisation
# ---------------------------------------------------------------------------

def cliffs_delta(x, y):
    """Cliff's delta effect size between two 1-D arrays."""
    x = np.asarray(x)
    y = np.asarray(y)
    return (np.sum(x[:, None] > y) - np.sum(x[:, None] < y)) / (len(x) * len(y))


def save_plotly_figure(fig, outdir: str, name: str, show: bool = True):
    """Save a Plotly figure as HTML + PNG and optionally display it."""
    os.makedirs(outdir, exist_ok=True)
    fig.write_html(os.path.join(outdir, f"{name}.html"))
    fig.write_image(os.path.join(outdir, f"{name}.png"), width=1200, height=800, scale=2)
    if show:
        fig.show()


def plot_by_genotype(df, features, feature_titles, hover_data, outdir="report_figures"):
    """Box plots per feature, grouped by genotype with Kruskal-Wallis p-value."""
    for feat in features:
        groups = [g[feat].values for _, g in df.groupby("genotype")]
        _, p_kw = kruskal(*groups)

        fig = px.box(
            df, x="genotype", y=feat, color="genotype",
            points="all", hover_data=hover_data,
            title=f"{feature_titles[feat]} (Kruskal-Wallis p={p_kw:.3g})",
        )
        fig.update_traces(jitter=0.35, marker=dict(size=9, opacity=0.8))
        fig.update_layout(showlegend=False)
        save_plotly_figure(fig, outdir, f"{feat}_by_genotype")


def plot_wt_vs_mutant(df, features, feature_titles, hover_data, outdir="report_figures"):
    """Box plots per feature, WT vs all mutants, with MWU p-value and Cliff's delta."""
    df = df.copy()
    df["WT_vs_mutant"] = np.where(df["genotype"] == "WT", "WT", "Mutant")

    for feat in features:
        wt = df[df["WT_vs_mutant"] == "WT"][feat]
        mut = df[df["WT_vs_mutant"] == "Mutant"][feat]

        _, p_u = mannwhitneyu(wt, mut, alternative="two-sided")
        delta = cliffs_delta(wt.values, mut.values)

        fig = px.box(
            df, x="WT_vs_mutant", y=feat, color="WT_vs_mutant",
            points="all", hover_data=hover_data,
            title=(
                f"{feature_titles[feat]} — WT vs Mutant "
                f"(MWU p={p_u:.3g}, Cliff's delta={delta:.2f})"
            ),
        )
        fig.update_traces(jitter=0.35, marker=dict(size=9, opacity=0.8))
        fig.update_layout(showlegend=False)
        save_plotly_figure(fig, outdir, f"{feat}_WT_vs_mutant")
