"""
Tests for the config-driven classifier dispatch in src/classification.py.

Covers:
- ``make_classifier()`` honours ``classification.method`` and the matching
  per-method hyperparameter block (kernel, C, gamma, solver, etc.).
- ``plot_feature_importance`` skips gracefully on non-linear SVC kernels.
- ``run_cross_validation`` runs the optional grid search over (C, gamma) when
  ``classification.svc.grid_search.enabled`` is true and reports best params.

The classifier module caches its config at import time. We monkeypatch the
cached attribute rather than reloading.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src import classification as C
from utils import Config


def _make_blobs(n_per_class: int = 12, n_features: int = 6, seed: int = 0):
    """Three loose Gaussian blobs, well separated in feature space."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(3, n_features))
    Xs, ys = [], []
    for k, c in enumerate(centers):
        Xs.append(rng.normal(loc=c, scale=1.0, size=(n_per_class, n_features)))
        ys.extend([f"class_{k}"] * n_per_class)
    return np.vstack(Xs), np.array(ys)


def _set_classification_cfg(monkeypatch, method: str, **overrides) -> None:
    """Swap the cached classification config with a test fixture.

    ``overrides`` accepts nested dicts (e.g. ``svc={'kernel': 'linear'}``)
    and is applied after the defaults to keep call sites short.
    """
    cfg = {
        "method": method,
        "svc": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "grid_search": {
                "enabled": False,
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", 0.1, 1.0],
            },
        },
        "lda": {"solver": "svd"},
        "logistic": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    }
    for section, vals in overrides.items():
        cfg[section] = {**cfg[section], **vals} if isinstance(cfg.get(section), dict) else vals
    monkeypatch.setattr(C, "_CFG", Config(cfg))


# ---------------------------------------------------------------------------
# make_classifier dispatches on the configured method
# ---------------------------------------------------------------------------

def test_make_classifier_svc_rbf_default(monkeypatch):
    _set_classification_cfg(monkeypatch, "svc")
    clf = C.make_classifier()
    assert isinstance(clf, SVC)
    assert clf.kernel == "rbf"
    assert clf.C == 1.0
    assert clf.gamma == "scale"


def test_make_classifier_svc_linear_with_overrides(monkeypatch):
    _set_classification_cfg(monkeypatch, "svc", svc={"kernel": "linear", "C": 0.5, "gamma": "auto", "grid_search": {"enabled": False, "C": [1.0], "gamma": ["scale"]}})
    clf = C.make_classifier()
    assert isinstance(clf, SVC)
    assert clf.kernel == "linear"
    assert clf.C == 0.5
    assert clf.gamma == "auto"


def test_make_classifier_svc_numeric_gamma(monkeypatch):
    _set_classification_cfg(monkeypatch, "svc", svc={"kernel": "rbf", "C": 1.0, "gamma": 0.01, "grid_search": {"enabled": False, "C": [1.0], "gamma": ["scale"]}})
    clf = C.make_classifier()
    assert isinstance(clf.gamma, float)
    assert clf.gamma == pytest.approx(0.01)


def test_make_classifier_lda(monkeypatch):
    _set_classification_cfg(monkeypatch, "lda", lda={"solver": "lsqr"})
    clf = C.make_classifier()
    assert isinstance(clf, LinearDiscriminantAnalysis)
    assert clf.solver == "lsqr"


def test_make_classifier_logistic(monkeypatch):
    _set_classification_cfg(monkeypatch, "logistic", logistic={"C": 0.25, "max_iter": 500, "solver": "liblinear"})
    clf = C.make_classifier()
    assert isinstance(clf, LogisticRegression)
    assert clf.C == pytest.approx(0.25)
    assert clf.max_iter == 500
    assert clf.solver == "liblinear"


def test_resolve_method_rejects_bad_name(monkeypatch):
    _set_classification_cfg(monkeypatch, "svc")
    with pytest.raises(ValueError, match="must be svc, lda, or logistic"):
        C.make_classifier("random_forest")


def test_make_classifier_explicit_override_beats_config(monkeypatch):
    _set_classification_cfg(monkeypatch, "lda")
    # Even with config.method=lda, an explicit override should win.
    clf = C.make_classifier("logistic")
    assert isinstance(clf, LogisticRegression)


# ---------------------------------------------------------------------------
# plot_feature_importance: linear backends produce a figure, non-linear skip
# ---------------------------------------------------------------------------

def test_plot_feature_importance_skips_rbf(monkeypatch, capsys):
    _set_classification_cfg(monkeypatch, "svc")  # rbf default
    pipeline = C.Pipeline([
        ("scaler", C.StandardScaler()),
        ("clf", C.make_classifier()),
    ])
    X, y = _make_blobs()
    fig = C.plot_feature_importance(pipeline, X, y, [f"f{i}" for i in range(X.shape[1])], "svc", "multiclass")
    assert fig is None
    out = capsys.readouterr().out
    assert "skipped" in out.lower() or "non-linear" in out.lower()


def test_plot_feature_importance_runs_for_linear_svc(monkeypatch):
    _set_classification_cfg(monkeypatch, "svc", svc={"kernel": "linear", "C": 1.0, "gamma": "scale", "grid_search": {"enabled": False, "C": [1.0], "gamma": ["scale"]}})
    pipeline = C.Pipeline([
        ("scaler", C.StandardScaler()),
        ("clf", C.make_classifier()),
    ])
    X, y = _make_blobs()
    fig = C.plot_feature_importance(pipeline, X, y, [f"f{i}" for i in range(X.shape[1])], "svc", "multiclass")
    assert fig is not None
    assert "SVC" in fig.layout.title.text


def test_plot_feature_importance_runs_for_logistic_and_lda(monkeypatch):
    for method, cls in (("logistic", LogisticRegression), ("lda", LinearDiscriminantAnalysis)):
        _set_classification_cfg(monkeypatch, method)
        pipeline = C.Pipeline([
            ("scaler", C.StandardScaler()),
            ("clf", C.make_classifier()),
        ])
        X, y = _make_blobs()
        fig = C.plot_feature_importance(pipeline, X, y, [f"f{i}" for i in range(X.shape[1])], method, "multiclass")
        assert fig is not None, f"feature importance for {method} should produce a figure"


# ---------------------------------------------------------------------------
# Cross-validation runs each backend end-to-end without crashing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["svc", "lda", "logistic"])
def test_run_cross_validation_each_backend(monkeypatch, method):
    _set_classification_cfg(monkeypatch, method)
    X, y = _make_blobs(n_per_class=8)
    pipeline = C.Pipeline([
        ("scaler", C.StandardScaler()),
        ("clf", C.make_classifier()),
    ])
    scores, fig, info = C.run_cross_validation(
        pipeline, method, X, y,
        classification_mode="multiclass", cv=3, save=False,
    )
    assert len(scores) == 3
    assert info["scheme"] == "stratified"
    assert 0.0 <= info["mean_accuracy"] <= 1.0
    assert fig.layout.title.text  # non-empty title


# ---------------------------------------------------------------------------
# SVC grid search over (C, gamma)
# ---------------------------------------------------------------------------

def test_svc_grid_search_reports_best_params(monkeypatch, capsys):
    _set_classification_cfg(
        monkeypatch, "svc",
        svc={
            "kernel": "rbf", "C": 1.0, "gamma": "scale",
            "grid_search": {
                "enabled": True,
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", 0.1, 1.0],
            },
        },
    )
    X, y = _make_blobs(n_per_class=8)
    pipeline = C.Pipeline([
        ("scaler", C.StandardScaler()),
        ("clf", C.make_classifier()),
    ])
    scores, fig, info = C.run_cross_validation(
        pipeline, "svc", X, y,
        classification_mode="multiclass", cv=3, save=False,
    )
    assert "best_params" in info
    assert "best_score" in info
    assert set(info["best_params"]).issuperset({"C", "gamma"})
    assert info["best_score"] >= info["mean_accuracy"] - 1e-9
    assert any(r["params"] == info["best_params"] for r in info["cv_results_top5"])

    out = capsys.readouterr().out
    assert "grid search" in out.lower()
    assert "best" in out.lower()
    assert "grid search best" in fig.layout.title.text.lower()


def test_svc_grid_search_skipped_when_disabled(monkeypatch):
    _set_classification_cfg(monkeypatch, "svc")  # grid_search.enabled=False
    X, y = _make_blobs(n_per_class=8)
    pipeline = C.Pipeline([
        ("scaler", C.StandardScaler()),
        ("clf", C.make_classifier()),
    ])
    _, _, info = C.run_cross_validation(
        pipeline, "svc", X, y,
        classification_mode="multiclass", cv=3, save=False,
    )
    assert "best_params" not in info
    assert "best_score" not in info


def test_svc_grid_search_not_triggered_for_lda(monkeypatch):
    """Even with svc.grid_search.enabled=True, LDA runs do not grid search."""
    _set_classification_cfg(
        monkeypatch, "lda",
        svc={
            "kernel": "rbf", "C": 1.0, "gamma": "scale",
            "grid_search": {"enabled": True, "C": [1.0], "gamma": ["scale"]},
        },
    )
    X, y = _make_blobs(n_per_class=8)
    pipeline = C.Pipeline([
        ("scaler", C.StandardScaler()),
        ("clf", C.make_classifier()),
    ])
    _, _, info = C.run_cross_validation(
        pipeline, "lda", X, y,
        classification_mode="multiclass", cv=3, save=False,
    )
    assert "best_params" not in info
