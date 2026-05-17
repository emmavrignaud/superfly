"""
Smoke + correctness tests for the configurable embedding dispatcher in
src/latent_space.py.

The dispatcher reads ``latent_space.embedding.method`` from config.yaml at
import time. To exercise all three backends from a single test session we
monkeypatch the cached config rather than reloading the module.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import latent_space as L
from utils import Config


def _make_X(n_samples: int = 30, n_features: int = 12, seed: int = 0) -> np.ndarray:
    """Three loose Gaussian blobs in n_features dimensions."""
    rng = np.random.default_rng(seed)
    per_blob = n_samples // 3
    centers = rng.normal(scale=5.0, size=(3, n_features))
    blobs = [rng.normal(loc=c, scale=1.0, size=(per_blob, n_features)) for c in centers]
    X = np.vstack(blobs)
    return X.astype(float)


def _set_embedding(monkeypatch, method: str, use_pca: bool = True) -> None:
    """Swap the cached latent_space config to use the requested method.

    Mirrors the production layout where ``use_pca`` and
    ``pca_explained_variance`` live under ``latent_space.embedding`` (not at
    the top level of the latent_space block).
    """
    new = Config(
        {
            "representation_method": "baseline",
            "seed": 42,
            "embedding": {
                "method": method,
                "use_pca": use_pca,
                "pca_explained_variance": 0.95,
                "umap": {"n_neighbors": 5, "min_dist": 0.1, "metric": "euclidean", "n_components": 3},
                "tsne": {"perplexity": 5, "learning_rate": "auto", "init": "pca", "metric": "euclidean", "n_components": 3},
                "pca":  {"n_components": 3},
            },
            "autoencoder": {
                "latent_dim": 4, "hidden_dims": [8], "epochs": 1, "batch_size": 8,
                "learning_rate": 0.001, "weight_decay": 0.0, "val_fraction": 0.2,
                "patience": 1, "verbose": False,
            },
        }
    )
    monkeypatch.setattr(L, "_CFG", new)


@pytest.mark.parametrize("method", ["pca", "tsne", "umap"])
def test_embed_returns_correct_shape(monkeypatch, method):
    _set_embedding(monkeypatch, method)
    X = _make_X()
    Y, info = L.embed(X)

    if method == "umap":
        try:
            import umap  # noqa: F401
        except Exception:
            assert info["method"] == "pca", "missing umap-learn should trigger PCA fallback"
            return

    assert info["method"] == method
    assert Y.shape[0] == X.shape[0]
    assert Y.shape[1] == 3


@pytest.mark.parametrize("method", ["pca", "tsne"])
def test_embed_info_reports_tuning_values(monkeypatch, method):
    _set_embedding(monkeypatch, method)
    X = _make_X()
    _, info = L.embed(X)
    assert "n_components" in info
    if method == "tsne":
        assert "perplexity" in info
        assert info["init"] == "pca"
    if method == "pca":
        assert "explained_variance" in info


def test_embed_unknown_method_raises(monkeypatch):
    bad = Config(dict(L._CFG))
    bad.embedding = Config(dict(L._CFG.embedding))
    bad.embedding["method"] = "not_a_real_method"
    monkeypatch.setattr(L, "_CFG", bad)
    with pytest.raises(ValueError, match="must be 'umap', 'tsne', or 'pca'"):
        L.embed(_make_X())


def test_tsne_perplexity_clamped_for_tiny_n(monkeypatch, capsys):
    _set_embedding(monkeypatch, "tsne")
    L._CFG.embedding.tsne["perplexity"] = 50.0  # way larger than N
    X = _make_X(n_samples=9, n_features=4)
    Y, info = L.embed(X)
    assert Y.shape == (X.shape[0], 3)
    assert info["perplexity"] < 50.0
    out = capsys.readouterr().out
    assert "clamping" in out


def test_make_embedding_figure_includes_method_in_title(monkeypatch):
    _set_embedding(monkeypatch, "pca")
    import pandas as pd

    X = _make_X(n_samples=9)
    Y, _ = L.embed(X)
    meta = pd.DataFrame(
        {
            "ordered_id": list(range(9)),
            "genotype": ["wt", "wt", "wt", "het", "het", "het", "homo", "homo", "homo"],
            "run": ["r1"] * 9,
        }
    )
    fig = L.make_embedding_figure(Y, meta, title="test", method="pca")
    assert "test" in fig.layout.title.text
    assert "pca" in fig.layout.title.text


def test_maybe_apply_pca_reads_embedding_block_on(monkeypatch):
    _set_embedding(monkeypatch, "pca", use_pca=True)
    X = _make_X(n_samples=30, n_features=12, seed=0)
    Xt, info = L.maybe_apply_pca(X)
    assert info["use_pca"] is True
    assert "explained_variance" in info
    # Variance is in (0, 1] when the requested target is reached.
    assert 0.0 < info["explained_variance"] <= 1.0
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] <= X.shape[1]


def test_maybe_apply_pca_reads_embedding_block_off(monkeypatch):
    _set_embedding(monkeypatch, "pca", use_pca=False)
    X = _make_X(n_samples=20, n_features=8, seed=1)
    Xt, info = L.maybe_apply_pca(X)
    assert info["use_pca"] is False
    # X passes through unchanged when PCA is off.
    assert np.array_equal(Xt, X)
    assert info["explained_variance"] == 1.0
