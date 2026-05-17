"""
src/latent_space.py

Latent-space analysis helpers for ordered_tracks-based genotype studies.
"""

from __future__ import annotations

import json
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from src.classification import genotype_category_order
from src.features import extract_behavioral_features, aggregate_per_fly_features
from src.plot_colors import genotype_color_map_for_dataframe
from src.representation_learning import fit_autoencoder_latent
from utils import load_config as _load_config  # noqa: E402


_CFG = _load_config(_REPO_ROOT / "config.yaml").latent_space


def maybe_apply_autoencoder(X: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
    method = str(_CFG.representation_method).strip().lower()
    if method != "autoencoder":
        return X, {"used_autoencoder": False, "representation_method": method}
    ae = _CFG.autoencoder
    Z, info = fit_autoencoder_latent(
        X=X,
        latent_dim=int(ae.latent_dim),
        hidden_dims=tuple(ae.hidden_dims),
        epochs=int(ae.epochs),
        batch_size=int(ae.batch_size),
        learning_rate=float(ae.learning_rate),
        weight_decay=float(ae.weight_decay),
        val_fraction=float(ae.val_fraction),
        patience=int(ae.patience),
        seed=int(_CFG.seed),
        verbose=bool(ae.verbose),
    )
    out = {"representation_method": method}
    out.update(info)
    print(f"[latent_space] {name} representation -> autoencoder latent shape={Z.shape}")
    return Z, out


def load_run_roi_bounds(run_dir: str | Path) -> dict[str, tuple[int, int, int, int]]:
    """
    Return vial ROI bounds for one run, hard-failing if unavailable.
    """
    run_dir = Path(run_dir)
    params_path = run_dir / "run_params.json"
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        roi = params.get("roi", {})
        if roi:
            return {k: tuple(map(int, v)) for k, v in roi.items()}

    roi_json = run_dir / "vial_rois.json"
    if roi_json.exists():
        with open(roi_json, "r", encoding="utf-8") as f:
            roi = json.load(f)
        if roi:
            return {k: tuple(map(int, v)) for k, v in roi.items()}

    raise FileNotFoundError(
        f"Missing ROI metadata for run '{run_dir}'. Expected 'run_params.json[\"roi\"]' "
        f"or 'vial_rois.json'."
    )


def add_roi_relative_x(
    df: pd.DataFrame,
    run_to_roi: dict[str, dict[str, tuple[int, int, int, int]]],
    run_col: str = "run",
    vial_col: str = "vial_id",
) -> pd.DataFrame:
    """
    Add x_rel in [0, 1] using vial ROI x-bounds (x0/x1) per run.
    """
    if run_col not in df.columns:
        raise ValueError(f"Column '{run_col}' is required for ROI-relative x normalization.")
    if vial_col not in df.columns:
        raise ValueError(f"Column '{vial_col}' is required for ROI-relative x normalization.")

    out = df.copy()
    x_rel = np.full(len(out), np.nan, dtype=float)

    for run_name, idx in out.groupby(run_col).groups.items():
        if run_name not in run_to_roi:
            raise KeyError(f"No ROI entry for run '{run_name}'.")
        rois = run_to_roi[run_name]
        sub = out.loc[idx]
        for vial_name, idx_vial in sub.groupby(vial_col).groups.items():
            if vial_name not in rois:
                raise KeyError(f"Run '{run_name}' missing ROI for vial '{vial_name}'.")
            x0, _, x1, _ = rois[vial_name]
            denom = float(x1 - x0)
            if denom <= 0:
                raise ValueError(
                    f"Invalid ROI bounds for run '{run_name}', vial '{vial_name}': "
                    f"x0={x0}, x1={x1}"
                )
            x_rel[idx_vial] = ((out.loc[idx_vial, "x"].astype(float) - x0) / denom).clip(0.0, 1.0)

    out["x_rel"] = x_rel
    if out["x_rel"].isna().any():
        raise ValueError("x_rel contains NaN after ROI-relative normalization.")
    return out


def _fly_ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["ordered_id"].astype(str).unique().tolist())


def build_xy_plus_features_matrix(
    df_frames: pd.DataFrame,
    df_fly: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """
    Build X1: flattened fixed-length (x_rel, y) trajectory + aggregate features.
    """
    fly_ids = _fly_ids(df_frames)
    lengths = (
        df_frames.groupby("ordered_id")["frame"]
        .count()
        .reindex(fly_ids)
        .fillna(0)
        .astype(int)
        .values
    )
    T = int(np.median(lengths[lengths > 0])) if np.any(lengths > 0) else 1
    T = max(T, 1)

    traj_blocks: list[np.ndarray] = []
    for fid in fly_ids:
        g = df_frames[df_frames["ordered_id"].astype(str) == fid].sort_values("frame")
        xy = g[["x_rel", "y"]].to_numpy(dtype=float)
        if len(xy) >= T:
            xy = xy[:T]
        else:
            pad = np.zeros((T - len(xy), 2), dtype=float)
            xy = np.vstack([xy, pad])
        traj_blocks.append(xy.reshape(-1))

    traj_matrix = np.vstack(traj_blocks) if traj_blocks else np.zeros((0, 2 * T))

    cols_drop = {"ordered_id", "vial_id", "genotype", "run"}
    numeric_cols = [c for c in df_fly.columns if c not in cols_drop and np.issubdtype(df_fly[c].dtype, np.number)]
    meta = (
        df_fly[["ordered_id", "genotype", "run"]].copy()
        .assign(ordered_id=lambda d: d["ordered_id"].astype(str))
        .set_index("ordered_id")
        .reindex(fly_ids)
    )
    feats = (
        df_fly.set_index(df_fly["ordered_id"].astype(str))[numeric_cols]
        .reindex(fly_ids)
        .to_numpy(dtype=float)
    )

    X = np.hstack([traj_matrix, feats])
    X = StandardScaler().fit_transform(X)
    meta = meta.reset_index().rename(columns={"index": "ordered_id"})
    return X, meta, T


def _fd_edges(values: np.ndarray, min_bins: int = 8, max_bins: int = 60) -> np.ndarray:
    v = values[np.isfinite(values)]
    if len(v) < 2:
        return np.array([0.0, 1.0], dtype=float)
    q25, q75 = np.percentile(v, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return np.linspace(v.min(), v.max() + 1e-9, min_bins + 1)
    bw = 2.0 * iqr / (len(v) ** (1.0 / 3.0))
    if bw <= 0:
        return np.linspace(v.min(), v.max() + 1e-9, min_bins + 1)
    n_bins = int(np.ceil((v.max() - v.min()) / bw))
    n_bins = max(min_bins, min(max_bins, n_bins))
    return np.linspace(v.min(), v.max() + 1e-9, n_bins + 1)


def build_hist_kinematics_matrix(
    df_frames: pd.DataFrame,
    signals: tuple[str, ...] = ("velocity", "turning_angle", "angular_velocity", "acceleration"),
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    """
    Build X2: concatenated per-fly normalized histograms for selected kinematic signals.
    """
    fly_ids = _fly_ids(df_frames)
    edges = {sig: _fd_edges(df_frames[sig].to_numpy(dtype=float)) for sig in signals}

    blocks: list[np.ndarray] = []
    for fid in fly_ids:
        g = df_frames[df_frames["ordered_id"].astype(str) == fid]
        vec: list[np.ndarray] = []
        for sig in signals:
            h, _ = np.histogram(g[sig].to_numpy(dtype=float), bins=edges[sig])
            h = h.astype(float)
            h = h / h.sum() if h.sum() > 0 else h
            vec.append(h)
        blocks.append(np.concatenate(vec))

    X = np.vstack(blocks) if blocks else np.zeros((0, 0))
    X = StandardScaler().fit_transform(X) if X.size else X
    meta = (
        df_frames.drop_duplicates("ordered_id")[["ordered_id", "genotype", "run"]]
        .assign(ordered_id=lambda d: d["ordered_id"].astype(str))
        .set_index("ordered_id")
        .reindex(fly_ids)
        .reset_index()
        .rename(columns={"index": "ordered_id"})
    )
    return X, meta, edges


def maybe_apply_pca(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Optional pre-embedding PCA whitening on the high-dim feature matrix.

    Toggle and target variance live under ``latent_space.embedding.{use_pca,
    pca_explained_variance}`` in config.yaml. The returned info dict always
    carries ``explained_variance`` (1.0 when PCA is off, since X passes
    through unchanged) so downstream reports can persist the number.
    """
    emb = _CFG.embedding
    if not bool(emb.use_pca) or X.size == 0:
        return X, {"use_pca": False, "n_components": X.shape[1] if X.ndim == 2 else 0, "explained_variance": 1.0}
    pca = PCA(n_components=float(emb.pca_explained_variance), svd_solver="full", random_state=int(_CFG.seed))
    Xt = pca.fit_transform(X)
    return Xt, {
        "use_pca": True,
        "target_variance": float(emb.pca_explained_variance),
        "n_components": int(pca.n_components_),
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def _embed_umap(X: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """UMAP backend. Falls back to PCA if umap-learn is not installed."""
    u = _CFG.embedding.umap
    n_components = int(u.n_components)
    try:
        import umap as _umap
    except Exception:
        print("[latent_space] umap-learn not found; falling back to PCA embedding.")
        return _embed_pca(X, seed, n_components_override=n_components, fallback_reason="umap_unavailable")

    model = _umap.UMAP(
        n_neighbors=int(u.n_neighbors),
        min_dist=float(u.min_dist),
        metric=str(u.metric),
        n_components=n_components,
        random_state=seed,
    )
    Y = model.fit_transform(X)
    return np.asarray(Y), {
        "method": "umap",
        "n_components": n_components,
        "n_neighbors": int(u.n_neighbors),
        "min_dist": float(u.min_dist),
        "metric": str(u.metric),
    }


def _embed_tsne(X: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """t-SNE backend. Clamps perplexity to a legal value for tiny N."""
    t = _CFG.embedding.tsne
    n_components = int(t.n_components)

    N = X.shape[0]
    requested = float(t.perplexity)
    perplexity = requested
    if N <= 1:
        return X[:, :n_components] if X.shape[1] >= n_components else X, {
            "method": "tsne",
            "n_components": n_components,
            "skipped": "n_samples<=1",
        }
    if perplexity >= N:
        perplexity = max(2.0, float(N - 1))
        print(f"[latent_space] tsne.perplexity {requested} >= n_samples={N}; clamping to {perplexity}.")

    lr_raw = t.learning_rate
    if isinstance(lr_raw, str) and lr_raw.strip().lower() == "auto":
        learning_rate = "auto"
    else:
        learning_rate = float(lr_raw)

    model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=str(t.init),
        metric=str(t.metric),
        random_state=seed,
    )
    Y = model.fit_transform(X)
    return np.asarray(Y), {
        "method": "tsne",
        "n_components": n_components,
        "perplexity": perplexity,
        "learning_rate": str(learning_rate),
        "init": str(t.init),
        "metric": str(t.metric),
    }


def _embed_pca(
    X: np.ndarray,
    seed: int,
    n_components_override: int | None = None,
    fallback_reason: str | None = None,
) -> tuple[np.ndarray, dict]:
    """PCA backend - parameter-free baseline, also used as UMAP's fallback."""
    n_components = int(n_components_override if n_components_override is not None else _CFG.embedding.pca.n_components)
    if X.size == 0:
        return X, {"method": "pca", "n_components": n_components, "skipped": "empty"}
    # PCA needs n_components <= min(n_samples, n_features). Cap it.
    nc = min(n_components, X.shape[0], X.shape[1])
    model = PCA(n_components=nc, random_state=seed)
    Y = model.fit_transform(X)
    info = {
        "method": "pca",
        "n_components": int(nc),
        "explained_variance": float(np.sum(model.explained_variance_ratio_)),
    }
    if fallback_reason is not None:
        info["fallback_from"] = fallback_reason
    return np.asarray(Y), info


def embed(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Dispatch to the embedding method named in config.yaml.

    Returns (Y, info). Y has shape (n_samples, n_components) and info is a
    dict that always contains "method" and "n_components" plus
    method-specific tuning values.

    This is visualisation only; classifiers (LDA / Logistic / SVC) operate on
    the pre-embedding feature matrix, not on Y.
    """
    method = str(_CFG.embedding.method).strip().lower()
    seed = int(_CFG.seed)
    if method == "umap":
        return _embed_umap(X, seed)
    if method == "tsne":
        return _embed_tsne(X, seed)
    if method == "pca":
        return _embed_pca(X, seed)
    raise ValueError(
        f"latent_space.embedding.method must be 'umap', 'tsne', or 'pca'; got {method!r}"
    )


def make_embedding_figure(
    emb: np.ndarray,
    meta: pd.DataFrame,
    title: str,
    method: str | None = None,
) -> go.Figure:
    """3D scatter of the low-dim embedding, coloured by genotype.

    If the embedding has fewer than 3 components it is zero-padded so the
    scatter still renders (the padded axes carry no information).
    """
    if emb.shape[1] < 3:
        emb = np.pad(emb, ((0, 0), (0, 3 - emb.shape[1])), mode="constant")
    d = pd.DataFrame(
        {
            "component_1": emb[:, 0],
            "component_2": emb[:, 1],
            "component_3": emb[:, 2],
            "genotype": meta["genotype"].values,
            "run": meta["run"].values,
            "ordered_id": meta["ordered_id"].values,
        }
    )
    full_title = f"{title} ({method})" if method else title
    pref = genotype_category_order(meta)
    color_discrete_map = genotype_color_map_for_dataframe(d, pref)
    cat_g = [g for g in pref if g in set(d["genotype"].astype(str))]
    for g in sorted(set(d["genotype"].astype(str)) - set(cat_g)):
        cat_g.append(g)

    fig = px.scatter_3d(
        d,
        x="component_1",
        y="component_2",
        z="component_3",
        color="genotype",
        hover_data=["ordered_id", "run"],
        category_orders={"genotype": cat_g},
        color_discrete_map=color_discrete_map,
        title=full_title,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.85))
    fig.update_layout(
        height=620,
        scene=dict(
            xaxis_title=f"{method or 'Embedding'} component 1",
            yaxis_title=f"{method or 'Embedding'} component 2",
            zaxis_title=f"{method or 'Embedding'} component 3",
        ),
    )
    return fig


def permanova_test(
    X: np.ndarray,
    groups: np.ndarray,
    permutations: int = 999,
    strata: np.ndarray | None = None,
) -> dict:
    """
    PERMANOVA with fallback permutation ANOVA on distance matrices.
    """
    D = squareform(pdist(X, metric="euclidean"))
    y = pd.Series(groups).astype(str).values
    grand = D[np.triu_indices_from(D, k=1)].mean()
    uniq = np.unique(y)

    def _pseudo_f(y_labels: np.ndarray) -> tuple[float, float, float]:
        ss_between = 0.0
        ss_within = 0.0
        for u in uniq:
            idx = np.where(y_labels == u)[0]
            if len(idx) < 2:
                continue
            sub = D[np.ix_(idx, idx)]
            m = sub[np.triu_indices_from(sub, k=1)].mean() if len(idx) > 1 else 0.0
            ss_between += len(idx) * (m - grand) ** 2
            ss_within += np.sum((sub - m) ** 2)
        f = ss_between / max(ss_within, 1e-12)
        r2 = ss_between / max(ss_between + ss_within, 1e-12)
        return f, ss_between, r2

    f_obs, _, r2_obs = _pseudo_f(y)
    rng = np.random.default_rng(42)
    count = 0

    if strata is None:
        for _ in range(permutations):
            yp = rng.permutation(y)
            f_perm, _, _ = _pseudo_f(yp)
            if f_perm >= f_obs:
                count += 1
        method = "permutation_global"
    else:
        strata = pd.Series(strata).astype(str).values
        strata_levels = np.unique(strata)
        for _ in range(permutations):
            yp = y.copy()
            for s in strata_levels:
                idx = np.where(strata == s)[0]
                if len(idx) > 1:
                    yp[idx] = rng.permutation(yp[idx])
            f_perm, _, _ = _pseudo_f(yp)
            if f_perm >= f_obs:
                count += 1
        method = "permutation_stratified_by_run"

    p = (count + 1) / (permutations + 1)
    return {
        "method": method,
        "pseudo_f": float(f_obs),
        "p_value": float(p),
        "r2": float(r2_obs),
        "permutations": int(permutations),
    }


def run_latent_space_analysis(df_raw: pd.DataFrame) -> dict:
    """
    End-to-end latent analysis from raw compact tracks + metadata columns.
    Requires: ordered_id, frame, x, y, vial_id, genotype, run.
    """
    required = {"ordered_id", "frame", "x", "y", "vial_id", "genotype", "run"}
    miss = required - set(df_raw.columns)
    if miss:
        raise ValueError(f"Missing required columns for latent analysis: {sorted(miss)}")

    run_to_roi = {}
    for run_name, sub in df_raw.groupby("run"):
        run_dir = sub["run_dir"].iloc[0] if "run_dir" in sub.columns else None
        if not run_dir:
            raise ValueError("run_dir column is required for ROI lookup in latent analysis.")
        run_to_roi[str(run_name)] = load_run_roi_bounds(run_dir)

    # Guardrail: pooled runs from one cohort/day should keep consistent vial->genotype mapping.
    vial_maps: dict[str, dict[str, str]] = {}
    for run_name, sub in df_raw.groupby("run"):
        m = (
            sub[["vial_id", "genotype"]]
            .dropna()
            .drop_duplicates()
            .groupby("vial_id")["genotype"]
            .first()
            .to_dict()
        )
        vial_maps[str(run_name)] = {str(k): str(v) for k, v in m.items()}
    if len(vial_maps) > 1:
        ref_run = sorted(vial_maps.keys())[0]
        ref = vial_maps[ref_run]
        for run_name, mapping in vial_maps.items():
            if mapping != ref:
                raise ValueError(
                    f"Inconsistent vial->genotype mapping across runs: '{ref_run}' != '{run_name}'. "
                    "Please pool only runs with consistent layout."
                )

    df_norm = add_roi_relative_x(df_raw, run_to_roi)

    df_feat = extract_behavioral_features(df_norm)
    df_fly = aggregate_per_fly_features(df_feat, pause_threshold=1.0)
    meta = df_norm.drop_duplicates("ordered_id").set_index("ordered_id")[["genotype", "run"]]
    df_fly["genotype"] = df_fly["ordered_id"].map(meta["genotype"])
    df_fly["run"] = df_fly["ordered_id"].map(meta["run"])

    X1, m1, T = build_xy_plus_features_matrix(df_feat, df_fly)
    X1_eff, pca1 = maybe_apply_pca(X1)
    X1_repr, repr1 = maybe_apply_autoencoder(X1_eff, "analysis1")
    emb1, embed1_info = embed(X1_repr)
    fig1 = make_embedding_figure(
        emb1,
        m1,
        "Fly trajectory and kinematic embedding",
        method=embed1_info.get("method"),
    )

    X2, m2, edges = build_hist_kinematics_matrix(df_feat)
    X2_eff, pca2 = maybe_apply_pca(X2)
    X2_repr, repr2 = maybe_apply_autoencoder(X2_eff, "analysis2")
    emb2, embed2_info = embed(X2_repr)
    fig2 = make_embedding_figure(
        emb2,
        m2,
        "Fly kinematic distribution embedding",
        method=embed2_info.get("method"),
    )

    perma1 = {
        "run_aware": permanova_test(X1_repr, m1["genotype"].values, strata=m1["run"].values),
        "pooled": permanova_test(X1_repr, m1["genotype"].values, strata=None),
    }
    perma2 = {
        "run_aware": permanova_test(X2_repr, m2["genotype"].values, strata=m2["run"].values),
        "pooled": permanova_test(X2_repr, m2["genotype"].values, strata=None),
    }

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(clf, X2_repr, m2["genotype"].values, cv=cv, scoring="accuracy")
    clf.fit(X2_repr, m2["genotype"].values)
    imp = clf.feature_importances_
    top = np.argsort(imp)[-20:][::-1]
    fig_imp = go.Figure(
        go.Bar(
            x=imp[top][::-1],
            y=[f"bin_{i}" for i in top][::-1],
            orientation="h",
        )
    )
    fig_imp.update_layout(title=f"RF histogram-bin importance (cv acc={rf_scores.mean():.3f})", height=560)

    return {
        "df_frames": df_feat,
        "df_fly": df_fly,
        "analysis1": {
            "X": X1_repr,
            "meta": m1,
            "embedding_fig": fig1,
            "umap_fig": fig1,                # deprecated alias; prefer embedding_fig
            "embedding": embed1_info,
            "permanova": perma1,
            "pca": pca1,
            "representation": repr1,
            "T": T,
        },
        "analysis2": {
            "X": X2_repr,
            "meta": m2,
            "embedding_fig": fig2,
            "umap_fig": fig2,                # deprecated alias; prefer embedding_fig
            "embedding": embed2_info,
            "permanova": perma2,
            "pca": pca2,
            "representation": repr2,
            "edges": edges,
        },
        "rf_importance_fig": fig_imp,
    }
