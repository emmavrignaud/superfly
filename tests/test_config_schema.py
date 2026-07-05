"""
Schema validation for config.yaml.

Locks down the contract the active pipeline expects so a typo or accidental
deletion is caught at test time, not three hours into a tracking run.
"""

from __future__ import annotations

import pytest

REQUIRED_SECTIONS = [
    "roboflow", "video", "calibration", "features", "tracker", "watershed",
    "pipeline", "preprocessing", "visualization", "roi", "latent_space",
    "classification",
]

REQUIRED_KEYS = {
    "roboflow": ["model_id", "inference_api_url"],
    "video": ["fallback_fps"],
    "calibration": ["px_per_cm", "length_unit", "time_unit"],
    "features": ["kinematic_three_families"],
    "tracker": [
        "detection_confidence_rfdetr", "confidence", "lost_track_buffer",
        "minimum_matching_threshold", "minimum_consecutive_frames",
        "asso_func", "brownian_pos_noise",
        "aspect_weight", "behavioral_weight",
        "jump_factor", "jump_iou_threshold", "jump_inertia",
    ],
    "watershed": [
        "enabled", "area_outlier_k", "max_flies_per_blob",
        "min_distance_factor", "min_region_area_fraction",
        "debug", "debug_max_images",
    ],
    "pipeline": ["expected_per_vial", "skip_tracked"],
    "preprocessing": ["bg_gain", "bg_white_level", "bg_percentile", "bg_sample_stride"],
    "visualization": ["enabled", "overlay_source", "fps_out", "show_ids"],
    "roi": ["use_saved_roi", "n_vials", "gap_ratio"],
    "latent_space": ["representation_method", "seed", "embedding"],
    "classification": ["method", "svc", "lda", "logistic"],
}


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_required_section_present(config_yaml: dict, section: str) -> None:
    assert section in config_yaml, f"config.yaml is missing the {section!r} section"
    assert isinstance(config_yaml[section], dict), f"config.yaml:{section} must be a mapping"


@pytest.mark.parametrize(
    "section,key",
    [(s, k) for s, keys in REQUIRED_KEYS.items() for k in keys],
)
def test_required_key_present(config_yaml: dict, section: str, key: str) -> None:
    assert key in config_yaml.get(section, {}), (
        f"config.yaml:{section}.{key} is required by the active pipeline"
    )


def test_video_fallback_fps_is_positive(config_yaml: dict) -> None:
    fps = config_yaml["video"]["fallback_fps"]
    assert isinstance(fps, (int, float)) and fps > 0, "video.fallback_fps must be a positive number"


def test_pipeline_expected_per_vial_is_positive_int(config_yaml: dict) -> None:
    n = config_yaml["pipeline"]["expected_per_vial"]
    assert isinstance(n, int) and n > 0, "pipeline.expected_per_vial must be a positive integer"


def test_roi_n_vials_is_positive_int(config_yaml: dict) -> None:
    n = config_yaml["roi"]["n_vials"]
    assert isinstance(n, int) and n > 0, "roi.n_vials must be a positive integer"


def test_roi_gap_ratio_in_unit_range(config_yaml: dict) -> None:
    r = config_yaml["roi"]["gap_ratio"]
    assert isinstance(r, (int, float)) and 0.0 <= float(r) <= 1.0, (
        "roi.gap_ratio must be a number in [0, 1]"
    )


def test_calibration_px_per_cm_is_positive(config_yaml: dict) -> None:
    s = config_yaml["calibration"]["px_per_cm"]
    assert isinstance(s, (int, float)) and s > 0, "calibration.px_per_cm must be a positive number"


def test_calibration_length_unit_is_valid(config_yaml: dict) -> None:
    u = config_yaml["calibration"]["length_unit"]
    assert u in ("cm", "px"), f"calibration.length_unit must be 'cm' or 'px', got {u!r}"


def test_calibration_time_unit_is_valid(config_yaml: dict) -> None:
    u = config_yaml["calibration"]["time_unit"]
    assert u in ("s", "frame"), f"calibration.time_unit must be 's' or 'frame', got {u!r}"


def test_visualization_enabled_is_bool(config_yaml: dict) -> None:
    v = config_yaml["visualization"]["enabled"]
    assert isinstance(v, bool), "visualization.enabled must be a boolean"


def test_features_kinematic_three_families_is_bool(config_yaml: dict) -> None:
    v = config_yaml["features"]["kinematic_three_families"]
    assert isinstance(v, bool), "features.kinematic_three_families must be a boolean"


def test_latent_space_embedding_method_is_valid(config_yaml: dict) -> None:
    m = config_yaml["latent_space"]["embedding"]["method"]
    assert m in ("umap", "tsne", "pca"), (
        f"latent_space.embedding.method must be 'umap', 'tsne', or 'pca', got {m!r}"
    )


@pytest.mark.parametrize("backend,required", [
    ("umap", ["n_neighbors", "min_dist", "metric", "n_components"]),
    ("tsne", ["perplexity", "learning_rate", "init", "metric", "n_components"]),
    ("pca",  ["n_components"]),
])
def test_latent_space_embedding_backend_keys(config_yaml: dict, backend: str, required: list[str]) -> None:
    block = config_yaml["latent_space"]["embedding"].get(backend, {})
    for key in required:
        assert key in block, (
            f"latent_space.embedding.{backend}.{key} is required by the embedding dispatcher"
        )


def test_latent_space_pca_toggle_under_embedding(config_yaml: dict) -> None:
    """PCA pre-embedding toggle moved under latent_space.embedding."""
    emb = config_yaml["latent_space"]["embedding"]
    assert "use_pca" in emb, "latent_space.embedding.use_pca is required"
    assert "pca_explained_variance" in emb, "latent_space.embedding.pca_explained_variance is required"
    assert isinstance(emb["use_pca"], bool), "latent_space.embedding.use_pca must be a boolean"
    v = emb["pca_explained_variance"]
    assert isinstance(v, (int, float)) and 0.0 < float(v) <= 1.0, (
        "latent_space.embedding.pca_explained_variance must be in (0, 1]"
    )


def test_classification_method_is_valid(config_yaml: dict) -> None:
    m = config_yaml["classification"]["method"]
    assert m in ("svc", "lda", "logistic"), (
        f"classification.method must be 'svc', 'lda', or 'logistic', got {m!r}"
    )


@pytest.mark.parametrize("backend,required", [
    ("svc",      ["kernel", "C", "gamma", "grid_search"]),
    ("lda",      ["solver"]),
    ("logistic", ["C", "max_iter", "solver"]),
])
def test_classification_backend_keys(config_yaml: dict, backend: str, required: list[str]) -> None:
    block = config_yaml["classification"].get(backend, {})
    for key in required:
        assert key in block, (
            f"classification.{backend}.{key} is required by the classifier dispatcher"
        )


def test_classification_svc_grid_search_schema(config_yaml: dict) -> None:
    gs = config_yaml["classification"]["svc"]["grid_search"]
    assert isinstance(gs.get("enabled"), bool), (
        "classification.svc.grid_search.enabled must be a boolean"
    )
    for key in ("C", "gamma"):
        assert isinstance(gs.get(key), list) and len(gs[key]) >= 1, (
            f"classification.svc.grid_search.{key} must be a non-empty list"
        )
    for c in gs["C"]:
        assert isinstance(c, (int, float)) and c > 0, (
            f"classification.svc.grid_search.C values must be positive numbers, got {c!r}"
        )
    for g in gs["gamma"]:
        ok = isinstance(g, str) and g in ("scale", "auto")
        ok = ok or (isinstance(g, (int, float)) and g > 0)
        assert ok, (
            f"classification.svc.grid_search.gamma values must be 'scale', 'auto', or positive number, got {g!r}"
        )


def test_classification_augmentation_schema(config_yaml: dict) -> None:
    """augmentation block: enabled bool + non-empty transform list of known names."""
    import re
    aug = config_yaml["classification"].get("augmentation")
    assert isinstance(aug, dict), "classification.augmentation must be a mapping"
    assert isinstance(aug.get("enabled"), bool), (
        "classification.augmentation.enabled must be a boolean"
    )
    tfs = aug.get("transforms")
    assert isinstance(tfs, list) and len(tfs) >= 1, (
        "classification.augmentation.transforms must be a non-empty list"
    )
    flips = {"identity", "flip_x", "flip_y", "flip_xy"}
    rot_pat = re.compile(r"^rot-?\d+(?:\.\d+)?$")
    bad = [t for t in tfs if t not in flips and not rot_pat.match(str(t))]
    assert not bad, (
        f"classification.augmentation.transforms contains unsupported names {bad!r}; "
        f"allowed: {sorted(flips)} or rot<angle> (e.g. rot15, rot-30)"
    )


def test_classification_masking_schema(config_yaml: dict) -> None:
    """masking block: enabled bool + mask_fraction in (0, 1) with 1/fraction >= 2."""
    m = config_yaml["classification"].get("masking")
    assert isinstance(m, dict), "classification.masking must be a mapping"
    assert isinstance(m.get("enabled"), bool), "classification.masking.enabled must be a boolean"
    mf = m.get("mask_fraction")
    assert isinstance(mf, (int, float)) and 0.0 < float(mf) < 1.0, (
        f"classification.masking.mask_fraction must be in (0, 1); got {mf!r}"
    )
    n_copies = int(round(1.0 / float(mf)))
    assert n_copies >= 2, (
        f"classification.masking.mask_fraction={mf} yields only {n_copies} block(s); "
        f"use a smaller fraction so 1/fraction >= 2"
    )


def test_classification_polynomial_schema(config_yaml: dict) -> None:
    """polynomial block: enabled bool + degree int|list + interaction_only bool|list."""
    p = config_yaml["classification"].get("polynomial")
    assert isinstance(p, dict), "classification.polynomial must be a mapping"
    assert isinstance(p.get("enabled"), bool), "classification.polynomial.enabled must be a boolean"
    deg = p.get("degree")
    if isinstance(deg, list):
        assert deg and all(isinstance(d, int) and d >= 1 for d in deg), (
            f"classification.polynomial.degree list must contain positive ints; got {deg!r}"
        )
    else:
        assert isinstance(deg, int) and deg >= 1, (
            f"classification.polynomial.degree must be a positive int or a list; got {deg!r}"
        )
    inter = p.get("interaction_only")
    if isinstance(inter, list):
        assert inter and all(isinstance(v, bool) for v in inter), (
            f"classification.polynomial.interaction_only list must be booleans; got {inter!r}"
        )
    else:
        assert isinstance(inter, bool), (
            f"classification.polynomial.interaction_only must be a bool or list; got {inter!r}"
        )
    assert isinstance(p.get("include_bias"), bool), (
        "classification.polynomial.include_bias must be a boolean"
    )


def test_watershed_schema(config_yaml: dict) -> None:
    """watershed block: types + positivity constraints."""
    w = config_yaml["watershed"]
    assert isinstance(w.get("enabled"), bool), "watershed.enabled must be a boolean"
    assert isinstance(w.get("debug"), bool), "watershed.debug must be a boolean"
    for key in ("area_outlier_k", "min_distance_factor", "min_region_area_fraction"):
        v = w.get(key)
        assert isinstance(v, (int, float)) and v > 0, (
            f"watershed.{key} must be a positive number, got {v!r}"
        )
    mfb = w.get("max_flies_per_blob")
    assert isinstance(mfb, int) and mfb >= 2, (
        f"watershed.max_flies_per_blob must be an int >= 2, got {mfb!r}"
    )
    dmi = w.get("debug_max_images")
    assert isinstance(dmi, int) and dmi >= 1, (
        f"watershed.debug_max_images must be an int >= 1, got {dmi!r}"
    )
    mraf = w["min_region_area_fraction"]
    assert 0.0 < float(mraf) < 1.0, (
        f"watershed.min_region_area_fraction must be in (0, 1), got {mraf!r}"
    )


def test_no_legacy_stitching_block_in_main_config(config_yaml: dict) -> None:
    """The active pipeline does not read config.yaml's stitching block.

    If it reappears, somebody is silently re-introducing dead config that the
    runtime code will ignore. Legacy stitching params live in
    legacy/stitching_config.yaml.
    """
    assert "stitching" not in config_yaml, (
        "Found a 'stitching:' section in config.yaml. "
        "Move post-hoc stitching params to legacy/stitching_config.yaml."
    )


def test_no_top_level_pca_keys_in_latent_space(config_yaml: dict) -> None:
    """Pre-embedding PCA toggles live under embedding:, not at latent_space:.

    Catches drift where someone re-adds latent_space.use_pca at the top level.
    """
    ls = config_yaml["latent_space"]
    assert "use_pca" not in ls, (
        "latent_space.use_pca moved under latent_space.embedding.use_pca"
    )
    assert "pca_explained_variance" not in ls, (
        "latent_space.pca_explained_variance moved under latent_space.embedding.pca_explained_variance"
    )
