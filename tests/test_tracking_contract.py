"""
Contract test for the tracker seam (src/tracking.py).

`export_tracks_xy_tuple_csv_one_config` is the one door between our pipeline and
the vendored OC-SORT tracker. Every value it takes from config.yaml is a
required, keyword-only parameter with no default. This test pins that: if
someone reintroduces a default, config.yaml quietly stops being the single
source of truth (a caller could drop a value and still "work" with a stale
fallback). Making these required turns that silent drift into a loud TypeError.

Hermetic: inspects the signature only — no video, no Roboflow, no OC-SORT.
"""

from __future__ import annotations

import inspect

import pytest

from src.tracking import export_tracks_xy_tuple_csv_one_config

# Parameters whose values come from config.yaml. None may carry a default.
CONFIG_DRIVEN_PARAMS = [
    # roboflow + inference
    "inference_api_url",
    "inference_mode",
    "local_weights_path",
    "local_resolution",
    "local_num_classes",
    "local_optimize_for_gpu",
    "repo_root",
    # tracker: detection + association
    "detection_confidence_rfdetr",
    "confidence",
    "lost_track_buffer",
    "minimum_matching_threshold",
    "minimum_consecutive_frames",
    "asso_func",
    "brownian_pos_noise",
    "aspect_weight",
    "behavioral_weights",
    "overlap_weight_scale",
    "jump_factor",
    "jump_iou_threshold",
    "jump_inertia",
    "inertia",
    "delta_t",
    "overlap_iou_scale",
    "edge_fraction",
    "expected_count",
    "w_under",
    "w_over",
    # tracker.ghost_detection
    "ghost_detection_enabled",
    "ghost_offset_fraction",
    "ghost_confidence",
    "ghost_occlusion_max_gap",
    "ghost_top_exit_px",
]


@pytest.mark.parametrize("name", CONFIG_DRIVEN_PARAMS)
def test_config_driven_param_is_required(name: str) -> None:
    """Each config-driven parameter must be required (no default)."""
    sig = inspect.signature(export_tracks_xy_tuple_csv_one_config)
    assert name in sig.parameters, f"{name!r} is missing from the wrapper signature"
    param = sig.parameters[name]
    assert param.default is inspect.Parameter.empty, (
        f"{name!r} has a default ({param.default!r}); values that live in "
        f"config.yaml must be required so config stays the single source of truth."
    )


@pytest.mark.parametrize("name", CONFIG_DRIVEN_PARAMS)
def test_config_driven_param_is_keyword_only(name: str) -> None:
    """Config-driven params are keyword-only, so call sites read like config.foo=..."""
    sig = inspect.signature(export_tracks_xy_tuple_csv_one_config)
    assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
