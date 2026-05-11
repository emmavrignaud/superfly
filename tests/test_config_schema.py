"""
Schema validation for config.yaml.

Locks down the contract the active pipeline expects so a typo or accidental
deletion is caught at test time, not three hours into a tracking run.
"""

from __future__ import annotations

import pytest

REQUIRED_SECTIONS = ["roboflow", "video", "tracker", "pipeline", "preprocessing", "visualization", "roi"]

REQUIRED_KEYS = {
    "roboflow": ["model_id", "inference_api_url"],
    "video": ["fallback_fps"],
    "tracker": [
        "detection_confidence_rfdetr", "confidence", "lost_track_buffer",
        "minimum_matching_threshold", "minimum_consecutive_frames",
        "asso_func", "brownian_pos_noise",
        "aspect_weight", "behavioral_weight",
        "jump_factor", "jump_iou_threshold", "jump_inertia",
    ],
    "pipeline": ["expected_per_vial"],
    "preprocessing": ["bg_gain", "bg_white_level", "bg_percentile", "bg_sample_stride"],
    "visualization": ["overlay_source", "fps_out", "show_ids"],
    "roi": ["use_saved_roi"],
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
