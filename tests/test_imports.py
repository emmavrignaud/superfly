"""
Smoke tests: every name advertised by src/__init__.py must actually import.

Catches the class of bug the notebook had (calling
``assign_vials_and_compact_ids`` after it had been renamed) before runtime.
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED_PUBLIC_NAMES = [
    # preprocessing
    "gui_pick_roi_and_range",
    "preprocess_bgsub_gui",
    # tracking
    "export_tracks_xy_tuple_csv_one_config",
    # io / format conversion (lives in src.wide_long, re-exported via src.stitching)
    "wide_to_long",
    # roi
    "draw_and_save_vial_rois",
    "assign_ordered_ids_left_to_right",
    "assign_vials_and_ordered_ids",
    # features
    "add_kinematics",
    "add_area_covered",
    "add_path_tortuosity",
    "extract_behavioral_features",
    "aggregate_per_fly_features",
    # classification
    "make_classifier",
    "prepare_xy",
    "prepare_target",
    "run_cross_validation",
    "run_classifier",
    # visualization
    "render_vial_overlay_video",
]


@pytest.fixture(scope="module")
def src_module():
    return importlib.import_module("src")


@pytest.mark.parametrize("name", EXPECTED_PUBLIC_NAMES)
def test_public_name_is_importable(src_module, name: str) -> None:
    assert hasattr(src_module, name), f"src package is missing public name {name!r}"
    obj = getattr(src_module, name)
    assert callable(obj), f"src.{name} exists but is not callable (got {type(obj)})"


def test_utils_save_run_params_importable() -> None:
    from utils import save_run_params

    assert callable(save_run_params)


def test_wide_long_module_directly_importable() -> None:
    from src.wide_long import wide_to_long as wl_direct
    from src.stitching import wide_to_long as wl_stub

    assert wl_direct is wl_stub, "src.stitching must re-export the same wide_to_long object"
