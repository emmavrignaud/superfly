"""
Tests for the src.stitching backwards-compatibility shim.

The post-hoc stitching code lives in legacy/stitching.py; src/stitching.py is a
thin re-export of wide_to_long. Old call sites that say
``from src.stitching import build_tracklets`` should still work, but emit a
DeprecationWarning so anyone touching them is nudged toward the legacy module.
"""

from __future__ import annotations

import warnings

import pytest


def test_wide_to_long_is_silently_re_exported() -> None:
    """The active pipeline imports wide_to_long via src.stitching all over the
    place. That path must not warn."""
    import src.stitching as stitch
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = stitch.wide_to_long


def test_unknown_attribute_raises_attribute_error() -> None:
    import src.stitching as stitch
    with pytest.raises(AttributeError):
        _ = stitch.this_name_does_not_exist


def test_legacy_function_is_forwarded_with_deprecation_warning() -> None:
    """build_tracklets lives in legacy/stitching.py. Importing it from
    src.stitching should still resolve, but with a DeprecationWarning."""
    import src.stitching as stitch
    with pytest.warns(DeprecationWarning, match="legacy.stitching"):
        bt = stitch.build_tracklets
    assert callable(bt)


def test_legacy_module_imports_directly() -> None:
    """legacy.stitching itself must import (reads legacy/stitching_config.yaml)."""
    import importlib
    legacy = importlib.import_module("legacy.stitching")
    assert hasattr(legacy, "build_tracklets")
    assert hasattr(legacy, "stitch")
    assert hasattr(legacy, "link_score")
