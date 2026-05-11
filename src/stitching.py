"""
src.stitching (compatibility shim)
==================================

The post-hoc tracklet stitching machinery (``build_tracklets``, ``stitch``,
``link_score``, etc.) is no longer part of the active pipeline. Its full body
now lives in ``legacy/stitching.py`` and is only imported by the deprecated
``legacy/grid_search_stitching_params.py`` script.

The active pipeline only ever needs ``wide_to_long`` for wide -> long format
conversion; that function has moved to ``src.wide_long``. This module is kept
as a re-export so existing imports (``from src.stitching import wide_to_long``)
keep working without code changes.

If you are looking for the legacy stitching algorithms, import them from
``legacy.stitching`` instead.
"""

from __future__ import annotations

import warnings

from src.wide_long import wide_to_long  # noqa: F401  (re-export)


def __getattr__(name: str):
    """Forward any other name lookup to legacy.stitching with a deprecation warning."""
    if name == "wide_to_long":
        return wide_to_long
    try:
        from legacy import stitching as _legacy_stitching
    except ImportError as e:
        raise AttributeError(
            f"src.stitching no longer provides '{name}'. "
            f"The legacy stitching module could not be imported: {e}"
        ) from e
    if hasattr(_legacy_stitching, name):
        warnings.warn(
            f"src.stitching.{name} is deprecated; import from legacy.stitching instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_legacy_stitching, name)
    raise AttributeError(f"module 'src.stitching' has no attribute {name!r}")


__all__ = ["wide_to_long"]
