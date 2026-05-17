"""
Tests for geometric trajectory augmentation in src/features.py.

The augmentation step expands the *row* count of the raw trajectory frame
(new copies under each transform) without changing the feature schema, and
tags each row with ``aug_group`` so callers can keep augmented copies of the
same fly inside a single CV fold via GroupKFold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as F


def _two_flies_raw() -> pd.DataFrame:
    n = 6
    rows = []
    for fid, x0, y0 in [("flyA", 100.0, 50.0), ("flyB", 200.0, 80.0)]:
        for i in range(n):
            rows.append({
                "frame": i,
                "ordered_id": fid,
                "x": x0 + i * 3.0,
                "y": y0 + i * 2.0,
                "vial_id": "vial1",
                "genotype": "WT",
                "run": "runX",
            })
    return pd.DataFrame(rows)


def test_default_transforms_expand_rows_4x_and_preserve_columns():
    df = _two_flies_raw()
    aug = F.augment_trajectories_geometric(df)
    assert len(aug) == 4 * len(df)
    # aug_group is added; every other original column survives
    for col in df.columns:
        assert col in aug.columns, f"augmented frame dropped column {col!r}"
    assert "aug_group" in aug.columns


def test_aug_group_matches_original_ordered_id():
    df = _two_flies_raw()
    aug = F.augment_trajectories_geometric(df, transforms=["identity", "flip_x"])
    flipped = aug[aug["ordered_id"].str.endswith("__aug:flip_x")]
    assert (flipped["aug_group"] == flipped["ordered_id"].str.replace("__aug:flip_x", "", regex=False)).all()
    identity = aug[~aug["ordered_id"].str.contains("__aug:")]
    assert (identity["aug_group"] == identity["ordered_id"]).all()


def test_flip_x_negates_x_and_leaves_y_alone():
    df = _two_flies_raw()
    aug = F.augment_trajectories_geometric(df, transforms=["identity", "flip_x"])
    orig = aug[~aug["ordered_id"].str.contains("__aug:")].sort_values(["ordered_id", "frame"]).reset_index(drop=True)
    flip = aug[aug["ordered_id"].str.endswith("__aug:flip_x")].sort_values(["ordered_id", "frame"]).reset_index(drop=True)
    np.testing.assert_allclose(flip["x"].to_numpy(), -orig["x"].to_numpy())
    np.testing.assert_allclose(flip["y"].to_numpy(),  orig["y"].to_numpy())


def test_identity_round_trips_through_feature_extraction():
    df = _two_flies_raw()
    aug = F.augment_trajectories_geometric(df, transforms=["identity"])
    feats_orig = F.extract_behavioral_features(df)
    feats_aug = F.extract_behavioral_features(aug.drop(columns=["aug_group"]))
    # All kinematic columns must match between the original and the identity copy.
    shared = [c for c in feats_orig.columns if c in feats_aug.columns and c not in {"ordered_id"}]
    pd.testing.assert_frame_equal(
        feats_orig[shared].reset_index(drop=True),
        feats_aug[shared].reset_index(drop=True),
        check_dtype=False,
    )


def test_unknown_transform_raises():
    df = _two_flies_raw()
    with pytest.raises(ValueError, match="unknown geometric transforms"):
        F.augment_trajectories_geometric(df, transforms=["identity", "rotate_37_deg"])


def test_empty_transform_list_raises():
    df = _two_flies_raw()
    with pytest.raises(ValueError, match="non-empty"):
        F.augment_trajectories_geometric(df, transforms=[])


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def test_rot180_matches_flip_xy_around_per_fly_mean():
    """180-degree rotation around each fly's mean is equivalent to negating the
    centred displacement, then restoring the mean. Verifies the per-fly anchor."""
    df = _two_flies_raw()
    aug = F.augment_trajectories_geometric(df, transforms=["identity", "rot180"])
    orig = aug[~aug["ordered_id"].str.contains("__aug:")].sort_values(["ordered_id", "frame"]).reset_index(drop=True)
    rot = aug[aug["ordered_id"].str.endswith("__aug:rot180")].sort_values(["ordered_id", "frame"]).reset_index(drop=True)
    for fid in orig["aug_group"].unique():
        o = orig[orig["aug_group"] == fid]
        r = rot[rot["aug_group"] == fid]
        cx, cy = o["x"].mean(), o["y"].mean()
        np.testing.assert_allclose(r["x"].to_numpy(), 2 * cx - o["x"].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(r["y"].to_numpy(), 2 * cy - o["y"].to_numpy(), atol=1e-9)


def test_rot90_swaps_components_correctly():
    df = _two_flies_raw()
    aug = F.augment_trajectories_geometric(df, transforms=["identity", "rot90"])
    orig = aug[~aug["ordered_id"].str.contains("__aug:")].sort_values(["ordered_id", "frame"]).reset_index(drop=True)
    rot = aug[aug["ordered_id"].str.endswith("__aug:rot90")].sort_values(["ordered_id", "frame"]).reset_index(drop=True)
    for fid in orig["aug_group"].unique():
        o = orig[orig["aug_group"] == fid]
        r = rot[rot["aug_group"] == fid]
        cx, cy = float(o["x"].mean()), float(o["y"].mean())
        dx = o["x"].to_numpy() - cx
        dy = o["y"].to_numpy() - cy
        np.testing.assert_allclose(r["x"].to_numpy(), -dy + cx, atol=1e-9)
        np.testing.assert_allclose(r["y"].to_numpy(),  dx + cy, atol=1e-9)


def test_negative_rotation_parses():
    df = _two_flies_raw()
    aug = F.augment_trajectories_geometric(df, transforms=["identity", "rot-45"])
    assert (aug["ordered_id"].str.endswith("__aug:rot-45")).any()


# ---------------------------------------------------------------------------
# Masking (block / temporal jackknife)
# ---------------------------------------------------------------------------

def test_masking_yields_one_over_fraction_copies():
    df = _two_flies_raw()  # 2 flies x 6 frames
    out = F.augment_trajectories_masking(df, mask_fraction=0.5)  # 2 copies
    assert set(out["ordered_id"].str.extract(r"__aug:(maskblock\d+)$")[0].dropna()) == {
        "maskblock0", "maskblock1",
    }
    # 2 copies x 2 flies = 4 groups; each has 6 - 3 = 3 frames kept (half block dropped).
    for (orig, oid), g in out.groupby(["aug_group", "ordered_id"]):
        assert len(g) == 3


def test_masking_blocks_are_complementary():
    """Together, copies k=0..N-1 must drop every frame of a fly exactly once."""
    df = _two_flies_raw()
    out = F.augment_trajectories_masking(df, mask_fraction=1 / 3)  # 3 copies
    # Sum of kept frames across copies = (N-1) * total -> every frame appears in N-1 copies.
    for fid, g in df.groupby("ordered_id"):
        total = len(g)
        kept_total = (out["aug_group"] == fid).sum()
        assert kept_total == (3 - 1) * total, (
            f"fly {fid}: kept_total={kept_total}, expected {(3 - 1) * total}"
        )


def test_masking_is_deterministic():
    df = _two_flies_raw()
    a = F.augment_trajectories_masking(df, mask_fraction=0.25)
    b = F.augment_trajectories_masking(df, mask_fraction=0.25)
    pd.testing.assert_frame_equal(
        a.sort_values(["aug_group", "ordered_id", "frame"]).reset_index(drop=True),
        b.sort_values(["aug_group", "ordered_id", "frame"]).reset_index(drop=True),
    )


def test_masking_rejects_bad_inputs():
    df = _two_flies_raw()
    with pytest.raises(ValueError):
        F.augment_trajectories_masking(df, mask_fraction=0.0)
    with pytest.raises(ValueError):
        F.augment_trajectories_masking(df, mask_fraction=1.0)
    with pytest.raises(ValueError):
        # 0.99 -> 1/0.99 ~ 1 copy, not augmentation
        F.augment_trajectories_masking(df, mask_fraction=0.99)
