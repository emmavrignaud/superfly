"""
tests/run_analysis_integration.py

Usage:
    python tests/run_analysis_integration.py
    python tests/run_analysis_integration.py --stage 1
    python tests/run_analysis_integration.py --stage 2
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import time

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str, exc: Exception) -> None:
    print(f"  [FAIL] {msg}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Stage 1 — Synthetic smoke tests
# ---------------------------------------------------------------------------

def _make_fake_df_feat(n_flies=20, n_frames=120, n_runs=2, rng=None) -> pd.DataFrame:
    """Minimal frame-level feature DataFrame that mimics extract_behavioral_features output."""
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []
    vials = ["vial1", "vial2", "vial3"]
    genotypes = ["WT", "A90V", "M337V"]
    runs = [f"run_{i}_13DPE_n00{i+1}" for i in range(n_runs)]
    for run in runs:
        for fly_i in range(n_flies):
            fly_id = f"{run}::fly_{fly_i}"
            vial = vials[fly_i % len(vials)]
            geno = genotypes[fly_i % len(genotypes)]
            x, y = 0.0, 0.0
            for frame in range(n_frames):
                dx, dy = rng.normal(0, 0.1), rng.normal(0, 0.1)
                x, y = x + dx, y + dy
                speed = float(np.sqrt(dx**2 + dy**2)) * 10
                rows.append({
                    "ordered_id": fly_id,
                    "frame": frame,
                    "x": x, "y": y,
                    "dx": dx, "dy": dy,
                    "dt": 1.0,
                    "step_distance": float(np.sqrt(dx**2 + dy**2)),
                    "velocity": speed,
                    "heading": float(np.arctan2(dy, dx)),
                    "turning_angle": rng.uniform(-0.3, 0.3),
                    "angular_velocity": rng.uniform(-0.3, 0.3),
                    "acceleration": rng.uniform(0, 0.05),
                    "distance_traveled": float(frame * 0.1),
                    "tortuosity": rng.uniform(1.0, 2.0),
                    "area_covered": rng.uniform(0.5, 5.0),
                    "vial_id": vial,
                    "genotype": geno,
                    "run": run,
                })
    return pd.DataFrame(rows)


def test_aggregate_binned():
    from src.features import aggregate_per_fly_features_binned
    df = _make_fake_df_feat(n_flies=6, n_frames=90)
    result = aggregate_per_fly_features_binned(df, n_bins=3, min_frames=2)
    assert not result.empty, "binned aggregation returned empty DataFrame"
    assert "time_bin" in result.columns, "time_bin column missing"
    assert result["time_bin"].nunique() == 3, f"expected 3 bins, got {result['time_bin'].nunique()}"
    _ok(f"aggregate_per_fly_features_binned: {len(result)} rows, bins={sorted(result['time_bin'].unique())}")


def test_multitask_autoencoder():
    from src.representation_learning import fit_multitask_autoencoder
    rng = np.random.default_rng(1)
    N, D = 60, 12
    X = rng.standard_normal((N, D)).astype(np.float64)
    y_age = rng.integers(0, 4, size=N)
    y_age[:5] = -1  # some unknown
    y_time = rng.integers(0, 3, size=N)
    y_geno = rng.integers(0, 3, size=N)

    Z, info = fit_multitask_autoencoder(
        X, y_age=y_age, y_time_bin=y_time, y_genotype=y_geno,
        n_age_classes=4, n_time_bins=3, n_genotypes=3,
        age_weight=1.0, time_weight=0.5, genotype_weight=0.5,
        latent_dim=8, hidden_dims=(32, 16), epochs=20, batch_size=16,
        verbose=True,
    )
    assert Z.shape == (N, 8), f"unexpected latent shape {Z.shape}"
    assert info["used_autoencoder"]
    assert info["multitask"]
    _ok(
        f"fit_multitask_autoencoder: Z={Z.shape}  "
        f"age_acc={info.get('age_accuracy')}  "
        f"time_acc={info.get('time_accuracy')}  "
        f"geno_acc={info.get('genotype_accuracy')}"
    )


def test_feature_significance():
    from src.statistics import feature_significance_report
    rng = np.random.default_rng(2)
    n = 80
    df = pd.DataFrame({
        "genotype": (["WT"] * 20 + ["A90V"] * 20 + ["G287S"] * 20 + ["M337V"] * 20),
        "mean_velocity": np.concatenate([
            rng.normal(2.0, 0.5, 20),   # WT
            rng.normal(1.2, 0.5, 20),   # A90V — clearly different
            rng.normal(2.1, 0.5, 20),   # G287S — similar to WT
            rng.normal(0.8, 0.5, 20),   # M337V — very different
        ]),
        "pause_fraction": rng.uniform(0, 1, n),
        "tortuosity": rng.normal(1.5, 0.3, n),
    })
    features = ["mean_velocity", "pause_fraction", "tortuosity"]
    results, vol_fig, bar_fig = feature_significance_report(df, features, group_col="genotype")
    assert len(results) == len(features)
    assert "kw_p_adj" in results.columns
    assert "max_abs_delta" in results.columns
    # mean_velocity should come out significant
    mv_row = results[results["feature"] == "mean_velocity"].iloc[0]
    assert mv_row["kw_p"] < 0.05, f"mean_velocity KW p={mv_row['kw_p']} not significant"
    _ok(f"feature_significance_report: {len(results)} features tested, volcano+bar figs created")


def test_permutation_test(tmp_dir):
    from src.classification import run_permutation_test
    from src.features import classification_feature_columns
    rng = np.random.default_rng(3)
    n = 60
    feats = classification_feature_columns()
    data = {f: rng.standard_normal(n) for f in feats}
    data["genotype"] = (["WT"] * 15 + ["A90V"] * 15 + ["G287S"] * 15 + ["M337V"] * 15)
    # Make mean_velocity actually separate genotypes so real accuracy > chance
    data["mean_velocity"] = (
        [2.0] * 15 + [0.5] * 15 + [1.5] * 15 + [0.2] * 15
    ) + rng.normal(0, 0.3, n)
    df = pd.DataFrame(data)

    result = run_permutation_test(
        df, outdir=tmp_dir,
        classification_mode="multiclass",
        cv=3, n_permutations=50,  # fast for the test
        save_files=False,
    )
    assert "real_score" in result
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert len(result["null_scores"]) == 50
    _ok(
        f"run_permutation_test: real={result['real_score']:.3f}  "
        f"p={result['p_value']:.4g}  null_mean={result['null_scores'].mean():.3f}"
    )


def run_stage1():
    _section("Stage 1 — Synthetic smoke tests")
    import tempfile
    tmp = tempfile.mkdtemp()

    tests = [
        ("aggregate_per_fly_features_binned", test_aggregate_binned),
        ("fit_multitask_autoencoder (3 heads)", test_multitask_autoencoder),
        ("feature_significance_report", test_feature_significance),
        ("run_permutation_test", lambda: test_permutation_test(tmp)),
    ]
    passed = 0
    for name, fn in tests:
        try:
            t0 = time.time()
            fn()
            print(f"     ({time.time()-t0:.1f}s)")
            passed += 1
        except Exception as e:
            _fail(name, e)

    print(f"\n  {passed}/{len(tests)} synthetic tests passed.")
    return passed == len(tests)


# ---------------------------------------------------------------------------
# Stage 2 — Real-data integration test
# ---------------------------------------------------------------------------

RUN_DIRS = [
    "../outputs/run_80_13DPE_n002/",
    "../outputs/run_77_28DPE_n003/",
    "../outputs/run_114_31DPE_n005/",
    "../outputs/run_66_41DPE_n004/",
]


def run_stage2():
    _section("Stage 2 — Real-data integration test")

    import os
    from src.classification import map_vial_to_genotype, run_classifier, run_permutation_test
    from src.features import (
        extract_behavioral_features,
        aggregate_per_fly_features,
        classification_feature_columns,
    )
    from src.statistics import feature_significance_report
    from src.latent_space import run_latent_space_analysis, write_latent_space_report

    out_dir = os.path.join(_REPO, "outputs", "classification", "test_overnight_run")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output dir: {out_dir}")

    # --- Load data ---
    parts = []
    for rd in RUN_DIRS:
        rd_abs = os.path.join(_REPO, "notebooks", rd)
        if not os.path.isdir(rd_abs):
            rd_abs = os.path.join(_REPO, rd.lstrip("../"))
        if not os.path.isdir(rd_abs):
            print(f"  [SKIP] {rd} not found at {rd_abs}")
            continue
        try:
            d = map_vial_to_genotype(rd_abs)
            run_tag = os.path.basename(rd_abs.rstrip("/\\"))
            d["run"] = run_tag
            d["run_dir"] = rd_abs
            d["ordered_id"] = run_tag + "::" + d["ordered_id"].astype(str)
            parts.append(d)
            _ok(f"Loaded {run_tag}: {len(d)} frames, {d['ordered_id'].nunique()} flies")
        except Exception as e:
            _fail(f"Loading {rd}", e)

    if not parts:
        print("  No runs loaded — aborting Stage 2.")
        return False

    df_raw = pd.concat(parts, ignore_index=True)
    print(f"\n  Combined: {df_raw.shape}  genotypes: {df_raw['genotype'].value_counts().to_dict()}")

    # --- Feature extraction ---
    print("\n  Extracting behavioral features...")
    t0 = time.time()
    df_feat = extract_behavioral_features(df_raw)
    df_agg = aggregate_per_fly_features(df_feat, pause_threshold=1.0)
    meta = df_raw.drop_duplicates("ordered_id").set_index("ordered_id")[["genotype", "run"]]
    df_agg = df_agg.join(meta, on="ordered_id").dropna(subset=["genotype"])
    _ok(f"Features extracted: {df_agg.shape[0]} flies in {time.time()-t0:.1f}s")

    FEATURES = classification_feature_columns()
    groups = df_agg["run"].to_numpy()

    # --- C: Feature significance ---
    print("\n  Running feature significance (C)...")
    try:
        sig, vol, bar = feature_significance_report(df_agg, FEATURES)
        vol.write_html(os.path.join(out_dir, "feature_significance_volcano.html"))
        bar.write_html(os.path.join(out_dir, "feature_significance_effect_sizes.html"))
        _ok(f"Significance done. Top feature: {sig.iloc[0]['feature']}  p={sig.iloc[0]['kw_p']:.4g}")
        print(sig[["feature", "kw_p", "kw_p_adj", "max_abs_delta"]].to_string(index=False))
    except Exception as e:
        _fail("Feature significance (C)", e)

    # --- D: Permutation test ---
    print("\n  Running permutation test (D, n=200)...")
    try:
        for mode in ["multiclass", "binary"]:
            r = run_permutation_test(
                df_agg, outdir=out_dir,
                classification_mode=mode, cv=4, groups=groups,
                n_permutations=200, save_files=True,
            )
            _ok(f"[{mode}] real={r['real_score']:.3f}  p={r['p_value']:.4g}")
    except Exception as e:
        _fail("Permutation test (D)", e)

    # --- A+B: Latent space with multitask AE (temporal + genotype heads) ---
    print("\n  Running latent space analysis (A+B — multitask AE)...")
    try:
        latent = run_latent_space_analysis(df_raw)
        rel = write_latent_space_report(latent, out_dir)
        _ok(f"Latent space report written: {rel}")
        a4 = latent.get("analysis4", {})
        if a4 and not a4.get("meta", pd.DataFrame()).empty:
            r4 = a4.get("representation", {})
            _ok(
                f"Analysis 4 (temporal bins): "
                f"time_acc={r4.get('time_accuracy')}  "
                f"age_acc={r4.get('age_accuracy')}  "
                f"geno_acc={r4.get('genotype_accuracy')}"
            )
        else:
            print("  [WARN] Analysis 4 was empty.")
    except Exception as e:
        _fail("Latent space (A+B)", e)

    print(f"\n  All Stage 2 outputs in: {out_dir}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=0,
                        help="1=synthetic only, 2=real data only, 0=both (default)")
    args = parser.parse_args()

    t_start = time.time()
    ok1, ok2 = True, True

    if args.stage in (0, 1):
        ok1 = run_stage1()

    if args.stage in (0, 2):
        ok2 = run_stage2()

    elapsed = time.time() - t_start
    _section(f"Done in {elapsed/60:.1f} min  —  Stage1={'OK' if ok1 else 'FAIL'}  Stage2={'OK' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)
