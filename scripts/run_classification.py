#!/usr/bin/env python
"""
scripts/run_classification.py

CLI: outputs/<run_dir> -> classifier results + Plotly figures.

Stages
------
1. Load ordered_tracks from run dir, map genotypes from run_params.json
2. Extract behavioural features (kinematics, area, tortuosity)
3. Aggregate per-fly features
4. Run chosen classifier(s) with cross-validation
5. Produce per-genotype and WT-vs-mutant box plots

Usage
-----
python scripts\\run_classification.py ^
    --run-dir    outputs\\my_run ^
    --output-dir outputs\\my_run\\classification

Use --help for all options.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classification pipeline: ordered_tracks.csv -> results + figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True,
                   help="Path to a run directory under outputs/ (must contain "
                        "run_params.json and ordered_tracks.csv)")
    p.add_argument("--output-dir", default=None,
                   help="Directory for figures and outputs "
                        "(default: <run-dir>/classification)")
    p.add_argument("--modes", nargs="+", default=["multiclass", "binary"],
                   choices=["multiclass", "binary"],
                   help="Classification modes")
    p.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    p.add_argument("--pause-threshold", type=float, default=1.0,
                   help="Pause threshold in configured velocity unit (see calibration in config.yaml)")
    p.add_argument("--no-box-plots", action="store_true",
                   help="Skip per-feature box plots")
    p.add_argument("--report-site", action="store_true",
                   help="Write navigable multi-page report site")
    p.add_argument("--latent-space", action="store_true",
                   help="Run latent-space analysis and export latent report page")
    return p


def main():
    args = build_parser().parse_args()

    # Defer heavy imports until after --help is resolved
    from src.classification import (
        map_vial_to_genotype,
        run_classifier,
        plot_by_genotype,
        plot_wt_vs_mutant,
        write_classification_report_site,
    )
    from src.features import (
        extract_behavioral_features,
        aggregate_per_fly_features,
        aggregate_feature_plot_titles,
        augment_trajectories_geometric,
        augment_trajectories_masking,
        classification_feature_columns,
    )
    from src.latent_space import run_latent_space_analysis
    from utils import load_config

    import os
    import pandas as pd

    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _cls_cfg = load_config(_REPO_ROOT / "config.yaml").classification
    _cfg_aug = getattr(_cls_cfg, "augmentation", None)
    aug_enabled = bool(getattr(_cfg_aug, "enabled", False))
    aug_transforms = list(getattr(_cfg_aug, "transforms", []) or [])
    _cfg_mask = getattr(_cls_cfg, "masking", None)
    mask_enabled = bool(getattr(_cfg_mask, "enabled", False))

    features = classification_feature_columns()
    feature_titles = aggregate_feature_plot_titles(features)
    output_dir = args.output_dir or os.path.join(args.run_dir, "classification")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading run from: {args.run_dir}")
    df_raw = map_vial_to_genotype(args.run_dir)
    df_raw["run"] = os.path.basename(os.path.normpath(args.run_dir))
    df_raw["run_dir"] = args.run_dir

    if aug_enabled and aug_transforms:
        print(f"Applying geometric augmentation: {aug_transforms} "
              f"({len(aug_transforms)}x rows)")
        df_for_feat = augment_trajectories_geometric(df_raw, transforms=aug_transforms)
    else:
        df_for_feat = df_raw
    if mask_enabled:
        mf = float(getattr(_cfg_mask, "mask_fraction", 0.10))
        n_copies = int(round(1.0 / mf))
        print(f"Applying masking augmentation: mask_fraction={mf} -> {n_copies} copies per fly")
        df_masked = augment_trajectories_masking(
            df_for_feat if not aug_enabled else df_raw,
            mask_fraction=mf,
        )
        # When geometric is also on, masking acts on the originals and is appended,
        # so aug_group still refers to the original ordered_id (no nested aug).
        df_for_feat = pd.concat([df_for_feat, df_masked], ignore_index=True)

    print("Extracting behavioural features...")
    df_feat = extract_behavioral_features(df_for_feat)

    print("Aggregating per-fly features...")
    df_agg = aggregate_per_fly_features(df_feat, pause_threshold=args.pause_threshold)

    # Merge genotype back onto per-fly aggregation
    genotype_map = df_for_feat.drop_duplicates("ordered_id").set_index("ordered_id")["genotype"]
    df_agg["genotype"] = df_agg["ordered_id"].map(genotype_map)
    df_agg = df_agg.dropna(subset=["genotype"])

    # Carry the aug_group label through aggregation so GroupKFold can keep all
    # augmented copies of one fly in a single CV fold (no leakage across folds).
    aug_groups = None
    if "aug_group" in df_for_feat.columns:
        ag_map = df_for_feat.drop_duplicates("ordered_id").set_index("ordered_id")["aug_group"]
        df_agg["aug_group"] = df_agg["ordered_id"].map(ag_map)
        aug_groups = df_agg["aug_group"].to_numpy()

    hover_data = ["ordered_id"]

    # ------------------------------------------------------------------
    # Classification (single method, chosen in config.yaml under
    # ``classification.method``). One run per mode in --modes.
    # ------------------------------------------------------------------
    for mode in args.modes:
        print(f"\n=== classification [{mode}] ===")
        run_classifier(
            df=df_agg,
            outdir=output_dir,
            classification_mode=mode,
            cv=args.cv,
            plot_importance=True,
            groups=aug_groups,
        )

    # ------------------------------------------------------------------
    # Box plots
    # ------------------------------------------------------------------
    # Boxplots and the report site reflect biology, not augmentation — use
    # original rows only so duplicated trajectories don't inflate per-genotype
    # distributions or run-level stats.
    if "aug_group" in df_agg.columns:
        df_agg_orig = df_agg[df_agg["ordered_id"] == df_agg["aug_group"]].copy()
    else:
        df_agg_orig = df_agg

    if not args.no_box_plots:
        print("\n=== Per-genotype box plots ===")
        plot_by_genotype(df_agg_orig, features, feature_titles, hover_data, outdir=output_dir)

        print("=== WT vs Mutant box plots ===")
        plot_wt_vs_mutant(df_agg_orig, features, feature_titles, hover_data, outdir=output_dir)

    # ------------------------------------------------------------------
    # Latent-space analysis (optional)
    # ------------------------------------------------------------------
    latent_rel = None
    if args.latent_space:
        print("\n=== Latent-space analysis ===")
        latent = run_latent_space_analysis(df_raw)
        latent_dir = os.path.join(output_dir, "latent_space")
        os.makedirs(latent_dir, exist_ok=True)

        _method = latent["analysis1"].get("embedding", {}).get("method", "embedding")
        xy_html_name = f"{_method}_xy_kinematics.html"
        hist_html_name = f"{_method}_hist_kinematics.html"
        latent["analysis1"]["embedding_fig"].write_html(os.path.join(latent_dir, xy_html_name))
        latent["analysis2"]["embedding_fig"].write_html(os.path.join(latent_dir, hist_html_name))
        latent["rf_importance_fig"].write_html(os.path.join(latent_dir, "rf_importance.html"))

        # Standalone latent summary page (linked from report site when enabled)
        p1a = latent["analysis1"]["permanova"]["run_aware"]
        p1b = latent["analysis1"]["permanova"]["pooled"]
        p2a = latent["analysis2"]["permanova"]["run_aware"]
        p2b = latent["analysis2"]["permanova"]["pooled"]
        r1 = latent["analysis1"].get("representation", {})
        r2 = latent["analysis2"].get("representation", {})
        pca1 = latent["analysis1"].get("pca", {})
        pca2 = latent["analysis2"].get("pca", {})

        # Persist a machine-readable summary so PCA variance + embedding choice
        # are easy to inspect later. Drop non-JSON-safe values from sub-dicts.
        def _jsonable(d):
            out = {}
            for k, v in (d or {}).items():
                try:
                    json.dumps(v)
                    out[k] = v
                except (TypeError, ValueError):
                    out[k] = str(v)
            return out

        with open(os.path.join(latent_dir, "latent_report.json"), "w", encoding="utf-8") as fjson:
            json.dump(
                {
                    "embedding_method": _method,
                    "analysis1": {
                        "embedding": _jsonable(latent["analysis1"].get("embedding")),
                        "pca": _jsonable(pca1),
                        "representation": _jsonable(r1),
                        "permanova": {"run_aware": p1a, "pooled": p1b},
                    },
                    "analysis2": {
                        "embedding": _jsonable(latent["analysis2"].get("embedding")),
                        "pca": _jsonable(pca2),
                        "representation": _jsonable(r2),
                        "permanova": {"run_aware": p2a, "pooled": p2b},
                    },
                },
                fjson,
                indent=2,
            )

        def _fmt_pca(p):
            if not p:
                return "n/a"
            if not p.get("use_pca", False):
                return "off"
            return (
                f"on, kept {p.get('n_components', 'n/a')} dims, "
                f"explained_variance={p.get('explained_variance', 0.0):.3f}"
            )

        latent_page = os.path.join(output_dir, "latent_space_report.html")
        with open(latent_page, "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>",
                        "<title>Latent-space report</title>",
                        "<style>body{font-family:system-ui,sans-serif;margin:1rem 2rem;max-width:1200px;} .card{background:#f7f7f8;padding:0.6rem 0.8rem;border-radius:6px;margin:0.6rem 0;} iframe{width:100%;height:560px;border:1px solid #ddd;margin:0.5rem 0 1.2rem;}</style>",
                        "</head><body>",
                        "<h1>Latent-space report</h1>",
                        f"<div class='card'><strong>Embedding method</strong> - {_method}</div>",
                        f"<div class='card'><strong>Analysis 1 representation</strong> - method={r1.get('representation_method', 'baseline')}, used_autoencoder={r1.get('used_autoencoder', False)}, latent_dim={r1.get('latent_dim', 'n/a')}</div>",
                        f"<div class='card'><strong>Analysis 1 pre-embedding PCA</strong> - {_fmt_pca(pca1)}</div>",
                        f"<div class='card'><strong>Analysis 2 representation</strong> - method={r2.get('representation_method', 'baseline')}, used_autoencoder={r2.get('used_autoencoder', False)}, latent_dim={r2.get('latent_dim', 'n/a')}</div>",
                        f"<div class='card'><strong>Analysis 2 pre-embedding PCA</strong> - {_fmt_pca(pca2)}</div>",
                        f"<div class='card'><strong>Analysis 1 PERMANOVA (run-aware, primary)</strong> - method={p1a['method']}, pseudo-F={p1a['pseudo_f']:.4f}, p={p1a['p_value']:.4g}, R2={p1a['r2']:.4f}</div>",
                        f"<div class='card'><strong>Analysis 1 PERMANOVA (pooled, exploratory)</strong> - method={p1b['method']}, pseudo-F={p1b['pseudo_f']:.4f}, p={p1b['p_value']:.4g}, R2={p1b['r2']:.4f}</div>",
                        f"<div class='card'><strong>Analysis 2 PERMANOVA (run-aware, primary)</strong> - method={p2a['method']}, pseudo-F={p2a['pseudo_f']:.4f}, p={p2a['p_value']:.4g}, R2={p2a['r2']:.4f}</div>",
                        f"<div class='card'><strong>Analysis 2 PERMANOVA (pooled, exploratory)</strong> - method={p2b['method']}, pseudo-F={p2b['pseudo_f']:.4f}, p={p2b['p_value']:.4g}, R2={p2b['r2']:.4f}</div>",
                        f"<h2>Fly trajectory and kinematic embedding ({_method}, 3D)</h2>",
                        f"<iframe src='latent_space/{xy_html_name}'></iframe>",
                        f"<h2>Fly kinematic distribution embedding ({_method}, 3D)</h2>",
                        f"<iframe src='latent_space/{hist_html_name}'></iframe>",
                        "<h2>Random Forest Importance</h2>",
                        "<iframe src='latent_space/rf_importance.html'></iframe>",
                        "</body></html>",
                    ]
                )
            )
        latent_rel = "latent_space_report.html"
        print("Latent-space figures saved to:", latent_dir)

    # ------------------------------------------------------------------
    # Navigable report site (optional)
    # ------------------------------------------------------------------
    if args.report_site:
        entry = write_classification_report_site(
            df=df_agg_orig,
            features=features,
            feature_titles=feature_titles,
            hover_data=hover_data,
            out_dir=output_dir,
            trial_column="run",
            report_title="Classification and latent-space report",
            pooled_cv=args.cv,
            pooled_cv_groups=None,
            per_trial_cv=args.cv,
            entry_filename="classification_report.html",
            latent_page_filename=latent_rel,
        )
        print("Report site entry:", entry)

    print("\nDone. Figures saved to:", output_dir)


if __name__ == "__main__":
    main()
