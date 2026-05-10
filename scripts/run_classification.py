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
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FEATURES = [
    "mean_velocity",
    "median_velocity",
    "std_velocity",
    "pause_fraction",
    "mean_abs_turning_angle",
    "mean_abs_angular_velocity",
    "total_distance_traveled",
    "tortuosity",
    "area_covered",
]

FEATURE_TITLES = {
    "mean_velocity": "Mean velocity (px/s)",
    "median_velocity": "Median velocity (px/s)",
    "std_velocity": "Velocity std (px/s)",
    "pause_fraction": "Pause fraction",
    "mean_abs_turning_angle": "Mean |turning angle| (rad)",
    "mean_abs_angular_velocity": "Mean |angular velocity| (rad/s)",
    "total_distance_traveled": "Total distance traveled (px)",
    "tortuosity": "Path tortuosity",
    "area_covered": "Area covered (px²)",
}


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
    p.add_argument("--models", nargs="+", default=["lda", "logistic", "svc"],
                   choices=["lda", "logistic", "svc"],
                   help="Classifiers to run")
    p.add_argument("--modes", nargs="+", default=["multiclass", "binary"],
                   choices=["multiclass", "binary"],
                   help="Classification modes")
    p.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    p.add_argument("--pause-threshold", type=float, default=1.0,
                   help="Velocity threshold (px/s) for pause detection")
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
    from src.features import extract_behavioral_features, aggregate_per_fly_features
    from src.latent_space import run_latent_space_analysis

    import os
    output_dir = args.output_dir or os.path.join(args.run_dir, "classification")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading run from: {args.run_dir}")
    df_raw = map_vial_to_genotype(args.run_dir)
    df_raw["run"] = os.path.basename(os.path.normpath(args.run_dir))
    df_raw["run_dir"] = args.run_dir

    print("Extracting behavioural features...")
    df_feat = extract_behavioral_features(df_raw)

    print("Aggregating per-fly features...")
    df_agg = aggregate_per_fly_features(df_feat, pause_threshold=args.pause_threshold)

    # Merge genotype back onto per-fly aggregation
    genotype_map = df_raw.drop_duplicates("ordered_id").set_index("ordered_id")["genotype"]
    df_agg["genotype"] = df_agg["ordered_id"].map(genotype_map)
    df_agg = df_agg.dropna(subset=["genotype"])

    hover_data = ["ordered_id"]

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    for model_name in args.models:
        for mode in args.modes:
            print(f"\n=== {model_name.upper()} [{mode}] ===")
            run_classifier(
                df=df_agg,
                outdir=output_dir,
                model_name=model_name,
                classification_mode=mode,
                cv=args.cv,
                plot_importance=True,
            )

    # ------------------------------------------------------------------
    # Box plots
    # ------------------------------------------------------------------
    if not args.no_box_plots:
        print("\n=== Per-genotype box plots ===")
        plot_by_genotype(df_agg, FEATURES, FEATURE_TITLES, hover_data, outdir=output_dir)

        print("=== WT vs Mutant box plots ===")
        plot_wt_vs_mutant(df_agg, FEATURES, FEATURE_TITLES, hover_data, outdir=output_dir)

    # ------------------------------------------------------------------
    # Latent-space analysis (optional)
    # ------------------------------------------------------------------
    latent_rel = None
    if args.latent_space:
        print("\n=== Latent-space analysis ===")
        latent = run_latent_space_analysis(df_raw)
        latent_dir = os.path.join(output_dir, "latent_space")
        os.makedirs(latent_dir, exist_ok=True)

        latent["analysis1"]["umap_fig"].write_html(os.path.join(latent_dir, "umap_xy_kinematics.html"))
        latent["analysis2"]["umap_fig"].write_html(os.path.join(latent_dir, "umap_hist_kinematics.html"))
        latent["rf_importance_fig"].write_html(os.path.join(latent_dir, "rf_importance.html"))

        # Standalone latent summary page (linked from report site when enabled)
        p1a = latent["analysis1"]["permanova"]["run_aware"]
        p1b = latent["analysis1"]["permanova"]["pooled"]
        p2a = latent["analysis2"]["permanova"]["run_aware"]
        p2b = latent["analysis2"]["permanova"]["pooled"]
        r1 = latent["analysis1"].get("representation", {})
        r2 = latent["analysis2"].get("representation", {})
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
                        f"<div class='card'><strong>Analysis 1 representation</strong> - method={r1.get('representation_method', 'baseline')}, used_autoencoder={r1.get('used_autoencoder', False)}, latent_dim={r1.get('latent_dim', 'n/a')}</div>",
                        f"<div class='card'><strong>Analysis 2 representation</strong> - method={r2.get('representation_method', 'baseline')}, used_autoencoder={r2.get('used_autoencoder', False)}, latent_dim={r2.get('latent_dim', 'n/a')}</div>",
                        f"<div class='card'><strong>Analysis 1 PERMANOVA (run-aware, primary)</strong> - method={p1a['method']}, pseudo-F={p1a['pseudo_f']:.4f}, p={p1a['p_value']:.4g}, R2={p1a['r2']:.4f}</div>",
                        f"<div class='card'><strong>Analysis 1 PERMANOVA (pooled, exploratory)</strong> - method={p1b['method']}, pseudo-F={p1b['pseudo_f']:.4f}, p={p1b['p_value']:.4g}, R2={p1b['r2']:.4f}</div>",
                        f"<div class='card'><strong>Analysis 2 PERMANOVA (run-aware, primary)</strong> - method={p2a['method']}, pseudo-F={p2a['pseudo_f']:.4f}, p={p2a['p_value']:.4g}, R2={p2a['r2']:.4f}</div>",
                        f"<div class='card'><strong>Analysis 2 PERMANOVA (pooled, exploratory)</strong> - method={p2b['method']}, pseudo-F={p2b['pseudo_f']:.4f}, p={p2b['p_value']:.4g}, R2={p2b['r2']:.4f}</div>",
                        "<h2>Fly trajectory and kinematic embedding (3D)</h2>",
                        "<iframe src='latent_space/umap_xy_kinematics.html'></iframe>",
                        "<h2>Fly kinematic distribution embedding (3D)</h2>",
                        "<iframe src='latent_space/umap_hist_kinematics.html'></iframe>",
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
            df=df_agg,
            features=FEATURES,
            feature_titles=FEATURE_TITLES,
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
