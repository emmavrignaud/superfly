#!/usr/bin/env python
"""
scripts/run_classification.py

CLI: compact_tracks.csv -> classifier results + Plotly figures.

Stages
------
1. Load compact_tracks, map genotypes from filename
2. Extract behavioural features (kinematics, area, tortuosity)
3. Aggregate per-fly features
4. Run chosen classifier(s) with cross-validation
5. Produce per-genotype and WT-vs-mutant box plots

Usage
-----
python scripts\\run_classification.py ^
    --tracks     outputs\\my_run\\compact_tracks.csv ^
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
        description="Classification pipeline: compact_tracks.csv -> results + figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracks", required=True,
                   help="Path to compact_tracks.csv")
    p.add_argument("--output-dir", required=True,
                   help="Directory for figures and outputs")
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
    return p


def main():
    args = build_parser().parse_args()

    # Defer heavy imports until after --help is resolved
    from src.classification import (
        map_vial_to_genotype,
        run_classifier,
        plot_by_genotype,
        plot_wt_vs_mutant,
    )
    from src.features import extract_behavioral_features, aggregate_per_fly_features

    print(f"\nLoading tracks from: {args.tracks}")
    df_raw = map_vial_to_genotype(args.tracks)

    print("Extracting behavioural features...")
    df_feat = extract_behavioral_features(df_raw)

    print("Aggregating per-fly features...")
    df_agg = aggregate_per_fly_features(df_feat, pause_threshold=args.pause_threshold)

    # Merge genotype back onto per-fly aggregation
    genotype_map = df_raw.drop_duplicates("compact_id").set_index("compact_id")["genotype"]
    df_agg["genotype"] = df_agg["compact_id"].map(genotype_map)
    df_agg = df_agg.dropna(subset=["genotype"])

    hover_data = ["compact_id"]

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    for model_name in args.models:
        for mode in args.modes:
            print(f"\n=== {model_name.upper()} [{mode}] ===")
            run_classifier(
                df=df_agg,
                outdir=args.output_dir,
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
        plot_by_genotype(df_agg, FEATURES, FEATURE_TITLES, hover_data, outdir=args.output_dir)

        print("=== WT vs Mutant box plots ===")
        plot_wt_vs_mutant(df_agg, FEATURES, FEATURE_TITLES, hover_data, outdir=args.output_dir)

    print("\nDone. Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
