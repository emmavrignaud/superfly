"""
src/__init__.py

Re-exports the public API so users can do:
    from src import preprocess_bgsub_gui
    from src import export_tracks_xy_tuple_csv_one_config
    from src import wide_to_long, build_tracklets, stitch
    from src import assign_vials_and_compact_ids
    from src import extract_behavioral_features, run_classifier
"""

from src.preprocessing import (
    gui_pick_roi_and_range,
    preprocess_bgsub_gui,
)

from src.tracking import (
    export_tracks_xy_tuple_csv_one_config,
)

from src.stitching import (
    wide_to_long,
    Tracklet,
    build_tracklets,
    simulate_position,
    link_score,
    build_cost_matrix,
    stitch_per_vial,
    stitch_general,
    stitch,
)

from src.roi import (
    draw_and_save_vial_rois,
    assign_compact_ids_left_to_right,
    assign_vials_and_compact_ids,
)

from src.features import (
    add_kinematics,
    add_area_covered,
    add_path_tortuosity,
    extract_behavioral_features,
    aggregate_per_fly_features,
)

from src.classification import (
    map_vial_to_genotype,
    make_classifier,
    prepare_xy,
    prepare_target,
    run_cross_validation,
    plot_feature_importance,
    run_classifier,
    cliffs_delta,
    save_plotly_figure,
    plot_by_genotype,
    plot_wt_vs_mutant,
)

from src.visualization import (
    render_vial_overlay_video,
)

__all__ = [
    # preprocessing
    "gui_pick_roi_and_range",
    "preprocess_bgsub_gui",
    # tracking
    "export_tracks_xy_tuple_csv_one_config",
    # stitching
    "wide_to_long",
    "Tracklet",
    "build_tracklets",
    "simulate_position",
    "link_score",
    "build_cost_matrix",
    "stitch_per_vial",
    "stitch_general",
    "stitch",
    # roi
    "draw_and_save_vial_rois",
    "assign_compact_ids_left_to_right",
    "assign_vials_and_compact_ids",
    # features
    "add_kinematics",
    "add_area_covered",
    "add_path_tortuosity",
    "extract_behavioral_features",
    "aggregate_per_fly_features",
    # classification
    "map_vial_to_genotype",
    "make_classifier",
    "prepare_xy",
    "prepare_target",
    "run_cross_validation",
    "plot_feature_importance",
    "run_classifier",
    "cliffs_delta",
    "save_plotly_figure",
    "plot_by_genotype",
    "plot_wt_vs_mutant",
    # visualization
    "render_vial_overlay_video",
]
