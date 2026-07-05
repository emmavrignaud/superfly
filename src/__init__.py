"""
src/__init__.py

Re-exports the public API so users can do:
    from src import preprocess_bgsub_gui
    from src import export_tracks_xy_tuple_csv_one_config
    from src import wide_to_long
    from src import assign_vials_and_ordered_ids
    from src import extract_behavioral_features, run_classifier
"""

from src.preprocessing import (
    capture_crop_params_gui,
    gui_pick_roi_and_range,
    preprocess_bgsub_gui,
)

from src.tracking import (
    export_tracks_xy_tuple_csv_one_config,
)

from src.stitching import (
    wide_to_long,
)

from src.roi import (
    draw_and_save_vial_rois,
    load_vial_rois,
    assign_ordered_ids_left_to_right,
    assign_vials_and_ordered_ids,
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
    genotype_category_order,
    make_figure_by_genotype,
    make_figure_wt_vs_mutant,
    plot_by_genotype,
    plot_wt_vs_mutant,
    write_classification_html_report,
)

from src.visualization import (
    render_vial_overlay_video,
)

__all__ = [
    # preprocessing
    "capture_crop_params_gui",
    "gui_pick_roi_and_range",
    "preprocess_bgsub_gui",
    # tracking
    "export_tracks_xy_tuple_csv_one_config",
    # stitching (wide_to_long still used for format conversion)
    "wide_to_long",
    # roi
    "draw_and_save_vial_rois",
    "load_vial_rois",
    "assign_ordered_ids_left_to_right",
    "assign_vials_and_ordered_ids",
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
    "genotype_category_order",
    "make_figure_by_genotype",
    "make_figure_wt_vs_mutant",
    "plot_by_genotype",
    "plot_wt_vs_mutant",
    "write_classification_html_report",
    # visualization
    "render_vial_overlay_video",
]
