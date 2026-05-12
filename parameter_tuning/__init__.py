"""HOTA evaluation + hyperparameter-tuning infrastructure.

Three layers:
    build_mot_files.build(...)  -- write MOTChallenge-format files for one
                                   (video, tracker) pair.
    run_hota.score(...)         -- run TrackEval on a prepared mot_inputs dir
                                   and return the metrics dict.
    score_run.score_run(...)    -- end-to-end: take a tracker run folder, find
                                   its GT in the labeler session folder, build
                                   MOT files, score, write hota.json. Returns
                                   None when no GT exists for the run's video.

Import from submodules explicitly:
    from parameter_tuning.score_run import score_run
    from parameter_tuning.build_mot_files import build
    from parameter_tuning.run_hota import score
"""
