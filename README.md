# Drosophila Climbing Behaviour: Fly Tracking and Genotype Classification

End-to-end pipeline for the EPFL fly-climbing assay: raw video in, per-fly
trajectories and genotype predictions out.

![fly-insect-hands](https://github.com/user-attachments/assets/7118b6f8-30ee-4a85-b593-62c744745e03)


## Pipeline at a glance

```mermaid
flowchart LR
    A[Raw video<br/>*.mp4] --> B[Preprocessing<br/>crop + temporal trim<br/>background subtraction]
    B --> C[Vial ROIs<br/>vial_rois.json]
    A --> C
    B --> D[RF-DETR detection<br/>+ OC-SORT tracking<br/>ocsort_tracks.csv]
    C --> D
    D --> E[Wide to long<br/>ocsort_tracks_long.csv]
    E --> F[Vial assignment<br/>+ ordered IDs<br/>ordered_tracks.csv]
    C --> F
    D --> G[Diagnostics<br/>metrics_report.html]
    F --> G
    F --> H[Track Overlays<br/>overlay_ordered.mp4]
    F --> I[Behavioural features<br/>kinematics, area, tortuosity]
    I --> J[Genotype classification<br/>LDA / Logistic / SVC]
```

Every step except detection and classification runs locally with no GPU.
Detection calls Roboflow's hosted RF-DETR over HTTP; classification is pure
scikit-learn.


## Quick start

```bash
git clone <your-repo-url>
cd superfly

# Conda (recommended):
conda env create -f environment.yml
conda activate fly-tracking

# Or pip:
# pip install -r requirements.txt

# Add a Roboflow API key for detection:
cp creds_config.example.yaml creds_config.yaml   # then paste the key into creds_config.yaml
```

`creds_config.example.yaml` is the template; the real `creds_config.yaml` is
gitignored.


## Running the pipeline

A run is defined entirely in **`config.yaml`**: the video to track, the tracker
settings, and which stages run. Edit `config.yaml`, then launch it. The notebook
and scripts only read config; you do not edit them to change a run.

### Step 1: configure the run in `config.yaml`

Sections are grouped **Meta** (inputs and environment), **Pipeline** (stages in
execution order), and **Downstream analysis**:

| Section | What it controls |
|---|---|
| `video` | `raw_path` (the clip to track) + `fallback_fps` when metadata is missing |
| `roboflow` | `model_id` and `inference_api_url` for RF-DETR detection |
| `calibration` | `px_per_cm` + output units (cm/px, s/frame) for behavioural features |
| `preprocessing` | `enabled` + background subtraction (percentile, gain, white level, codec) |
| `roi` | Whether to reuse saved vial/crop ROIs or always reopen the GUI |
| `tracker` | RF-DETR confidence, every OC-SORT knob (jump round, behavioral weights), `cached_detections`, and `ghost_detection` |
| `watershed` | Splitting detection boxes that contain two touching flies |
| `pipeline` | `expected_per_vial`: flies per vial, for diagnostics and ghost gating |
| `visualization` | Track Overlays (`enabled`, default false) + `overlay_source`, fps, style |
| `features` | Optional kinematic feature families |
| `latent_space` | PCA / t-SNE / UMAP embedding settings |
| `classification` | Active classifier (`method`) + per-backend hyperparameters |

### Step 2: launch it

Every script reads `config.yaml`. Tracking output goes to
`data/outputs/run_<N>_<short_name>/`. CLI flags affect the current run only; they
do not edit config.

There are four ways to run the tracking pipeline. They share the same engine, the
same `config.yaml`, and the same outputs, so pick by how much GUI you want.

#### Option 1: `scripts/app.py` (setup window)

One window that stays open for the whole setup: pick a video with Browse, draw
the crop and frame trim, then draw each vial and set its fly count and colour. Two
checkboxes control the run (write overlays, reuse a saved ROI). Clicking finish hands off the run to `run_tracking.py`

```bash
python scripts/app.py                            # pick the video inside the window
python scripts/app.py --video data/raw/clip.mp4  # preselect a video, skip the first Browse
```



#### Option 2: `notebooks/01_tracking_pipeline.ipynb`

You can use this notebook if you prefer seeing the pipeline stages one by one. 

```bash
jupyter lab notebooks/01_tracking_pipeline.ipynb
```

#### Option 3: `scripts/run_tracking.py`

One video, full pipeline, driven from the command line.

```bash
python scripts/run_tracking.py                            # track config.video.raw_path
python scripts/run_tracking.py --video data/raw/clip.mp4  # track a different file instead
python scripts/run_tracking.py --reuse-roi                # reuse saved crop + ROIs, no drawing GUI
python scripts/run_tracking.py --overlay                  # also write Track Overlays this run
python scripts/run_tracking.py --no-overlay               # skip overlays even if config enables them
```

With no overlay flag, overlays follow `visualization.enabled` in config. The
`--reuse-roi` flag is what `app.py` passes on handoff, and it is also handy for
scripted runs off an already saved ROI.

#### Option 4: `scripts/run_all.py`

Stage 1 tracks video(s); Stage 2 runs behavioural significance analysis on the
tracked runs. Every run is fresh by default, so each video is re-tracked. Set
`pipeline.skip_tracked: true` in `config.yaml` to skip videos that already have
`ordered_tracks.csv`, which is useful for resuming an interrupted batch.

```bash
python scripts/run_all.py --data-root data/raw              # track every video in the folder, then analyse
python scripts/run_all.py --data-root data/raw --draw-first # draw all crop + ROIs upfront, then track unattended
```


#### `scripts/run_classification.py`

Classify genotypes from a finished run folder (`ordered_tracks.csv` and
`run_params.json` required). Classifier settings come from config.

```bash
python scripts/run_classification.py --run-dir data/outputs/run_5_13DPE_n001  # required: the run folder
python scripts/run_classification.py --run-dir <run> --report-site            # also write an HTML report
```


## Outputs

Each tracking run produces:

```
data/outputs/run_<N>_<short_name>/
├── <video_name>.mp4                          # hardlink/copy of the raw input
├── crop_roi.json                             # crop window + temporal trim (if preprocessed)
├── vial_rois.json                            # per-vial bounding boxes
├── detections_raw.csv                        # raw RF-DETR detections (detection cache)
├── ocsort_tracks.csv                         # tracker output (wide format)
├── ocsort_tracks_long.csv                    # long format, one row per (frame, id)
├── ordered_tracks.csv                        # vial-assigned, left-to-right ordered_id
├── tracker_log.json                          # detection, suppression + ghost/exit log
├── run_params.json                           # full param snapshot per stage
├── metrics_report.{md,html}                  # diagnostics report
├── <video_stem>_detections_RF-DETR.mp4       # Track Overlays (when enabled or --overlay)
├── overlay_raw_ocsort.mp4
└── overlay_ordered.mp4
```

`data/outputs/` and `data/annotations/` are gitignored. Only `.gitkeep` stubs are tracked.


## Repository layout

```
superfly/
├── config.yaml                # the one file you edit to define a run
├── creds_config.example.yaml  # Roboflow key template; copy to creds_config.yaml
├── environment.yml            # conda env spec
├── requirements.txt           # pip mirror
├── utils.py                   # shared helpers (config loading, run outputs)
├── roi_library.json           # cached crop + vial ROIs keyed by video stem
├── RF-DETR_model/             # trained fly-detection model
│   └── weights.pt
├── external/                  # vendored TrackEval for HOTA / MOT scoring
├── data/                      # raw/ input videos + outputs/ tracked runs
├── notebooks/
│   ├── 01_tracking_pipeline.ipynb
│   └── 02_classification_analysis.ipynb
├── scripts/
│   ├── app.py                 # setup window: crop + vial ROIs + colours, hands off to run_tracking
│   ├── run_tracking.py        # single video, full render
│   ├── run_all.py             # batch tracking (+ --draw-first) then analysis
│   ├── run_classification.py  # genotype classification on ordered_tracks
│   ├── populate_roi_library.py
│   └── export_to_roboflow.py  # dataset export for model training
├── src/
│   │   # -- tracking pipeline --
│   ├── preprocessing.py       # background subtraction + crop/trim GUI
│   ├── roi.py                 # vial ROI drawing + assign_vials_and_ordered_ids
│   ├── tracking.py            # RF-DETR + OC-SORT orchestration
│   ├── watershed_split.py     # split boxes holding two touching flies
│   ├── wide_long.py           # wide -> long format conversion
│   ├── visualization.py       # overlay video rendering
│   ├── metrics.py             # run_diagnostics, per-vial counts, HOTA
│   │   # -- vendored tracker (OC-SORT, lightly modified) --
│   ├── ocsort.py              # OC-SORT implementation
│   ├── kalmanfilter.py        # Kalman filter used per track
│   ├── association.py         # IoU variants + Hungarian matching
│   │   # -- downstream behavioural analysis --
│   ├── features.py            # kinematics, area, tortuosity
│   ├── statistics.py          # Kruskal-Wallis + Cliff's delta significance
│   ├── classification.py      # LDA / Logistic / SVC + plots
│   ├── latent_space.py        # PCA / t-SNE / UMAP embeddings
│   ├── forecasting.py         # next-step movement MLP (genotype-conditioned)
│   ├── plot_colors.py         # stable per-genotype Plotly colours
│   └── ui_context.py          # PyQt GUI helpers
├── parameter_tuning/          # ground-truth, grid search + HOTA scoring
├── labeler/                   # standalone detection-labelling GUI
├── tests/                     # hermetic unit tests (pytest)
└── legacy/                    # deprecated post-hoc stitching (not used)
```


## Verifying the setup

```bash
pytest tests/ -v
```

The tests need no Roboflow key, no videos, and no GPU, so a green run confirms
the environment is installed correctly.


## Acknowledgements

- RF-DETR: [Roboflow](https://roboflow.com)
- OC-SORT: [boxmot](https://github.com/mikel-brostrom/boxmot)
- Dataset: EPFL fly-climbing assay (hTDP43 mutant line)
