# Drosophila Climbing Behaviour — Fly Tracking & Genotype Classification

End-to-end pipeline: raw video → tracked trajectories → ML genotype classification.

---

![fly-insect-hands](https://github.com/user-attachments/assets/7118b6f8-30ee-4a85-b593-62c744745e03)


## Quick-start

```bash
# 1. Clone / copy the repo
git clone <your-repo-url>
cd Claude_Professional_Repo

# 2. Create environment (conda recommended)
conda env create -f environment.yml
conda activate fly-tracking

# — OR — plain pip
pip install -r requirements.txt

# 3. Run the tracking pipeline
python scripts/run_tracking.py \
    --video      data/my_experiment.mp4 \
    --output-dir outputs/my_run \
    --api-key    YOUR_ROBOFLOW_KEY \
    --model-id   YOUR_MODEL_ID

# 4. Run classification
python scripts/run_classification.py \
    --tracks     outputs/my_run/compact_tracks.csv \
    --output-dir outputs/my_run/classification
```

Both scripts respect `config.yaml` defaults and accept `--help` for all options.

---

## Expected data folder layout

```
data/
├── my_experiment.mp4        # raw climbing video
└── ...
outputs/
└── my_run/
    ├── tracks_wide_format.csv
    ├── tracks_xy_stitched_long.csv
    ├── vial_rois.json
    ├── compact_tracks.csv
    ├── overlay_vials_stitched.mp4
    └── classification/
        └── report_figures/
```

Data files are **not committed** (too large). Only `data/.gitkeep` and
`outputs/.gitkeep` are tracked so the folders exist after a fresh clone.

---

## Module overview

| Module | Responsibility |
|---|---|
| `src/preprocessing.py` | Interactive background-subtraction GUI |
| `src/tracking.py` | Roboflow RF-DETR + OC-SORT → wide CSV |
| `src/stitching.py` | Hungarian track stitching across gaps |
| `src/roi.py` | Vial ROI drawing, assignment, compact IDs |
| `src/features.py` | Kinematics, area, tortuosity, aggregation |
| `src/classification.py` | LDA/Logistic/SVC classifiers + Plotly figures |
| `src/visualization.py` | Overlay video rendering |

All public functions are re-exported from `src/__init__.py`:

```python
from src import export_tracks_xy_tuple_csv_one_config
from src import stitch_wide_csv_to_long, assign_vials_and_compact_ids
from src import extract_behavioral_features, run_classifier
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_tracking_pipeline.ipynb` | Run & inspect each tracking stage |
| `notebooks/02_classification_analysis.ipynb` | Feature exploration + genotype classification |

---

## Configuration

Edit `config.yaml` to tune tracker parameters, stitching penalties, etc.
CLI flags always override the config file.

---

## Citation / acknowledgements

- RF-DETR: [Roboflow](https://roboflow.com)
- OC-SORT: [boxmot](https://github.com/mikel-brostrom/boxmot)
- Dataset: EPFL fly-climbing assay (hTDP43 mutant line)
