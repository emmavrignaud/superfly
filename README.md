


![fly-insect-hands](https://github.com/user-attachments/assets/7118b6f8-30ee-4a85-b593-62c744745e03)


# Fly Tracking + Genotype Classification (ML4Science / McCabe Lab)

This project implements an end-to-end pipeline to detect, track, and analyze individual Drosophila in climbing assay videos (RING-style). The motivation is to quantify how TDP-43 mutations, associated with ALS, affect fly motricity beyond the average velocity output of the original FreeClimber workflow.

The pipeline produces:
- per-frame trajectory CSVs with consistent fly identities as much as possible  
- an overlay video for qualitative validation of tracking  
- per-fly behavioral features and baseline genotype classification results  

---

## Repository contents

bibii


---


---

## Pipeline overview

### 1. Optional preprocessing: ROI selection and background subtraction

An optional preprocessing step improves detection quality by reducing background clutter:
- interactive selection of a region of interest and frame range using an OpenCV GUI  
- computation of an average background from the selected frames  
- generation of a preprocessed grayscale video where flies appear with higher contrast  

Main function:
- preprocess_bgsub_gui_cv2_avg_background(video_path, ...)

Output:
- a preprocessed mp4 video (default suffix `_pp`)

---

### 2. Detection and tracking (RF-DETR + OC-SORT)

Each frame is processed by an object detector and the resulting detections are linked over time using a tracker.

Detection:
- performed using a RF-DETR model accessed via inference.get_model
- configurable confidence threshold

Tracking:
- performed with OC-SORT from the boxmot library
- tuned to tolerate fast, non-linear motion and short detection gaps
- relies on motion consistency rather than appearance cues, which is appropriate for visually indistinguishable flies

Main function:
- export_tracks_xy_tuple_csv_one_config(...)

Output format:
- wide CSV where each row corresponds to a frame
- each track ID has its own column
- cells contain "(x, y)" coordinates of bounding-box centers or are empty if the track is absent

Output file:
- tracks_wide_format.csv

---

### 3. Manual vial ROI definition

To enable per-vial analysis, rectangular regions corresponding to the six vials are manually defined:
- ROIs are drawn interactively on a reference frame
- ROIs are assumed to be fixed throughout the video
- ROIs are saved in left-to-right order

Main function:
- draw_and_save_vial_rois(video_path, roi_json_path, n_vials=6)

Output file:
- vial_rois.json

---

### 4. Offline stitching of tracklets

Even with optimized tracking, fly identities may fragment due to occlusions, missed detections, or rapid motion. To address this, fragmented tracklets are stitched offline.

Procedure:
- convert the wide tracking CSV into a long format (frame, id, x, y)
- summarize each original track ID into a tracklet with start/end frames and positions
- estimate a characteristic step size from the data
- compute a cost for linking pairs of tracklets based on spatial distance and temporal gap
- solve a one-to-one assignment problem to link tracklets consistently

Assignment is solved using the Hungarian algorithm when available, with a greedy fallback otherwise.

Main function:
- stitch_wide_csv_to_long(input_csv, output_stitched_long, max_gap, ...)

Output file:
- tracks_xy_stitched_long.csv, including a stitched_id column

---

### 5. Vial assignment and compact ID relabeling

After stitching:
- each detection is assigned to a vial based on the ROI definitions
- detections outside all vials are discarded
- stitched IDs are relabeled into compact consecutive IDs, ordered left-to-right within each vial
- the video frame rate can be attached to facilitate kinematic analysis

Main function:
- assign_vials_and_compact_ids(stitched_csv, roi_json, out_csv, fps=...)

Output file:
- compact_tracks.csv

---

### 6. Overlay video generation

For qualitative validation, an overlay video is rendered:
- fly positions are drawn on the original video
- colors encode vial identity
- shading encodes compact fly ID within each vial
- optional numeric ID labels can be displayed

Main function:
- render_vial_overlay_video(video_path, csv_path, out_mp4, ...)

Output file:
- overlay_vials_shaded.mp4

---

## Feature extraction and analysis

From compact_tracks.csv, frame-level kinematics and trajectory-level features are computed.

Frame-level quantities:
- displacement, velocity, acceleration
- turning angle and angular velocity
- cumulative distance traveled

Trajectory-level quantities:
- area covered (convex hull)
- path tortuosity

Main functions:
- extract_behavioral_features(df)
- aggregate_per_fly_features(df_features, pause_threshold=1.0)

---

## Statistical analysis and classification

The extracted per-fly features are used for:
- genotype-wise comparisons using non-parametric statistics
  - Kruskal–Wallis tests
  - Mann–Whitney U tests with Cliff’s delta
- simple classification baselines to distinguish genotypes

Classifiers implemented:
- Linear Discriminant Analysis
- Logistic Regression
- Linear Support Vector Classifier

All classifiers are run within a standardized pipeline with feature scaling and cross-validation.

Main function:
- run_classifier(df_fly, model_name="lda", classification_mode="multiclass")

Figures are saved as both static PNGs and interactive HTML files.

---

## Libraries and dependencies

Core processing:
- opencv-python (cv2) for video I/O, GUI tools, and rendering
- numpy for numerical operations
- pandas for CSV handling and data manipulation
- standard Python libraries: os, pathlib, json, ast, math, logging, warnings, dataclasses, typing

Detection and tracking:
- inference for loading and running the RF-DETR detection model
- supervision for handling detector outputs
- boxmot for the OC-SORT tracker

Stitching, geometry, and statistics:
- scipy.optimize.linear_sum_assignment for Hungarian assignment
- scipy.stats for non-parametric statistical tests
- scipy.spatial.ConvexHull for trajectory area estimation

Classification:
- scikit-learn for preprocessing, models, and cross-validation

Visualization:
- plotly for statistical plots and model evaluation figures
- kaleido for exporting Plotly figures to image files

Notebook utilities:
- IPython.display for inline video and image display

---

## Installation

A minimal setup can be created as follows:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -U pip
pip install opencv-python numpy pandas supervision boxmot scipy scikit-learn plotly kaleido inference
