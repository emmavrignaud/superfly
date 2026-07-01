# Superfly Analysis Pipeline — Overview for Collaborators

This document describes the full pipeline: from raw videos to behavioral features,
statistical significance, representation learning, and movement forecasting.
It is intended to help a collaborator understand what has been built, what the
results mean, and where to focus for making publication-quality plots.

---

## Experiment Design

**Animal model:** *Drosophila melanogaster* expressing human TDP-43 variants
associated with ALS (amyotrophic lateral sclerosis).

**Genotypes (one per vial, 6 vials per video):**
| Label | Mutation | Notes |
|-------|----------|-------|
| WT | Wild-type (no mutation) | Control |
| A90V | TDP-43 A90V | Mild |
| G287S | TDP-43 G287S | |
| G294A | TDP-43 G294A | |
| A315T | TDP-43 A315T | |
| M337V | TDP-43 M337V | Strongest phenotype |

**Time points:** 6, 9, 13, 16, 21, 24, 28, 31, 35, 41 days post-eclosion (DPE).
6 replicate videos per time point = ~59 videos total.

**Video format:** ~300–500 frames, ~30 fps, 1024×768 px.
Calibration: **29.0 px/cm**, outputs in cm and seconds.

---

## Step 1 — Video Preprocessing

Each raw video is:
1. **Cropped** to the vial region (stored in `roi_library.json` under `preprocessing`).
2. **Background-subtracted** using a per-pixel temporal median (85th percentile over
   sampled frames, stride=30). Output: `*_pp.mp4` (enhanced) + `*_raw_cropped.mp4`.

---

## Step 2 — Object Detection + Tracking

### Detection
**RF-DETR** (Roboflow Detection Transformer) detects individual flies per frame.
- Primary confidence threshold: **0.55**
- Low-confidence range [0.10, 0.55) feeds a secondary "jump" pass only.

### Tracking — OC-SORT with behavioral extensions
Detected bounding boxes are linked across frames using **OC-SORT**
(Observation-Centric SORT), extended with:

**Behavioral cost matrix additions:**
| Weight | Feature | Value |
|--------|---------|-------|
| `speed` | Speed consistency between detection and tracker | 0.2 |
| `turning_angle` | Heading change consistency | 0.2 |
| `pause` | Pause state consistency | 0.2 |
| `acceleration` | Acceleration consistency | 0.2 |
| `overlap_weight_scale` | Boost all weights when flies overlap | 6× |

**Association parameters (tuned via grid search over 165,888 configs):**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `minimum_matching_threshold` | 0.05 | Min score to keep a match |
| `inertia` | 0.0 | Direction consistency weight |
| `delta_t` | -1 | Use full history for velocity estimate |
| `jump_factor` | 2.0 | Search radius multiplier for occluded flies |
| `lost_track_buffer` | 400 frames | ~13s before dropping a lost track |

**Grid search result:** HOTA improved from 0.764 → **0.806** (+5.5%),
ID-switches reduced from 1358 → **1320** across two GT-annotated videos.

**Two-pass ghost detection:** when a fly disappears behind another,
a synthetic "ghost" detection is injected to prevent ID loss.

**Vial assignment:** after tracking, each track is assigned to a vial
(and therefore a genotype) based on its median x-position relative to
manually drawn vial ROI bounding boxes.

**Output:** `ordered_tracks.csv` — one row per (fly × frame) with columns:
`ordered_id`, `vial_id`, `genotype`, `frame`, `x`, `y`, `fps`.

---

## Step 3 — Kinematic Feature Extraction

From `ordered_tracks.csv`, kinematics are computed per frame in `src/features.py`:

```
dx, dy, speed, heading, angular_velocity, acceleration_x, acceleration_y
```

Then **per-fly aggregate features** are computed (one row per fly):

### Movement activity
| Feature | Definition |
|---------|-----------|
| `pause_fraction` | Fraction of frames where speed < 1 cm/s |
| `pause_count` | Number of contiguous pause episodes |
| `mean_pause_duration` | Mean duration of pause episodes (s) |
| `max_pause_duration` | Longest single pause (s) |
| `latency_to_first_movement` | Seconds before first non-pause frame |
| `mean_speed_early` | Mean speed in first 10 s of recording |

### Burst (active movement) episodes
| Feature | Definition |
|---------|-----------|
| `burst_count` | Number of contiguous active episodes (speed ≥ 1 cm/s) |
| `mean_burst_duration` | Mean burst duration (s) |
| `mean_burst_speed` | Mean speed *within* bursts (cm/s) |

### Velocity / acceleration statistics
| Feature | Definition |
|---------|-----------|
| `mean_velocity`, `median_velocity`, `std_velocity` | Distribution of speed |
| `mean_velocity_x/y`, `std_velocity_x/y`, `median_velocity_x/y` | Per-axis |
| `mean_acceleration`, `std_acceleration_x/y`, `median_acceleration_y` | Acceleration stats |

### Path shape
| Feature | Definition |
|---------|-----------|
| `total_distance_traveled` | Cumulative path length (cm) |
| `total_distance_x`, `total_distance_y` | Axis-specific distance |
| `tortuosity` | Path length / straight-line displacement (1 = straight) |
| `area_covered` | Convex hull area of trajectory (cm²) |

### Turning / direction
| Feature | Definition |
|---------|-----------|
| `mean_abs_turning_angle` | Mean absolute frame-to-frame heading change (°) |
| `mean_abs_angular_velocity` | Mean absolute angular velocity (°/s) |
| `reversal_rate` | Fraction of frames with heading change > 150° |

**Total: 35 features** (with `KINEMATIC_THREE_FAMILIES=True`).

---

## Step 4 — Statistical Significance Testing

All tests in `src/statistics.py`. For each feature:

1. **Kruskal-Wallis** (non-parametric ANOVA across all 6 genotypes)
2. **Benjamini-Hochberg FDR correction** on the 35 KW p-values (α = 0.05)
3. **Mann-Whitney U** pairwise (WT vs. each mutant) — not FDR corrected,
   used for directionality only
4. **Cliff's delta** effect size per pair:
   `δ = [#(mutant > WT) - #(mutant < WT)] / (n_mutant × n_WT)` ∈ [-1, 1]
   - δ > 0 → mutant *higher* than WT
   - δ < 0 → mutant *lower* than WT
   - |δ| > 0.33 = medium effect, |δ| > 0.47 = large effect

### Results on full dataset (2,218 flies, 59 videos)

**31 / 35 features significant** (FDR < 0.05).

Top features by effect size (|Cliff's δ| vs. strongest mutant):

| Feature | KW p (adj) | Max |δ| | Strongest contrast |
|---------|-----------|------|-------------------|
| `pause_count` | 2.3e-31 | 0.531 | M337V lower |
| `burst_count` | 8.6e-31 | 0.527 | M337V lower |
| `total_distance_x` | 3.8e-25 | 0.481 | M337V lower |
| `total_distance_traveled` | 3.4e-17 | 0.389 | M337V lower |
| `tortuosity` | 3.3e-14 | 0.381 | M337V lower |
| `mean_burst_duration` | 3.4e-13 | 0.345 | M337V higher |
| `mean_abs_angular_velocity` | 3.0e-12 | 0.346 | M337V lower |
| `median_velocity` | 1.2e-12 | 0.306 | M337V higher |
| `mean_velocity_y` | 1.2e-12 | 0.329 | M337V lower |
| `std_acceleration_y` | 1.8e-11 | 0.191 | A315T higher |

**Pattern:** M337V is the most severely affected genotype — fewer pauses, fewer
bursts, less total movement, less tortuosity. This is consistent with progressive
motor impairment. A315T shows a distinct pattern (higher acceleration variability).

**Not significant (4 / 35):**
`mean_pause_duration`, `median_acceleration_y`, `mean_velocity_x`, and one other.

---

## Step 5 — Significance Report (HTML)

The interactive HTML report at `outputs/analysis/significance_report/significance_report.html` contains:
- Summary stat cards (n flies, n runs, n significant)
- Volcano plot (effect size vs. -log10 p-value)
- Effect size heatmap (all genotypes × all features)
- Box plots per significant feature, coloured by genotype
- Full results table with all p-values and effect sizes

**Genotype colour scheme:**
```
WT     = #4361ee (blue)
A90V   = #f72585 (pink)
G287S  = #7209b7 (purple)
G294A  = #3a0ca3 (dark purple)
A315T  = #4cc9f0 (cyan)
M337V  = #f77f00 (orange)
```

---

## Step 6 — Representation Learning (Autoencoder)

`src/latent_space.py` trains a **multi-task autoencoder** to embed each fly's
behavior into a low-dimensional latent space that is simultaneously informed by
age, position in the recording, and genotype.

### Architecture

```
Input X
  └─ Encoder: Linear(input_dim → 512) → ReLU → Linear(512 → 256) → ReLU
                → Linear(256 → latent_dim=64) → ReLU
                                ↓
                          z (64-dim)
                                ↓
  ┌─────────────────────────────────────────────────┐
  │ Decoder:  Linear(64 → 256) → ReLU              │  ← reconstruction head
  │           → Linear(256 → 512) → ReLU           │
  │           → Linear(512 → input_dim)             │
  └─────────────────────────────────────────────────┘
  ┌──────────────────────┐
  │ age_head:            │  ← predicts DPE class (6/9/13/16/21/24/28/31/35/41d)
  │   Linear(64 → 64)    │
  │   → ReLU → Linear(64 → n_age_classes)           │
  └──────────────────────┘
  ┌──────────────────────┐
  │ time_bin_head:       │  ← predicts third of recording (early/mid/late)
  │   Linear(64 → 32)    │
  │   → ReLU → Linear(32 → n_time_bins=3)           │
  └──────────────────────┘
  ┌──────────────────────┐
  │ genotype_head:       │  ← predicts genotype (WT/A90V/G287S/G294A/A315T/M337V)
  │   Linear(64 → 32)    │
  │   → ReLU → Linear(32 → 6)                       │
  └──────────────────────┘
```

**Loss:** `L = L_recon + 1.0×L_age + 0.5×L_time + 0.5×L_genotype`

All cross-entropy for classification heads. Pure NumPy + Adam optimizer.

**Training:** 300 epochs, batch 64, lr=1e-3, early stopping patience=30.

### Four analysis variants

| Analysis | Input | Purpose |
|----------|-------|---------|
| A1 | Raw (x,y) trajectory + aggregate features | Does trajectory shape separate genotypes? |
| A2 | Kinematic histograms (speed, turning, accel) | Does distribution shape matter? |
| A3 | Aggregate features only | Compact feature-based embedding |
| A4 | Features split into time bins (early/mid/late) | Does behavior change within a recording? |

**Embeddings:** t-SNE (perplexity=20), UMAP (n_neighbors=5), PCA.
**Significance:** PERMANOVA pseudo-F on pairwise distance matrix.

---

## Step 7 — Movement Forecasting (ForecastMLP)

`src/forecasting.py` asks: *can we predict a fly's next displacement from its
recent trajectory history, and does knowing its genotype help?*

### Architecture

```
Input: window of W=20 frames × 5 features
       + genotype one-hot (6 dims)
       = 106-dimensional input vector

ForecastMLP:
  Linear(106 → 256) → ReLU → Dropout(0.1)
  → Linear(256 → 128) → ReLU → Dropout(0.1)
  → Linear(128 → 2)   ← output: (dx_per_s, dy_per_s)
```

**Window features:** `[dx_per_s, dy_per_s, speed, heading, angular_velocity]`

**Training:** MSE on next-frame (dx, dy), Adam lr=1e-3, 200 epochs,
early stopping patience=20, batch size 256.

### Ablation study
Three models are compared:
1. **Constant-velocity baseline** — just repeat the last (dx, dy)
2. **MLP without genotype** — trajectory only
3. **MLP with genotype** — full model

**Result:** Genotype conditioning gives **11–23% MSE reduction** per genotype
compared to the no-genotype model. This suggests that genotype encodes
systematic differences in movement dynamics beyond what trajectory alone captures.

> ⚠️ Note: the current evaluation does not use leave-one-run-out cross-validation,
> so the per-genotype gain numbers may be slightly optimistic. Proper CV is a
> recommended next step.

---

## File Structure

```
superfly/
├── data/
│   └── raw/                  # Raw videos, organised by DPE + replicate
├── outputs/
│   ├── run_NNN_<DPE>_n<rep>/ # One folder per tracked video
│   │   ├── ordered_tracks.csv        # Main output: per-fly per-frame tracks
│   │   ├── tracks_long_format.csv    # Wide→long intermediate
│   │   ├── ocsort_tracks.csv         # Raw tracker output (wide format)
│   │   ├── detections_raw.csv        # RF-DETR detections
│   │   ├── vial_rois.json            # Vial bounding boxes
│   │   └── run_params.json           # Config snapshot
│   └── analysis/
│       └── significance_report/
│           └── significance_report.html   # ← Main results report
├── src/
│   ├── features.py       # Feature extraction
│   ├── statistics.py     # Significance tests + HTML report
│   ├── latent_space.py   # Autoencoder + embeddings
│   ├── forecasting.py    # ForecastMLP
│   ├── tracking.py       # OC-SORT + RF-DETR pipeline
│   ├── roi.py            # Vial ROI drawing + assignment
│   └── preprocessing.py  # Background subtraction
├── scripts/
│   ├── run_all.py        # Batch tracking + analysis
│   └── draw_rois.py      # ROI drawing GUI
├── roi_library.json      # Stored crop + vial ROI params for all videos
└── config.yaml           # All pipeline parameters
```

---

## Suggested Next Steps for Plots

The HTML report already has interactive box plots, but for publication consider:

1. **Box plots per feature, grouped by genotype** — use the `outputs/analysis/significance_report/` Plotly figures or re-plot with seaborn/matplotlib for finer control. Data is in `df_agg` (one row per fly, columns = features + genotype).

2. **Trajectory heat maps** — overlay all fly paths per genotype, coloured by speed, to visually show the movement difference.

3. **Age × genotype interaction** — features like `pause_count` likely change with DPE. A line plot of feature mean ± SEM vs. DPE, one line per genotype, would show disease progression.

4. **Latent space UMAP** — colour points by genotype and age. Check whether WT and mutants separate, and whether there's an age gradient.

5. **Feature correlation matrix** — many features are correlated (pause_count and burst_count are almost mirror images). A clustered heatmap would help identify independent behavioral dimensions.

6. **Cliff's delta heatmap** — genotypes × features, coloured by δ direction and size. Already in the report but can be made publication-quality.

---

## How to Re-run the Analysis

```bash
# Just the classification/significance report (fast, ~1 min):
venv/Scripts/python.exe scripts/run_all.py --skip-tracking

# Full tracking + analysis (hours):
venv/Scripts/python.exe scripts/run_all.py

# Latent space analyses (notebook):
jupyter notebook notebooks/
```

To load the aggregated feature data directly in Python:
```python
import pandas as pd
from pathlib import Path
from src.features import aggregate_per_fly_features
from src import load_run

# Example: load one run
df_frames, meta = load_run("outputs/run_NNN_13DPE_n001")
df_agg = aggregate_per_fly_features(df_frames)
```

---

*Generated 2026-06-07. Pipeline by Emma Vrignaud.*
