# Working Memory - Behavioral & Neural Data Analysis Platform

## Important Note

**This codebase is highly customized for a specific research project and experimental setup.** The code, file paths, data structures, and analysis workflows are tailored to particular hardware configurations, data formats, and research needs. As such, it may not be directly applicable to other projects without significant modifications.

**If you're interested in using or adapting this code for your own research, please feel free to reach out!** I'm happy to help explain the codebase structure, discuss how it might be adapted to your needs, or collaborate on modifications.

## Overview

A Streamlit-based platform for managing, analyzing, and visualizing data from Go/No-Go behavioral experiments (Bpod) and Neuropixels electrophysiology recordings. The application provides project management, interactive analysis dashboards, and batch offline processing pipelines for neuroscience research.

## Tech Stack

| Category | Libraries |
|---|---|
| Web Framework | Streamlit |
| Data | Pandas, NumPy, SciPy |
| Visualization | Plotly, Altair, Matplotlib |
| Machine Learning | scikit-learn |
| Statistics | statsmodels |
| Neural Modeling | nemos, pynapple |
| MATLAB I/O | scipy.io (`loadmat`) |

## Getting Started

```bash
streamlit run db.py
```

The app launches a multi-page Streamlit interface with Home, Projects, and Neuropixels Monitoring pages.

## Project Structure

```
DB/
├── db.py                        # Streamlit entry point, page navigation
├── Home.py                      # Dashboard: running projects overview
├── Project.py                   # Project view: session table, analysis dispatch
├── global_dataset_page.py       # Global training performance dashboard
├── npxl_monitoring.py           # Neuropixels recording status & analysis
├── functions.py                 # Core utility functions
│
├── Analysis/
│   ├── GNG_bpod_analysis/       # Go/No-Go behavioral analysis (metrics, psychometric
│   │                            #   curves, licking, biases, latency maps)
│   └── NPXL_analysis/           # Neuropixels neural analysis
│       ├── single_unit_offline_analysis/  # Offline single-unit pipeline + GLM sub-module
│       └── NPXL_offline_analysis/         # Batch processing (with SLURM support)
│
├── load_data/                   # Data loading (Bpod .mat, Educage, MATLAB scripts)
├── support/                     # Utilities (compression, audio analysis, tests)
└── users_data/                  # Per-user project data & CSVs
```

## Application Pages

### Home (`Home.py`)
Displays the user's running projects in an editable table. Allows adding and saving new project entries.

### Projects (`Project.py`)
The main analysis workspace. Select a project to view its session table, then select sessions to launch behavioral or neural analysis. Dispatches to the appropriate analysis module based on project type.

### Neuropixels Monitoring (`npxl_monitoring.py`)
Tracks the processing status of Neuropixels recordings (SpikeGLX, Kilosort, Phy, TPrime, Bombcell stages). Also embeds single-unit analysis, population heatmaps, and decoding panels for selected recordings.

### Global Dataset (`global_dataset_page.py`)
Dashboard for the global training performance dataset across all animals and sessions. Provides filtering, region-wise performance breakdowns, and cross-animal comparisons.

## Analysis Modules

### Behavioral Analysis (`GNG_bpod_analysis/`)

| File | Purpose |
|---|---|
| `GNG_Bpod_Analysis.py` | Entry point: tabs for single-session, multi-session, and multi-animal analysis |
| `metric.py` | d-prime, hit/miss/FA/CR rates, pairwise stimulus d-prime, classifier metrics (ROC-AUC) |
| `psychometric_curves.py` | Weibull & sigmoid fitting, threshold estimation, slope progression across sessions |
| `psychometric_curves_plotting.py` | Psychometric curve visualization with boundary markers and outlier filtering |
| `licking_and_outcome.py` | Lick rates by stimulus, first-lick latency, learning curves, daily activity, cumulative trial analysis |
| `latency_map.py` | Normalized latency maps, boundary vs. polar statistics |
| `biases.py` | Choice bias (previous-trial carry-over), stimulus bias |
| `stats_tests.py` | Statistical testing utilities |
| `GNG_bpod_general.py` | Shared helpers: stimulus parsing, early-response filtering, Plotly config |
| `colors.py` | Color schemes, subject color maps, marker symbols |

### Neural Analysis (`NPXL_analysis/`)

| File | Purpose |
|---|---|
| `NPXL_Preprocessing.py` | Find Kilosort folders, load cluster info, extract spike matrices, align to behavioral events |
| `npxl_single_unit_analysis.py` | Interactive single-unit panel: PSTH, tuning curves, selectivity metrics |
| `population_analysis.py` | Population heatmaps with time navigation, best-stimulus distribution, ML decoding (stimulus/choice/outcome) |
| `NPXL_study_notebook.py` | OFC/ACx region comparison: active unit detection, selectivity, category sensitivity |

### GLM Sub-module (`NPXL_analysis/single_unit_offline_analysis/GLM/`)

Poisson regression models (via `nemos`) for quantifying task-variable contributions to single-neuron firing rates. Includes design matrix construction, model fitting, partial contribution analysis, and an interactive Streamlit panel.

### ML Decoding Pipeline (`population_analysis.py`)

Population-level decoding of stimulus identity, Go/NoGo choice, and trial outcome using scikit-learn classifiers (Logistic Regression, SVM, LDA). Supports time-resolved sliding-window analysis with stratified cross-validation.

## Data Pipeline

```
.mat files (Bpod)  ──►  load_bpod_data.py  ──►  per-session CSV
Educage text files  ──►  educage_data_formmater.py  ──►  per-session CSV
                                                            │
                                            concat_global_csv.py
                                                            │
                                                    ▼
                                    {project}_experimental_data.csv
                                                    │
                            ┌───────────────────────┼───────────────────────┐
                            ▼                       ▼                       ▼
                  GNG_bpod_analysis         NPXL_Preprocessing      global_dataset_page
                  (behavioral metrics,     (spike matrices,         (cross-animal
                   psychometric curves,     event windows)           performance
                   licking, biases)              │                   dashboard)
                                                 ▼
                                    npxl_single_unit_analysis
                                    population_analysis
                                    GLM / ML decoding
```

Neuropixels recordings follow a separate preprocessing path: raw SpikeGLX data is spike-sorted with Kilosort, curated in Phy, then loaded by `NPXL_Preprocessing.py` into event-aligned matrices for analysis.
