# Working Memory - Behavioral & Neural Data Analysis Platform

## ⚠️ Important Note

**This codebase is highly customized for a specific research project and experimental setup.** The code, file paths, data structures, and analysis workflows are tailored to particular hardware configurations, data formats, and research needs. As such, it may not be directly applicable to other projects without significant modifications.

**If you're interested in using or adapting this code for your own research, please feel free to reach out!** I'm happy to help explain the codebase structure, discuss how it might be adapted to your needs, or collaborate on modifications.

## Overview

Working Memory is a comprehensive data analysis platform designed for neuroscience research, specifically focused on behavioral experiments and neural recordings. The platform provides an integrated environment for managing, analyzing, and visualizing experimental data from various paradigms including Go/No-Go behavioral tasks and Neuropixels recordings.

## 🧠 Key Features

### 1. **Multi-Modal Data Support**
- **Behavioral Data (Bpod)**: Analysis of Go/No-Go classification tasks
- **Neuropixels Recordings**: Single-unit and population-level neural analysis
- **Educage System**: Automated home-cage behavioral monitoring
- **FRA (Frequency Response Area)**: Auditory response characterization

### 2. **Interactive Web Interface**
- Built with Streamlit for intuitive, browser-based interaction
- Real-time data visualization and analysis
- Project-based organization with user-specific workspaces
- Integrated file management and data processing

### 3. **Comprehensive Analysis Suite**

#### Behavioral Analysis (Go/No-Go Tasks)
- **Performance Metrics**: Hit rates, false alarms, d-prime calculations
- **Psychometric Curves**: Stimulus-response relationship analysis
- **Learning Curves**: Performance tracking over time
- **Licking Behavior**: Detailed analysis of licking patterns and latencies
- **Bias Analysis**: Response bias detection and quantification
- **Multi-session Comparisons**: Cross-session performance analysis
- **GLM Analysis**: Generalized Linear Models for behavioral pattern analysis

#### Neuropixels Analysis
- **Single Unit Analysis**: Spike sorting, PSTH, raster plots
- **Population Analysis**: Multi-unit activity patterns
- **Cross-session Analysis**: Longitudinal neural tracking
- **Advanced Population Dynamics**: Complex multi-dimensional analysis
- **Quality Control**: Automated spike sorting validation
- **GLM for Neural Encoding**: Poisson regression models to quantify task variable contributions to firing rates
- **Machine Learning Decoding Pipeline**: Comprehensive ML framework for stimulus, choice, and outcome decoding

### 4. **Data Management**
- **Automated Data Loading**: MATLAB and Python integration for seamless data import
- **Session Concatenation**: Automated merging of experimental sessions
- **Project Organization**: Hierarchical project structure with metadata tracking
- **Export Capabilities**: CSV export for external analysis

## 📊 Analysis Capabilities

### Behavioral Metrics
- **Signal Detection Theory**: d-prime, criterion, hit/miss rates
- **Psychophysics**: Psychometric curve fitting and analysis
- **Temporal Dynamics**: Trial-by-trial performance tracking
- **Response Patterns**: Licking behavior and reaction time analysis

### Neural Analysis
- **Spike Train Analysis**: PSTH, ISI, firing rate calculations
- **Population Dynamics**: Multi-unit correlation and synchrony
- **Stimulus Encoding**: Response selectivity and tuning curves
- **Cross-modal Integration**: Behavior-neural correlations

### 🔬 Generalized Linear Models (GLM)

The platform includes comprehensive GLM analysis capabilities for both behavioral and neural data:

#### Single-Unit GLM Analysis
- **Poisson Regression Models**: Fit GLMs to quantify how task variables contribute to single neuron firing rates
- **Task Variable Encoding**: Model contributions of stimulus identity, trial type (Go/NoGo), and outcome (Hit/Miss/FA/CR)
- **Model Diagnostics**: Coefficient analysis, R-squared calculations, and predicted vs. actual firing rate comparisons
- **Feature Standardization**: Automated feature scaling for robust model fitting
- **Interactive Visualization**: Real-time GLM coefficient plots and model fit diagnostics

#### Behavioral GLM Analysis
- **Licking Pattern Modeling**: GLM analysis of licking behavior patterns
- **Trial-by-Trial Predictions**: Model behavioral responses based on task variables
- **Outcome Prediction**: Quantify relationships between stimuli, choices, and outcomes

### 🤖 Machine Learning Pipeline

The platform features a comprehensive machine learning framework for neural population decoding:

#### Population Decoding Framework
- **Stimulus Decoding**: Decode stimulus identity from population neural activity
  - Multiple classifier options: Logistic Regression, SVM (RBF kernel), LDA
  - Stratified K-fold cross-validation for robust performance assessment
  - Confusion matrices and classification reports
  - Feature importance analysis
  - Time-resolved decoding with sliding window analysis

- **Choice Decoding**: Predict Go vs. NoGo decisions from ensemble activity
  - Binary classification of lick (Go) vs. no-lick (NoGo) trials
  - Balanced class sampling for robust training
  - Performance metrics with chance level comparison
  - Temporal dynamics of decision signal formation

- **Outcome Decoding**: Predict reward vs. punishment outcomes from population activity
  - Reward (Hit) vs. Punishment (Miss/False Alarm/CR) classification
  - Cross-validated accuracy metrics
  - Time-resolved analysis of outcome signal representation

#### Time-Resolved Decoding
- **Sliding Window Analysis**: Track when different signals become decodable over time
- **Simultaneous Multi-Signal Decoding**: Decode stimulus, choice, and outcome signals simultaneously
- **Temporal Precision**: Configurable window sizes and step sizes for fine temporal resolution
- **Event-Aligned Analysis**: Decoding accuracy aligned to task events (stimulus onset, response, outcome)

#### ML Pipeline Features
- **Preprocessing Pipeline**: StandardScaler integration for feature normalization
- **Cross-Validation Framework**: Stratified K-fold with balanced class sampling
- **Multiple Classifier Support**: Pipeline-based implementation supporting various sklearn classifiers
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC calculations
- **Interactive Parameter Controls**: Real-time adjustment of classifier parameters, time windows, and analysis settings

### Visualization
- Interactive plots using Plotly for dynamic exploration
- Heatmaps for population-level activity
- Time-series plots for behavioral performance
- Statistical overlays and confidence intervals
- GLM coefficient visualizations
- ML decoding accuracy plots with temporal resolution

## 🗂️ Project Structure

```
DB/
├── Analysis/                    # Analysis modules
│   ├── GNG_bpod_analysis/      # Go/No-Go behavioral analysis
│   └── NPXL_analysis/          # Neuropixels neural analysis
├── load_data/                  # Data loading and preprocessing
├── users_data/                 # User-specific project data
├── Home.py                     # Main dashboard
├── Project.py                  # Project management interface
├── db.py                       # Application entry point
└── functions.py                # Core utility functions
```

## 🔧 Core Functionality

### Data Processing Pipeline
1. **Raw Data Import**: Load .mat files from Bpod, Neuropixels systems
2. **Preprocessing**: Automated quality control and formatting
3. **Analysis**: Comprehensive behavioral and neural metrics
4. **Visualization**: Interactive plots and statistical summaries
5. **Export**: Results available in multiple formats

### Session Management
- **Project-based Organization**: Group related experiments
- **Session Tracking**: Metadata for each experimental session
- **Quality Control**: Automated validation of data integrity
- **Cross-session Analysis**: Longitudinal comparisons

## 📈 Analysis Modules

### Behavioral Analysis (`GNG_bpod_analysis/`)
- `metric.py`: Performance metrics and signal detection theory
- `psychometric_curves.py`: Psychophysical analysis
- `licking_and_outcome.py`: Detailed licking behavior analysis
- `biases.py`: Response bias detection and correction
- `stats_tests.py`: Statistical validation and testing

### Neural Analysis (`NPXL_analysis/`)
- `npxl_single_unit_analysis.py`: Individual neuron characterization, GLM analysis for single units
- `population_analysis.py`: Multi-unit activity patterns, ML decoding pipeline (stimulus/choice/outcome)
- `cross_session_analysis.py`: Longitudinal neural tracking
- `NPXL_Preprocessing.py`: Data cleaning and spike sorting

### GLM & ML Analysis Modules
- **Single-Unit GLM**: `fit_glm_single_unit()` - Poisson regression for task variable encoding
- **Population ML Decoding**: `advanced_population_analysis_panel()` - Comprehensive decoding framework
- **Time-Resolved Decoding**: Sliding window analysis for temporal dynamics of neural signals
- **Behavioral GLM**: GLM analysis integrated into behavioral analysis workflows

