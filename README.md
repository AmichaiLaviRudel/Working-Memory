# Working Memory - Behavioral & Neural Data Analysis Platform

## Overview

Working Memory is a comprehensive data analysis platform designed for neuroscience research, specifically focused on behavioral experiments and neural recordings. The platform provides an integrated environment for managing, analyzing, and visualizing experimental data from various paradigms including Go/No-Go behavioral tasks and Neuropixels recordings.

## 🧠 Key Features

### 1. **Multi-Modal Data Support**
- **Behavioral Data (Bpod)**: Analysis of Go/No-Go discrimination tasks
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

#### Neuropixels Analysis
- **Single Unit Analysis**: Spike sorting, PSTH, raster plots
- **Population Analysis**: Multi-unit activity patterns
- **Cross-session Analysis**: Longitudinal neural tracking
- **Advanced Population Dynamics**: Complex multi-dimensional analysis
- **Quality Control**: Automated spike sorting validation

### 4. **Data Management**
- **Automated Data Loading**: MATLAB and Python integration for seamless data import
- **Session Concatenation**: Automated merging of experimental sessions
- **Project Organization**: Hierarchical project structure with metadata tracking
- **Export Capabilities**: CSV export for external analysis

## 🚀 Getting Started

### Prerequisites
- Python environment with required packages
- MATLAB (for session concatenation)
- Streamlit

### Installation & Setup
1. Navigate to Anaconda Navigator
2. Activate the DeepLabCut environment:
   ```bash
   activate C:\Users\Owner\miniconda3\envs\DEEPLABCUT
   ```
3. Launch the application:
   ```bash
   streamlit run "Z:\Shared\Amichai\General\code\DB\db.py"
   ```

### Quick Start
1. **Home Page**: View and manage your running projects
2. **Projects Page**: Select a project to analyze specific experimental sessions
3. **Neuropixels Monitoring**: Monitor and analyze neural recording data

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

### Visualization
- Interactive plots using Plotly for dynamic exploration
- Heatmaps for population-level activity
- Time-series plots for behavioral performance
- Statistical overlays and confidence intervals

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
- `npxl_single_unit_analysis.py`: Individual neuron characterization
- `population_analysis.py`: Multi-unit activity patterns
- `cross_session_analysis.py`: Longitudinal neural tracking
- `NPXL_Preprocessing.py`: Data cleaning and spike sorting

## 🎯 Use Cases

### Research Applications
- **Learning & Memory**: Track behavioral acquisition and retention
- **Sensory Processing**: Analyze stimulus-response relationships
- **Neural Encoding**: Understand how behavior relates to neural activity
- **Longitudinal Studies**: Monitor changes across experimental sessions

### Data Types Supported
- **Bpod Behavioral Data**: Go/No-Go discrimination tasks
- **Neuropixels Recordings**: High-density neural recordings
- **Educage Data**: Automated home-cage monitoring
- **Custom Protocols**: Flexible analysis framework

