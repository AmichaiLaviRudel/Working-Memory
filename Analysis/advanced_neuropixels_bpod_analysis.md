# Advanced Analyses for Neuropixels ↔ Bpod Data

## Single-Unit Analysis
For each isolated neuron in AC or OFC, we provide a comprehensive suite of analyses and interactive visualizations to quantify tuning, temporal dynamics, and task modulation.

- [x] **Stimulus selectivity (tuning curves)**
  - Computes firing-rate tuning across stimuli using a configurable analysis window relative to event onset.
  - Returns the tuning mean and SEM per stimulus and identifies the best stimulus (maximum response).
  - UI: The Single Unit tab shows the tuning curve with shaded SEM and a metric readout of the best stimulus value.
  - Implementation: `compute_stimulus_selectivity(event_windows_data, stimuli_outcome_df, unit_idx, window)`.

- [x] **Peri-stimulus time histogram (PSTH) and metrics**
  - PSTH computed from event-aligned windows; baseline and response statistics extracted.
  - Metrics include: baseline rate/STD, response magnitude, onset and peak latencies, FWHM, rise/decay times, trial variability, signal-to-noise, and suppression/excitation classification.
  - Implementation highlights: `calculate_psth_metrics(...)`, integrated in the Single Unit UI.

- [x] **Significance testing and p-values**
  - Mann–Whitney U test compares baseline (≤ 0 s) vs post-stimulus (> 0 s) periods on the averaged PSTH.
  - Supports computing and saving p-values across units; UI shows total vs significant counts and allows filtering to significant units (α configurable).
  - Implementation: `compute_psth_pvalues_from_event_windows(event_windows_matrix, event_times, bin_size, window)`; p-values are used throughout the UI for filtering and labeling.

- [x] **Go/No‑Go coding**
  - Computes d′ and ROC AUC for Go vs NoGo discrimination using trial-averaged rates in a specified time window.
  - Implementation: `compute_go_nogo_coding(...)` with outcome-based trial masks.

- [x] **Outcome modulation**
  - Tests reward modulation (e.g., Hit vs non-rewarded) using Mann–Whitney U over trial-averaged rates; reports p-value and effect size.
  - Implementation: `compute_outcome_modulation(...)`.

- [x] **Choice probability (CP)**
  - Estimates the probability that a blind observer could predict choice from the neuron’s trial-by-trial activity.
  - Implementation: `compute_choice_probability(...)`.

- [ ] **GLM (task encoding)**
  - Fits a GLM to quantify how task variables contribute to firing; exposes coefficients and basic model diagnostics. TODO BETTER!
  - Implementation: `fit_glm_single_unit(...)` and `plot_advanced_unit_analysis(...)` panels.

- [ ] **Lick modulation**
  - Planned: PSTHs aligned to lick events and peri-lick modulation metrics (UI hooks already load lick-aligned matrices).

### How to use (UI)
- Open the monitoring app’s Behavior Analysis → Single Unit tab.
- Select a unit and set the analysis window; view PSTH, tuning curve (with best stimulus), and advanced metrics.
- Toggle “Only significant units (p < α)” to restrict analyses/plots to significant neurons where relevant.

## Population Analysis
Explore collective dynamics in AC or OFC:

### Implementation Progress:
**Phase 1: Core Infrastructure** ✅
- [x] Created `population_analysis_advanced.py` module
- [x] Base `PopulationAnalyzer` class with data preparation
- [x] Cross-validation framework
- [x] Trial labeling utilities

**Phase 2: Decoding Analyses** ✅
- [x] `StimulusDecoder` class implemented
- [x] `ChoiceDecoder` class implemented  
- [x] `OutcomeDecoder` class implemented
- [x] Time-resolved decoding support for all three signal types
- [x] UI integration (Advanced tab in monitoring interface)
- [x] Fixed continuous stimulus labeling issue with quantile-based binning
- [x] Comprehensive time-resolved analysis with simultaneous stimulus, choice, and outcome decoding

[x] **Stimulus decoding** – apply linear classifiers to population activity to decode stimulus identity; use cross-validation to assess accuracy.
   - ✅ Implemented: Logistic Regression, SVM, LDA classifiers
   - ✅ Cross-validation with stratified K-fold
   - ✅ Confusion matrices and classification reports
   - ✅ Feature importance analysis
   - ✅ Interactive parameter controls in UI

[x] **Choice decoding** – predict lick (Go) vs no-lick (NoGo) trials from ensemble activity; assess decision signal strength.
   - ✅ Implemented: Go vs NoGo classification
   - ✅ Multiple classifier options
   - ✅ Time window selection
   - ✅ Performance metrics with chance level comparison
   - ✅ Time-resolved decoding analysis

[x] **Outcome decoding** – predict reward vs punishment outcomes from population activity; assess outcome signal representation.
   - ✅ Implemented: Reward (Hit) vs Punishment (Miss/False Alarm/CR) classification
   - ✅ Multiple classifier options (Logistic Regression, SVM, LDA)
   - ✅ Cross-validation with performance metrics
   - ✅ Time-resolved decoding analysis
   - ✅ Interactive UI integration with dedicated analysis panel

### Time-Resolved Decoding Framework ⏱️

The system now supports comprehensive time-resolved decoding across three key neural signals:

**1. Stimulus Decoding Over Time**
- Sliding window analysis to track when stimulus information becomes decodable
- Support for both individual stimulus identity and Go/NoGo identity grouping
- Configurable window size and step size for temporal precision

**2. Action/Choice Decoding Over Time** 
- Tracks the emergence and evolution of lick vs no-lick decision signals
- Based on behavioral outcomes: Hit/False Alarm (lick) vs Miss/CR (no-lick)
- Reveals decision formation dynamics in neural populations

**3. Outcome Decoding Over Time**
- Monitors reward vs punishment signal representation over time
- Reward: Hit trials; Punishment: Miss/False Alarm/CR trials
- Shows how outcome expectations and feedback are encoded

**Integrated Analysis Features:**
- ✅ Simultaneous multi-signal decoding in single plot
- ✅ Individual chance level lines for each signal type
- ✅ Event onset markers for temporal reference
- ✅ Interactive parameter controls (window size, step size, classifier type)
- ✅ Cross-validated accuracy with error handling for insufficient samples

**Usage:** Select "Time-Resolved Decoding" in the Advanced Population Analysis panel, choose which signals to decode (Stimulus, Choice, Outcome), and configure temporal parameters.

**Phase 3: Dimensionality Reduction & RSA** ✅
- [x] `DimensionalityReducer` class implemented
- [x] `RepresentationalSimilarityAnalyzer` class implemented
- [x] UI integration for Phase 3 analyses
- [x] Interactive parameter controls and visualizations
- [x] Fixed RSA distance matrix computation error
- [x] Enhanced plot labels to show actual stimulus values (e.g., frequency in kHz) instead of generic indices

[x] **Low-dimensional trajectories** – apply PCA/dPCA to trial-averaged population activity to visualize state-space trajectories by condition.
   - ✅ Implemented: PCA with explained variance analysis
   - ✅ Condition-averaged trajectories visualization
   - ✅ Interactive component selection and time window controls
   - ✅ Comprehensive variance analysis and plotting

[x] **Dimensionality Reduction (PCA, dPCA, UMAP)** – Visualize and separate variance linked to stimuli, outcomes, or time. **Output:** low-D trajectories linked to task variables.
   - ✅ Implemented: PCA and UMAP algorithms
   - ✅ Multi-panel visualization (variance, projections, trajectories)
   - ✅ Trial-by-trial and condition-averaged analyses
   - ✅ Color-coded by stimulus and choice conditions

[x] **Representational Similarity Analysis (RSA)** – Measure dissimilarities between task conditions in neural space. Useful for exploring geometry of representations.
   - ✅ Implemented: Multiple distance metrics (correlation, euclidean, cosine)
   - ✅ Representational Dissimilarity Matrix (RDM) computation
   - ✅ Hierarchical clustering of conditions
   - ✅ Interactive metric selection and visualization
   - ✅ Fixed correlation metric distance matrix computation
   - ✅ Added robust error handling and validation
   - ✅ Display actual stimulus values instead of generic labels (e.g., "Stim_4.5" instead of "Stim_0")

**Phase 4: jPCA Analysis** ✅
- [x] `jPCAAnalyzer` class implemented
- [x] Skew-symmetric matrix decomposition and SVD
- [x] jPCA pair identification and rotation matrix computation
- [x] Comprehensive visualization with singular values, trajectories, rotation matrices, and phase space plots
- [x] Integration into dimensionality reduction analysis panel

[x] **jPCA (joint Principal Component Analysis)** – Identify rotational dynamics in neural population activity by finding skew-symmetric components in cross-covariance matrices.
   - ✅ Implemented: Cross-covariance matrix computation between consecutive time points
   - ✅ Skew-symmetric matrix decomposition using SVD
   - ✅ jPCA pair identification based on similar singular values
   - ✅ Rotation matrix computation for each jPCA pair
   - ✅ Multi-panel visualization (singular values, trajectories, rotation matrices, phase space)
   - ✅ Interactive parameter controls (PCA components, max skew)
   - ✅ Integration with existing dimensionality reduction framework

[] **Latency differences** – estimate onset latency of stimulus or decision signals in AC vs OFC.

[] **Functional coupling** – compute noise correlations across areas and spike-field coherence between AC spikes and OFC LFP (and vice versa).

[] **Cross-area decoding transfer** – train decoders in AC and test in OFC (and vice versa) to assess shared representations.

[] **Granger causality / information flow** – apply directed metrics to estimate flow of information (AC→OFC or OFC→AC) during different trial epochs.

[] **Representational similarity** – compute dissimilarity matrices across tones and outcomes; assess clustering and geometry of population codes.

[] **Latent Factor Inference (LFADS, GPFA)** – Model underlying latent neural states on a trial-by-trial basis. Useful for uncovering shared or hidden dynamics.


[] **Temporal Generalization Matrices** – Cross-time decoding to test stability of codes. **Output:** 2D matrix of decoding accuracy (train-time × test-time).


[] **State-Space and Dynamics Modeling** – Use jPCA, HMMs, or switching dynamical systems to explore neural trajectories, transitions, and attractors.

## Multi-Area Analysis (AC ↔ OFC)
Joint analyses across probes:

[] **Cross-Area Coupling Metrics** – Coherence (LFP/LFP or spike/LFP), cross-correlation. Detect synchronized rhythms and their timing.

[] **Shared Subspace Analysis (CCA, projections)** – Identify joint modes of activity. **Output:** canonical correlation dimensions, subspace alignment.

[] **Information Transfer (TE, Granger Causality)** – Measure directed influence between areas. **Output:** TE/GC metrics per direction and time window.

[] **Decoding Generalization Across Areas** – Train decoder in one area, test in the other. Explore representational compatibility and redundancy.

[] **Neural Manifold Alignment** – Align low-D spaces (e.g. Procrustes). Discover whether AC/OFC encode similar task axes differently.
