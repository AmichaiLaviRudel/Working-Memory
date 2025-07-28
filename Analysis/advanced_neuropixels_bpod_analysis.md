# Advanced Analyses for Neuropixels ↔ Bpod Data

## Single-Unit Analysis
For each isolated neuron in AC or OFC, advanced analyses can reveal how it encodes stimuli and task events beyond basic firing rates:

[] **stimulus selectivity** - compute per-unit frequency tuning curves (e.g., firing rate across tone frequencies); extract best frequency and response bandwidth.
[] **go/no‑go coding** - use d′ or ROC AUC to measure how well single neurons discriminate Go vs No-Go stimuli.

[] **outcome modulation** - compare responses between rewarded (Hit) and non-rewarded (Miss/FA) trials to identify reward-related signals.

[] **lick modulation** - align spikes to lick times; assess motor-related activity using peri-lick histograms.

[] **choice probability/d′** - calculate choice probability (CP) or trial-by-trial correlations between spike counts and Go/No-Go choice.

[] **Generalized Linear Models (GLMs) for task encoding** – Fit single-neuron spike trains with a GLM (e.g. Poisson regression) to quantify how behavioral variables affect firing. **Input:** a design matrix of trial events (stimulus identity, Go/No-Go cue, choice, reward timing, etc.) and the neuron’s spike counts. **Output:** coefficients (“tuning weights”) for each factor, plus goodness-of-fit. Helps find "functional fingerprints" of each neuron.

[] **Spike-Triggered Averages (STAs)** – Compute the average of stimuli preceding each spike to estimate the neuron’s receptive field. **Input:** time-varying stimulus or behavioral signal. **Output:** the STA, e.g. preferred tone frequency or timing features.

[] **Mutual Information Analysis** – Measure information content between spike trains and conditions. **Input:** conditions (Go vs No-Go, tones), and neuron spike counts. **Output:** information in bits; non-parametric insight into nonlinear encoding.

[] **Time-Resolved Selectivity Metrics** – Sliding-window analysis to track when a neuron differentiates task conditions. **Input:** aligned spike trains; **Output:** selectivity (e.g. d′ or AUC) as a function of time.

## Population Analysis
Explore collective dynamics in AC or OFC:

[] **Latency differences** – estimate onset latency of stimulus or decision signals in AC vs OFC.

[] **Functional coupling** – compute noise correlations across areas and spike-field coherence between AC spikes and OFC LFP (and vice versa).

[] **Cross-area decoding transfer** – train decoders in AC and test in OFC (and vice versa) to assess shared representations.

[] **Granger causality / information flow** – apply directed metrics to estimate flow of information (AC→OFC or OFC→AC) during different trial epochs.

[] **Stimulus decoding** – apply linear classifiers to population activity to decode stimulus identity; use cross-validation to assess accuracy.

[] **Choice decoding** – predict lick (Go) vs no-lick (NoGo) trials from ensemble activity; assess decision signal strength.

[] **Low-dimensional trajectories** – apply PCA/dPCA to trial-averaged population activity to visualize state-space trajectories by condition.

[] **Representational similarity** – compute dissimilarity matrices across tones and outcomes; assess clustering and geometry of population codes.

[] **Dimensionality Reduction (PCA, dPCA, UMAP)** – Visualize and separate variance linked to stimuli, outcomes, or time. **Output:** low-D trajectories linked to task variables.

[] **Latent Factor Inference (LFADS, GPFA)** – Model underlying latent neural states on a trial-by-trial basis. Useful for uncovering shared or hidden dynamics.

[] **Stimulus/Outcome Decoders** – Predict trial labels from ensemble activity. **Output:** classification accuracy, feature weights. Explores what’s decodable from the population.

[] **Temporal Generalization Matrices** – Cross-time decoding to test stability of codes. **Output:** 2D matrix of decoding accuracy (train-time × test-time).

[] **Representational Similarity Analysis (RSA)** – Measure dissimilarities between task conditions in neural space. Useful for exploring geometry of representations.

[] **State-Space and Dynamics Modeling** – Use jPCA, HMMs, or switching dynamical systems to explore neural trajectories, transitions, and attractors.

## Multi-Area Analysis (AC ↔ OFC)
Joint analyses across probes:

[] **Cross-Area Coupling Metrics** – Coherence (LFP/LFP or spike/LFP), cross-correlation. Detect synchronized rhythms and their timing.

[] **Shared Subspace Analysis (CCA, projections)** – Identify joint modes of activity. **Output:** canonical correlation dimensions, subspace alignment.

[] **Information Transfer (TE, Granger Causality)** – Measure directed influence between areas. **Output:** TE/GC metrics per direction and time window.

[] **Decoding Generalization Across Areas** – Train decoder in one area, test in the other. Explore representational compatibility and redundancy.

[] **Neural Manifold Alignment** – Align low-D spaces (e.g. Procrustes). Discover whether AC/OFC encode similar task axes differently.
