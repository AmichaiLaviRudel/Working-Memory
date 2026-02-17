"""
Latency map analysis for Go first-lick behavior.

Implements the "Averaged Normalized Latency Map" described by the user:

1. Spatial alignment: distance from category boundary in octaves.
2. Temporal normalization: per-session Z-scoring of first-lick latency.
3. Binning & aggregation across animals / expertise levels.
4. Statistical testing: boundary vs polar bins (ANOVA + Tukey HSD).

The main entry points are:

- compute_normalized_latency_map
- run_boundary_vs_polar_stats
- plot_latency_map_group
- plot_latency_map_single_session

This module is intentionally dataframe‑driven and does not depend on
Streamlit; it can be imported from the existing GNG UI code or used
stand‑alone from a script / notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from scipy import stats


@dataclass
class LatencyMapResult:
    """Container for the aggregated latency map."""

    latency_map: pd.DataFrame  # columns: Expertise, Dist_oct_center, Mean_Z, SEM_Z, N
    trials_z: pd.DataFrame     # trial-level with Dist_oct and Latency_Z


def _zscore_by_session(
    df: pd.DataFrame,
    session_col: str | list[str],
    latency_col: str,
    z_col: str = "Latency_Z",
) -> pd.DataFrame:
    """
    Add a Z-scored latency column, computed per group.

    session_col can be a single column name or a list of columns
    (e.g. ["SessionID", "Trial Type"]) to Z-score within each subgroup.
    Z = (Latency_trial - mu_group) / sigma_group
    """
    df = df.copy()
    x = df[latency_col].astype(float)
    # Build groupby keys — works for both str and list[str]
    if isinstance(session_col, str):
        group_keys = df[session_col]
    else:
        group_keys = [df[c] for c in session_col]
    grouped = x.groupby(group_keys)
    mu = grouped.transform("mean")
    sigma = grouped.transform(lambda s: s.std(ddof=1))
    z = (x - mu) / sigma
    # Replace inf / NaN from zero-variance groups
    df[z_col] = z.where(np.isfinite(z), other=np.nan)
    return df


def compute_normalized_latency_map(
    trials: pd.DataFrame,
    *,
    mouse_col: str = "MouseName",
    session_col: str = "SessionID",
    stim_freq_col: str = "StimulusFreq_Hz",
    boundary_freq_col: str = "BoundaryFreq_Hz",
    latency_col: str = "FirstLickLatency_s",
    expertise_col: str = "ExpertiseLevel",
    bin_width: float = 0.1,
    dist_range: Optional[Tuple[float, float]] = None,
    z_col: str = "Latency_Z",
) -> LatencyMapResult:
    """
    Compute an averaged, normalized latency map across trials.

    Parameters
    ----------
    trials:
        Trial-level dataframe. Each row is a trial.

    mouse_col, session_col:
        Columns identifying subject and session (used for information only;
        Z-scoring is done per *session_col*).

    stim_freq_col:
        Column with stimulus frequency (same units as boundary).

    boundary_freq_col:
        Column with category boundary frequency for the corresponding trial /
        session. If you have one boundary per session, simply broadcast that
        value to all trials in the session.

    latency_col:
        Column with first-lick latency in seconds.

    expertise_col:
        Categorical expertise label, e.g. 'Novice', 'Mid', 'Expert'.

    bin_width:
        Width of octave-distance bins.

    dist_range:
        Optional (min, max) range in octave space. If None, computed from data.

    Returns
    -------
    LatencyMapResult
        - latency_map: aggregated mean/SEM Z per expertise × octave bin.
        - trials_z: trial-level dataframe with Dist_oct and Latency_Z columns.
    """

    required_cols = {
        mouse_col,
        session_col,
        stim_freq_col,
        boundary_freq_col,
        latency_col,
        expertise_col,
    }
    missing = required_cols - set(trials.columns)
    if missing:
        raise KeyError(f"compute_normalized_latency_map: missing columns {sorted(missing)}")

    df = trials.copy()

    # Step 1: Spatial alignment in octave distance
    stim = df[stim_freq_col].astype(float)
    boundary = df[boundary_freq_col].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Dist_oct"] = np.log2(stim / boundary)

    # Step 2: Z-score within each session
    df = _zscore_by_session(df, session_col=session_col, latency_col=latency_col, z_col=z_col)

    # Drop trials without finite Dist or Z
    df = df[np.isfinite(df["Dist_oct"]) & np.isfinite(df[z_col])].copy()
    if df.empty:
        raise ValueError("compute_normalized_latency_map: no valid trials after filtering.")

    # Step 3: Binning in octave space
    if dist_range is None:
        min_oct = float(np.floor(df["Dist_oct"].min() / bin_width) * bin_width)
        max_oct = float(np.ceil(df["Dist_oct"].max() / bin_width) * bin_width)
    else:
        min_oct, max_oct = dist_range

    if min_oct >= max_oct:
        raise ValueError("compute_normalized_latency_map: invalid dist_range after inspection.")

    edges = np.arange(min_oct, max_oct + bin_width * 1.01, bin_width, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Assign each trial to a bin
    bin_idx = np.digitize(df["Dist_oct"].values, edges) - 1
    # Keep only in-range bins
    mask = (bin_idx >= 0) & (bin_idx < len(centers))
    df = df.loc[mask].copy()
    df["BinIndex"] = bin_idx[mask]
    df["Dist_oct_center"] = centers[df["BinIndex"].values]

    # Step 3: Aggregation by expertise × bin
    def _agg(group: pd.DataFrame) -> pd.Series:
        z = group[z_col].astype(float)
        n = z.size
        mean_z = float(np.nanmean(z)) if n > 0 else np.nan
        sem_z = float(np.nanstd(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        return pd.Series(
            {
                "Mean_Z": mean_z,
                "SEM_Z": sem_z,
                "N": n,
            }
        )

    latency_map = (
        df.groupby([expertise_col, "BinIndex", "Dist_oct_center"], as_index=False)
        .apply(_agg)
    )

    latency_map.rename(columns={expertise_col: "Expertise"}, inplace=True)

    return LatencyMapResult(latency_map=latency_map, trials_z=df)


def run_boundary_vs_polar_stats(
    trials_z: pd.DataFrame,
    *,
    dist_col: str = "Dist_oct",
    z_col: str = "Latency_Z",
    expertise_col: str = "ExpertiseLevel",
    boundary_width: float = 0.1,
    polar_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Compare boundary vs polar bins using one-way ANOVA + Tukey HSD.

    This operates on the trial-level Z-scored dataframe returned from
    compute_normalized_latency_map (LatencyMapResult.trials_z).

    For each expertise level, we:
    - define Region = Boundary if |Dist_oct| <= boundary_width / 2
                    or Polar    if |Dist_oct| >= polar_threshold
    - run one-way ANOVA on Z with factor Region
    - run Tukey HSD post-hoc between Boundary and Polar

    Returns a long-format dataframe with per-expertise statistics, including
    ANOVA F/p and Tukey summary.
    """

    required = {dist_col, z_col, expertise_col}
    missing = required - set(trials_z.columns)
    if missing:
        raise KeyError(f"run_boundary_vs_polar_stats: missing columns {sorted(missing)}")

    df = trials_z.copy()
    df = df[np.isfinite(df[dist_col]) & np.isfinite(df[z_col])].copy()

    # Label trials as Boundary / Polar / Other
    abs_dist = df[dist_col].abs()
    df["Region"] = np.where(
        abs_dist <= (boundary_width / 2.0),
        "Boundary",
        np.where(abs_dist >= polar_threshold, "Polar", "Other"),
    )
    df = df[df["Region"].isin(["Boundary", "Polar"])].copy()
    if df.empty:
        raise ValueError("run_boundary_vs_polar_stats: no Boundary/Polar trials after filtering.")

    rows = []

    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "statsmodels is required for Tukey HSD. Install via `pip install statsmodels`."
        ) from exc

    for expertise, sub in df.groupby(expertise_col):
        if sub["Region"].nunique() < 2:
            continue

        boundary_z = sub.loc[sub["Region"] == "Boundary", z_col].values
        polar_z = sub.loc[sub["Region"] == "Polar", z_col].values

        if boundary_z.size < 2 or polar_z.size < 2:
            continue

        # One-way ANOVA across the two regions
        fval, pval = stats.f_oneway(boundary_z, polar_z)

        # Tukey HSD
        tukey = pairwise_tukeyhsd(endog=sub[z_col], groups=sub["Region"], alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

        rows.append(
            {
                "Expertise": expertise,
                "ANOVA_F": float(fval),
                "ANOVA_p": float(pval),
                "Tukey_summary": tukey_df,
            }
        )

    return pd.DataFrame(rows)


def plot_latency_map_group(
    latency_map: pd.DataFrame,
    *,
    novice_label: str = "Novice",
    mid_label: str = "Mid",
    expert_label: str = "Expert",
) -> go.Figure:
    """
    Group-level latency map (Figure 4F-style).

    Plots Mean_Z ± SEM_Z vs Dist_oct_center for each expertise level on the
    same axis with shaded error regions.
    """

    required_cols = {"Expertise", "Dist_oct_center", "Mean_Z", "SEM_Z"}
    missing = required_cols - set(latency_map.columns)
    if missing:
        raise KeyError(f"plot_latency_map_group: missing columns {sorted(missing)}")

    fig = go.Figure()

    color_map = {
        novice_label: "gray",
        mid_label: "skyblue",
        expert_label: "indigo",
    }

    for expertise, df_e in latency_map.groupby("Expertise"):
        df_e = df_e.sort_values("Dist_oct_center")
        x = df_e["Dist_oct_center"].values
        y = df_e["Mean_Z"].values
        sem = df_e["SEM_Z"].values

        color = color_map.get(expertise, "black")

        # Central line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=str(expertise),
                line=dict(color=color, width=2),
            )
        )

        # Shaded SEM band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y - sem, (y + sem)[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Vertical dashed line at category boundary (0 octaves)
    fig.add_vline(
        x=0.0,
        line_dash="dash",
        line_color="black",
        line_width=1,
    )

    fig.update_layout(
        title="Averaged Normalized Latency Map",
        xaxis_title="Distance from Boundary (octaves)",
        yaxis_title="Normalized First Lick Latency (Z-score)",
        template="simple_white",
    )

    return fig


def plot_latency_map_single_session(
    df_session: pd.DataFrame,
    *,
    stim_freq_col: str = "StimulusFreq_Hz",
    boundary_freq_col: str = "BoundaryFreq_Hz",
    latency_col: str = "FirstLickLatency_s",
    bin_width: float = 0.1,
) -> go.Figure:
    """
    Single-session latency map for one animal/session.

    This is a light wrapper around the main computation: it builds octave
    distance and per-session Z-scores for the provided session dataframe only.
    """

    # Reuse the group function, but we don't care about expertise labels here.
    df = df_session.copy()
    df["ExpertiseLevel"] = "Session"  # dummy label
    result = compute_normalized_latency_map(
        df,
        mouse_col="MouseName" if "MouseName" in df.columns else "ExpertiseLevel",
        session_col="SessionID" if "SessionID" in df.columns else "Session",
        stim_freq_col=stim_freq_col,
        boundary_freq_col=boundary_freq_col,
        latency_col=latency_col,
        expertise_col="ExpertiseLevel",
        bin_width=bin_width,
    )

    fig = plot_latency_map_group(result.latency_map, novice_label="Session", mid_label="Session", expert_label="Session")
    fig.update_layout(title="Normalized First Lick Latency vs Distance from Boundary (single session)")
    return fig


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Compute and plot Averaged Normalized Latency Map from a trial-level CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Required columns (defaults):
              - MouseName
              - SessionID
              - StimulusFreq_Hz
              - BoundaryFreq_Hz
              - FirstLickLatency_s
              - ExpertiseLevel
            """
        ),
    )
    parser.add_argument("csv", help="Path to trial-level CSV file.")
    parser.add_argument("--mouse-col", default="MouseName")
    parser.add_argument("--session-col", default="SessionID")
    parser.add_argument("--stim-freq-col", default="StimulusFreq_Hz")
    parser.add_argument("--boundary-freq-col", default="BoundaryFreq_Hz")
    parser.add_argument("--latency-col", default="FirstLickLatency_s")
    parser.add_argument("--expertise-col", default="ExpertiseLevel")
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--output-html", default="latency_map.html")

    args = parser.parse_args()

    trials_csv = pd.read_csv(args.csv)
    result = compute_normalized_latency_map(
        trials_csv,
        mouse_col=args.mouse_col,
        session_col=args.session_col,
        stim_freq_col=args.stim_freq_col,
        boundary_freq_col=args.boundary_freq_col,
        latency_col=args.latency_col,
        expertise_col=args.expertise_col,
        bin_width=args.bin_width,
    )

    stats_df = run_boundary_vs_polar_stats(
        result.trials_z,
        dist_col="Dist_oct",
        z_col="Latency_Z",
        expertise_col=args.expertise_col,
    )

    fig = plot_latency_map_group(result.latency_map)
    fig.write_html(args.output_html)

    print("Saved latency map figure to", args.output_html)
    if not stats_df.empty:
        print("ANOVA / Tukey results (per expertise):")
        for _, row in stats_df.iterrows():
            print(f"Expertise: {row['Expertise']}, F={row['ANOVA_F']:.3f}, p={row['ANOVA_p']:.3g}")
            print(row["Tukey_summary"])
            print()

