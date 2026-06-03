#!/usr/bin/env python3
"""Standalone CLI script for NPXL group-level decoder analysis (cluster / offline use).

Runs ACx/OFC choice decoders on every valid session in the NPXL monitoring CSV,
decoding both the behavior target (lick vs withhold) and the ground-truth target
(Go tone vs No-Go tone). Single-area sessions are supported — the script decodes
whatever area(s) are available.

Results are saved as a flat CSV with one row per (session × area).

Usage
-----
python npxl_group_decoder_batch.py \\
    --monitoring_csv  /path/to/npxl_monitoring.csv \\
    --output          /path/to/group_decoder_results.csv \\
    [--decode_window  0.0 0.25] \\
    [--aggregation    Mean] \\
    [--classifier     "Logistic Regression"] \\
    [--random_state   42] \\
    [--min_trials     5] \\
    [--no_histology]

Output files (histology suffix appended to each base name)
----------------------------------------------------------
*_histology.csv / *_no_histology.csv
    Per-session × area decoder metrics (see columns below).
*_joint_probs_*.csv
    Per-session joint conditional probabilities by ACx/OFC GT-decoder prediction
    (dual-area sessions only).
*_marginal_probs_*.csv
    Per-session marginal conditional probabilities P(outcome | area GT prediction).
*_single_area_probs_*.csv
    Per-session conditional probabilities when only one area is available.
*_psychometric_*.csv
    Per-session P(Go) by stimulus frequency (kHz) for mouse, GT, and GT-decoder predictions.
    Rows with error_trials_only=True are Miss/FA trials only (psychometric on errors).
*_model_agreement.csv
    Per-session P(lick matches shared decoder) by ACx/OFC agreement state (dual-area only).
*_psychometric_model_agree.csv
    Per-session P(behavior match) vs stimulus on models-agree trials (dual-area only).
*_psychometric_model_disagree.csv
    Per-session P(behavior match) vs stimulus on model-disagree trials, split by ACx vs OFC (dual-area only).
*_kappa_agreement.csv
    Per-session chance-corrected agreement (Cohen/Fleiss kappa) among Mouse, GT, ACx, and OFC.
*_kappa_by_stimulus.csv
    Per-session stimulus-resolved kappa rows for group-level binning/plotting.
*_failures_*.csv
    Sessions that raised an exception.

Decoder metrics columns
-----------------------
animal, date, session_type, session_dprime, session_hit_rate, area, n_units, n_trials, n_folds,
accuracy, precision, recall, roc_auc,          (behavior target)
accuracy_gt, precision_gt, recall_gt, roc_auc_gt,  (ground-truth target)
acx_beta, ofc_beta, beta_diff, beta_diff_ci_low, beta_diff_ci_high,
histology_fallback

Agreement probability columns match npxl_agreement_decoder (GT-decoder OOF predictions).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap: ensure both the Code root and DB directory are importable.
# This allows:
#   - `import npxl_agreement_decoder`  (same folder as this script)
#   - `from Analysis.* import ...`     (relative to Code root)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()        # .../Code/DB/
_CODE_ROOT = _HERE.parent.resolve()            # .../Code/

for _p in [str(_CODE_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Import pure computation functions from the Streamlit page.
# The _in_streamlit() guard in that module prevents the UI code from executing
# on import, so this is safe even outside a Streamlit server context.
from npxl_agreement_decoder import (  # noqa: E402
    UNIT_THRESHOLD,
    _aggregate_joint_probabilities_across_sessions,
    _aggregate_kappa_agreement_across_sessions,
    _aggregate_marginal_probabilities_across_sessions,
    _aggregate_psychometric_across_sessions,
    _build_group_summary_table,
    _run_one_session_batch,
    load_valid_sessions,
)


# Windows → Linux path remapping table.
# The monitoring CSV stores Windows paths; on the cluster these must be translated.
# Add or modify entries to match your cluster's mount points.
_PATH_REMAPS: list[tuple[str, str]] = [
    (r"Z:\Shared\Amichai", "/ems/elsc-labs/mizrahi-a/Shared/Amichai"),
    (r"Z:/Shared/Amichai", "/ems/elsc-labs/mizrahi-a/Shared/Amichai"),
]


def _remap_path(path: str) -> str:
    """Translate a Windows path from the monitoring CSV to the cluster Linux path."""
    for win_prefix, linux_prefix in _PATH_REMAPS:
        if path.startswith(win_prefix):
            remainder = path[len(win_prefix):].replace("\\", "/")
            return linux_prefix + remainder
    # Also handle plain backslash → forward slash for paths not matched above
    return path.replace("\\", "/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NPXL group-level decoder analysis for cluster batch jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--monitoring_csv",
        required=True,
        help="Path to the NPXL monitoring CSV (the same file used by the Streamlit app).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output CSV path for per-session × area results.",
    )
    p.add_argument(
        "--decode_window",
        nargs=2,
        type=float,
        default=[0.0, 0.25],
        metavar=("START_S", "STOP_S"),
        help="Decode window in seconds relative to tone onset.",
    )
    p.add_argument(
        "--aggregation",
        default="Mean",
        choices=["Mean", "Sum"],
        help="How to aggregate spike counts across the decode window.",
    )
    p.add_argument(
        "--classifier",
        default="Logistic Regression",
        choices=["RBF SVM", "Logistic Regression", "Linear SVM"],
        help="Classifier for the out-of-fold decoder.",
    )
    p.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for CV splits and classifiers.",
    )
    p.add_argument(
        "--min_trials",
        type=int,
        default=5,
        help="Minimum trials per class required to run a decoder for a session.",
    )
    p.add_argument(
        "--no_histology",
        action="store_true",
        help=(
            "Use all good+MUA units instead of histology-matched units only. "
            "Gives more units per session but loses the area-label guarantee."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    monitoring_csv = os.path.abspath(args.monitoring_csv)
    output_path = os.path.abspath(args.output)
    decode_window: tuple[float, float] = (args.decode_window[0], args.decode_window[1])
    use_histology = not args.no_histology

    # Append a histology suffix to the output filename so runs with different
    # filter settings never overwrite each other.
    histology_suffix = "histology" if use_histology else "no_histology"
    base, ext = os.path.splitext(output_path)
    output_path = f"{base}_{histology_suffix}{ext}"

    print("=" * 70)
    print("NPXL Group-Level Decoder  —  batch mode")
    print("=" * 70)
    print(f"  Monitoring CSV : {monitoring_csv}")
    print(f"  Output CSV     : {output_path}")
    print(f"  Decode window  : {decode_window[0]:.3f} – {decode_window[1]:.3f} s")
    print(f"  Aggregation    : {args.aggregation}")
    print(f"  Classifier     : {args.classifier}")
    print(f"  Random state   : {args.random_state}")
    print(f"  Min trials/cls : {args.min_trials}")
    print(f"  Use histology  : {use_histology}")
    if _PATH_REMAPS:
        print(f"  Path remapping : {_PATH_REMAPS[0][0]!r} → {_PATH_REMAPS[0][1]!r}")
    print()

    print(f"Loading sessions from: {monitoring_csv}")
    sessions_df = load_valid_sessions(monitoring_csv)
    n_sessions = len(sessions_df)
    print(
        f"Found {n_sessions} valid sessions "
        f"(>= {UNIT_THRESHOLD} good+MUA units in at least one area).\n"
    )

    if sessions_df.empty:
        print("No valid sessions — nothing to decode. Exiting.")
        return

    records: list[dict] = []
    joint_prob_records: list[dict] = []
    marginal_prob_records: list[dict] = []
    single_area_prob_records: list[dict] = []
    psychometric_prob_records: list[dict] = []
    model_agreement_records: list[dict] = []
    psychometric_model_agree_records: list[dict] = []
    psychometric_model_disagree_records: list[dict] = []
    kappa_agreement_records: list[dict] = []
    kappa_by_stimulus_records: list[dict] = []
    failures: list[dict] = []
    t_start = time.time()

    for i, (_, row) in enumerate(sessions_df.iterrows(), 1):
        session_dir = _remap_path(str(row.get("current_dir", "")).strip())
        label = str(row.get("session_label", f"session_{i}"))
        elapsed = time.time() - t_start
        eta_str = ""
        if i > 1:
            per_session = elapsed / (i - 1)
            remaining = per_session * (n_sessions - i + 1)
            eta_str = f"  ETA {remaining / 60:.1f} min"

        print(f"[{i:3d}/{n_sessions}] {label}{eta_str}", end=" ... ", flush=True)

        try:
            (
                batch_records,
                joint_recs,
                marginal_recs,
                single_recs,
                psych_recs,
                agree_recs,
                psych_agree_recs,
                psych_disagree_recs,
                kappa_recs,
                kappa_stim_recs,
            ) = _run_one_session_batch(
                session_dir,
                row,
                decode_window=decode_window,
                aggregation=args.aggregation,
                classifier_type=args.classifier,
                random_state=args.random_state,
                min_trials_per_class=args.min_trials,
                use_histology=use_histology,
            )
            records.extend(batch_records)
            joint_prob_records.extend(joint_recs)
            marginal_prob_records.extend(marginal_recs)
            single_area_prob_records.extend(single_recs)
            psychometric_prob_records.extend(psych_recs)
            model_agreement_records.extend(agree_recs)
            psychometric_model_agree_records.extend(psych_agree_recs)
            psychometric_model_disagree_records.extend(psych_disagree_recs)
            kappa_agreement_records.extend(kappa_recs)
            kappa_by_stimulus_records.extend(kappa_stim_recs)
            areas = [r["area"] for r in batch_records]
            fallbacks = [r for r in batch_records if r.get("histology_fallback")]
            prob_note = ""
            if joint_recs:
                prob_note = f" +{len(joint_recs)} joint"
            elif single_recs:
                prob_note = f" +{len(single_recs)} single-area probs"
            if psych_recs:
                prob_note += f" +{len(psych_recs)} psychometric"
            if kappa_recs:
                prob_note += f" +{len(kappa_recs)} kappa"
            note = " [histology fallback]" if fallbacks else ""
            print(f"OK ({', '.join(areas)}){prob_note}{note}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"session": label, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)

    total_elapsed = time.time() - t_start
    print(f"\nCompleted in {total_elapsed / 60:.1f} min.")

    # --- Save results ---
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    output_stem = os.path.splitext(output_path)[0]

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_path, index=False)

    joint_probs_df = pd.DataFrame(joint_prob_records)
    joint_probs_path = f"{output_stem}_joint_probs.csv"
    if not joint_probs_df.empty:
        joint_probs_df.to_csv(joint_probs_path, index=False)

    marginal_probs_df = pd.DataFrame(marginal_prob_records)
    marginal_probs_path = f"{output_stem}_marginal_probs.csv"
    if not marginal_probs_df.empty:
        marginal_probs_df.to_csv(marginal_probs_path, index=False)

    single_area_probs_df = pd.DataFrame(single_area_prob_records)
    single_area_probs_path = f"{output_stem}_single_area_probs.csv"
    if not single_area_probs_df.empty:
        single_area_probs_df.to_csv(single_area_probs_path, index=False)

    psychometric_df = pd.DataFrame(psychometric_prob_records)
    psychometric_path = f"{output_stem}_psychometric.csv"
    if not psychometric_df.empty:
        psychometric_df.to_csv(psychometric_path, index=False)

    model_agreement_df = pd.DataFrame(model_agreement_records)
    model_agreement_path = f"{output_stem}_model_agreement.csv"
    if not model_agreement_df.empty:
        model_agreement_df.to_csv(model_agreement_path, index=False)

    psychometric_model_agree_df = pd.DataFrame(psychometric_model_agree_records)
    psychometric_model_agree_path = f"{output_stem}_psychometric_model_agree.csv"
    if not psychometric_model_agree_df.empty:
        psychometric_model_agree_df.to_csv(psychometric_model_agree_path, index=False)

    psychometric_model_disagree_df = pd.DataFrame(psychometric_model_disagree_records)
    psychometric_model_disagree_path = f"{output_stem}_psychometric_model_disagree.csv"
    if not psychometric_model_disagree_df.empty:
        psychometric_model_disagree_df.to_csv(psychometric_model_disagree_path, index=False)

    kappa_agreement_df = pd.DataFrame(kappa_agreement_records)
    kappa_agreement_path = f"{output_stem}_kappa_agreement.csv"
    if not kappa_agreement_df.empty:
        kappa_agreement_df.to_csv(kappa_agreement_path, index=False)

    kappa_by_stimulus_df = pd.DataFrame(kappa_by_stimulus_records)
    kappa_by_stimulus_path = f"{output_stem}_kappa_by_stimulus.csv"
    if not kappa_by_stimulus_df.empty:
        kappa_by_stimulus_df.to_csv(kappa_by_stimulus_path, index=False)

    n_rows = len(results_df)
    n_success = n_sessions - len(failures)
    areas_present = sorted(results_df["area"].unique()) if not results_df.empty else []
    print(f"\nSaved {n_rows} rows ({n_success}/{n_sessions} sessions decoded, "
          f"areas: {areas_present}) to:\n  {output_path}")
    if not joint_probs_df.empty:
        print(f"Saved {len(joint_probs_df)} joint-probability rows to:\n  {joint_probs_path}")
    if not marginal_probs_df.empty:
        print(f"Saved {len(marginal_probs_df)} marginal-probability rows to:\n  {marginal_probs_path}")
    if not single_area_probs_df.empty:
        print(
            f"Saved {len(single_area_probs_df)} single-area probability rows to:\n"
            f"  {single_area_probs_path}"
        )
    if not psychometric_df.empty:
        print(f"Saved {len(psychometric_df)} psychometric rows to:\n  {psychometric_path}")
    if not model_agreement_df.empty:
        print(f"Saved {len(model_agreement_df)} model-agreement rows to:\n  {model_agreement_path}")
    if not psychometric_model_agree_df.empty:
        print(
            f"Saved {len(psychometric_model_agree_df)} psychometric model-agree rows to:\n"
            f"  {psychometric_model_agree_path}"
        )
    if not psychometric_model_disagree_df.empty:
        print(
            f"Saved {len(psychometric_model_disagree_df)} psychometric model-disagree rows to:\n"
            f"  {psychometric_model_disagree_path}"
        )
    if not kappa_agreement_df.empty:
        print(f"Saved {len(kappa_agreement_df)} kappa-agreement rows to:\n  {kappa_agreement_path}")
    if not kappa_by_stimulus_df.empty:
        print(
            f"Saved {len(kappa_by_stimulus_df)} stimulus-resolved kappa rows to:\n"
            f"  {kappa_by_stimulus_path}"
        )

    # --- Summary table ---
    if not results_df.empty:
        print("\nSummary (mean accuracy by session_type × area):")
        summary = _build_group_summary_table(results_df)
        print(summary[["session_type", "area", "n_sessions",
                        "mean_accuracy", "sem_accuracy",
                        "mean_accuracy_gt", "sem_accuracy_gt"]].to_string(index=False))

    if not joint_probs_df.empty:
        pooled_joint = _aggregate_joint_probabilities_across_sessions(joint_probs_df)
        print("\nPooled joint conditional probabilities (all dual-area sessions, GT decoders):")
        joint_cols = [
            "condition",
            "n_condition",
            "p_mouse_go",
            "p_mouse_nogo",
            "p_gt_go",
            "p_gt_nogo",
        ]
        if pooled_joint.empty or not set(joint_cols).issubset(pooled_joint.columns):
            print("  (no pooled joint rows — check per-session joint sidecar CSV)")
        else:
            print(pooled_joint[joint_cols].to_string(index=False, float_format="%.3f"))

    if not kappa_agreement_df.empty:
        pooled_kappa = _aggregate_kappa_agreement_across_sessions(kappa_agreement_df)
        print("\nMean session-level kappa agreement (all dual-area sessions, GT decoders):")
        kappa_cols = [
            "n_sessions",
            "n_trials",
            "mean_cohen_mouse_acx",
            "mean_cohen_mouse_ofc",
            "mean_cohen_acx_ofc",
            "mean_fleiss_mouse_acx_ofc",
            "mean_fleiss_mouse_acx_ofc_gt",
        ]
        if pooled_kappa.empty or not set(kappa_cols).issubset(pooled_kappa.columns):
            print("  (no pooled kappa rows — check per-session kappa sidecar CSV)")
        else:
            print(pooled_kappa[kappa_cols].to_string(index=False, float_format="%.3f"))

    if not marginal_probs_df.empty:
        pooled_marginal = _aggregate_marginal_probabilities_across_sessions(marginal_probs_df)
        print("\nPooled marginal conditional probabilities (all sessions, GT decoders):")
        print(
            pooled_marginal[
                ["expression", "n_condition", "p_go", "p_nogo"]
            ].to_string(index=False, float_format="%.3f")
        )

    if not psychometric_df.empty:
        pooled_psych = _aggregate_psychometric_across_sessions(psychometric_df)
        psych_cols = [
            "stimulus",
            "error_trials_only",
            "trials",
            "p_mouse_go",
            "p_gt_go",
        ]
        if "p_acx_go" in pooled_psych.columns:
            psych_cols.append("p_acx_go")
        if "p_ofc_go" in pooled_psych.columns:
            psych_cols.append("p_ofc_go")
        for error_only, label in ((False, "all trials"), (True, "error trials only")):
            subset = pooled_psych[pooled_psych["error_trials_only"] == error_only]
            if subset.empty:
                continue
            print(f"\nPooled psychometric agreement by stimulus ({label}, GT decoders):")
            print(subset[psych_cols].to_string(index=False, float_format="%.3f"))

    # --- Failures ---
    if failures:
        failures_path = f"{os.path.splitext(output_path)[0]}_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"\nFailed: {len(failures)} sessions — see {failures_path}")
        for f in failures:
            print(f"  • {f['session']}: {f['error']}")
    else:
        print("\nAll sessions decoded successfully.")


if __name__ == "__main__":
    main()
