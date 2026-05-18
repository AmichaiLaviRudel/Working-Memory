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

Output columns
--------------
animal, date, session_type, session_dprime, area, n_units, n_trials, n_folds,
accuracy, precision, recall, roc_auc,          (behavior target)
accuracy_gt, precision_gt, recall_gt, roc_auc_gt,  (ground-truth target)
acx_beta, ofc_beta, beta_diff, beta_diff_ci_low, beta_diff_ci_high,
histology_fallback
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
    _build_group_summary_table,
    _run_one_session_batch,
    load_valid_sessions,
)


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
    failures: list[dict] = []
    t_start = time.time()

    for i, (_, row) in enumerate(sessions_df.iterrows(), 1):
        session_dir = str(row.get("current_dir", "")).strip()
        label = str(row.get("session_label", f"session_{i}"))
        elapsed = time.time() - t_start
        eta_str = ""
        if i > 1:
            per_session = elapsed / (i - 1)
            remaining = per_session * (n_sessions - i + 1)
            eta_str = f"  ETA {remaining / 60:.1f} min"

        print(f"[{i:3d}/{n_sessions}] {label}{eta_str}", end=" ... ", flush=True)

        try:
            batch_records = _run_one_session_batch(
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
            areas = [r["area"] for r in batch_records]
            fallbacks = [r for r in batch_records if r.get("histology_fallback")]
            note = " [histology fallback]" if fallbacks else ""
            print(f"OK ({', '.join(areas)}){note}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"session": label, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)

    total_elapsed = time.time() - t_start
    print(f"\nCompleted in {total_elapsed / 60:.1f} min.")

    # --- Save results ---
    results_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results_df.to_csv(output_path, index=False)

    n_rows = len(results_df)
    n_success = n_sessions - len(failures)
    areas_present = sorted(results_df["area"].unique()) if not results_df.empty else []
    print(f"\nSaved {n_rows} rows ({n_success}/{n_sessions} sessions decoded, "
          f"areas: {areas_present}) to:\n  {output_path}")

    # --- Summary table ---
    if not results_df.empty:
        print("\nSummary (mean accuracy by session_type × area):")
        summary = _build_group_summary_table(results_df)
        print(summary[["session_type", "area", "n_sessions",
                        "mean_accuracy", "sem_accuracy",
                        "mean_accuracy_gt", "sem_accuracy_gt"]].to_string(index=False))

    # --- Failures ---
    if failures:
        failures_path = output_path.replace(".csv", "_failures.csv")
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"\nFailed: {len(failures)} sessions — see {failures_path}")
        for f in failures:
            print(f"  • {f['session']}: {f['error']}")
    else:
        print("\nAll sessions decoded successfully.")


if __name__ == "__main__":
    main()
