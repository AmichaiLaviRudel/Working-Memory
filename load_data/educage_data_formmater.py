import pandas as pd
import ast
from pathlib import Path
import numpy as np

# Source file path
path = r"Z:\\Shared\\Noam\\results\\pilot_amichai_07_09_2025\\pilot_amichai_07_09_2025.txt"

# Read file
df = pd.read_csv(path)
print(df.columns)
# Normalize column names robustly (spaces, slashes, dashes -> underscores; lowercase)
def _normalize(col):
    c = str(col).strip().lower()
    for ch in [" ", "\\", "/", "-"]:
        c = c.replace(ch, "_")
    while "__" in c:
        c = c.replace("__", "_")
    return c

df.columns = [_normalize(c) for c in df.columns]

# Build absolute start datetime and licks datetime lists to compute RT
if all(col in df.columns for col in ["date", "start_time"]):
    df["start_dt"] = pd.to_datetime(df["date"] + " " + df["start_time"])

def _parse_licks_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x)
    except Exception:
        return []

if "licks_time" in df.columns:
    df["licks_time_list"] = df["licks_time"].apply(_parse_licks_list)
    # Convert to absolute datetimes on the same date as the trial
    if "start_dt" in df.columns:
        df["licks_dt_list"] = [
            [pd.to_datetime(f"{d} {t}") for t in lst]
            for d, lst in zip(df["date"], df["licks_time_list"]) 
        ]
        # First lick RT in milliseconds
        def _first_rt_ms(row):
            lst = row["licks_dt_list"]
            if not lst:
                return pd.NA
            return (lst[0] - row["start_dt"]).total_seconds() * 1000.0
        df["rt_first_ms"] = df.apply(_first_rt_ms, axis=1)

        # Licks relative to start in seconds as None or numpy array([...])
        def _licks_rel(row):
            lst = row["licks_dt_list"]
            if not lst:
                return None
            vals = [(t - row["start_dt"]).total_seconds() for t in lst]
            return np.array(vals, dtype=float)
        df["licks_rel"] = df.apply(_licks_rel, axis=1)

# Prepare cleaned stimulus names: drop .npz and replace '-' with '.'
if "stim_name" in df.columns:
    df["stim_name_clean"] = (
        df["stim_name"].astype(str)
          .str.replace(".npz", "", regex=False)
          .str.replace("-", ".", regex=False)
    )

# Group by mouse ID, date, and level; include trial lists per group
# Sort within groups by start_time to preserve chronological order
sort_cols = ["mouse_id", "date", "level", "start_time"]
use_cols = [c for c in sort_cols if c in df.columns]
df_sorted = df.sort_values(use_cols)

grouped = (
    df_sorted
      .groupby(["mouse_id", "date", "level"], as_index=False)
      .agg(
          n_trials=("stim_name", "size"),
          go_no_go=("go_no_go", list),
          stim_name=("stim_name_clean", list) if "stim_name_clean" in df_sorted.columns else ("stim_name", list),
          score=("score", list) if "score" in df_sorted.columns else ("go_no_go", list),
          rt_first_ms=("rt_first_ms", list) if "rt_first_ms" in df_sorted.columns else ("stim_name", lambda s: [pd.NA] * len(s)),
          licks=("licks_rel", list) if "licks_rel" in df_sorted.columns else ("stim_name", lambda s: [None] * len(s)),
      )
      .sort_values(["date", "mouse_id", "level"]) 
)
print(df.columns)
print(grouped.head())

# Save grouped to parent of the source path
base_dir = Path(path).resolve().parent

# Normalize/capitalize trial types and outcomes
def _cap_trial_types(lst):
    out = []
    for v in lst:
        s = str(v).replace("_", " ").replace("-", " ").strip().lower()
        if s in {"no go", "nogo", "no  go"}:
            out.append("NoGo")
        elif s == "go":
            out.append("Go")
        else:
            out.append(s.title())
    return out

def _map_outcomes(lst):
    mapping = {
        "hit": "Hit",
        "miss": "Miss",
        "fa": "False Alarm",
        "false alarm": "False Alarm",
        "false_alarm": "False Alarm",
        "cr": "CR",
        "correct rejection": "CR",
        "correct_rejection": "CR",
    }
    out = []
    for v in lst:
        s = str(v).replace("_", " ").strip().lower()
        out.append(mapping.get(s, s.title()))
    return out

if "go_no_go" in grouped.columns:
    grouped["go_no_go"] = grouped["go_no_go"].apply(_cap_trial_types)
if "score" in grouped.columns:
    grouped["score"] = grouped["score"].apply(_map_outcomes)

# Convert grouped stimuli names to float values scaled by 0.1 (e.g., '7.07' -> 0.707)
def _scale_stimuli_list(lst):
    out = []
    for v in lst:
        try:
            out.append(round(float(str(v)) * 0.1,3))
        except Exception:
            out.append(np.nan)
    return out

if "stim_name" in grouped.columns:
    grouped["stim_name"] = grouped["stim_name"].apply(_scale_stimuli_list)
tones_per_class = grouped.apply(
    lambda r: len({s for t, s in zip(r["go_no_go"], r["stim_name"]) if str(t).lower() == "go"}),
    axis=1,
)

grouped_out = (
    grouped.assign(Tones_per_class=tones_per_class, N_Boundaries=1)
      .rename(columns={
          "mouse_id": "MouseName",
          "date": "SessionDate",
          "level": "Notes",
          "go_no_go": "TrialTypes",
          "score": "Outcomes",
          "stim_name": "Stimuli",
          "licks": "Licks",
      })
      .assign(FilePath=path)[
          ["MouseName", "SessionDate", "TrialTypes", "Outcomes", "Stimuli", "Licks", "Notes", "FilePath", "Tones_per_class", "N_Boundaries"]
      ]
)
out_csv = Path(r"Z:\\Shared\\Amichai\\Code\\DB\\users_data\\Amichai\\Educage_experimental_data.csv")
# Format Stimuli lists without commas, space-separated within brackets
def _format_space_list(lst):
    try:
        vals = []
        for v in lst:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            vals.append(f"{float(v):.3f}")
        return "[" + " ".join(vals) + "]"
    except Exception:
        return str(lst)

_export_df = grouped_out.copy()
if "Stimuli" in _export_df.columns:
    _export_df["Stimuli"] = _export_df["Stimuli"].apply(_format_space_list)

_export_df.to_csv(out_csv, index=False)
print(f"Grouped summary saved to: {out_csv}")


