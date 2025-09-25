import pandas as pd
import ast
from pathlib import Path
import numpy as np
import re

# Source file path
path1 = r"Z:\Shared\Noam\results\pilot_amichai_08_09_2025\pilot_amichai_08_09_2025.txt"
path2 = r"Z:\Shared\Noam\results\experiments\experiments.txt"
# Read file
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

df = pd.concat([df1, df2])

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
    # Vectorized lick processing via explode/groupby
    if "start_dt" in df.columns:
        trial_id = df.index
        licks_exploded = (
            df[["date", "start_dt", "licks_time_list"]]
            .assign(trial_id=trial_id)
            .explode("licks_time_list")
        )
        # Drop rows with no licks
        licks_exploded = licks_exploded.dropna(subset=["licks_time_list"]) if not licks_exploded.empty else licks_exploded
        if not licks_exploded.empty:
            licks_exploded["lick_dt"] = pd.to_datetime(licks_exploded["date"] + " " + licks_exploded["licks_time_list"].astype(str), errors="coerce")
            # Compute relative seconds and keep only < 4s
            rel_sec = (licks_exploded["lick_dt"] - licks_exploded["start_dt"]).dt.total_seconds()
            licks_exploded["rel_sec"] = rel_sec
            licks_exploded = licks_exploded[np.isfinite(licks_exploded["rel_sec"])]
            licks_exploded = licks_exploded[licks_exploded["rel_sec"] < 4]

            # First lick RT in ms per trial
            first_rt = (
                licks_exploded.groupby("trial_id")["rel_sec"].min().mul(1000.0)
            )
            # All licks list per trial
            licks_list = (
                licks_exploded.groupby("trial_id")["rel_sec"].apply(list)
            )
            df["rt_first_ms"] = first_rt.reindex(trial_id).astype(float)
            df["licks_rel"] = licks_list.reindex(trial_id)
        else:
            df["rt_first_ms"] = pd.NA
            df["licks_rel"] = None

# Prepare cleaned stimulus names: drop .npz and replace '-' with '.'
if "stim_name" in df.columns:
    df["stim_name_clean"] = (
        df["stim_name"].astype(str)
          .str.replace(".npz", "", regex=False)
          .str.replace("-", ".", regex=False)
    )

# Exclude dummy probe mouse_id before processing
df = df[df["mouse_id"] != "000799EB9B"]

# Vectorized normalization for trial types and outcomes BEFORE grouping
if "go_no_go" in df.columns:
    _tt = (
        df["go_no_go"].astype(str).str.replace("_", " ").str.replace("-", " ").str.strip().str.lower()
    )
    df["go_no_go_norm"] = np.where(_tt.isin(["no go", "nogo", "no  go"]), "NoGo",
                             np.where(_tt.eq("go"), "Go", _tt.str.title()))
if "score" in df.columns:
    _sc = df["score"].astype(str).str.replace("_", " ").str.strip().str.lower()
    _map = {
        "hit": "Hit",
        "miss": "Miss",
        "fa": "False Alarm",
        "false alarm": "False Alarm",
        "false_alarm": "False Alarm",
        "cr": "CR",
        "correct rejection": "CR",
        "correct_rejection": "CR",
    }
    df["score_norm"] = _sc.map(_map).fillna(_sc.str.title())

# Group by mouse ID, date, and level; include trial lists per group
# Sort within groups by start_time to preserve chronological order
sort_cols = ["mouse_id", "date", "level", "start_time"]
use_cols = [c for c in sort_cols if c in df.columns]
df_sorted = df.sort_values(use_cols)

# Prepare numeric stimulus value column once
if "stim_name_clean" in df_sorted.columns:
    df_sorted["stim_value"] = pd.to_numeric(df_sorted["stim_name_clean"], errors="coerce").astype(float) * 0.1
elif "stim_name" in df_sorted.columns:
    df_sorted["stim_value"] = pd.to_numeric(df_sorted["stim_name"], errors="coerce").astype(float) * 0.1

grouped = (
    df_sorted
      .groupby(["mouse_id", "date", "level"], as_index=False)
      .agg(
          n_trials=("stim_value", "size"),
          go_no_go=("go_no_go_norm", list) if "go_no_go_norm" in df_sorted.columns else ("go_no_go", list),
          stim_name=("stim_value", list),
          score=("score_norm", list) if "score_norm" in df_sorted.columns else ("score", list) if "score" in df_sorted.columns else ("go_no_go", list),
          rt_first_ms=("rt_first_ms", list) if "rt_first_ms" in df_sorted.columns else ("stim_value", lambda s: [pd.NA] * len(s)),
          licks=("licks_rel", list) if "licks_rel" in df_sorted.columns else ("stim_value", lambda s: [None] * len(s)),
          start_time=("start_time", list) if "start_time" in df_sorted.columns else ("stim_value", lambda s: [pd.NA] * len(s)),
      )
      .sort_values(["date", "mouse_id", "level"]) 
)

# Save grouped to parent of the source path
base_dir = Path(path2).resolve().parent

# Per-row normalization removed: done pre-grouping

# Extract tones_per_class and N_Boundaries directly from level formatted as xT_yB
def _parse_level_tones(level):
    try:
        m = re.search(r"(\d+)\s*[tT]", str(level))
        return int(m.group(1)) if m else 1
    except Exception:
        return 1

def _parse_level_boundaries(level):
    try:
        m = re.search(r"(\d+)\s*[bB]", str(level))
        return int(m.group(1)) if m else 1
    except Exception:
        return 1
tones_per_class = grouped["level"].apply(_parse_level_tones)

n_boundaries = grouped["level"].apply(_parse_level_boundaries)


grouped_out = (
    grouped.assign(Tones_per_class=tones_per_class, N_Boundaries=n_boundaries)
      .rename(columns={
          "mouse_id": "MouseName",
          "date": "SessionDate",
          "level": "Notes",
          "go_no_go": "TrialTypes",
          "score": "Outcomes",
          "stim_name": "Stimuli",
          "licks": "Licks",
          "start_time": "StartTime",
      })
      .assign(FilePath=path2)[
          ["MouseName", "SessionDate", "TrialTypes", "Outcomes", "Stimuli", "Licks", "StartTime", "Notes", "FilePath", "Tones_per_class", "N_Boundaries"]
      ]
)
# Sort final dataframe by combined datetime (SessionDate + first StartTime if available)
def _first_start_time(lst):
    try:
        return lst[0] if isinstance(lst, list) and len(lst) > 0 else None
    except Exception:
        return None

_first_times = grouped_out["StartTime"].apply(_first_start_time) if "StartTime" in grouped_out.columns else None
if _first_times is not None:
    _sort_dt = pd.to_datetime(
        grouped_out["SessionDate"].astype(str) + " " + _first_times.astype(str), errors="coerce"
    )
else:
    _sort_dt = pd.to_datetime(grouped_out["SessionDate"], errors="coerce")

grouped_out = grouped_out.assign(_sort_dt=_sort_dt).sort_values(["_sort_dt", "MouseName", "Notes"]).drop(columns=["_sort_dt"])  
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


