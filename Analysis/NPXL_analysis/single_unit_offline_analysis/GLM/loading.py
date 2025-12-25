"""
Data loading functions for GLM analysis.
"""
import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import pynapple as nap

# Suppress pynapple conversion warnings
nap.nap_config.suppress_conversion_warnings = True


# ============================================================================
# Unit Type Mapping
# ============================================================================

UNIT_TYPE_NUMERIC_MAP = {
    0: 'noise',
    1: 'good',
    2: 'mua',
    3: 'non-somatic'
}


# ============================================================================
# Spike Loading Functions
# ============================================================================

def load_probe_spikes(
    base_path: str, 
    imec_name: str, 
    region_name: str
) -> Tuple[List[nap.Ts], List[str], List[str], List[int], str]:
    """
    Load spikes from a single probe with unit type labels.
    
    Parameters
    ----------
    base_path : str
        Base recording path
    imec_name : str
        Probe name (e.g., 'imec0', 'imec1')
    region_name : str
        Brain region name (e.g., 'ACx', 'OFC')
        
    Returns
    -------
    spike_list : list of nap.Ts
        List of spike trains
    unit_types : list of str
        List of unit type labels
    regions : list of str
        List of region labels
    cluster_ids : list of int
        List of original cluster IDs
    probe_path : str
        Path to the probe directory
    """
    # Extract recording name without 'catgt_' prefix for subdirectory naming
    base_name = base_path.split('\\')[-1]
    rec_name = base_name.replace('catgt_', '') if base_name.startswith('catgt_') else base_name
    probe_path = f"{base_path}\\{rec_name}_{imec_name}"
    
    # Load spike data
    spike_times = np.load(f"{probe_path}\\{imec_name}_ks4\\spike_times_sec_adj.npy")
    spike_clusters = np.load(f"{probe_path}\\{imec_name}_ks4\\spike_clusters.npy")
    unit_labels = pd.read_csv(f"{probe_path}\\bombcell\\unit_labels.tsv", sep="\t")

    # Create unit type map
    unit_type_map = {}
    for unit_id, row in unit_labels.iterrows():
        label_raw = row.iloc[0]
        
        # Handle both numeric and string formats
        if isinstance(label_raw, (int, float, np.integer, np.floating)):
            label = UNIT_TYPE_NUMERIC_MAP.get(int(label_raw), 'unknown')
        else:
            label_str = str(label_raw).upper()
            if 'GOOD' in label_str:
                label = 'good'
            elif 'MUA' in label_str:
                label = 'mua'
            elif 'NON-SOMA' in label_str or 'NONSOMATIC' in label_str:
                label = 'non-somatic'
            elif 'NOISE' in label_str:
                label = 'noise'
            else:
                label = 'unknown'
        
        unit_type_map[int(unit_id)] = label
    
    # Build lists for reindexing
    spike_list = []
    unit_types = []
    regions = []
    cluster_ids = []
    
    for cl in np.unique(spike_clusters):
        unit_times = spike_times[spike_clusters == cl]
        spike_list.append(nap.Ts(unit_times))
        unit_types.append(unit_type_map.get(int(cl), 'unknown'))
        regions.append(region_name)
        cluster_ids.append(int(cl))
    
    print(f"Loaded {len(spike_list)} units from {region_name} ({imec_name})")
    print(f"  Unit types: {pd.Series(unit_types).value_counts().to_dict()}")
    
    return spike_list, unit_types, regions, cluster_ids, probe_path


def load_all_probes(base_path: str) -> Tuple[nap.TsGroup, str, str]:
    """
    Load spikes from all probes and combine into a single TsGroup.
    
    Parameters
    ----------
    base_path : str
        Base recording path
        
    Returns
    -------
    spikes : nap.TsGroup
        Combined spike data with metadata
    probe_path_acx : str
        Path to ACx probe directory
    probe_path_ofc : str
        Path to OFC probe directory
    """
    # Load both probes
    spike_list_acx, unit_types_acx, regions_acx, cluster_ids_acx, probe_path_acx = load_probe_spikes(
        base_path, 'imec0', 'ACx'
    )
    
    spike_list_ofc, unit_types_ofc, regions_ofc, cluster_ids_ofc, probe_path_ofc = load_probe_spikes(
        base_path, 'imec1', 'OFC'
    )
    
    # Combine into single lists
    all_spikes = spike_list_acx + spike_list_ofc
    all_unit_types = unit_types_acx + unit_types_ofc
    all_regions = regions_acx + regions_ofc
    all_cluster_ids = cluster_ids_acx + cluster_ids_ofc
    
    spike_dict_reindexed = {i: spk for i, spk in enumerate(all_spikes)}
    spikes = nap.TsGroup(spike_dict_reindexed)
    
    # Add metadata (unit_type, region, original cluster_id)
    spikes.set_info(
        unit_type=all_unit_types,
        region=all_regions,
        cluster_id=all_cluster_ids
    )
    
    print(f"\nCombined: {len(spikes)} total units")
    print(f"By region: {pd.Series(all_regions).value_counts().to_dict()}")
    print(f"By unit type: {pd.Series(all_unit_types).value_counts().to_dict()}")
    
    return spikes, probe_path_acx, probe_path_ofc


# ============================================================================
# Event Loading Functions
# ============================================================================

def load_events(
    base_path: str, 
    probe_path: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load behavioral events (licks, tone onsets, stimuli/outcome data).
    
    Parameters
    ----------
    base_path : str
        Base recording path
    probe_path_acx : str
        Path to ACx probe directory (for analysis output)
        
    Returns
    -------
    licks : np.ndarray
        Lick times
    tone_onset : np.ndarray
        Tone onset times
    stimuli_outcome_df : pd.DataFrame
        DataFrame with stimulus and outcome information
    """
    base_name = base_path.split('\\')[-1]
    # Remove 'catgt_' prefix if present
    rec_name = base_name.replace('catgt_', '') if base_name.startswith('catgt_') else base_name

    licks = np.loadtxt(os.path.join(base_path, f"{rec_name}_tcat.nidq.xd_0_2_0.txt"))
    tone_onset = np.loadtxt(os.path.join(base_path, f"{rec_name}_tcat.nidq.xd_0_1_100.txt"))
    stimuli_outcome_df = pd.read_csv(os.path.join(probe_path, "analysis_output", "stimuli_outcome_df.csv"))
    
    return licks, tone_onset, stimuli_outcome_df


def filter_spikes(
    spikes: nap.TsGroup, 
    unit_type: str = 'good', 
    rate_threshold: float = 1.0
) -> nap.TsGroup:
    """
    Filter spikes by unit type and minimum firing rate.
    
    Parameters
    ----------
    spikes : nap.TsGroup
        Spike data
    unit_type : str
        Unit type to keep ('good', 'mua', etc.)
    rate_threshold : float
        Minimum firing rate threshold in Hz
        
    Returns
    -------
    nap.TsGroup
        Filtered spike data
    """
    spikes = spikes.getby_category("unit_type")[unit_type]
    spikes = spikes.getby_threshold("rate", rate_threshold)
    return spikes

