import os
from mtscomp import decompress
from pathlib import Path
import numpy as np

def decompress_bin_files(root_dir, overwrite=False):
    """
    Recursively decompress .ap.cbin and .lf.cbin files using mtscomp.

    Parameters:
    - root_dir (str or Path): The root directory to search for cbin files.
    - overwrite (bool): Whether to overwrite existing .ap.bin files.
    """
    root_dir = Path(root_dir)

    for cbin_path in root_dir.rglob("*.cbin"):
        try:
            if cbin_path.suffix != ".cbin":
                continue

            bin_path = cbin_path.with_suffix('.bin')
            ch_path = cbin_path.with_suffix('.ch')
            
            if not ch_path.exists():
                print(f"[SKIP] {cbin_path} missing corresponding .ch file: {ch_path}")
                continue
            
            if bin_path.exists() and not overwrite:
                print(f"[SKIP] {bin_path} already exists.")
                continue

            print(f"[DECOMPRESSING] {cbin_path}")
            decompress(
                cbin_path,
                ch_path,
                bin_path
            )
            print(f"[DONE] -> {bin_path}")
        except Exception as e:
            print(f"[ERROR] Failed to decompress {cbin_path}: {e}")
            continue

# Example usage:
# Set your root directory here:
root_dir = r"E:\Amichai - bad recs\Group3\G3A3\G3A3_rec2_g0\G3A3_rec2_g0_imec0"
decompress_bin_files(root_dir, overwrite=False)
