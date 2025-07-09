import os
from mtscomp import compress
from pathlib import Path
import numpy as np

def compress_bin_files(root_dir, overwrite=False):
    """
    Recursively compress .ap.bin and .lf.bin files using mtscomp.

    Parameters:
    - root_dir (str or Path): The root directory to search for bin files.
    - overwrite (bool): Whether to overwrite existing .ap.cbin files.
    """
    root_dir = Path(root_dir)

    for bin_path in root_dir.rglob("*.bin"):
        try:
            if bin_path.suffix != ".bin":
                continue

            cbin_path = bin_path.with_suffix('.cbin')
            if cbin_path.exists() and not overwrite:
                print(f"[SKIP] {cbin_path} already exists.")
                continue

            print(f"[COMPRESSING] {bin_path}")
            compress(
                bin_path,
                cbin_path,
                cbin_path.with_suffix('.ch'),
                sample_rate = 30000.00,
                n_channels = 385,
                dtype=np.int16
            )
            print(f"[DONE] -> {cbin_path}")
        except:
            print(f"[ERROR] Failed to compress {bin_path}.")
            continue

# Example usage:
# Set your root directory here:
root_dir = r"E:\Amichai - bad recs"
compress_bin_files(root_dir, overwrite=False)
