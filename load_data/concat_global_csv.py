import os
import pandas as pd
from pathlib import Path


def concat_csv_files(input_folder: str, output_file: str) -> str:
    """Concatenate all CSV files in a folder into a single CSV file.
    
    Args:
        input_folder: Path to folder containing CSV files
        output_file: Path for the output combined CSV file
    
    Returns:
        Path to the saved output file
    """
    folder = Path(input_folder)
    csv_files = list(folder.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_folder}")
    
    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        dfs.append(df)
        print(f"Loaded {csv_path.name}: {len(df)} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"Saved {output_file} with {len(combined)} total rows from {len(csv_files)} files")
    
    return output_file


if __name__ == "__main__":
    input_folder = r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\global"
    output_file = r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\global_training.csv"
    
    concat_csv_files(input_folder, output_file)
