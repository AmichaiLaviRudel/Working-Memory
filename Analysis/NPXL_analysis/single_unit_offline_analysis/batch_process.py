"""
Batch processing script for NPXL offline analysis.

This script searches for all folders starting with "catgt" in the main recordings directory
and runs the analysis pipeline for each folder.
"""
import sys
import os
import traceback
from pathlib import Path

# Add the workspace root to Python path before any Analysis imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if 'single_unit_offline_analysis' in current_dir or 'NPXL_offline_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
elif 'NPXL_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(current_dir))
else:
    # Fallback: try to find the workspace root by going up directories
    test_dir = current_dir
    for _ in range(4):
        if os.path.exists(os.path.join(test_dir, 'Analysis', 'NPXL_analysis')):
            workspace_root = test_dir
            break
        test_dir = os.path.dirname(test_dir)
    else:
        workspace_root = current_dir

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from Analysis.NPXL_analysis.single_unit_offline_analysis.main import main


def find_catgt_folders(base_dir: str) -> list:
    """
    Find all folders starting with "catgt" in the base directory and subdirectories.
    
    Parameters:
    -----------
    base_dir : str
        Base directory to search in
    
    Returns:
    --------
    list
        List of full paths to folders starting with "catgt"
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Base directory does not exist: {base_dir}")
        return []
    
    catgt_folders = []
    
    # Search in base directory and all subdirectories
    for folder in base_path.rglob("*"):
        if folder.is_dir() and folder.name.startswith("catgt"):
            catgt_folders.append(str(folder))
    
    # Sort for consistent processing order
    catgt_folders.sort()
    
    return catgt_folders


def batch_process_all_recordings(base_dir: str = r"Z:\Shared\Amichai\Data\pipeline_output"):
    """
    Process all recordings in folders starting with "catgt".
    
    Parameters:
    -----------
    base_dir : str
        Base directory to search for catgt folders
    """
    print("=" * 80)
    print("NPXL Batch Processing Script")
    print("=" * 80)
    print(f"\nSearching for folders starting with 'catgt' in: {base_dir}")
    
    # Find all catgt folders
    catgt_folders = find_catgt_folders(base_dir)
    
    if not catgt_folders:
        print(f"\nNo folders starting with 'catgt' found in {base_dir}")
        return
    
    print(f"\nFound {len(catgt_folders)} folder(s) to process:")
    for i, folder in enumerate(catgt_folders, 1):
        print(f"  {i}. {folder}")
    
    # Process each folder
    successful = []
    failed = []
    
    for i, parent_dir in enumerate(catgt_folders, 1):
        print("\n" + "=" * 80)
        print(f"Processing folder {i}/{len(catgt_folders)}: {os.path.basename(parent_dir)}")
        print(f"Full path: {parent_dir}")
        print("=" * 80)
        
        try:
            # Run the main analysis function
            main(parent_dir=parent_dir)
            successful.append(parent_dir)
            print(f"\n✓ Successfully completed: {parent_dir}")
            
        except KeyboardInterrupt:
            print("\n\nBatch processing interrupted by user.")
            print(f"Processed {i-1}/{len(catgt_folders)} folders before interruption.")
            break
            
        except Exception as e:
            error_msg = f"Error processing {parent_dir}: {str(e)}"
            print(f"\n✗ {error_msg}")
            print("\nFull traceback:")
            traceback.print_exc()
            failed.append((parent_dir, str(e)))
            # Continue with next folder instead of stopping
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"\nTotal folders found: {len(catgt_folders)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n✓ Successfully processed folders:")
        for folder in successful:
            print(f"  - {folder}")
    
    if failed:
        print("\n✗ Failed folders:")
        for folder, error in failed:
            print(f"  - {folder}")
            print(f"    Error: {error}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch process NPXL recordings in folders starting with 'catgt'"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=r"Z:\Shared\Amichai\Data\pipeline_output",
        help="Base directory to search for catgt folders (default: Z:\\Shared\\Amichai\\Data\\pipeline_output)"
    )
    
    args = parser.parse_args()
    
    batch_process_all_recordings(base_dir=args.base_dir)

