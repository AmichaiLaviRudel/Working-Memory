import os
import argparse

def delete_done_files(root_dir):
    """
    Recursively walk through root_dir and delete all files ending with .done
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.done'):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


root_dir = r"Z:\Shared\Amichai\Behavior\data\Group_5"
delete_done_files(root_dir)