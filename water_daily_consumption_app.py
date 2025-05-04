# Importing necessary libraries
from scipy.io import loadmat
import numpy as np
import os
from datetime import datetime, date
import tkinter as tk
from tkinter import filedialog, messagebox

# Default baseline folder
BASELINE_FOLDER = r"Z:\Shared\Amichai\Behavior\data"

# Function to calculate water consumed in a session
def water_in_session(file_path):
    try:
        mat_data = loadmat(file_path)

        # Extract SessionData
        session_data = mat_data['SessionData'][0, 0]
        trial_settings = session_data['TrialSettings'][0]
        raw_events = session_data['RawEvents'][0, 0]
        raw_event_trials = raw_events['Trial'][0]  # Extract trials

        water = 0  # Initialize total water counter

        for idx, trial in enumerate(raw_event_trials):
            try:
                # Access 'States' for the trial
                states = trial['States'][0, 0]
                if 'Reward' in states.dtype.names:
                    reward_times = states['Reward'][0][0][0][0]
                    if not np.isnan(reward_times):
                        water += trial_settings[idx]['GUI'][0, 0]['RewardAmount'][0, 0]
            except Exception:
                continue

        return water

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return 0  # Return 0 if the file is invalid


# Function to find files created today
def find_files_created_today_in_session_data(base_path):
    today = date.today()
    today_files = []

    for root, dirs, files in os.walk(base_path):
        if 'session' in root.lower():
            for file in files:
                file_path = os.path.join(root, file)
                creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if creation_time.date() == today:
                    today_files.append(file_path)

    return today_files


# Function to run the main logic
def run_analysis():
    folder_path = filedialog.askdirectory(initialdir=BASELINE_FOLDER, title="Select the Main Folder")
    if not folder_path:
        messagebox.showerror("Error", "No folder selected.")
        return

    files_created_today = find_files_created_today_in_session_data(folder_path)
    if not files_created_today:
        messagebox.showinfo("Info", "No files created today in session data folders.")
        return

    mice = []
    total_water = 0

    for file in files_created_today:
        mouse_name = file.split('\\')[-1].split('_')[0]
        mice.append(mouse_name)
        total_water += water_in_session(file)

    n_mice = len(np.unique(mice))
    water_needed = (n_mice * 1500) - total_water

    # Show the results
    result_message = (
        f"Total water consumed today: {total_water:.2f} ml\n"
        f"Number of unique mice: {n_mice}\n"
        f"Please give: {water_needed:.2f} ml"
    )
    messagebox.showinfo("Results", result_message)


# Create the GUI application
def create_app():
    root = tk.Tk()
    root.title("Water Analysis App")

    label = tk.Label(root, text="Click the button below to select the main folder and run the analysis:")
    label.pack(pady=20)

    analyze_button = tk.Button(root, text="Select Folder and Analyze", command=run_analysis)
    analyze_button.pack(pady=10)

    quit_button = tk.Button(root, text="Quit", command=root.quit)
    quit_button.pack(pady=10)

    root.mainloop()


# Run the application
if __name__ == "__main__":
    create_app()
