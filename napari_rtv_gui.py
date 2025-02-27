import tkinter as tk
from tkinter import messagebox, filedialog
import math
import subprocess


# Function to handle the form submission
def submit_form():
    try:
        # Get values from the entries
        folder_paths = folder_paths_entry.get()
        channel_patterns = channel_patterns_entry.get()
        voxel_resolution = voxel_resolution_entry.get()
        timepoint_range = timepoint_range_entry.get()
        timepoint_step_size = timepoint_step_size_entry.get()
        max_timepoints = max_timepoints_entry.get()

        # Validate and process inputs
        folder_paths = list(map(str, folder_paths.split(',')))
        channel_patterns = list(map(str, channel_patterns.split(',')))
        voxel_resolution = tuple(map(int, voxel_resolution.split(',')))
        timepoint_range = list(map(int, timepoint_range.split(','))) if timepoint_range else [0]
        timepoint_step_size = int(timepoint_step_size)
        max_timepoints = math.inf if max_timepoints == 'inf' else int(max_timepoints)

        # Format as command-line arguments
        cmd = [
            "python", "napari_rtv.py",
            "--folder-paths", ','.join(folder_paths),
            "--channel-patterns", ','.join(channel_patterns),
            "--voxel-resolution", ','.join(map(str, voxel_resolution)),
            "--timepoint-range", ','.join(map(str, timepoint_range)),
            "--timepoint-step-size", str(timepoint_step_size),
            "--max-timepoints", "inf" if max_timepoints == math.inf else str(max_timepoints)
        ]

        # Execute the command in a new process
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Function to browse for a single folder
def browse_folder():
    folder = filedialog.askdirectory()
    if folder:
        folder_paths_entry.delete(0, tk.END)  # Clear the entry field
        folder_paths_entry.insert(0, folder)  # Set the new folder path


# Creating the main window
root = tk.Tk()
root.title("napari-rtv")

# Grid configuration
root.grid_columnconfigure(0, weight=1, minsize=150)  # Labels
root.grid_columnconfigure(1, weight=2, minsize=300)  # Entries
root.grid_columnconfigure(2, weight=0)  # Buttons
for i in range(7):
    root.grid_rowconfigure(i, weight=1)

# Set default values for the fields
default_values = {
    'folder_paths': '',
    'channel_patterns': '',
    'voxel_resolution': '108,108,108',
    'timepoint_range': '0',
    'timepoint_step_size': '1',
    'max_timepoints': 'inf'
}


# Function to add input fields
def add_input_field(label_text, default_value, row, has_browse=False):
    label = tk.Label(root, text=label_text)
    label.grid(row=row, column=0, sticky="w", padx=5, pady=5)

    entry = tk.Entry(root, width=50)
    entry.insert(0, default_value)
    entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)

    if has_browse:
        button = tk.Button(root, text="Browse", command=browse_folder)
        button.grid(row=row, column=2, padx=5, pady=5)

    return entry


# Create the form fields with default values
folder_paths_entry = add_input_field("Folder Paths (comma-separated)", default_values['folder_paths'], 0, True)
channel_patterns_entry = add_input_field("Channel Patterns (comma-separated)", default_values['channel_patterns'], 1)
voxel_resolution_entry = add_input_field("Voxel Resolution (z,y,x in nm)", default_values['voxel_resolution'], 2)
timepoint_range_entry = add_input_field("Timepoint Range (start or start,end)", default_values['timepoint_range'], 3)
timepoint_step_size_entry = add_input_field("Timepoint Step Size", default_values['timepoint_step_size'], 4)
max_timepoints_entry = add_input_field("Max Timepoints (inf or a whole number)", default_values['max_timepoints'], 5)

# Submit button
submit_button = tk.Button(root, text="Submit", command=submit_form)
submit_button.grid(row=6, column=0, columnspan=3, pady=10)

# Running the GUI
root.mainloop()
