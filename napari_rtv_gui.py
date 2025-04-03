from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QTabWidget,
    QVBoxLayout, QHBoxLayout, QMainWindow, QCheckBox
)
from PyQt5.QtGui import QIcon, QFont
import sys
import math
import subprocess
import os


class dataAuditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Default values
        self.default_values = {
            'folder_paths': '',
            'channel_patterns': '',
            'voxel_resolution': '108,108,108',
            'timepoint_range': '0',
            'timepoint_step_size': '1',
            'max_timepoints': 'inf'
        }

        # Add the Main Label
        title_layout = QVBoxLayout()
        hboxlayout = QHBoxLayout()
        title_label = QLabel('Data Auditor')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Sans Serif', 24))
        hboxlayout.addWidget(title_label)
        title_layout.addLayout(hboxlayout)

        layout = QVBoxLayout()

        # Input fields
        self.folder_paths_entry = self.add_input_field(layout, "Folder Paths (comma-separated)", self.default_values['folder_paths'], True)
        layout.addStretch(1)
        self.channel_patterns_entry = self.add_input_field(layout, "Channel Patterns (comma-separated)", self.default_values['channel_patterns'])
        layout.addStretch(1)
        self.voxel_resolution_entry = self.add_input_field(layout, "Voxel Resolution (z,y,x in nm)", self.default_values['voxel_resolution'])
        layout.addStretch(1)
        self.timepoint_range_entry = self.add_input_field(layout, "Timepoint Range (start or start,end)", self.default_values['timepoint_range'])
        layout.addStretch(1)
        self.timepoint_step_size_entry = self.add_input_field(layout, "Timepoint Step Size", self.default_values['timepoint_step_size'])
        layout.addStretch(1)
        self.max_timepoints_entry = self.add_input_field(layout, "Max Timepoints (inf or a whole number)", self.default_values['max_timepoints'])

        # Submit button
        submit_layout = QVBoxLayout()
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit_form)
        submit_hboxlayout = QHBoxLayout()
        submit_hboxlayout.addWidget(submit_button)
        submit_layout.addLayout(submit_hboxlayout)

        main_layout.addLayout(title_layout, 1)
        main_layout.addLayout(layout, 10)
        main_layout.addLayout(submit_layout, 1)

        self.setLayout(main_layout)

    def add_input_field(self, layout, label_text, default_value, has_browse=False):
        hboxlayout = QHBoxLayout()
        label = QLabel(label_text)
        entry = QLineEdit()
        entry.setText(default_value)
        entry.setToolTip(entry.text())
        entry.textChanged.connect(lambda text, e=entry: self.update_tooltip(e, text))
        hboxlayout.addWidget(label)
        hboxlayout.setStretch(0,1)
        hboxlayout.addWidget(entry)
        hboxlayout.setStretch(1,1)

        if has_browse:
            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(lambda: self.browse_folder(entry))
            hboxlayout.addWidget(browse_button)
            hboxlayout.setStretch(0, 1)
            hboxlayout.setStretch(1, 4)
            hboxlayout.setStretch(2,1)

        layout.addLayout(hboxlayout, 1)

        return entry

    def update_tooltip(self, entry, text):
        entry.setToolTip(text)

    def browse_folder(self, entry):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            entry.setText(folder)

    def submit_form(self):
        try:
            # Get values from the entries
            folder_paths = self.folder_paths_entry.text()
            channel_patterns = self.channel_patterns_entry.text()
            voxel_resolution = self.voxel_resolution_entry.text()
            timepoint_range = self.timepoint_range_entry.text()
            timepoint_step_size = self.timepoint_step_size_entry.text()
            max_timepoints = self.max_timepoints_entry.text()

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
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


class visualizeTrainingDataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Default values
        self.default_values = {
            'zarr_path': '',
            'timepoint_range': '0',
        }

        # Add the Main Label
        title_layout = QVBoxLayout()
        hboxlayout = QHBoxLayout()
        title_label = QLabel('Visualize Training Data')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Sans Serif', 24))
        hboxlayout.addWidget(title_label)
        title_layout.addLayout(hboxlayout)

        layout = QVBoxLayout()

        # Input fields
        self.zarr_path_entry = self.add_input_field(layout, "Zarr Path", self.default_values['zarr_path'], True)
        layout.addStretch(1)
        self.timepoint_range_entry = self.add_input_field(layout, "Timepoint Range (start or start,end)", self.default_values['timepoint_range'])
        layout.addStretch(1)
        stitch_hboxlayout = QHBoxLayout()
        label = QLabel('Stitch')
        stitch_hboxlayout.addWidget(label)
        stitch_hboxlayout.setStretch(0,1)
        stitch_checkbox = QCheckBox()
        stitch_hboxlayout.addWidget(stitch_checkbox)
        stitch_hboxlayout.setStretch(1,1)
        layout.addLayout(stitch_hboxlayout, 1)
        self.stitch_checkbox = stitch_checkbox

        # Submit button
        submit_layout = QVBoxLayout()
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit_form)
        submit_hboxlayout = QHBoxLayout()
        submit_hboxlayout.addWidget(submit_button)
        submit_layout.addLayout(submit_hboxlayout)

        main_layout.addLayout(title_layout, 1)
        main_layout.addLayout(layout, 10)
        main_layout.addLayout(submit_layout, 1)

        self.setLayout(main_layout)

    def add_input_field(self, layout, label_text, default_value, has_browse=False):
        hboxlayout = QHBoxLayout()
        label = QLabel(label_text)
        entry = QLineEdit()
        entry.setText(default_value)
        entry.setToolTip(entry.text())
        entry.textChanged.connect(lambda text, e=entry: self.update_tooltip(e, text))
        hboxlayout.addWidget(label)
        hboxlayout.setStretch(0,1)
        hboxlayout.addWidget(entry)
        hboxlayout.setStretch(1,1)

        if has_browse:
            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(lambda: self.browse_folder(entry))
            hboxlayout.addWidget(browse_button)
            hboxlayout.setStretch(0, 1)
            hboxlayout.setStretch(1, 4)
            hboxlayout.setStretch(2,1)

        layout.addLayout(hboxlayout, 1)

        return entry

    def update_tooltip(self, entry, text):
        entry.setToolTip(text)

    def browse_folder(self, entry):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            entry.setText(folder)

    def submit_form(self):
        try:
            # Get values from the entries
            zarr_path = self.zarr_path_entry.text()
            timepoint_range = self.timepoint_range_entry.text()

            # Validate and process inputs
            timepoint_range = list(map(int, timepoint_range.split(','))) if timepoint_range else [0]

            # Format as command-line arguments
            script_name = 'read_zarr3.py'
            if self.stitch_checkbox.isChecked():
                script_name = 'stitch_zarr3.py'
            cmd = [
                "python", script_name,
                "--zarr-path", zarr_path,
                "--timepoint-range", ','.join(map(str, timepoint_range)),
            ]

            # Execute the command in a new process
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tabs = None
        self.setWindowTitle("napari-rtv")
        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()

        self.tabs.addTab(dataAuditorTab(), "Data Auditor")
        self.tabs.addTab(visualizeTrainingDataTab(), "Visualize Training Data")

        self.setCentralWidget(self.tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)),'icons','abcIcon.ico')))
    window.setMinimumSize(600, 480)
    window.resize(600, 480)
    window.show()
    sys.exit(app.exec_())
