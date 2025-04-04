import argparse
import copy
import json
import math
import os

import napari
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QSizePolicy, QLabel
from napari.qt.threading import thread_worker
import numpy as np
import cpptiff
from pathlib import Path
import time
from datetime import datetime, timedelta
import re
from magicgui import magicgui
from napari.utils.notifications import show_error
from functools import partial

from copy import deepcopy
from typing import Optional
from packaging.version import parse as parse_version
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import qthrottled
from napari.components.layerlist import Extent
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Layer, Vectors
from napari.qt import QtViewer
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info

NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version('0.4.16')


def copy_layer_le_4_16(layer: Layer, name: str = ''):
    res_layer = deepcopy(layer)
    # this deepcopy is not optimal for labels and images layers
    if isinstance(layer, (Image, Labels)):
        res_layer.data = layer.data

    res_layer.metadata['viewer_name'] = name

    res_layer.events.disconnect()
    res_layer.events.source = res_layer
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
        emitter.source = res_layer
    return res_layer


def copy_layer(layer: Layer, name: str = ''):
    if not NAPARI_GE_4_16:
        return copy_layer_le_4_16(layer, name)

    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata['viewer_name'] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ('thumbnail', 'name'):
            continue
        if (
                isinstance(getattr(klass, event_name, None), property)
                and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


def center_cross_on_mouse(
        viewer_model: napari.components.viewer_model.ViewerModel,
):
    """move the cross to the mouse position"""

    if not getattr(viewer_model, 'mouse_over_canvas', True):
        # There is no way for napari 0.4.15 to check if mouse is over sending canvas.
        show_info(
            'Mouse is not over the canvas. You may need to click on the canvas.'
        )
        return

    viewer_model.dims.current_step = tuple(
        np.round(
            [
                max(min_, min(p, max_)) / step
                for p, (min_, max_, step) in zip(
                viewer_model.cursor.position, viewer_model.dims.range
            )
            ]
        ).astype(int)
    )


action_manager.register_action(
    name='napari:move_point',
    command=center_cross_on_mouse,
    description='Move dims point to mouse position',
    keymapprovider=ViewerModel,
)

action_manager.bind_shortcut('napari:move_point', 'C')


class own_partial:
    """
    Workaround for deepcopy not copying partial functions
    (Qt widgets are not serializable)
    """

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
            self,
            filenames: list,
            stack: bool,
            plugin: Optional[str] = None,
            layer_type: Optional[str] = None,
            **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class CrossWidget(QCheckBox):
    """
    Widget to control the cross layer. because of the performance reason
    the cross update is throttled
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__('Add cross layer')
        self.viewer = viewer
        self.setChecked(False)
        self.stateChanged.connect(self._update_cross_visibility)
        self.layer = None
        self.viewer.dims.events.order.connect(self.update_cross)
        self.viewer.dims.events.ndim.connect(self._update_ndim)
        self.viewer.dims.events.current_step.connect(self.update_cross)
        self._extent = None

        self._update_extent()
        self.viewer.dims.events.connect(self._update_extent)

    @qthrottled(leading=False)
    def _update_extent(self):
        """
        Calculate the extent of the data.

        Ignores the the cross layer itself in calculating the extent.
        """
        if NAPARI_GE_4_16:
            layers = [
                layer
                for layer in self.viewer.layers
                if layer is not self.layer
            ]
            self._extent = self.viewer.layers.get_extent(layers)
        else:
            extent_list = [
                layer.extent
                for layer in self.viewer.layers
                if layer is not self.layer
            ]
            self._extent = Extent(
                data=None,
                world=self.viewer.layers._get_extent_world(extent_list),
                step=self.viewer.layers._get_step_size(extent_list),
            )
        self.update_cross()

    def _update_ndim(self, event):
        if self.layer in self.viewer.layers:
            self.viewer.layers.remove(self.layer)
        self.layer = Vectors(name='.cross', ndim=event.value, edge_color='#FFFF00')
        # self.layer.edge_width = 1.5
        self.layer.blending = 'opaque'
        self.layer.edge_width = 2.0
        self.layer.vector_style = 'line'
        # self.layer.edge_color = '#FFFF00'
        self.update_cross()

    def _update_cross_visibility(self, state):
        if state:
            self.viewer.layers.append(self.layer)
        else:
            self.viewer.layers.remove(self.layer)
        self.update_cross()

    def update_cross(self):
        if self.layer not in self.viewer.layers:
            return

        point = self.viewer.dims.current_step
        vec = []
        for i, (lower, upper) in enumerate(self._extent.world.T):
            if (upper - lower) / self._extent.step[i] == 1:
                continue
            point1 = list(point)
            point1[i] = (lower + self._extent.step[i] / 2) / self._extent.step[
                i
            ]
            point2 = [0 for _ in point]
            point2[i] = (upper - lower) / self._extent.step[i]
            vec.append((point1, point2))
        if np.any(self.layer.scale != self._extent.step):
            self.layer.scale = self._extent.step
        self.layer.data = vec


class ExampleWidget(QWidget):
    """
    Dummy widget showcasing how to place additional widgets to the right
    of the additional viewers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.btn = QPushButton('Perform action')
        self.spin = QDoubleSpinBox()
        layout = QVBoxLayout()
        layout.addWidget(self.spin)
        layout.addWidget(self.btn)
        layout.addStretch(1)
        self.setLayout(layout)


class MultipleViewerWidget(QSplitter):
    """The main widget of the example."""

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.viewer_model1 = ViewerModel(title='model1')
        self.viewer_model2 = ViewerModel(title='model2')
        # Add the scale bar
        self.viewer_model1.scale_bar.visible = True
        self.viewer_model1.scale_bar.unit = 'nm'
        # Add the scale bar
        self.viewer_model2.scale_bar.visible = True
        self.viewer_model2.scale_bar.unit = 'nm'
        self._block = False
        self.qt_viewer1 = QtViewerWrap(viewer, self.viewer_model1)
        self.qt_viewer2 = QtViewerWrap(viewer, self.viewer_model2)

        ''''''

        # Function to sync the zoom level across all viewers to the one that is being zoomed
        def sync_zoom_to_active(viewers, active_viewer):
            zoom_level = active_viewer.camera.zoom
            # Apply the same zoom level to the other viewers
            for viewer in viewers:
                if viewer != active_viewer:
                    viewer.camera.zoom = zoom_level

        # Function to be used with the zoom event for syncing
        def on_zoom_event(event, viewer, viewers):
            sync_zoom_to_active(viewers, viewer)

        # Function to connect zoom events for syncing
        def connect_zoom_events(viewers):
            for viewer in viewers:
                # Create a partial function with the correct 'viewer' and 'viewers' bound
                event_handler = partial(on_zoom_event, viewer=viewer, viewers=viewers)
                viewer.camera.events.zoom.connect(event_handler)

        # List of all viewers
        viewers = [self.viewer, self.viewer_model1, self.viewer_model2]

        # Connect zoom events for all viewers
        connect_zoom_events(viewers)

        # self.tab_widget = QTabWidget()
        # w1 = ExampleWidget()
        # w2 = ExampleWidget()
        # self.tab_widget.addTab(w1, 'Sample 1')
        # self.tab_widget.addTab(w2, 'Sample 2')

        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Vertical)
        viewer_splitter.addWidget(self.qt_viewer2)
        viewer_splitter.addWidget(self.qt_viewer1)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        # self.setSizes([20, 1])

        self.addWidget(viewer_splitter)

        '''
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(viewer.window._qt_viewer)  # Main viewer
        main_splitter.addWidget(self.qt_viewer1)  # Viewer 1 below main viewer
        main_splitter.setSizes([2, 1])  # Adjust the relative sizes of main viewer and viewer 1

        # Add main_splitter and viewer 2 to the horizontal layout
        self.addWidget(main_splitter)  # Add vertical splitter (main + viewer 1)
        self.addWidget(self.qt_viewer2)  # Viewer 2 to the right of the main splitter

        self.setSizes([3, 1])
        '''

        # self.addWidget(self.tab_widget)

        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_model1.dims.events.current_step.connect(self._point_update)
        self.viewer_model2.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)
        self.viewer_model1.events.status.connect(self._status_update)
        self.viewer_model2.events.status.connect(self._status_update)

    def _status_update(self, event):
        self.viewer.status = event.value

    def _reset_view(self):
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            self.viewer_model1.layers.selection.active = None
            self.viewer_model2.layers.selection.active = None
            return

        self.viewer_model1.layers.selection.active = self.viewer_model1.layers[
            event.value.name
        ]
        self.viewer_model2.layers.selection.active = self.viewer_model2.layers[
            event.value.name
        ]

    def _point_update(self, event):
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if model.dims is event.source:
                continue
            if len(self.viewer.layers) != len(model.layers):
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model1.dims.order = order
            self.viewer_model2.dims.order = order
            return

        order[-3:] = order[-2], order[-1], order[-3]
        self.viewer_model1.dims.order = order
        # order[-3:] = order[-2], order[-3], order[-1]
        # self.viewer_model1.dims.order = order
        # order = list(self.viewer.dims.order)
        # order[-3:] = order[-1], order[-2], order[-3]
        # self.viewer_model2.dims.order = order

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events"""
        self.viewer_model1.layers.insert(
            event.index, copy_layer(event.value, 'model1')
        )
        self.viewer_model2.layers.insert(
            event.index, copy_layer(event.value, 'model2')
        )
        self.viewer.dims.axis_labels = axis_labels
        self.viewer_model1.dims.axis_labels = axis_labels
        self.viewer_model2.dims.axis_labels = axis_labels
        # self.viewer_model1.dims.set_point(0, 0)
        # self.viewer_model1.dims.nsteps = (1,) + self.viewer_model1.dims.nsteps[1:]
        # self.viewer.window._qt_viewer.dims.slider_widgets[3].hide()
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                own_partial(self._property_sync, name)
            )

        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            event.value.events.labels_update.connect(self._set_data_refresh)
            self.viewer_model1.layers[
                event.value.name
            ].events.set_data.connect(self._set_data_refresh)
            self.viewer_model2.layers[
                event.value.name
            ].events.set_data.connect(self._set_data_refresh)
            event.value.events.labels_update.connect(self._set_data_refresh)
            self.viewer_model1.layers[
                event.value.name
            ].events.labels_update.connect(self._set_data_refresh)
            self.viewer_model2.layers[
                event.value.name
            ].events.labels_update.connect(self._set_data_refresh)
        if event.value.name != '.cross':
            self.viewer_model1.layers[event.value.name].events.data.connect(
                self._sync_data
            )
            self.viewer_model2.layers[event.value.name].events.data.connect(
                self._sync_data
            )

        event.value.events.name.connect(self._sync_name)

        self._order_update()

    def _sync_name(self, event):
        """sync name of layers"""
        index = self.viewer.layers.index(event.source)
        self.viewer_model1.layers[index].name = event.source.name
        self.viewer_model2.layers[index].name = event.source.name

    def _sync_data(self, event):
        """sync data modification from additional viewers"""
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """remove layer in all viewers"""
        self.viewer_model1.layers.pop(event.index)
        self.viewer_model2.layers.pop(event.index)

    def _layer_moved(self, event):
        """update order of layers"""
        dest_index = (
            event.new_index
            if event.new_index < event.index
            else event.new_index + 1
        )
        self.viewer_model1.layers.move(event.index, dest_index)
        self.viewer_model2.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            setattr(
                self.viewer_model1.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
            setattr(
                self.viewer_model2.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
        finally:
            self._block = False


def extract_timepoint(filename):
    """Extracts the numeric timepoint before 't' in the filename."""
    match = re.search(r'(\d+)t', filename)
    return int(match.group(1)) if match else float('inf')


def natural_sort_key(path):
    """Generate a sort key that interprets numbers within strings numerically."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.stem)]


def load_z_stack(files, range=None, step_size=1, max_timepoints=math.inf):
    """Load Z-stacks grouped by timepoints into a 4D array."""
    if range is None:
        range = [0]
    timepoint_dict = {}

    # Group files by timepoint
    for file in files:
        timepoint = extract_timepoint(file.name)
        if timepoint is not None:
            timepoint_dict.setdefault(timepoint, []).append(file)

    num_timepoints = len(timepoint_dict)
    if len(range) == 1:
        range.append(num_timepoints)
    sorted_keys = sorted(timepoint_dict.keys())
    if range[0] != 0 or range[1] != num_timepoints:
        sorted_keys = sorted_keys[range[0]:range[1]]
    if step_size > 1:
        sorted_keys = sorted_keys[::step_size]
    if max_timepoints < len(sorted_keys):
        sorted_keys = sorted_keys[-max_timepoints:]

    z_slices = []
    for timepoint in sorted_keys:
        z_files = sorted(timepoint_dict[timepoint], key=lambda f: int(re.search(r'_(\d+)msecAbs_', f.stem).group(1)))
        for z_file in z_files:
            z_slices.append(cpptiff.read_tiff(str(z_file)))

    return np.stack(z_slices, axis=0)


def get_newest_mod_time(folder_path, channel_pattern, paths=None):
    """Get the latest modification time among all TIFF files in the folder."""
    if not paths:
        paths = Path(folder_path).glob('*' + channel_pattern + '*.tif')
    return max((f.stat().st_mtime for f in paths), default=0)


def is_file_old_enough(file_path, newest_mod_time, min_age_seconds=60):
    """Check if a file is at least `min_age_seconds` old or not the newest file in the folder."""
    file_mod_time = file_path.stat().st_mtime
    return (time.time() - file_mod_time > min_age_seconds) or (file_mod_time < newest_mod_time)


@thread_worker
def new_files(folder_paths, channel_patterns, loaded_z_files, min_age_seconds, data, layers, max_timepoints):
    # Monitor folder for new files
    while napari.current_viewer() is not None:
        for folder_path in folder_paths:
            path = Path(folder_path)
            for channel_pattern in channel_patterns:
                paths = path.glob('*' + channel_pattern + '*.tif')
                newest_mod_time = get_newest_mod_time(folder_path, channel_pattern, paths)

                if newest_mod_time:
                    new_z_files = {
                        f for f in paths
                        if f not in loaded_z_files and is_file_old_enough(f, newest_mod_time, min_age_seconds)
                    }
                    if new_z_files:
                        print(f"New Z-stacks found: {new_z_files}")
                        loaded_z_files.update(new_z_files, max_timepoints)

                        # Load only the new files and update combined data
                        if channel_pattern in data:
                            new_data = load_z_stack(new_z_files, max_timepoints=max_timepoints)
                            data[channel_pattern] = np.concatenate((data[channel_pattern], new_data),
                                                                   axis=0)  # Append new data to combined_data
                        else:
                            data[channel_pattern] = load_z_stack(new_z_files, max_timepoints=max_timepoints)
                        if data[channel_pattern].shape[0] > max_timepoints:
                            data[channel_pattern] = data[channel_pattern][-max_timepoints:, :, :, :]
                        if channel_pattern in layers:
                            layers[channel_pattern].data = data[channel_pattern]
                        else:
                            layer_update = {'channel_pattern': channel_pattern, 'data': data[channel_pattern]}
                            layer = yield layer_update
                            layers[channel_pattern] = layer
        time.sleep(1)


class PNGViewerWidget(QWidget):
    def __init__(self, viewer, layer_to_image_map):
        """Widget to show a PNG for the selected layer."""
        super().__init__()

        self.viewer = viewer
        self.layer_to_image_map = layer_to_image_map  # Dict {layer_name: image_path}

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.current_pixmap = None  # Store the original pixmap

        # Connect to Napari layer selection changes
        self.viewer.layers.selection.events.active.connect(self.update_image)

        # Initially update image when widget is created
        self.update_image()

    def update_image(self, event=None):
        """Update the PNG display when the active layer changes."""
        active_layer = self.viewer.layers.selection.active
        if active_layer and active_layer.name in self.layer_to_image_map:
            image_path = self.layer_to_image_map[active_layer.name]
            self.current_pixmap = QPixmap(image_path)
            self.resize_image(self.size())  # Resize immediately
        else:
            self.label.clear()
            self.current_pixmap = None

    def resize_image(self, new_size):
        """Resize the image to 10% of the widget size."""
        if not self.current_pixmap:
            return  # No image loaded

        # Set the image size to 10% of the widget's size
        scale_factor = 0.8  # 10% of widget size

        # Calculate new size based on widget size
        scaled_size = QSize(int(new_size.width() * scale_factor), int(new_size.height() * scale_factor))

        # Scale image while keeping aspect ratio
        scaled_pixmap = self.current_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Handle widget resize events to update the image size."""
        new_size = event.size()  # Get the new size of the widget
        self.resize_image(new_size)
        super().resizeEvent(event)  # Call base method to avoid recursion


# Function to crop data based on bounding box
def crop_3d_with_bounding_box(image_layer, layer, mip_size):
    current_z = viewer.dims.current_step[1]
    mip_half_size = int(mip_size // 2)
    total_slices = image_layer.data.shape[1]

    # Initial range calculations
    z_min = current_z - mip_half_size
    z_max = current_z + mip_half_size + 1

    # Adjust for out-of-bounds on the lower end
    if z_min < 0:
        z_max = min(total_slices, z_max + abs(z_min))
        z_min = 0

    # Adjust for out-of-bounds on the upper end
    if z_max > total_slices:
        z_min = max(0, z_min - (z_max - total_slices))
        z_max = total_slices

    # Check if the range is valid
    if z_max - z_min < mip_size:
        show_error(
            f"Cannot create MIP with size {mip_size}: Not enough slices in the z-dimension."
        )
        return
    if mip_size != 1:
        cropped_data = image_layer.data[:, z_min:z_max, :, :]
        cropped_data = np.max(cropped_data, axis=1)
    else:
        cropped_data = image_layer.data[:, current_z, :, :]
    cropped_data = np.expand_dims(cropped_data, axis=1)
    viewer.add_image(cropped_data, name=layer + ' MIP', scale=voxel_resolution)


if __name__ == "__main__":
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder-paths', type=lambda s: list(map(Path, s.split(','))), required=True,
                    help="Paths to the folder containing Z-stacks separated by comma")
    ap.add_argument('--channel-patterns', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Channel patterns separated by comma")
    ap.add_argument('--voxel-resolution', type=lambda s: tuple(map(int, s.split(','))), required=True,
                    help="Voxel resolution as z,y,x in nm.")
    ap.add_argument('--timepoint-range', type=lambda s: list(map(int, s.split(','))), default=[0],
                    help="Comma separated timepoint range for start and end. End timepoint is optional. Ex. 0,10 or 0")
    ap.add_argument('--timepoint-step-size', type=int, default=1,
                    help="Timpoint step size.")
    ap.add_argument('--max-timepoints', type=float, default=math.inf,
                    help="The max amount of timepoints that will be displayed at a time.")
    args = ap.parse_args()

    folder_paths = args.folder_paths
    channel_patterns = args.channel_patterns
    if len(args.voxel_resolution) != 3:
        ap.error("Voxel resolution must have exactly three values (y,x,z).")
    voxel_resolution = (1,) + args.voxel_resolution
    timepoint_range = args.timepoint_range
    timepoint_step_size = args.timepoint_step_size
    max_timepoints = args.max_timepoints
    axis_labels = ('t', 'z', 'y', 'x')

    viewer = napari.Viewer()
    viewer.window._qt_window.showMaximized()
    # viewer.window._qt_window.showFullScreen()

    # dock_widget1 = SingleViewerWidget(viewer)
    # dock_widget2 = SingleViewerWidget(viewer)
    dock_widget = MultipleViewerWidget(viewer)
    cross = CrossWidget(viewer)

    viewer.window.add_dock_widget(dock_widget, name='yz and xz planes', area='right')
    viewer.window.add_dock_widget(cross, name='Cross', area='left')
    window_geometry = viewer.window.geometry()
    # viewer.window._qt_window.resize(200,200)

    loaded_z_files = set()  # Track loaded files
    min_age_seconds = 60  # Minimum age in seconds before a file is considered complete

    # Track the newest file's modification time
    newest_mod_time = 0
    data = {}
    layers = {}
    channel_patterns_copy = copy.deepcopy(channel_patterns)
    channel_patterns = set()
    # Define the pattern for chunked files
    pattern = re.compile(r'\d+x_\d+y_\d+z')
    for folder_path in folder_paths:
        for channel_pattern in channel_patterns_copy:
            all_files = [
                f for f in Path(folder_path).glob('*' + channel_pattern + '*.tif')
                if is_file_old_enough(f, newest_mod_time, min_age_seconds)
            ]
            # Extract matching patterns and add to the set
            for f in all_files:
                if match := pattern.search(f.name):
                    channel_patterns.add(f'{channel_pattern}*{match.group(0)}')
    if channel_patterns:
        channel_patterns = sorted(channel_patterns)
    else:
        channel_patterns = channel_patterns_copy
    '''
    layer_to_image_map = {}
    for channel_pattern in channel_patterns:
        layer_to_image_map[channel_pattern] = '/clusterfs/nvme2/Data/20240911_Korra_Foundation/20250218_mem_histone/fish1_24hpf_halo_jfx649/roi1/DSH1/DSH_PNG/DSH1_DSH_dx_000y_000z_000t_0000_elapsed_47.758s.png'
    widget = PNGViewerWidget(viewer, layer_to_image_map)
    viewer.window.add_dock_widget(widget, name="Layer PNG Viewer", area="left")
    '''
    pbr = napari.utils.progress(total=len(channel_patterns))

    # Initial load with all existing files
    for folder_path in folder_paths:
        for channel_pattern in channel_patterns:
            newest_mod_time = get_newest_mod_time(folder_path, channel_pattern)
            if newest_mod_time:
                initial_files = [
                    f for f in Path(folder_path).glob('*' + channel_pattern + '*.tif') if
                    is_file_old_enough(f, newest_mod_time, min_age_seconds)
                ]
                print(f"Initial Z-stacks found: {initial_files}")
                loaded_z_files.update(initial_files)
                data[channel_pattern] = load_z_stack(initial_files, timepoint_range, timepoint_step_size,
                                                     max_timepoints)  # Store combined data
                if data[channel_pattern].shape[0] > max_timepoints:
                    data[channel_pattern] = data[channel_pattern][-max_timepoints:, :, :, :]
                pbr.update(1)

                plane_parameters = {
                    'position': (32, 32, 32),
                    'normal': (1, 1, 1),
                    'enabled': True
                }

                layers[channel_pattern] = viewer.add_image(
                    data[channel_pattern],
                    name=channel_pattern,
                    colormap="gray",
                    scale=voxel_resolution
                )
                order = list(viewer.dims.order)
                if order == [0, 1, 2, 3]:
                    order[-3:] = order[-1], order[-2], order[-3]
                    viewer.dims.order = order
                viewer.dims.axis_labels = axis_labels
    pbr.close()
    worker = new_files(folder_paths, channel_patterns, loaded_z_files, min_age_seconds, data, layers, max_timepoints)

    def on_layer_change(event):
        selected = list(viewer.layers.selection)
        if not selected:
            return

        current_layer = selected[0]
        layer_name = current_layer.name
        if layer_name not in channel_patterns:
            return
        curr_data_quality_audit = Path(os.path.join(folder_paths[0], f'data_quality_audit_{layer_name}.json'))
        with curr_data_quality_audit.open() as f:
            data_quality_audit_new_dict = json.load(f)
            audit_widget.status.value = data_quality_audit_new_dict.get("quality_metric", "keep")
            audit_widget.until.value = data_quality_audit_new_dict.get("keep_until", 0)
            audit_widget.notes.value = data_quality_audit_new_dict.get("auditor_notes", "")
    viewer.layers.selection.events.changed.connect(on_layer_change)

    # Path to save the JSON file
    data_quality_audit = Path(os.path.join(folder_paths[0], f'data_quality_audit_{channel_patterns[-1]}.json'))

    def save_to_json(status: str, until: int, notes: str):
        """Save the current radio button and notes to a JSON file."""
        data = {"quality_metric": status, 'keep_until': until, "auditor_notes": notes}
        selected = list(viewer.layers.selection)
        if not selected:
            return

        current_layer = selected[0]
        layer_name = current_layer.name
        if layer_name not in channel_patterns:
            return
        curr_data_quality_audit = Path(os.path.join(folder_paths[0], f'data_quality_audit_{layer_name}.json'))
        with curr_data_quality_audit.open("w") as f:
            json.dump(data, f, indent=4)
        os.chmod(curr_data_quality_audit, 0o775)

    quality_metric = 'keep'
    keep_until_max = int((len(all_files) / len(channel_patterns)) * len(channel_patterns_copy))
    keep_until = keep_until_max
    auditor_notes = ''

    if os.path.exists(data_quality_audit):
        with data_quality_audit.open() as f:
            data_quality_audit_dict = json.load(f)
            quality_metric = data_quality_audit_dict['quality_metric']
            keep_until = data_quality_audit_dict['keep_until']
            auditor_notes = data_quality_audit_dict['auditor_notes']
    for channel_pattern in channel_patterns:
        curr_json_file = Path(os.path.join(folder_paths[0], f'data_quality_audit_{channel_pattern}.json'))
        if not os.path.exists(curr_json_file):
            data = {"quality_metric": quality_metric, "keep_until": keep_until, "auditor_notes": auditor_notes}
            with curr_json_file.open("w") as f:
                json.dump(data, f, indent=4)
            os.chmod(curr_json_file, 0o775)

    @magicgui(
        status={"label": "Quality Metric", "widget_type": "RadioButtons", "choices": ["keep", "maybe", "kill"],
                "value": quality_metric},
        until={"label": "Keep Until", "widget_type": "SpinBox", "min": 0, "max": keep_until_max, "value": keep_until,
               "step": 1},
        notes={"label": "Auditor Notes", "widget_type": "TextEdit", "value": auditor_notes},
        call_button="Save"
    )
    def audit_widget(status: str, until: int, notes: str):
        """Napari widget with radio buttons and a text box."""
        save_to_json(status, until, notes)


    def update_status(value):
        """Callback for status changes."""
        save_to_json(audit_widget.status.value, audit_widget.until.value, audit_widget.notes.value)


    def update_until(value):
        """Callback for keep_until changes."""
        save_to_json(audit_widget.status.value, audit_widget.until.value, audit_widget.notes.value)


    def update_notes():
        """Callback for notes changes."""
        save_to_json(audit_widget.status.value, audit_widget.until.value, audit_widget.notes.value)


    # Connect signals
    audit_widget.status.changed.connect(update_status)
    audit_widget.until.changed.connect(update_until)
    audit_widget.notes.native.textChanged.connect(update_notes)

    viewer.window.add_dock_widget(audit_widget, name="Audit", area="left")


    # Define a widget using magicgui
    @magicgui(
        call_button="Generate MIP",
        layer={"choices": lambda widget: [layer.name for layer in viewer.layers], "label": "Layer"},
        mip_size={"label": "Z Planes", "min": 1, "step": 2}
    )
    def crop_widget(
            layer: str,
            mip_size: int = 1
    ):
        if mip_size % 2 == 0:
            show_error("MIP Size must be an odd number.")
            return
        selected_layer = layers[layer]
        crop_3d_with_bounding_box(selected_layer, layer, mip_size)


    # Add the widget to the Napari viewer
    viewer.window.add_dock_widget(crop_widget, name="Create MIP", area="left")


    @worker.yielded.connect
    def update_layer(layer_update):
        """Update Napari layer with new data."""
        layers[layer_update['channel_pattern']] = viewer.add_image(
            layer_update['data'],
            name=layer_update['channel_pattern'],
            colormap="gray",
            scale=voxel_resolution
        )
        order = list(viewer.dims.order)
        if order == [0, 1, 2, 3]:
            order[-3:] = order[-1], order[-2], order[-3]
            viewer.dims.order = order
        return layers[layer_update['channel_pattern']]


    # Bind the function to a mouse click event
    '''
    @viewer.mouse_drag_callbacks.append
    def handle_mouse_event(viewer, event):
        if event.type == 'mouse_press' and event.button == 1:  # Left mouse button
            center_cross_on_mouse(viewer)
    '''

    # Add the scale bar
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'nm'

    worker.start()
    napari.run()
