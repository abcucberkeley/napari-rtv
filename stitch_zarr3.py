import argparse
import napari
import tensorstore as ts
import os
import json
import re
import numpy as np

if __name__ == "__main__":
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr-path', type=str, required=True,
                    help="Paths to the Zarr File")
    ap.add_argument('--timepoint-range', type=lambda s: list(map(int, s.split(','))), default=[0],
                    help="Comma separated timepoint range for start and end. End timepoint is optional. Ex. 0,10 or 0")
    args = ap.parse_args()

    zarr_path = args.zarr_path
    timepoint_range = args.timepoint_range
    base_name = os.path.basename(zarr_path)
    with open(f'{os.path.join(os.path.dirname(zarr_path), "metadata.json")}', 'r') as json_file:
        metadata = json.load(json_file)
    chunks = metadata['training_images'][base_name]

    dataset = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": zarr_path
            }
        },
        read=True
    ).result()

    if len(timepoint_range) == 1:
        timepoint_range.append(dataset.domain.shape[1])

    z_min = float('inf')
    y_min = float('inf')
    x_min = float('inf')
    z_max = 0
    y_max = 0
    x_max = 0
    for chunk_name, chunk in chunks['chunk_names'].items():
        z_min = min(z_min, chunk['bbox'][0])
        y_min = min(y_min, chunk['bbox'][1])
        x_min = min(x_min, chunk['bbox'][2])
        z_max = max(z_max, chunk['bbox'][3])
        y_max = max(y_max, chunk['bbox'][4])
        x_max = max(x_max, chunk['bbox'][5])

    stitched_image = np.zeros((timepoint_range[1] - timepoint_range[0], z_max - z_min, y_max - y_min, x_max - x_min),
                              dtype=np.uint16)
    border_mask = np.zeros_like(stitched_image[0,...])

    image_data = dataset[:, timepoint_range[0]:timepoint_range[1], ..., 0].read().result()

    # Create a list to store text positions and labels
    chunk_positions = []
    chunk_labels = []
    delimiters = r"[./]"
    for chunk_name, chunk in chunks['chunk_names'].items():
        curr_chunk_list = re.split(delimiters, chunk_name)
        curr_chunk_list.remove('c')
        if int(curr_chunk_list[1]) != 0:
            continue

        # Get the chunk's position within stitched_image
        z_start, y_start, x_start = chunk['bbox'][0] - z_min, chunk['bbox'][1] - y_min, chunk['bbox'][2] - x_min
        z_end, y_end, x_end = chunk['bbox'][3] - z_min, chunk['bbox'][4] - y_min, chunk['bbox'][5] - x_min

        # Compute chunk center
        y_center = (y_start + y_end) // 2
        x_center = (x_start + x_end) // 2

        # Add a point at every Z plane
        for z in range(chunk['bbox'][0] - z_min, chunk['bbox'][3] - z_min):
            chunk_positions.append([z, y_center, x_center])  # Ensure order is (Z, Y, X)
            chunk_labels.append(f'{curr_chunk_list[0]}')  # Keep the corresponding chunk label

        # Copy chunk data
        stitched_image[:, z_start:z_end, y_start:y_end, x_start:x_end] = \
            image_data[int(curr_chunk_list[0]), timepoint_range[0]:timepoint_range[1], ...]

        # Add white borders around the chunk
        border_mask[z_start:z_end, y_start, x_start:x_end] = 65535  # Top face (y=0)
        border_mask[z_start:z_end, y_end - 1, x_start:x_end] = 65535  # Bottom face (y=max)
        border_mask[z_start:z_end, y_start:y_end, x_start] = 65535  # Left face (x=0)
        border_mask[z_start:z_end, y_start:y_end, x_end - 1] = 65535  # Right face (x=max)

    # Convert to NumPy arrays for Napari
    chunk_positions = np.array(chunk_positions)
    #chunk_labels = [str(label) for label in chunk_labels]

    # Launch Napari with the loaded image
    viewer = napari.Viewer()
    viewer.add_image(stitched_image, name="Stitched Image",
                     scale=(1, metadata['dz'], metadata['xyPixelSize'], metadata['xyPixelSize']))
    # Add the border as a separate overlay layer
    viewer.add_image(border_mask, name="Chunk Borders",
                     scale=(metadata['dz'], metadata['xyPixelSize'], metadata['xyPixelSize']),
                     opacity=.25, blending="additive", colormap="gray")
    # Add text labels at chunk centers
    viewer.add_points(chunk_positions, name="Chunk Numbers",
                      scale=(metadata['dz'], metadata['xyPixelSize'], metadata['xyPixelSize']),
                      size=0, face_color="white", border_color="black", opacity=.25,
                      text={"string": chunk_labels, "size": 12, "color": "white"})
    axis_labels = ('t', 'z', 'y', 'x')
    viewer.dims.axis_labels = axis_labels

    napari.run()
