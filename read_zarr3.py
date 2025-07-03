import argparse
import json
import os

import napari
import numpy as np
import tensorstore as ts

if __name__ == "__main__":
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr-path', type=str, required=True,
                    help="Paths to the Zarr File")
    ap.add_argument('--timepoint-range', type=lambda s: list(map(int, s.split(','))), default=[0],
                    help="Comma separated timepoint range for start and end. End timepoint is optional. Ex. 0,10 or 0")
    ap.add_argument('--channel-range', type=lambda s: list(map(int, s.split(','))), default=[0],
                    help="Comma separated channel range for start and end. End channel is optional. Ex. 0,10 or 0")
    args = ap.parse_args()

    zarr_path = args.zarr_path
    timepoint_range = args.timepoint_range
    channel_range = args.channel_range
    pixel_sizes = [.097, .097, .097]
    base_name = os.path.basename(zarr_path)
    with open(f'{os.path.join(os.path.dirname(zarr_path), "metadata.json")}', 'r') as json_file:
        metadata = json.load(json_file)

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
        timepoint_range.append(dataset.domain.shape[0])

    if len(channel_range) == 1:
        channel_range.append(dataset.domain.shape[4])

    cube_size = metadata['cube_size']
    bbox = metadata['training_images'][base_name]['bbox']
    num_z_chunks = int(((dataset.shape[1]) / cube_size))
    num_y_chunks = int(((dataset.shape[2]) / cube_size))
    num_x_chunks = int(((dataset.shape[3]) / cube_size))
    num_chunks = num_z_chunks*num_y_chunks*num_x_chunks

    if metadata.get('augmentation_info'):
        num_z_chunks = metadata['augmentation_info']['num_z_chunks']
        num_y_chunks = metadata['augmentation_info']['num_y_chunks']
        num_x_chunks = metadata['augmentation_info']['num_x_chunks']
        num_chunks = metadata['augmentation_info']['num_chunks']

    viewer = napari.Viewer()
    for i in range(channel_range[0], channel_range[1]):
        image_data = np.zeros((num_chunks, timepoint_range[1] - timepoint_range[0], metadata['cube_size'],
                               metadata['cube_size'], metadata['cube_size']), dtype=np.uint16)
        curr_chunk = 0
        # Read the dataset as a NumPy array
        for z in range(num_z_chunks):
            for y in range(num_y_chunks):
                for x in range(num_x_chunks):
                    image_data[curr_chunk, ...] = dataset[timepoint_range[0]:timepoint_range[1], z*cube_size:(z+1)*cube_size, y*cube_size:(y+1)*cube_size, x*cube_size:(x+1)*cube_size, i].read().result()
                    curr_chunk += 1
        channel_name = metadata['channelPatterns'][i]
        viewer.add_image(image_data, name=channel_name, scale=(1, 1, pixel_sizes[0], pixel_sizes[1], pixel_sizes[2]))
    axis_labels = ('chunk', 't', 'z', 'y', 'x')
    viewer.dims.axis_labels = axis_labels
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'

    napari.run()
