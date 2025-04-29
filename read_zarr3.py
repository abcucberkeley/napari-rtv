import argparse
import json
import os
import napari
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
        timepoint_range.append(dataset.domain.shape[1])

    if len(channel_range) == 1:
        channel_range.append(dataset.domain.shape[5])

    viewer = napari.Viewer()
    for i in range(channel_range[0], channel_range[1]):
        # Read the dataset as a NumPy array
        image_data = dataset[:, timepoint_range[0]:timepoint_range[1], ..., i].read().result()
        channel_name = metadata['channelPatterns'][i]
        viewer.add_image(image_data, name=channel_name, scale=(1, 1, pixel_sizes[0], pixel_sizes[1], pixel_sizes[2]))
    axis_labels = ('chunk', 't', 'z', 'y', 'x')
    viewer.dims.axis_labels = axis_labels

    napari.run()
