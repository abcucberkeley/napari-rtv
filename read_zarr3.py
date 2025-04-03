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
    args = ap.parse_args()

    zarr_path = args.zarr_path
    timepoint_range = args.timepoint_range

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

    # Read the dataset as a NumPy array
    image_data = dataset[:, timepoint_range[0]:timepoint_range[1], ..., 0].read().result()

    # Launch Napari with the loaded image
    viewer = napari.Viewer()
    viewer.add_image(image_data, name="Chunks", scale=(1, 1, metadata['dz'], metadata['xyPixelSize'], metadata['xyPixelSize']))
    axis_labels = ('chunk', 't', 'z', 'y', 'x')
    viewer.dims.axis_labels = axis_labels

    napari.run()
