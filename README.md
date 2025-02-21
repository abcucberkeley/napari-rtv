# napari-rtv

## Conda

### Installation
````
git clone https://github.com/abcucberkeley/napari-rtv
cd napari-rtv
conda env create -f environment.yaml
conda activate napari-rtv
````
### Usage

You can call the napari_rtv.py script from the napari-rtv folder

#### Monitor a single folder and channel pattern
````
python napari_rtv.py --folder-paths /path/to/folder1 --channel-patterns pattern1 --voxel-resolution 108,108,108
````

#### Monitor multiple folders and channel patterns
````
python napari_rtv.py --folder-paths /path/to/folder1,/path/to/folder2 --channel-patterns pattern1,pattern2 --voxel-resolution 108,108,108
````

#### Monitor a single folder and channel pattern with a maximum amount of timepoints (latest timepoints will be used)
````
python napari_rtv.py --folder-paths /path/to/folder1 --channel-patterns pattern1 --voxel-resolution 108,108,108 --max-timepoints 2
````
