'''
Utility functions for file handling and data processing.

Functions:
- grab_time: Get the current time in the format YYYY-MM-DD_HH-MM-SS.
- get_file_timestamp: Get the timestamp of a file.
- flatten_timestamp: Flatten the timestamp to a file name format.
- get_latest_file: Get the latest file in a directory.
- load_latest_file: Load the latest file in a directory based on its file type.
- save_data: Save a dataframe to a specified file type.

Functions:

grab_time:
    Get the current time in the format YYYY-MM-DD_HH-MM-SS.
get_file_timestamp:
    Get the timestamp of a file.
flatten_timestamp:
    Flatten the timestamp to a file name format.
get_latest_file:
    Get the latest file in a directory.
    - directory: The directory to search for files.
    - file_type: Optional. The file type to filter by (e.g., 'csv', 'json').
    - latest_file: The latest file in the directory.
    - timestamp: The timestamp of the latest file.
load_latest_file:
    Load the latest file in a directory based on its file type.
    - file_path: The path to the file.
    - data: The loaded data, which could be a DataFrame, GeoDataFrame, or dict.
save_data:
    Save a dataframe to a specified file type.
    - data: The data to be saved.
    - data_type: The type of data, which should be one of the keys in DATA_PATHS.
    - file_type: The file format to save the data in.
    - timestamp: Optional. The timestamp to use in the file name.
    - output_file: Optional. The full path to save the file.
'''

import glob
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from .constants import *

# Get time right now in the format YYYY-MM-DD_HH-MM-SS
grab_time = lambda: datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Get the timestamp of a file
get_file_timestamp = lambda x: datetime.fromtimestamp(Path(x).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

# Flatten the timestamp to file name format
flatten_timestamp = lambda x: x.replace(' ', '_').replace(':', '-')

# Get latest file in a directory
def get_latest_file(directory, file_type=None):
    directory = Path(directory)
    if file_type:
        files = sorted(directory.glob(f'*.{file_type}'), key=lambda f: f.stat().st_mtime, reverse=True)
    else:
        files = sorted(directory.glob('*'), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if not files:
        raise FileNotFoundError(f"No files found in directory: {directory}")
    
    latest_file = files[0]
    timestamp = get_file_timestamp(latest_file)
    
    return latest_file, timestamp

def load_latest_file(directory, file_type=None):
    file_path, _ = get_latest_file(directory, file_type)
    file_path = Path(file_path)
    file_type = file_path.suffix[1:]
    
    if file_type == 'csv':
        data = pd.read_csv(file_path)
    elif file_type == 'geojson' or file_type == 'gpkg':
        data = gpd.read_file(file_path)
    elif file_type == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"File type {file_type} not supported")
    
    return data

def save_data(data, data_type, file_type, timestamp=None, output_file=None):
    # Get the data path
    data_path = DATA_PATHS[data_type]
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")
    
    # Get the file name
    if timestamp is None:
        timestamp = grab_time()
    file_name = f"{data_type}_{timestamp}.{file_type}"
    
    # Get the full file path
    file_path = data_path / file_name
    if output_file is not None:
        file_path = Path(output_file)
    
    # Save the data
    if file_type == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, str):
            with open(file_path, 'w') as f:
                f.write(data)
        else:
            raise ValueError(f"Data type {type(data)} not supported for file type {file_type}")
    elif file_type == 'geojson':
        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(file_path, driver='GeoJSON')
        else:
            raise ValueError(f"Data type {type(data)} not supported for file type {file_type}")
    elif file_type == 'json':
        if isinstance(data, dict):
            with open(file_path, 'w') as f:
                json.dump(data, f)
        else:
            raise ValueError(f"Data type {type(data)} not supported for file type {file_type}")
    elif file_type == 'gpkg':
        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(file_path, driver='GPKG')
        else:
            raise ValueError(f"Data type {type(data)} not supported for file type {file_type}")
    else:
        raise ValueError(f"File type {file_type} not supported")
    
    return file_path, get_file_timestamp(file_path)