"""
This module defines constants and file paths used throughout the KW EV Charging Optimization project.

Attributes:
    PROJECT_ROOT (Path): The root directory of the project.
    DATA_DIR (Path): The directory where data is stored.
    RAW_DATA_DIR (Path): The directory where raw data is stored.
    PROCESSED_DATA_DIR (Path): The directory where processed data is stored.
    NOTEBOOKS_DIR (Path): The directory where Jupyter notebooks are stored.
    SRC_MODELS_DIR (Path): The directory where model source code is stored.
    SRC_DATA_DIR (Path): The directory where data source code is stored.
    SRC_VISUALIZATION_DIR (Path): The directory where visualization source code is stored.
    DATA_PATHS (dict): A dictionary containing paths to various data files.
        Keys:
            'charging_stations' (Path): Path to the charging stations data.
            'potential_locations' (Path): Path to the potential locations data.
            'population_density' (Path): Path to the population density data.
            'grid_capacity' (Path): Path to the grid capacity data.
            'all_locations' (Path): Path to the processed all locations data.
            'population_analysis' (Path): Path to the processed population analysis data.
    KW_BOUNDS (dict): A dictionary containing the geographical boundaries of the KW region.
        Keys:
            'north' (float): Northern boundary of Waterloo.
            'south' (float): Southern boundary of Kitchener.
            'east' (float): Eastern boundary.
            'west' (float): Western boundary.
    KW_CENTER (list): The geographical center of the KW region.
        Elements:
            0 (float): Latitude of the center.
            1 (float): Longitude of the center.
    SEARCH_RADIUS_KM (int): The radius (in kilometers) for searching amenities.
    API_TIMEOUT (int): The timeout duration (in seconds) for API requests.
    MAX_RESULTS (int): The maximum number of results to return from API requests.
"""

from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
SRC_MODELS_DIR = PROJECT_ROOT / 'src/models'
SRC_DATA_DIR = PROJECT_ROOT / 'src/data'
SRC_VISUALIZATION_DIR = PROJECT_ROOT / 'src/visualization'

# File paths
DATA_PATHS = {
    'charging_stations': RAW_DATA_DIR / 'charging_stations',
    'potential_locations': RAW_DATA_DIR / 'potential_locations',
    'population_density': RAW_DATA_DIR / 'population_density',
    'grid_capacity': RAW_DATA_DIR / 'grid_capacity',
    'all_locations': PROCESSED_DATA_DIR / 'all_locations',
    'analyzed_locations': PROCESSED_DATA_DIR / 'analyzed_locations',
    'population_analysis': PROCESSED_DATA_DIR / 'population_analysis'
}

# Source code files
SOURCE_CODE_FILES = [
    NOTEBOOKS_DIR / '01_data_collection_and_exploration.ipynb',
    NOTEBOOKS_DIR / '02_demand_analysis.ipynb',
    SRC_DATA_DIR / 'utils.py',
    SRC_DATA_DIR / 'constants.py',
    SRC_DATA_DIR / 'api_client.py',
    SRC_MODELS_DIR / 'facility_location.py',
    SRC_VISUALIZATION_DIR / 'map_viz.py'
]

# KW Region bounds
KW_BOUNDS = {
    'north': 43.5445,  # Northern boundary of Waterloo
    'south': 43.3839,  # Southern boundary of Kitchener
    'east': -80.4013,  # Eastern boundary
    'west': -80.6247   # Western boundary
}

# Area parameters
KW_CENTER = [43.4516, -80.4925]  # Center of KW region
SEARCH_RADIUS_KM = 10  # Radius for searching amenities

# API configurations
API_TIMEOUT = 10  # seconds
MAX_RESULTS = 500