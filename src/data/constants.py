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
CHARGING_STATIONS_DIR = RAW_DATA_DIR / 'charging_stations'
POTENTIAL_LOCATIONS_DIR = RAW_DATA_DIR / 'potential_locations'
POPULATION_DENSITY_DIR = RAW_DATA_DIR / 'population_density'
GRID_CAPACITY_DIR = RAW_DATA_DIR / 'grid_capacity'

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