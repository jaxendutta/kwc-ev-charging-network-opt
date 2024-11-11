"""Initialize the project package."""
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create necessary directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

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