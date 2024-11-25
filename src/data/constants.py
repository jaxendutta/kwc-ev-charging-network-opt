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
import folium

# Supported folium colours
FOLIUM_COLORS = (lambda: list(folium.Icon.color_options))()

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
RESULTS_DIR = PROJECT_ROOT / 'results'
SRC_MODEL_DIR = PROJECT_ROOT / 'src/model'
SRC_DATA_DIR = PROJECT_ROOT / 'src/data'
SRC_VISUALIZATION_DIR = PROJECT_ROOT / 'src/visualization'

# File paths
DATA_PATHS = {
    'charging_stations': RAW_DATA_DIR / 'charging_stations',
    'potential_locations': RAW_DATA_DIR / 'potential_locations',
    'population': RAW_DATA_DIR / 'population',
    'ev_fsa': RAW_DATA_DIR / 'ev_fsa',
    'boundaries': RAW_DATA_DIR / 'boundaries',
    'integrated_analyzed_data': PROCESSED_DATA_DIR / 'integrated_analyzed_data',
    'demand_points': PROCESSED_DATA_DIR / 'demand_points',
    'ev_fsa_analyzed': PROCESSED_DATA_DIR / 'ev_fsa_analyzed',
    'optimization_inputs': PROCESSED_DATA_DIR / 'optimization_inputs',
}

# Source code files
SOURCE_CODE_FILES = [
    NOTEBOOKS_DIR / '01_data_collection.ipynb',
    NOTEBOOKS_DIR / '02_location_analysis.ipynb',
    NOTEBOOKS_DIR / '03_enhancement_analysis.ipynb',
    NOTEBOOKS_DIR / '04_data_preparation.ipynb',
    NOTEBOOKS_DIR / '05_optimization_model.ipynb',
    SRC_DATA_DIR / 'utils.py',
    SRC_DATA_DIR / 'constants.py',
    SRC_DATA_DIR / 'data_manager.py',
    SRC_MODEL_DIR / 'network_optimizer.py',
    SRC_VISUALIZATION_DIR / 'map_viz.py',
    SRC_VISUALIZATION_DIR / 'optimization_viz.py'
]

# Area parameters
KW_CENTER = [43.4516, -80.4925]  # Center of KW region
SEARCH_RADIUS_KM = 10  # Radius for searching amenities

# KWC Cities and Townships
KWC_CITIES = [
    'Waterloo, Ontario',
    'Kitchener, Ontario',
    'Cambridge, Ontario',
    'Wilmot, Ontario', 
    'Woolwich, Ontario', 
    'North Dumfries, Ontario', 
    'Wellesley, Ontario']

# KWC_SORTATION_AREAS
FSA_CODES = {
    'Waterloo': ['N2J', 'N2K', 'N2L', 'N2T', 'N2V', 'N2M'],                                    # N2J to N2L, N2T, N2V, N2M
    'Kitchener': ['N2A', 'N2B', 'N2C', 'N2E', 'N2G', 'N2H', 'N2M', 'N2N', 'N2P', 'N2R'],       # N2A to N2H, N2M to N2R
    'Cambridge': ['N1P', 'N1R', 'N1S', 'N1T', 'N3C', 'N3E', 'N3H'],                            # N1P to N1T, N3C to N3E, N3H
    'North Dumfries': ['N0B'],                    # N0B
    'Wellesley': ['N0B'],                         # N0B
    'Wilmot': ['N3A'],                            # N3A
    'Woolwich': ['N0B', 'N3A', 'N3B'],            # N0B, N3A, N3B
}

# Combine overlapping sortation areas
ALL_FSA_CODES = set()
for areas in FSA_CODES.values():
    ALL_FSA_CODES.update(areas)

ALL_FSA_CODES = list(ALL_FSA_CODES)


# API configurations
API_TIMEOUT = 10  # seconds
MAX_RESULTS = 500

# Lcation type dictionary
grouped_location_types = {
    # Parking
    'parking': 'parking',

    # Retail
    'antiques': 'retail',
    'appliance': 'retail',
    'cannabis': 'retail',
    'clothes': 'retail',
    'computer;electronics': 'retail',
    'convenience': 'retail',
    'craft': 'retail',
    'department_store': 'retail',
    'doityourself': 'retail',
    'e-cigarette': 'retail',
    'electronics': 'retail',
    'fabric': 'retail',
    'furniture': 'retail',
    'garden_centre': 'retail',
    'general': 'retail',
    'hardware': 'retail',
    'interior_decoration': 'retail',
    'machine_tools': 'retail',
    'mall': 'retail',
    'marketplace': 'retail',
    'musical_instrument': 'retail',
    'outdoor': 'retail',
    'pet': 'retail',
    'retail': 'retail',
    'second_hand': 'retail',
    'shoes': 'retail',
    'stationery': 'retail',
    'toys': 'retail',
    'vacuum_cleaner': 'retail',
    'variety_store': 'retail',
    'wholesale': 'retail',

    # Commercial
    'commercial': 'commercial',

    # Fuel
    'fuel': 'fuel',

    # Supermarket
    'supermarket': 'supermarket',

    # Food
    'bakery': 'food',
    'butcher': 'food',
    'cafe': 'food',
    'confectionery': 'food',
    'deli': 'food',
    'farm': 'food',
    'fast_food': 'food',
    'frozen_food': 'food',
    'greengrocer': 'food',
    'ice_cream': 'food',
    'nutrition_supplements': 'food',
    'restaurant': 'food',

    # Automotive
    'bicycle': 'automotive',
    'car': 'automotive',
    'car_repair': 'automotive',
    'car_wash': 'automotive',
    'caravan': 'automotive',
    'motorcycle': 'automotive',
    'tyres': 'automotive',

    # Charging Stations
    'charging_station': 'charging_station',

    # Services
    'bank': 'services',
    'charity': 'services',
    'clinic': 'services',
    'collector': 'services',
    'community_centre': 'services',
    'copyshop': 'services',
    'dentist': 'services',
    'doctors': 'services',
    'drinking_water': 'services',
    'driving_school': 'services',
    'funeral_directors': 'services',
    'groundskeeping': 'services',
    'hairdresser': 'services',
    'locksmith': 'services',
    'optician': 'services',
    'pet_grooming': 'services',
    'pharmacy': 'services',
    'product_pickup': 'services',
    'rental': 'services',
    'spa': 'services',
    'tailor': 'services',
    'taxi': 'services',
    'veterinary': 'services',
    'weight_loss': 'services',

    # Vacant
    'vacant': 'vacant',

    # Entertainment
    'alcohol': 'entertainment',
    'art': 'entertainment',
    'bar': 'entertainment',
    'brothel': 'entertainment',
    'cinema': 'entertainment',
    'dojo': 'entertainment',
    'erotic': 'entertainment',
    'events_venue': 'entertainment',
    'music': 'entertainment',
    'nightclub': 'entertainment',
    'pub': 'entertainment',
    'sports': 'entertainment',
    'theatre': 'entertainment',
    'wine': 'entertainment',
}

# Icons Dictionary
icons = {
    'school': 'graduation-cap',
    'park': 'tree',
    'restaurant': 'cutlery',
    'shopping': 'shopping-cart',
    'parking': 'car',
    'retail': 'shopping-cart',
    'commercial': 'building',
    'fuel': 'gas-pump',
    'supermarket': 'shopping-basket',
    'fast_food': 'hamburger',
    'charging_station': 'bolt',
    'cafe': 'coffee',
    'convenience': 'shopping-bag',
    'car': 'car',
    'bank': 'university',
    'alcohol': 'glass-martini',
    'pharmacy': 'medkit',
    'car_repair': 'wrench',
    'doityourself': 'toolbox',
    'clothes': 'tshirt',
    'pub': 'beer',
    'mall': 'shopping-mall',
    'department_store': 'store-alt',
    'variety_store': 'store-alt',
    'hairdresser': 'cut',
    'furniture': 'couch',
    'dentist': 'tooth',
    'electronics': 'tv',
    'marketplace': 'store',
    'vacant': 'ban',
    'car_wash': 'shower',
    'cinema': 'film',
    'wholesale': 'warehouse',
    'hardware': 'hammer',
    'stationery': 'pencil-alt',
    'cannabis': 'cannabis',
    'confectionery': 'candy-cane',
    'veterinary': 'paw',
    'caravan': 'caravan',
    'sports': 'futbol',
    'charity': 'hand-holding-heart',
    'optician': 'glasses',
    'bakery': 'bread-slice',
    'shoes': 'shoe-prints',
    'second_hand': 'recycle',
    'general': 'store',
    'garden_centre': 'seedling',
    'e-cigarette': 'smoking',
    'doctors': 'user-md',
    'tyres': 'car',
    'craft': 'paint-brush',
    'toys': 'puzzle-piece',
    'pet': 'paw',
    'bicycle': 'bicycle',
    'funeral_directors': 'cross',
    'wine': 'wine-glass-alt',
    'butcher': 'drumstick-bite',
    'spa': 'spa',
    'theatre': 'theater-masks',
    'outdoor': 'tree',
    'frozen_food': 'snowflake',
    'vacuum_cleaner': 'broom',
    'drinking_water': 'tint',
    'machine_tools': 'cogs',
    'antiques': 'hourglass',
    'events_venue': 'calendar-alt',
    'copyshop': 'copy',
    'groundskeeping': 'leaf',
    'clinic': 'clinic-medical',
    'erotic': 'heart',
    'pet_grooming': 'paw',
    'dojo': 'yin-yang',
    'appliance': 'plug',
    'farm': 'tractor',
    'motorcycle': 'motorcycle',
    'deli': 'utensils',
    'brothel': 'bed',
    'locksmith': 'key',
    'interior_decoration': 'paint-roller',
    'art': 'palette',
    'rental': 'home',
    'computer_electronics': 'laptop',
    'fabric': 'tshirt',
    'weight_loss': 'weight',
    'greengrocer': 'apple-alt',
    'bar': 'glass-martini',
    'collector': 'coins',
    'musical_instrument': 'guitar',
    'ice_cream': 'ice-cream',
    'tailor': 'cut',
    'product_pickup': 'box',
    'nightclub': 'music',
    'community_centre': 'users',
    'taxi': 'taxi',
    'music': 'music',
    'driving_school': 'car',
    'nutrition_supplements': 'capsules'
}