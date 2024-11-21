"""
Configuration file containing all data endpoints for the KW-CMA region.
"""
# Boundary of the KW region
ROW_BOUNDS = {
    'boundary': {
        'url': 'https://utility.arcgis.com/usrsvcs/servers/259750cc5e8c44f78c56d27ac28b04ed/rest/services/OpenData/OpenData/MapServer/16/query',
        'formats': ['geojson', 'json']
    }
}

# Census Data Endpoints
CENSUS_ENDPOINTS = {
    'housing': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Census_2021_Housing/FeatureServer/1/query',
        'formats': ['geojson', 'json']
    },
    'income': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Census_2021_Income/FeatureServer/1/query',
        'formats': ['geojson', 'json']
    },
    'labour': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Census_2021_Labour/FeatureServer/1/query',
        'formats': ['geojson', 'json']
    }
}

# Infrastructure Endpoints
INFRASTRUCTURE_ENDPOINTS = {
    'roads': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Roads/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    },
    'trails': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Trails/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    },
    'parking': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Parking_Public_Lots/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    },
    'buildings': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Building_Outlines/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    },
    'poi': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Points_of_Interest/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    }
}

# Land Use Endpoints
LAND_USE_ENDPOINTS = {
    'business_park': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Business_Park/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    },
    'business_areas': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Business_Improvement_Areas/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    },
    'downtown': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Downtown_Boundary/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    },
    'parks': {
        'url': 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Parks/FeatureServer/0/query',
        'formats': ['geojson', 'json']
    }
}

# Transportation Endpoints
TRANSPORTATION_ENDPOINTS = {
    'grt_routes': {
        'url': 'https://utility.arcgis.com/usrsvcs/servers/16e0ab66dadf4044a5be144e9d88effb/rest/services/OpenData/OpenData/MapServer/4/query',
        'formats': ['json', 'geojson']
    },
    'grt_stops': {
        'url': 'https://utility.arcgis.com/usrsvcs/servers/52c4134809a94f85b31a2e9553de1358/rest/services/OpenData/OpenData/MapServer/3/query',
        'formats': ['json', 'geojson']
    },
    'ion_routes': {
        'url': 'https://utility.arcgis.com/usrsvcs/servers/1aea43272d854f59a4d5d710fc8e2c17/rest/services/OpenData/OpenData/MapServer/6/query',
        'formats': ['json', 'geojson']
    },
    'ion_stops': {
        'url': 'https://utility.arcgis.com/usrsvcs/servers/f063d1fb147847f796ce8c024e117419/rest/services/OpenData/OpenData/MapServer/5/query',
        'formats': ['json', 'geojson']
    }
}

# Default query parameters
DEFAULT_PARAMS = {
    'outFields': '*',
    'where': '1=1',
    'returnGeometry': 'true'
}

def get_endpoint_url(endpoint_type: str, dataset: str, format: str = 'geojson') -> str:
    """Get the complete URL for a specific endpoint."""
    # Get the endpoints dictionary for the specified type
    endpoints = {
        'row_bounds': ROW_BOUNDS,
        'census': CENSUS_ENDPOINTS,
        'infrastructure': INFRASTRUCTURE_ENDPOINTS,
        'land_use': LAND_USE_ENDPOINTS,
        'transportation': TRANSPORTATION_ENDPOINTS
    }.get(endpoint_type)

    if not endpoints:
        raise ValueError(f"Unknown endpoint type: {endpoint_type}")

    # Get dataset configuration
    dataset_config = endpoints.get(dataset)
    if not dataset_config:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Get base URL
    url = dataset_config['url']

    # Check if format is supported
    if format not in dataset_config['formats']:
        raise ValueError(f"Format {format} not supported for {dataset}")

    return url