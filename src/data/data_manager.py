"""
Comprehensive data manager for KWC-CMA region infrastructure analysis.
Combines functionality from data_fetcher and population_processor.

Features:
1. Geographic Data Collection and Validation 
   - CMA boundary management
   - Census tract data
   - Infrastructure data
   - Land use data
   - Transportation networks

2. Population and Demographics
   - Census data processing
   - Population density calculations
   - Demographic analysis
   - Employment patterns

3. Infrastructure Analysis
   - Transportation networks
   - Land use patterns
   - Accessibility metrics
   - Coverage analysis

4. Data Integration and Quality
   - Cross-source validation
   - Data cleaning and standardization
   - Quality metrics
   - Cache management
"""

import os
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import logging
import osmnx as ox
import pyproj
import hashlib
import osmnx as ox
import pyproj
import hashlib
from typing import Any, Optional, List, Dict, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import osmnx as ox
import pyproj
from shapely.geometry import Point, Polygon, MultiPolygon, box, shape
from shapely import ops
from dotenv import load_dotenv
import hashlib
import warnings
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
from scipy.spatial.distance import pdist
from tabulate import tabulate
from scipy.spatial.distance import pdist, squareform, cdist
import zipfile

from .constants import *
from .utils import *
from .endpoints import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataError(Exception):
    """Base class for data-related errors."""
    pass

class DataSourceError(DataError):
    """Error when fetching from data source."""
    pass

class DataValidationError(DataError):
    """Error when validating data."""
    pass

class DataManager:
    """
    Unified data manager for KWC-CMA region.
    
    This class provides comprehensive functionality for:
    - Data collection from multiple sources
    - Geographic data processing and validation
    - Population and demographic analysis
    - Infrastructure and accessibility analysis
    - Data quality management and caching
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, 
                 cache_expiry: int = 7):
        """
        Initialize the data manager.
        
        Args:
            cache_dir: Directory for caching data
            cache_expiry: Cache expiry in days
        """
        load_dotenv()
        
        # Initialize cache settings
        self.cache_dir = cache_dir or Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(days=cache_expiry)
        
        # Initialize cache index
        self.cache_index_file = self.cache_dir / 'cache_index.json'
        self.cache_index = self._load_cache_index()
        
        # Initialize API keys
        self.ocmap_api_key = os.getenv("OCMAP_API_KEY")
        if not self.ocmap_api_key:
            raise ValueError("OpenChargeMap API key not found")
        
        # Initialize coordinate transformers
        self.utm_crs = 'EPSG:32617'  # UTM Zone 17N for KW region
        self.wgs84_crs = 'EPSG:4326'  # WGS84
        self.transformer = pyproj.Transformer.from_crs(
            self.utm_crs,
            self.wgs84_crs,
            always_xy=True
        )
        
        # Region identifiers
        self.region_codes = {
            'kwc_cma_code': "541",        # ROW/StatCan CMA code
            'kwc_cma_uid': "35541",       # Alternative CMA ID
            'dc_place_id': "Q175735",     # Data Commons place ID
            'un_area_code': "124"         # UN area code for Canada
        }
        
        # 2021 Census statistics
        self.region_stats = {
            'total_population': 715219,
            'households': 275940,
            'avg_household_size': 2.6
        }
        
        # Initialize data quality metrics
        self.quality_metrics = {
            'completeness': {},
            'consistency': {},
            'coverage': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Initialize warning counts
        self._warning_counts = {
            'validation_warnings': 0,
            'coverage_warnings': 0,
            'consistency_warnings': 0
        }

        # Column descriptions for datasets
        column_descriptions = {
            'CTUID': 'Census Tract Unique Identifier',
            'TOTAL_DWELLINGS': 'Total number of private dwellings',
            'MEDIAN_INCOME': 'Median household income in 2020 dollars',
            'TOTAL_LABOUR_FORCE': 'Total labour force aged 15 years and over',
            'ROAD_CLASS': 'Road classification type',
            'ROUTE_ID': 'Transit route identifier',
            'StopID': 'Transit stop identifier',
            'StopName': 'Transit stop name'
        }

    #
    # Cache Management Methods
    #
    
    def _load_cache_index(self) -> Dict:
        """Load or create cache index with versioning."""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r') as f:
                index = json.load(f)
                if self._validate_cache_index(index):
                    return index
        return self._create_new_cache_index()
    
    def _create_new_cache_index(self) -> Dict:
        """Create a new cache index."""
        index = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'entries': {}
        }
        self._save_cache_index(index)
        return index
    
    def _save_cache_index(self, index: Optional[Dict] = None):
        """Save cache index."""
        if index is not None:
            self.cache_index = index
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f)
    
    def _validate_cache_index(self, index: Dict) -> bool:
        """Validate cache index structure."""
        required_keys = ['version', 'created', 'entries']
        return all(key in index for key in required_keys)
    
    def _get_cache_key(self, endpoint_type: str, dataset: str, format: str) -> str:
        """
        Generate unique cache key based on endpoint type, dataset, and format.
        
        Args:
            endpoint_type: Type of endpoint (census, infrastructure, etc.)
            dataset: Specific dataset name
            format: Data format (geojson, json)
            
        Returns:
            str: Unique hash key for caching
        """
        # Create string to hash
        to_hash = f"{endpoint_type}_{dataset}_{format}"
        return hashlib.md5(to_hash.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_index['entries']:
            return False
        
        cache_time = datetime.fromisoformat(
            self.cache_index['entries'][cache_key]['timestamp']
        )
        return datetime.now() - cache_time < self.cache_expiry
    
    def _cache_data(self, key: str, data: Union[pd.DataFrame, gpd.GeoDataFrame],
                    metadata: Optional[Dict] = None):
        """Cache data with metadata."""
        # Generate file path
        file_path = self.cache_dir / f"{key}.{'geojson' if isinstance(data, gpd.GeoDataFrame) else 'parquet'}"
        
        # Save data
        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(file_path, driver='GeoJSON')
        else:
            data.to_parquet(file_path)
        
        # Update cache index
        self.cache_index['entries'][key] = {
            'timestamp': datetime.now().isoformat(),
            'path': str(file_path),
            'type': 'geojson' if isinstance(data, gpd.GeoDataFrame) else 'parquet',
            'metadata': metadata or {}
        }
        self._save_cache_index()
    
    def _get_from_cache(self, key: str) -> Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]:
        """Retrieve data from cache if valid."""
        if not self._is_cache_valid(key):
            return None
        
        entry = self.cache_index['entries'][key]
        file_path = Path(entry['path'])
        
        if not file_path.exists():
            return None
        
        try:
            if entry['type'] == 'geojson':
                return gpd.read_file(file_path)
            else:
                return pd.read_parquet(file_path)
        except Exception as e:
            logger.warning(f"❌ Error reading from cache: {str(e)}")
            return None
    
    #
    # API and Data Collection Methods
    #
    
    @sleep_and_retry
    @limits(calls=60, period=60)
    def _make_request(self, url: str, params: Optional[Dict] = None,
                     headers: Optional[Dict] = None, retry_count: int = 3) -> Dict:
        """
        Make rate-limited API request with retries.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            headers: Request headers
            retry_count: Number of retry attempts
            
        Returns:
            Dict containing response data
            
        Raises:
            DataSourceError: If request fails after all retries
        """
        if params is None:
            params = {}
            
        # Ensure required parameters
        params.update(DEFAULT_PARAMS)
        
        for attempt in range(retry_count):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                if not self._validate_response(data):
                    raise DataValidationError("Invalid response format")
                
                return data
                
            except Exception as e:
                if attempt == retry_count - 1:
                    logger.error(f"Request failed after {retry_count} attempts: {str(e)}")
                    raise DataSourceError(f"Failed to fetch data: {str(e)}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                continue
    
    def _validate_response(self, data: Dict) -> bool:
            """
            Validate API response structure.
            
            Args:
                data: Response data to validate
                
            Returns:
                bool: True if valid, False otherwise
            """
            # Check for error indicators
            if 'error' in data:
                logger.error(f"API error: {data['error']}")
                return False
            
            # Check for expected format based on response structure
            if 'features' in data:
                return len(data['features']) > 0 and 'geometry' in data['features'][0]
            elif 'data' in data:
                return len(data['data']) > 0
            
            return True

    def _validate_data(self, data: Union[pd.DataFrame, gpd.GeoDataFrame], dataset: str) -> bool:
        """Validate fetched data."""
        try:
            # Check for empty data
            if len(data) == 0:
                logger.warning(f"Empty dataset: {dataset}")
                return False
            
            # Debug: print received columns
            logger.info(f"\nReceived columns for {dataset}:")
            for col in data.columns:
                logger.info(f"  {col}")
            
            # Map of expected column names to possible alternatives
            column_mappings = {
                'CTUID': ['CTUID', 'CT_UID', 'CENSUS_TRACT_ID', 'GEOID', 'GEO_CODE'],
                'TOTAL_DWELLINGS': ['TOTAL_DWELLINGS', 'TOT_PRIV_DWELL', 'DWELLINGS', 'TOT_DWELL_HHLDS_TENURE_25SAM'],
                'MEDIAN_INCOME': ['MEDIAN_INCOME', 'TOT_INC_2020_DOLLARS_HHLDS_MED'],
                'TOTAL_LABOUR_FORCE': ['TOTAL_LABOUR_FORCE', 'TOT_INC_STAT_2020_15PLUS'],
                'ROAD_CLASS': ['ROAD_CLASS', 'ROAD_TYPE', 'CLASS', 'CARTO_CLASS', 'OPERATIONS_MTO_CLASS'],
                'ROUTE_ID': ['ROUTE_ID', 'ROUTEID', 'ROUTE_NUM', 'GRT_ROUTE_I', 'RouteID', 'Route'],
                'StopID': ['StopID', 'STOP_ID', 'STOPID', 'GRT_STOP_ID'],
                'StopName': ['StopName', 'STOP_NAME', 'STOPNAME', 'NAME']
            }
            
            # Required columns for each dataset type
            required_mappings = {
                'housing': ['CTUID', 'TOTAL_DWELLINGS'],
                'income': ['CTUID', 'MEDIAN_INCOME'],
                'labour': ['CTUID', 'TOTAL_LABOUR_FORCE'],
                'roads': ['ROAD_CLASS'],
                'grt_routes': ['ROUTE_ID'],
                'grt_stops': ['StopID'],
                'ion_routes': [],  # No strict requirements for ION routes
                'ion_stops': ['StopName']
            }
            
            if dataset in required_mappings:
                required_cols = required_mappings[dataset]
                missing_cols = []
                
                # Try to map columns to their standardized names
                for required_col in required_cols:
                    found = False
                    possible_names = column_mappings.get(required_col, [required_col])
                    
                    # First check if the standardized name already exists
                    if required_col in data.columns:
                        found = True
                        continue
                        
                    # Then check alternatives
                    for col_name in possible_names:
                        if col_name in data.columns:
                            # If found under alternative name, rename to standard name
                            data.rename(columns={col_name: required_col}, inplace=True)
                            logger.info(f"Mapped {col_name} to {required_col}")
                            found = True
                            break
                        
                    if not found:
                        # Special handling for composite IDs or alternative ways to get the data
                        if required_col == 'ROUTE_ID' and dataset == 'grt_routes':
                            if 'RouteID' in data.columns:
                                data['ROUTE_ID'] = data['RouteID']
                                found = True
                                logger.info("Using RouteID as ROUTE_ID")
                            elif 'Route' in data.columns:
                                data['ROUTE_ID'] = data['Route']
                                found = True
                                logger.info("Using Route as ROUTE_ID")
                    
                    if not found and required_col:  # Only add to missing if it was required
                        missing_cols.append(required_col)
                
                if missing_cols:
                    logger.warning(f"Missing required columns for {dataset}: {missing_cols}")
                    logger.info("\nExpected one of these for each missing column:")
                    for col in missing_cols:
                        logger.info(f"{col}: {column_mappings.get(col, [col])}")
                    return False
            
            # Validate geometry for GeoDataFrames
            if isinstance(data, gpd.GeoDataFrame):
                if not self._validate_geometry(data, dataset):
                    return False
            
            # Print final columns after mapping
            logger.info("\nFinal columns after mapping:")
            logger.info(data.columns.tolist())
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error for {dataset}: {str(e)}")
            return False
    
    def _validate_geometry(self, gdf: gpd.GeoDataFrame, dataset: str) -> bool:
        """Validate geometry of spatial data."""
        try:
            # Check CRS
            if gdf.crs != self.wgs84_crs:
                gdf.to_crs(self.wgs84_crs, inplace=True)
            
            # Check for invalid geometries
            invalid_geoms = ~gdf.geometry.is_valid
            if invalid_geoms.any():
                logger.warning(f"Found {invalid_geoms.sum()} invalid geometries in {dataset}")
                gdf.geometry = gdf.geometry.buffer(0)  # Try to fix
            
            # Check against CMA boundary
            try:
                boundary = self.get_cma_boundary()
                intersects = gdf.intersects(boundary.geometry.iloc[0])
                if not intersects.any():
                    logger.warning(f"No geometries intersect CMA boundary in {dataset}")
                    return False
            except Exception as e:
                logger.warning(f"Could not check CMA intersection: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Geometry validation error for {dataset}: {str(e)}")
            return False
    
    def _process_features(self, features: List[Dict]) -> gpd.GeoDataFrame:
        """
        Process GeoJSON features into GeoDataFrame.
        
        Args:
            features: List of GeoJSON feature dictionaries
                
        Returns:
            GeoDataFrame with processed features
        """
        # Pre-process features to handle null coordinates
        processed_features = self._process_geojson_coordinates(features)
        
        if not processed_features:
            raise DataValidationError("No valid features after processing")
        
        return gpd.GeoDataFrame.from_features(processed_features, crs=self.wgs84_crs)

    def _process_geojson_coordinates(self, features: List[Dict]) -> List[Dict]:
        """
        Pre-process GeoJSON features to handle null coordinates before cleaning.
        
        Args:
            features: List of GeoJSON feature dictionaries
                
        Returns:
            List of processed feature dictionaries
        """
        processed_features = []
        
        for feature in features:
            try:
                if not feature.get('geometry'):
                    continue
                    
                # Clean coordinates
                coords = feature['geometry']['coordinates']
                
                # Handle LineString coordinates with null values
                if feature['geometry']['type'] == 'LineString':
                    coords = [
                        [x, y] for x, y, *_ in coords 
                        if x is not None and y is not None
                    ]
                
                # Use existing coordinate cleaning
                cleaned_coords = self._clean_coordinates(coords)
                
                if not cleaned_coords:
                    continue
                
                # Create cleaned feature
                cleaned_feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': feature['geometry']['type'],
                        'coordinates': cleaned_coords
                    },
                    'properties': feature.get('properties', {})
                }
                
                processed_features.append(cleaned_feature)
                
            except Exception as e:
                logger.warning(f"❌ Error processing feature: {str(e)}")
                continue
        
        if not processed_features:
            raise DataValidationError("No valid features after processing")
        
        return processed_features
    
    def _clean_coordinates(self, coords: Union[List, Tuple]) -> Optional[List]:
        """
        Clean coordinate arrays by removing invalid values and validating reasonableness.
        
        Args:
            coords: Raw coordinates to clean
            
        Returns:
            Cleaned coordinates or None if invalid
        """
        if not coords:
            return None
        
        try:
            # Handle single coordinate pair
            if not isinstance(coords[0], (list, tuple)):
                # Basic coordinate validation
                lon, lat = coords[0], coords[1]
                
                # Check if coordinates are within reasonable range for Ontario
                # Ontario bounds roughly: lat 41.5-57.5°N, lon 74.5-95.5°W
                if not (41.5 <= lat <= 57.5 and -95.5 <= lon <= -74.5):
                    return None
                    
                # More specific check for reasonable KW region coordinates
                # KW region roughly: lat 43.2-43.7°N, lon 80.1-80.9°W
                if not (43.2 <= lat <= 43.7 and -80.9 <= lon <= -80.1):
                    logger.warning(f"Coordinate {coords} outside expected KW region")
                
                return coords
                
            # Handle coordinate arrays recursively
            cleaned = []
            for coord in coords:
                clean_coord = self._clean_coordinates(coord)
                if clean_coord:
                    cleaned.append(clean_coord)
            
            return cleaned if cleaned else None
                
        except Exception as e:
            logger.warning(f"❌ Error cleaning coordinates: {str(e)}")
            return None
    
    def _clean_feature_coordinates(self, feature: Dict) -> Optional[Dict]:
        """Clean coordinates in a GeoJSON feature, handling null values.
        
        Args:
            feature: GeoJSON feature dictionary
            
        Returns:
            Cleaned feature or None if invalid
        """
        try:
            if not feature.get('geometry') or not feature['geometry'].get('coordinates'):
                return None
                
            coords = feature['geometry']['coordinates']
            geom_type = feature['geometry']['type'].lower()
            
            def clean_point_coords(point_coords):
                """Clean single point coordinates."""
                if not point_coords or len(point_coords) < 2:
                    return None
                if any(c is None for c in point_coords[:2]):
                    return None
                return point_coords[:2]  # Take only x,y coordinates
                
            def clean_line_coords(line_coords):
                """Clean single linestring coordinates."""
                if not line_coords:
                    return None
                cleaned = []
                for point in line_coords:
                    clean_point = clean_point_coords(point)
                    if clean_point:
                        cleaned.append(clean_point)
                return cleaned if len(cleaned) >= 2 else None
                
            def clean_polygon_coords(poly_coords):
                """Clean single polygon coordinates."""
                if not poly_coords:
                    return None
                cleaned_rings = []
                for ring in poly_coords:
                    cleaned_ring = clean_line_coords(ring)
                    if cleaned_ring and len(cleaned_ring) >= 4:
                        cleaned_rings.append(cleaned_ring)
                return cleaned_rings if cleaned_rings else None
            
            # Clean coordinates based on geometry type
            if geom_type == 'point':
                cleaned_coords = clean_point_coords(coords)
                if not cleaned_coords:
                    return None
                    
            elif geom_type == 'linestring':
                cleaned_coords = clean_line_coords(coords)
                if not cleaned_coords:
                    return None
                    
            elif geom_type == 'polygon':
                cleaned_coords = clean_polygon_coords(coords)
                if not cleaned_coords:
                    return None
                    
            elif geom_type == 'multipoint':
                cleaned_coords = []
                for point in coords:
                    clean_point = clean_point_coords(point)
                    if clean_point:
                        cleaned_coords.append(clean_point)
                if not cleaned_coords:
                    return None
                    
            elif geom_type == 'multilinestring':
                cleaned_coords = []
                for line in coords:
                    clean_line = clean_line_coords(line)
                    if clean_line:
                        cleaned_coords.append(clean_line)
                if not cleaned_coords:
                    return None
                    
            elif geom_type == 'multipolygon':
                cleaned_coords = []
                for poly in coords:
                    clean_poly = clean_polygon_coords(poly)
                    if clean_poly:
                        cleaned_coords.append(clean_poly)
                if not cleaned_coords:
                    return None
                    
            else:
                logger.warning(f"Unsupported geometry type: {geom_type}")
                return None
            
            # Create cleaned feature
            return {
                'type': 'Feature',
                'geometry': {
                    'type': feature['geometry']['type'],
                    'coordinates': cleaned_coords
                },
                'properties': feature.get('properties', {})
            }
            
        except Exception as e:
            logger.warning(f"Error cleaning feature coordinates: {str(e)}")
            return None

    def _clean_geojson_response(self, response: Dict) -> Dict:
        """Clean a GeoJSON response by handling null coordinates.
        
        Args:
            response: Raw GeoJSON response
            
        Returns:
            Cleaned GeoJSON response
        """
        if 'features' not in response:
            raise DataValidationError("Invalid GeoJSON response")
            
        cleaned_features = []
        total_features = len(response['features'])
        cleaned_count = 0
        
        for feature in response['features']:
            try:
                cleaned_feature = self._clean_feature_coordinates(feature)
                if cleaned_feature:
                    cleaned_features.append(cleaned_feature)
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Error processing feature: {str(e)}")
                continue
        
        logger.info(f"Cleaned {cleaned_count}/{total_features} features")
        
        if not cleaned_features:
            raise DataValidationError("No valid features after cleaning")
        
        return {
            'type': 'FeatureCollection',
            'features': cleaned_features
        }

    def fetch_data(self, endpoint_type: str, dataset: str, format: str = 'geojson') -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Fetch data with debug logging."""
        try:
            # Generate cache key
            params = {
                'outFields': '*',
                'where': '1=1',
                'returnGeometry': 'true',
                'f': format,
                'outSR': '4326'  # WGS84
            }
            
            cache_key = self._get_cache_key(endpoint_type, dataset, format)
            
            # Check cache
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Fetch new data
            url = get_endpoint_url(endpoint_type, dataset, format)
            
            print(f"  \033[94m⏳\033[0m  {dataset:<20}", end='', flush=True)
            
            response = self._make_request(url, params)
            
            # Process response
            if format == 'geojson':
                # Clean response data
                cleaned_response = self._clean_geojson_response(response)
                data = gpd.GeoDataFrame.from_features(cleaned_response['features'], crs=self.wgs84_crs)
                
                # Clip to CMA boundary if available
                try:
                    boundary = self.get_cma_boundary()
                    data = gpd.clip(data, boundary.geometry.iloc[0])
                except Exception as e:
                    logger.warning(f"Could not clip to CMA boundary: {str(e)}")
            else:
                data = pd.DataFrame(response.get('features', []))
            
            # Validate and cache
            if self._validate_data(data, dataset):
                self._cache_data(cache_key, data, {
                    'endpoint': endpoint_type,
                    'dataset': dataset
                })
                
                print(f"\r  \033[92m✓\033[0m  {dataset:<20} \033[94m({len(data)} records)\033[0m")
                return data
            else:
                raise DataValidationError("Data validation failed")
                
        except Exception as e:
            print(f"\r  \033[91m✗\033[0m  {dataset:<20} \033[91mFailed: {str(e)}\033[0m")
            raise
    
    def fetch_potential_locations(self) -> pd.DataFrame:
        """
        Fetch potential charging station locations across all KWC-CMA municipalities.
        
        The KWC-CMA includes:
        - Cities: Kitchener, Waterloo, Cambridge
        - Townships: Woolwich, Wilmot, North Dumfries, Wellesley
        
        Returns:
            DataFrame containing potential locations with their attributes
        """
        try:
            # Check cache
            cache_key = 'potential_locations'
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                print("\033[1mFetching Potential Locations...\033[0m")
                print("=" * 60)
                print(f"📦 Loaded {len(cached_data)} cached locations"
                      f" from {self.cache_index['entries'][cache_key]['timestamp']}")
                return cached_data
            
            # Define location types
            tags = {
                'amenity': ['parking', 'fuel', 'charging_station'],
                'shop': ['mall', 'supermarket'],
                'building': ['commercial', 'retail']
            }
            
            locations = []
            total_features = 0
            
            # Query each municipality separately
            for place in KWC_CITIES:
                try:
                    print(f"📍 Fetching locations in {place}...")
                    
                    # Query using polygon
                    area = ox.geocode_to_gdf(place)
                    gdf = ox.features.features_from_polygon(
                        area.geometry.iloc[0],
                        tags=tags
                    )
                    
                    total_features += len(gdf)
                    print(f"  \033[92m✓\033[0m Found {len(gdf)} features in {place}!\n")
                    
                    # Process locations from this area
                    for idx, row in gdf.iterrows():
                        try:
                            geom = row.geometry
                            if geom.geom_type == 'Point':
                                lat, lon = geom.y, geom.x
                            else:
                                centroid = geom.centroid
                                lat, lon = centroid.y, centroid.x
                            
                            # Determine location type
                            location_type = None
                            for key in ['amenity', 'shop', 'building']:
                                if key in row and pd.notna(row[key]):
                                    location_type = str(row[key])
                                    break
                            if location_type is None:
                                continue
                            
                            # Get name and ensure it's meaningful
                            name = str(row.get('name', ''))
                            if not name:
                                if location_type == 'parking':
                                    name = f"Parking at {row.get('addr:street', 'Unknown Location')}"
                                elif location_type == 'fuel':
                                    name = f"Gas Station at {row.get('addr:street', 'Unknown Location')}"
                                else:
                                    name = f"{location_type.capitalize()} at {row.get('addr:street', 'Unknown Location')}"
                            
                            locations.append({
                                'name': name,
                                'latitude': lat,
                                'longitude': lon,
                                'location_type': location_type,
                                'address': str(row.get('addr:street', 'Unknown')),
                                'city': place.split(',')[0],
                                'postal_code': str(row.get('addr:postcode', 'Unknown')),
                                'municipality_type': 'City' if place.split(',')[0] in ['Kitchener', 'Waterloo', 'Cambridge'] else 'Township'
                            })
                            
                        except Exception as e:
                            logger.warning(f"❌ Error processing location in {place}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"❌ Error processing {place}: {str(e)}")
                    continue
            
            if not locations:
                raise ValueError("No locations found in any municipality")
            
            # Create DataFrame
            df = pd.DataFrame(locations)
            
            # Clip to CMA boundary
            try:
                boundary = self.get_cma_boundary()
                points = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs=self.wgs84_crs
                )
                within_cma = points.intersects(boundary.geometry.iloc[0])
                df = df[within_cma].copy()
            except Exception as e:
                logger.warning(f"Could not clip to CMA boundary: {str(e)}")
            
            print("\nLocation Summary:")
            print("-" * 50)
            print(f"Total features processed: {total_features}")
            print(f"Valid locations found: {len(df)}")
            print("\nBy Municipality:")
            print(df['city'].value_counts())
            print("\nBy Type:")
            print(df['location_type'].value_counts())
            
            # Cache results
            self._cache_data(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching potential locations: {str(e)}")
            raise

    def fetch_charging_stations(self) -> pd.DataFrame:
        """
        Fetch charging stations from OpenChargeMap API with robust error handling.
        """
        try:
            print("\033[1mFetching Charging Stations...\033[0m")
            print("=" * 60)
            
            # Make API request
            params = {
                'key': self.ocmap_api_key,
                'maxresults': 500,
                'countrycode': 'CA',
                'latitude': 43.4516,
                'longitude': -80.4925,
                'distance': 25,
                'distanceunit': 'KM'
            }
            
            response = requests.get(
                "https://api.openchargemap.io/v3/poi",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"📡 Retrieved {len(data)} stations from API")
            
            # Process stations with better error handling
            stations = []
            skipped = 0
            
            for poi in data:
                try:
                    # Safely get nested values with defaults
                    address_info = poi.get('AddressInfo') or {}
                    operator_info = poi.get('OperatorInfo') or {}
                    connections = poi.get('Connections') or []
                    
                    # Skip if no valid location
                    lat = address_info.get('Latitude')
                    lon = address_info.get('Longitude')
                    if not (lat and lon):
                        skipped += 1
                        continue
                    
                    # Process connections and charger counts safely
                    total_chargers = 0
                    charger_types = set()
                    
                    for conn in connections:
                        if not conn:
                            continue
                            
                        # Safely get connection quantity
                        quantity = conn.get('Quantity')
                        if isinstance(quantity, (int, float)) and quantity > 0:
                            total_chargers += quantity
                        else:
                            total_chargers += 1  # Assume 1 if not specified
                        
                        # Safely determine charger type
                        level = conn.get('Level') or {}
                        level_title = (level.get('Title') or '').lower()
                        
                        if level.get('IsFastChargeCapable'):
                            charger_types.add('Level 3')
                        elif 'level 2' in level_title:
                            charger_types.add('Level 2')
                        elif 'level 1' in level_title:
                            charger_types.add('Level 1')
                    
                    # Determine highest level charger available
                    charger_type = 'Unknown'
                    if 'Level 3' in charger_types:
                        charger_type = 'Level 3'
                    elif 'Level 2' in charger_types:
                        charger_type = 'Level 2'
                    elif 'Level 1' in charger_types:
                        charger_type = 'Level 1'
                    
                    # Get usage cost safely
                    usage_cost = poi.get('UsageCost') or 'Unknown'
                    if isinstance(usage_cost, (dict, list)):
                        usage_cost = 'Varies'
                    
                    stations.append({
                        'name': address_info.get('Title') or 'Unnamed Station',
                        'latitude': lat,
                        'longitude': lon,
                        'num_chargers': total_chargers,
                        'charger_type': charger_type,
                        'operator': operator_info.get('Title') or 'Unknown',
                        'address': address_info.get('AddressLine1') or 'Unknown',
                        'city': address_info.get('Town') or 'Unknown',
                        'postal_code': address_info.get('Postcode') or 'Unknown',
                        'usage_cost': usage_cost
                    })
                    
                except Exception as e:
                    skipped += 1
                    continue
            
            df = pd.DataFrame(stations)
            
            # Clip to CMA boundary
            try:
                boundary = self.get_cma_boundary()
                points = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs=self.wgs84_crs
                )
                within_cma = points.intersects(boundary.geometry.iloc[0])
                df = df[within_cma].copy()
            except Exception as e:
                print(f"  \033[93m⚠️  Could not clip to CMA boundary: {str(e)}\033[0m")
            
            # Print summary
            print(f"\n📊 Charging Station Summary:")
            print("-" * 50)
            print(f"Total API Results: {len(data)}")
            print(f"Successfully Processed: {len(stations)}")
            print(f"Skipped Records: {skipped}")
            print(f"Within CMA Boundary: {len(df)}")
            
            print("\nBy Charger Type:")
            for type_name, count in df['charger_type'].value_counts().items():
                print(f"  \033[94m●\033[0m {type_name}: {count}")
            
            print("\nBy Operator:")
            for operator, count in df['operator'].value_counts().head().items():
                print(f"  \033[92m●\033[0m {operator}: {count}")
            
            return df
            
        except Exception as e:
            print(f"\033[91m❌ Error fetching charging stations: {str(e)} \033[0m")
            raise

    def _get_charger_type(self, connections: List[Dict]) -> str:
        """
        Determine highest level charger type available.
        
        Args:
            connections: List of charger connection data
            
        Returns:
            String indicating charger type
        """
        if not connections:
            return 'Unknown'
        
        # Check for Level 3 (DC Fast Charging)
        for conn in connections:
            level = conn.get('Level', {})
            if level.get('IsFastChargeCapable'):
                return 'Level 3'
        
        # Check for Level 2
        for conn in connections:
            level = conn.get('Level', {})
            if level and 'level 2' in level.get('Title', '').lower():
                return 'Level 2'
        
        return 'Level 1'
    
    #
    # Geographic Data Processing Methods
    #
    
    def get_cma_boundary(self) -> gpd.GeoDataFrame:
        """
        Get KWC-CMA boundary with caching.
        
        Returns:
            GeoDataFrame containing CMA boundary
            
        Raises:
            DataSourceError: If boundary cannot be fetched or validated
        """
        try:
            # Check cache
            cache_key = 'cma_boundary'
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
            
            # Fetch using census housing endpoint
            url = get_endpoint_url('row_bounds', 'boundary', 'geojson')
            params = {
                'where': f"1=1",
                'outFields': '*',
                'outSR': '4326',
                'f': 'geojson'
            }
            
            response = self._make_request(url, params)
            
            # Process boundary
            boundary = gpd.GeoDataFrame.from_features(
                response['features'],
                crs=self.wgs84_crs
            )
            
            # Clean and validate
            boundary = self._clean_boundary(boundary)
            if not self._validate_boundary(boundary):
                raise DataValidationError("Invalid boundary data")
            
            # Cache result
            self._cache_data(cache_key, boundary)
            
            return boundary
            
        except Exception as e:
            logger.error(f"❌ Error fetching CMA boundary: {str(e)}")
            raise DataSourceError(f"Failed to get CMA boundary: {str(e)}")
    
    def _clean_boundary(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clean and process CMA boundary.
        
        Args:
            boundary: Raw boundary GeoDataFrame
            
        Returns:
            Cleaned boundary GeoDataFrame
        """
        try:
            """Clean boundary data by removing redundant columns and standardizing names."""
            # Keep only essential columns
            essential_cols = ['geometry', 'ShortName', 'LongName']
            other_cols = [col for col in gdf.columns 
                        if col not in essential_cols 
                        and not any(x in col.lower() for x in ['shape', 'area', 'length', 'objectid'])]
            
            cols_to_keep = essential_cols + other_cols
            gdf = gdf[[col for col in cols_to_keep if col in gdf.columns]]
            
            return gdf
            
        except Exception as e:
            logger.error(f"❌ Error cleaning boundary: {str(e)}")
            raise
    
    def _validate_boundary(self, boundary: gpd.GeoDataFrame) -> bool:
        """
        Validate CMA boundary data.
        
        Args:
            boundary: Boundary to validate
            
        Returns:
            bool indicating if boundary is valid
        """
        try:
            # Check basic requirements
            if len(boundary) == 0:
                logger.error("Empty boundary")
                return False
            
            # Validate geometry
            if not boundary.geometry.is_valid.all():
                logger.error("Invalid geometry in boundary")
                return False
            
            # Check area is reasonable (500-2000 km² is expected for KW CMA)
            area_km2 = float(
                boundary.to_crs({'proj':'cea'})
                .geometry.area.sum() / 1_000_000
            )
            
            if not (500 <= area_km2 <= 2000):
                logger.error(f"Unexpected area: {area_km2:.1f} km²")
                return False
            
            # Verify covers key cities by checking if major intersections are included
            key_points = {
                'Kitchener': Point(-80.4927, 43.4516),  # King & Victoria
                'Waterloo': Point(-80.5203, 43.4643),   # King & University
                'Cambridge': Point(-80.3123, 43.3601)   # Water & Main
            }
            
            for city, point in key_points.items():
                if not boundary.geometry.contains(point).any():
                    logger.error(f"Boundary does not contain {city}")
                    return False
            
            return True
                
        except Exception as e:
            logger.error(f"❌ Error validating boundary: {str(e)}")
            return False
    
    def clip_to_cma(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clip data to CMA boundary.
        
        Args:
            gdf: GeoDataFrame to clip
            
        Returns:
            Clipped GeoDataFrame
        """
        try:
            boundary = self.get_cma_boundary()
            
            # Ensure same CRS
            if gdf.crs != boundary.crs:
                gdf = gdf.to_crs(boundary.crs)
            
            # Clip to boundary
            clipped = gpd.clip(gdf, boundary.geometry.iloc[0])
            
            # Calculate clipped percentage
            original_size = len(gdf)
            clipped_size = len(clipped)
            if original_size > 0:
                clip_pct = (clipped_size / original_size) * 100
                logger.info(f"Clipped {original_size - clipped_size} features "
                          f"({clip_pct:.1f}% retained)")
            
            return clipped
            
        except Exception as e:
            logger.error(f"❌ Error clipping to CMA: {str(e)}")
            return gdf
    
    def validate_within_cma(self, coords: Union[Tuple[float, float], 
                                               List[Tuple[float, float]]]) -> bool:
        """
        Validate if coordinates fall within CMA boundary.
        
        Args:
            coords: Single coordinate pair or list of coordinates
            
        Returns:
            bool indicating if coordinates are within CMA
        """
        try:
            boundary = self.get_cma_boundary()
            
            if isinstance(coords[0], (int, float)):
                # Single coordinate pair
                point = Point(coords[1], coords[0])  # lon, lat
                return boundary.geometry.contains(point).any()
            else:
                # Multiple coordinates
                points = [Point(lon, lat) for lat, lon in coords]
                return all(boundary.geometry.contains(point).any() 
                         for point in points)
                
        except Exception as e:
            logger.error(f"❌ Error validating coordinates: {str(e)}")
            return False
    
    def calculate_distances(self, 
                          from_points: gpd.GeoDataFrame,
                          to_points: gpd.GeoDataFrame) -> np.ndarray:
        """
        Calculate distance matrix between two sets of points.
        
        Args:
            from_points: Origin points
            to_points: Destination points
            
        Returns:
            2D array of distances in kilometers
        """
        try:
            # Convert to UTM for accurate distances
            from_utm = from_points.to_crs(self.utm_crs)
            to_utm = to_points.to_crs(self.utm_crs)
            
            # Extract coordinates
            from_coords = np.column_stack(
                (from_utm.geometry.x, from_utm.geometry.y)
            )
            to_coords = np.column_stack(
                (to_utm.geometry.x, to_utm.geometry.y)
            )
            
            # Calculate distances using broadcasting
            dx = from_coords[:, np.newaxis, 0] - to_coords[:, 0]
            dy = from_coords[:, np.newaxis, 1] - to_coords[:, 1]
            
            distances = np.sqrt(dx**2 + dy**2) / 1000  # Convert to km
            
            return distances
            
        except Exception as e:
            logger.error(f"❌ Error calculating distances: {str(e)}")
            raise
    
    def get_service_areas(self, 
                         points: gpd.GeoDataFrame,
                         radii: List[float]) -> Dict[float, gpd.GeoDataFrame]:
        """
        Generate service areas for points at different radii.
        
        Args:
            points: Points to generate service areas for
            radii: List of radii in kilometers
            
        Returns:
            Dict mapping radius to service area GeoDataFrame
        """
        try:
            # Convert to UTM for accurate buffers
            points_utm = points.to_crs(self.utm_crs)
            
            service_areas = {}
            for radius in radii:
                # Create buffers
                buffers = points_utm.copy()
                buffers.geometry = points_utm.geometry.buffer(radius * 1000)
                
                # Dissolve overlapping areas
                dissolved = buffers.dissolve().reset_index()
                
                # Convert back to WGS84
                service_areas[radius] = dissolved.to_crs(self.wgs84_crs)
            
            return service_areas
            
        except Exception as e:
            logger.error(f"❌ Error generating service areas: {str(e)}")
            raise
    
    def calculate_coverage(self, 
                         service_area: gpd.GeoDataFrame,
                         population: gpd.GeoDataFrame) -> Dict[str, float]:
        """
        Calculate population coverage for a service area.
        
        Args:
            service_area: Service area polygons
            population: Population points or polygons
            
        Returns:
            Dict containing coverage metrics
        """
        try:
            # Ensure same CRS
            if service_area.crs != population.crs:
                population = population.to_crs(service_area.crs)
            
            # Calculate intersections
            covered = gpd.clip(population, service_area.geometry.iloc[0])
            
            # Calculate metrics
            total_pop = float(population['population'].sum())
            covered_pop = float(covered['population'].sum())
            coverage_pct = (covered_pop / total_pop * 100) if total_pop > 0 else 0
            
            return {
                'total_population': total_pop,
                'covered_population': covered_pop,
                'coverage_percentage': coverage_pct
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating coverage: {str(e)}")
            raise

    #
    # Population and Demographic Analysis Methods
    #
    
    def _fetch_statcan_data(self) -> pd.DataFrame:
        """
        Fetch Statistics Canada census data for the KW region.
        
        Returns:
        pd.DataFrame: DataFrame containing the population data.
        """
        url = "https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/page.cfm?Lang=E&SearchText=Kitchener%20%2D%20Cambridge%20%2D%20Waterloo&DGUIDlist=2021S0503541&GENDERlist=1,2,3&STATISTIClist=1,4&HEADERlist=0"
        
        try:
            tables = pd.read_html(url)
            df = tables[0]  # The first table contains the population data
            
            # Clean the DataFrame
            df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.replace('...', pd.NA)
            df = df.dropna(how='all', axis=1)
            df = df.dropna(how='all', axis=0)
            
            return df
        except Exception as e:
            print(f"Error fetching StatCan data: {str(e)}")
            return pd.DataFrame()

    def _fetch_osm_population_data(self) -> gpd.GeoDataFrame:
        """
        Fetch population-related features from OpenStreetMap for spatial analysis.
        Includes residential buildings and amenities.
        """
        try:
            # Define OSM tags for population-related features
            tags = {
                'building': ['residential', 'apartments', 'house'],
                'amenity': ['school', 'university', 'college'],
                'landuse': ['residential']
            }

            # Get CMA boundary for area
            boundary = self.get_cma_boundary()
            polygon = boundary.geometry.iloc[0]

            # Download using OSMnx with polygon
            features = ox.geometries_from_polygon(polygon, tags)

            if len(features) == 0:
                raise ValueError("No features found in OSM data")

            # Process features
            features = features.to_crs(self.wgs84_crs)

            return features

        except Exception as e:
            logger.error(f"Error fetching OSM data: {str(e)}")
            raise

    def _fetch_un_data(self) -> pd.DataFrame:
        """
        Fetch population data for Kitchener-Cambridge-Waterloo from the UN data site.
        
        Returns:
        pd.DataFrame: DataFrame containing the population data.
        """
        url = "https://data.un.org/Data.aspx?d=POP&f=tableCode:240;countryCode:124&c=2,3,6,8,10,12,14,16,17,18&s=_countryEnglishNameOrderBy:asc,refYear:desc,areaCode:asc&v=1#f_2"
        
        try:
            tables = pd.read_html(url)
            df = tables[1]  # This table contains the population data
            
            kcw_df = df[df['City'].str.contains('Kitchener-Cambridge-Waterloo', case=False, na=False)]
            
            kcw_df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in kcw_df.columns]
            kcw_df = kcw_df.loc[:, ~kcw_df.columns.duplicated()]
            kcw_df = kcw_df.replace('...', pd.NA)
            kcw_df = kcw_df.dropna(how='all', axis=1)
            kcw_df = kcw_df.dropna(how='all', axis=0)
            
            return kcw_df
        except Exception as e:
            print(f"Error fetching UN data: {str(e)}")
            return pd.DataFrame()

    def _fetch_datacommons_data(self) -> pd.DataFrame:
        """
        Fetch demographic data from Data Commons for KWC-CMA.
        Uses Data Commons API with Wikidata ID Q175735.
        """
        try:
            base_url = "https://api.datacommons.org/v1"
            
            # Use environment variable for API key
            headers = {
                'X-API-Key': os.getenv('DATACOMMONS_API_KEY')
            }
            
            # First get place data to confirm we have the right region
            place_url = f"{base_url}/place/wikidataId/Q175735"
            place_response = requests.get(place_url, headers=headers)
            place_response.raise_for_status()
            place_data = place_response.json()
            
            # Now get statistical variables
            stats_url = f"{base_url}/stat/value"
            
            # Request specific statistics we want
            params = {
                "place": "wikidataId/Q175735",
                "stat_var": [
                    "Count_Person",  # Total population
                    "Count_Household",  # Number of households
                    "Median_Income_Household",  # Median household income
                    "Count_Person_Employed"  # Employment
                ]
            }
            
            stats_response = requests.get(
                stats_url, 
                params=json.dumps(params), 
                headers=headers
            )
            stats_response.raise_for_status()
            stats_data = stats_response.json()
            
            # Process the data
            processed_data = []
            for stat_var, values in stats_data.get('value', {}).items():
                if values:  # Check if we have values
                    # Get the most recent value
                    recent_value = next(iter(values.items()))
                    processed_data.append({
                        'metric': stat_var,
                        'value': recent_value[1],
                        'date': recent_value[0]
                    })
            
            return pd.DataFrame(processed_data)
            
        except Exception as e:
            logger.error(f"Error fetching Data Commons data: {str(e)}")
            return pd.DataFrame()

    def get_population_data(self) -> gpd.GeoDataFrame:
        try:
            print("\033[1mCollecting Population Data from Available Sources\033[0m")
            print("=" * 50)

            # 1. Primary Data (Region of Waterloo Census)
            print("\n1️⃣\xa0Region of Waterloo Census Data:")
            print("-" * 40)
            primary_data = self.fetch_data('census', 'housing', 'geojson')

            primary_data['data_source'] = 'Region of Waterloo'
            
            # Convert OBJECTID to CTUID if GEO_CODE isn't available
            primary_data['CTUID'] = primary_data['OBJECTID'].astype(str)

            primary_data['population'] = primary_data['TOTAL_DWELLINGS']
            
            # Calculate area and density
            primary_data['area_km2'] = (
                primary_data.to_crs({'proj':'cea'})
                .geometry.area / 1_000_000
            )

            primary_data['population_density'] = (
                primary_data['population'] / primary_data['area_km2']
            )
            
            # Calculate regional totals for verification
            total_pop = primary_data['population'].sum()
            total_area = primary_data['area_km2'].sum()
            avg_density = total_pop / total_area
            
            print("    📊 Regional Summary:")
            print("    " + "-" * 36)
            print(f"    Total Population: {total_pop:,.0f}")
            print(f"    Total Area: {total_area:.1f} km²")
            print(f"    Average Regional Density: {avg_density:.1f} people/km²")
            
            # 2. Statistics Canada Data
            print("\n2️⃣\xa0Statistics Canada Data:")
            print("-" * 40)
            try:
                statcan_data = self._fetch_statcan_data()
                statcan_pop = int(statcan_data.iloc[1, 1].replace(',', ''))  # Extract the population value and convert to integer
                print(f"Population Estimate: {statcan_pop:,.0f}")
                primary_data['statcan_validation'] = statcan_pop
            except Exception as e:
                print(f"⚠️  StatCan data error: {str(e)}")
                statcan_pop = 0
            
            # 3. UN Data
            print("\n3️⃣\xa0UN Population Data:")
            print("-" * 40)
            try:
                un_data = self._fetch_un_data()
                if not un_data.empty:
                    un_pop = un_data['Value'].iloc[0]
                    un_data['data_source'] = 'UN Data'  # Add 'data_source' column to UN data
                    print(f"Population Estimate: {un_pop:,.0f}")
                    primary_data = pd.concat([primary_data, un_data[['data_source', 'Value']].rename(columns={'Value': 'population'})], ignore_index=True)
                else:
                    un_pop = 0
                    print("⚠️  No UN data available")
            except Exception as e:
                print(f"⚠️  UN data error: {str(e)}")
                un_pop = 0
            
            # Calculate confidence score based on available sources
            available_pops = [pop for pop in [statcan_pop, un_pop] if pop > 0]
            if available_pops:
                variations = [abs(total_pop - pop) / total_pop for pop in available_pops]
                confidence_score = 1.0 - (sum(variations) / len(variations))
            else:
                confidence_score = 0.8  # Default score if only primary source is available
            
            primary_data.loc[primary_data['data_source'] == 'Region of Waterloo', 'confidence_score'] = confidence_score
            
            print("\n📊 Source Comparison Summary:")
            print("-" * 50)
            print(f"Region of Waterloo: {total_pop:,.0f}")
            if statcan_pop > 0:
                print(f"Statistics Canada: {statcan_pop:,.0f}")
            if un_pop > 0:
                print(f"UN Data: {un_pop:,.0f}")
            
            print(f"\nConfidence Score: {confidence_score:.2f}")
            
            return primary_data
        
        except Exception as e:
            logger.error(f"Error getting population data: {str(e)}")
            raise

    def _clean_population_data(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clean and validate population data.
        
        Args:
            df: Population GeoDataFrame to clean
            
        Returns:
            Cleaned GeoDataFrame
        """
        # Handle missing values
        df['MEDIAN_INCOME'] = df['MEDIAN_INCOME'].fillna(
            df['MEDIAN_INCOME'].mean()
        )
        df['TOTAL_LABOUR_FORCE'] = df['TOTAL_LABOUR_FORCE'].fillna(
            df['population'] * df['employment_rate'].mean()
        )
        
        # Remove invalid values
        df = df[df['population'] > 0]
        df = df[df['population_density'] < 50000]  # Max realistic density
        
        return df
    
    def generate_population_grid(self, resolution_km: float = 0.5
                               ) -> gpd.GeoDataFrame:
        """
        Generate regular grid of population estimates.
        
        Args:
            resolution_km: Grid cell size in kilometers
            
        Returns:
            GeoDataFrame with population grid
        """
        try:
            # Get CMA boundary and population data
            boundary = self.get_cma_boundary()
            population = self.get_population_data()
            
            # Create grid
            bounds = boundary.total_bounds
            width = resolution_km / 111  # Convert km to degrees
            height = width
            
            rows = int((bounds[3] - bounds[1]) / height)
            cols = int((bounds[2] - bounds[0]) / width)
            
            grid_cells = []
            for i in range(rows):
                for j in range(cols):
                    minx = bounds[0] + j * width
                    miny = bounds[1] + i * height
                    maxx = minx + width
                    maxy = miny + height
                    
                    cell = box(minx, miny, maxx, maxy)
                    if boundary.geometry.iloc[0].intersects(cell):
                        grid_cells.append(cell)
            
            # Create GeoDataFrame
            grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=boundary.crs)
            
            # Calculate population for each cell
            grid['population'] = 0.0
            for idx, cell in grid.iterrows():
                intersecting = population[population.intersects(cell.geometry)]
                if len(intersecting) > 0:
                    # Area-weighted population assignment
                    for _, tract in intersecting.iterrows():
                        intersection = cell.geometry.intersection(tract.geometry)
                        weight = intersection.area / tract.geometry.area
                        grid.loc[idx, 'population'] += tract['population'] * weight
            
            grid['area_km2'] = (
                grid.to_crs({'proj':'cea'}).geometry.area / 1_000_000
            )
            grid['population_density'] = grid['population'] / grid['area_km2']
            
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error generating population grid: {str(e)}")
            raise
    
    def analyze_demographics(self) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive demographic analysis.
        
        Returns:
            Dict containing various demographic analyses
        """
        try:
            population = self.get_population_data()
            
            analyses = {}
            
            # Population distribution
            analyses['population'] = pd.DataFrame({
                'total_population': population['population'].sum(),
                'mean_density': population['population_density'].mean(),
                'median_density': population['population_density'].median(),
                'std_density': population['population_density'].std()
            }, index=[0])
            
            # Income analysis
            analyses['income'] = pd.DataFrame({
                'mean_income': population['MEDIAN_INCOME'].mean(),
                'median_income': population['MEDIAN_INCOME'].median(),
                'std_income': population['MEDIAN_INCOME'].std(),
                'q25_income': population['MEDIAN_INCOME'].quantile(0.25),
                'q75_income': population['MEDIAN_INCOME'].quantile(0.75)
            }, index=[0])
            
            # Employment analysis
            analyses['employment'] = pd.DataFrame({
                'total_labour_force': population['TOTAL_LABOUR_FORCE'].sum(),
                'mean_employment_rate': population['employment_rate'].mean(),
                'median_employment_rate': population['employment_rate'].median()
            }, index=[0])
            
            # Correlation analysis
            correlations = population[[
                'population_density',
                'MEDIAN_INCOME',
                'employment_rate'
            ]].corr()
            analyses['correlations'] = correlations
            
            # Geographic clusters
            from sklearn.cluster import KMeans
            
            # Prepare features for clustering
            features = population[[
                'population_density',
                'MEDIAN_INCOME',
                'employment_rate'
            ]].copy()
            
            # Standardize features
            features = (features - features.mean()) / features.std()
            
            # Perform clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            population['cluster'] = kmeans.fit_predict(features)
            
            # Analyze clusters
            cluster_stats = population.groupby('cluster').agg({
                'population': 'sum',
                'population_density': 'mean',
                'MEDIAN_INCOME': 'mean',
                'employment_rate': 'mean'
            }).round(2)
            
            analyses['clusters'] = cluster_stats
            
            return analyses
            
        except Exception as e:
            logger.error(f"❌ Error analyzing demographics: {str(e)}")
            raise
    
    def get_population_hotspots(self, threshold_percentile: float = 90
                              ) -> gpd.GeoDataFrame:
        """
        Identify population density hotspots.
        
        Args:
            threshold_percentile: Percentile threshold for hotspots
            
        Returns:
            GeoDataFrame with hotspot areas
        """
        try:
            population = self.get_population_data()
            
            # Calculate density threshold
            threshold = population['population_density'].quantile(
                threshold_percentile / 100
            )
            
            # Identify hotspots
            hotspots = population[
                population['population_density'] >= threshold
            ].copy()
            
            # Calculate additional metrics
            hotspots['total_population'] = hotspots['population']
            hotspots['density_ratio'] = (
                hotspots['population_density'] / 
                population['population_density'].mean()
            )
            
            # Add rank based on population density
            hotspots['rank'] = hotspots['population_density'].rank(
                ascending=False
            )
            
            return hotspots
            
        except Exception as e:
            logger.error(f"❌ Error identifying hotspots: {str(e)}")
            raise
    
    def calculate_density_based_score(self, point: Point, radius_km: float = 2.0) -> float:
        """
        Calculate population-based score for a location.
        
        Args:
            point: Location to score
            radius_km: Radius to consider
            
        Returns:
            Float score between 0 and 1
        """
        try:
            population = self.get_population_data()
            population = population[population['data_source'] == 'Region of Waterloo']
            
            # Convert to UTM for accurate distance calculation
            point_utm = gpd.GeoDataFrame(
                geometry=[point],
                crs="EPSG:4326"
            ).to_crs("EPSG:32617")
            
            pop_utm = population.to_crs("EPSG:32617")
            
            # Create buffer
            buffer = point_utm.geometry[0].buffer(radius_km * 1000)  # Convert km to meters
            
            # Find intersecting census tracts
            intersecting = pop_utm[pop_utm.geometry.intersects(buffer)]
            
            if len(intersecting) == 0:
                return 0.0
            
            # Calculate score components
            total_pop = intersecting['population'].sum()
            max_pop = population['population'].max()
            pop_score = min(total_pop / max_pop, 1.0)
            
            return float(pop_score)
                
        except Exception as e:
            logger.debug(f"Error calculating density score: {str(e)}")
            return 0.0
        

    #
    # Infrastructure Analysis Methods
    #
    def analyze_transportation_network(self) -> Dict[str, Union[Dict, gpd.GeoDataFrame]]:
        """
        Analyze transportation infrastructure including roads, transit, and accessibility.
        
        Returns:
            Dict containing transportation analysis results
        """
        try:
            # Fetch required data
            roads = self.fetch_data('infrastructure', 'roads', 'geojson')
            grt_routes = self.fetch_data('transportation', 'grt_routes', 'geojson')
            grt_stops = self.fetch_data('transportation', 'grt_stops', 'geojson')
            ion_routes = self.fetch_data('transportation', 'ion_routes', 'geojson')
            ion_stops = self.fetch_data('transportation', 'ion_stops', 'geojson')

            # Convert to UTM for accurate measurements
            utm_crs = 'EPSG:32617'  # UTM Zone 17N for KW region
            roads_utm = roads.to_crs(utm_crs)
            grt_routes_utm = grt_routes.to_crs(utm_crs)
            ion_routes_utm = ion_routes.to_crs(utm_crs)
            
            # Calculate road network statistics
            road_stats = roads_utm.groupby('ROAD_CLASS').agg({
                'geometry': ['count', lambda x: x.length.sum() / 1000]  # Convert to km
            }).reset_index()
            road_stats.columns = ['class', 'segment_count', 'length_km']
                
            # Calculate total lengths
            total_road_length = roads_utm.geometry.length.sum() / 1000  # Convert to km
            total_grt_length = grt_routes_utm.geometry.length.sum() / 1000
            total_ion_length = ion_routes_utm.geometry.length.sum() / 1000
            
            analysis = {
                'road_network': {
                    'total_length': float(total_road_length),
                    'stats': {
                        'segment_count': len(roads),
                        'by_class': road_stats.to_dict('records')
                    },
                    'density_km_per_km2': float(total_road_length / self._get_cma_area()),
                    'data': roads
                },
                'transit_network': {
                    'grt_routes': len(grt_routes),
                    'grt_stops': len(grt_stops),
                    'ion_routes': len(ion_routes),
                    'ion_stops': len(ion_stops),
                    'grt_length_km': float(total_grt_length),
                    'ion_length_km': float(total_ion_length),
                    'grt_routes_data': grt_routes,
                    'grt_stops_data': grt_stops,
                    'ion_routes_data': ion_routes,
                    'ion_stops_data': ion_stops
                }
            }
            
            # Load and use population data for coverage analysis
            population_data = load_latest_file(DATA_PATHS['population'], 'geojson')
            if population_data is not None:
                # Create service area buffers
                grt_buffers = self._create_service_buffers(grt_stops, [0.4]) # 400m walking distance
                ion_buffers = self._create_service_buffers(ion_stops, [0.8]) # 800m walking distance

                # Calculate population coverage
                grt_coverage = self._calculate_population_coverage(grt_buffers[0.4], population_data)
                ion_coverage = self._calculate_population_coverage(ion_buffers[0.8], population_data)

                analysis['transit_coverage'] = {
                    'grt': grt_coverage,
                    'ion': ion_coverage
                }
            else:
                logger.warning("Population data not available for coverage calculation")
                analysis['transit_coverage'] = {
                    'grt': {'coverage_percentage': 0},
                    'ion': {'coverage_percentage': 0}
                }

            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing transportation network: {str(e)}")
            raise

    def analyze_land_use(self) -> Dict[str, Dict]:
        """
        Analyze land use patterns and their implications for charging station placement.
        
        Returns:
            Dict containing land use analysis results
        """
        try:
            # Fetch land use data
            business_areas = self.fetch_data('land_use', 'business_areas', 'geojson')
            downtown = self.fetch_data('land_use', 'downtown', 'geojson')
            parks = self.fetch_data('land_use', 'parks', 'geojson')
            
            # Convert to UTM for accurate area calculations
            utm_crs = 'EPSG:32617'  # UTM Zone 17N for KW region
            land_use_data = {
                'business': business_areas.to_crs(utm_crs),
                'downtown': downtown.to_crs(utm_crs),
                'parks': parks.to_crs(utm_crs)
            }
            
            # Store original data for visualization
            land_use_orig = {
                'business': business_areas,
                'downtown': downtown,
                'parks': parks
            }
            
            # Calculate total CMA area
            total_area = self._get_cma_area()
            
            # Calculate areas and percentages
            analysis = {}
            for use_type, data_utm in land_use_data.items():
                area_km2 = float(data_utm.geometry.area.sum() / 1_000_000)
                analysis[use_type] = {
                    'area_km2': area_km2,
                    'percentage': float(area_km2 / total_area * 100),
                    'data': land_use_orig[use_type]  # Store original data for visualization
                }
            
            # Load population data for coverage analysis
            population_data = load_latest_file(DATA_PATHS['population'])
            if population_data is not None:
                # Convert population data to UTM
                population_utm = population_data.to_crs(utm_crs)
                
                # Calculate population coverage for each land use type
                for use_type, data_utm in land_use_data.items():
                    coverage = self._calculate_population_coverage(
                        data_utm, 
                        population_utm
                    )
                    analysis[f"{use_type}_coverage"] = coverage
            else:
                logger.warning("Population data not available for coverage calculation")   
            
            # Add summary statistics
            analysis['area_breakdown'] = {
                'total_area_km2': total_area,
                'business': analysis['business'],
                'downtown': analysis['downtown'],
                'parks': analysis['parks']
            }
            
            # Log analysis summary
            logger.info(f"Land Use Analysis Summary:")
            logger.info(f"Total Area: {total_area:.1f} km²")
            for use_type in ['business', 'downtown', 'parks']:
                logger.info(f"{use_type.title()}: "
                        f"{analysis[use_type]['area_km2']:.1f} km² "
                        f"({analysis[use_type]['percentage']:.1f}%)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing land use: {str(e)}")
            raise

    def _to_utm(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert GeoDataFrame to UTM Zone 17N for accurate measurements."""
        utm_crs = 'EPSG:32617'  # UTM Zone 17N for KW region
        if gdf.crs != utm_crs:
            return gdf.to_crs(utm_crs)
        return gdf

    def _calculate_population_coverage(self, service_area: gpd.GeoDataFrame, population: gpd.GeoDataFrame) -> Dict[str, float]:
        """Calculate population coverage for a service area.
        
        Args:
            service_area: Service area polygons
            population: Population data with geometry
            
        Returns:
            Dict containing coverage metrics
        """
        try:
            # Ensure same CRS
            if service_area.crs != population.crs:
                service_area = service_area.to_crs(population.crs)
            
            # Calculate intersection
            covered = gpd.overlay(population, service_area, how='intersection')
            
            # Calculate metrics
            total_pop = float(population.loc[population['data_source'] == 'Region of Waterloo', 'population'].sum())
            covered_pop = float(covered.loc[covered['data_source'] == 'Region of Waterloo', 'population'].sum())
            coverage_pct = (covered_pop / total_pop * 100) if total_pop > 0 else 0
            
            logger.info(f"Total population: {total_pop:,.0f}")
            logger.info(f"Covered population: {covered_pop:,.0f}")
            
            return {
                'total_population': total_pop,
                'covered_population': covered_pop,
                'coverage_percentage': coverage_pct
            }
            
        except Exception as e:
            logger.warning(f"Error calculating population coverage: {str(e)}")
            return {
                'total_population': 0,
                'covered_population': 0,
                'coverage_percentage': 0
            }
    
    def calculate_accessibility_score(self, point: Point) -> Dict[str, float]:
        """Calculate accessibility score based on available data."""
        try:
            scores = {}
            
            # Population density score (40%)
            pop_density_score = 0.0
            try:
                density = self.calculate_density_based_score(point)
                pop_density_score = min(density, 1.0)
            except Exception as e:
                logger.debug(f"Error calculating density score: {str(e)}")
            
            # Transit accessibility score (30%)
            transit_score = 0.0
            try:
                transit_data = self.analyze_transportation_network()
                if transit_data:
                    transit_score = min(
                        self.calculate_transit_accessibility(point, transit_data),
                        1.0
                    )
            except Exception as e:
                logger.debug(f"Error calculating transit score: {str(e)}")
            
            # Location type score (30%)
            location_score = 0.0
            try:
                location_weights = {
                    'parking': 0.8,
                    'fuel': 1.0,
                    'retail': 0.7,
                    'commercial': 0.9,
                    'supermarket': 0.8,
                    'other': 0.6
                }
                if hasattr(point, 'location_type'):
                    location_score = location_weights.get(point.location_type, 0.6)
            except Exception as e:
                logger.debug(f"Error calculating location score: {str(e)}")
            
            # Combine scores
            total_score = (
                0.4 * pop_density_score +
                0.3 * transit_score +
                0.3 * location_score
            )
            
            return {
                'total': total_score,
                'components': {
                    'population': pop_density_score,
                    'transit': transit_score,
                    'location': location_score
                }
            }
            
        except Exception as e:
            logger.debug(f"Error in accessibility scoring: {str(e)}")
            return {'total': 0.0, 'components': {}}
        
    def calculate_accessibility_scores_batch(self, points: gpd.GeoDataFrame, max_distance: float = 2.0) -> np.ndarray:
        """
        Calculate transit accessibility scores for multiple points efficiently.
        
        Args:
            points: GeoDataFrame containing points to score
            max_distance: Maximum distance to consider for transit access (km)
            
        Returns:
            Array of transit scores between 0 and 1
        """
        # Cache transit data if not already cached
        if not hasattr(self, '_transit_cache'):
            transport = self.analyze_transportation_network()
            
            # Convert stops to UTM and ensure we have point geometries
            grt_stops = transport['transit_network']['grt_stops_data'].to_crs(self.utm_crs)
            ion_stops = transport['transit_network']['ion_stops_data'].to_crs(self.utm_crs)
            
            self._transit_cache = {
                'grt_stops': grt_stops,
                'ion_stops': ion_stops
            }
        
        # Convert points to UTM and ensure we have point geometries
        points_utm = points.to_crs(self.utm_crs)
        
        # Calculate distances to nearest stops
        def calc_min_distances(stop_points):
            if len(stop_points) == 0:
                return np.full(len(points_utm), np.inf)
                
            # Create coordinate arrays using centroids for non-point geometries
            point_coords = np.column_stack([
                [geom.centroid.x if not geom.geom_type == 'Point' else geom.x for geom in points_utm.geometry],
                [geom.centroid.y if not geom.geom_type == 'Point' else geom.y for geom in points_utm.geometry]
            ])
            
            stop_coords = np.column_stack([
                [geom.centroid.x if not geom.geom_type == 'Point' else geom.x for geom in stop_points.geometry],
                [geom.centroid.y if not geom.geom_type == 'Point' else geom.y for geom in stop_points.geometry]
            ])
            
            # Calculate all distances efficiently using broadcasting
            distances = np.sqrt(
                ((point_coords[:, np.newaxis] - stop_coords) ** 2).sum(axis=2)
            )
            
            # Get minimum distance for each point
            return np.min(distances, axis=1) / 1000  # Convert to km
        
        # Calculate minimum distances to each type of stop
        grt_distances = calc_min_distances(self._transit_cache['grt_stops'])
        ion_distances = calc_min_distances(self._transit_cache['ion_stops'])
        
        # Calculate scores
        def distance_to_score(distances, max_dist):
            return np.clip(1 - (distances / max_dist), 0, 1)
        
        # Weight different transit types
        grt_scores = distance_to_score(grt_distances, 0.8)  # 800m for bus stops
        ion_scores = distance_to_score(ion_distances, 1.2)  # 1.2km for ION stops
        
        # Combine scores with weights
        return 0.6 * grt_scores + 0.4 * ion_scores

    def calculate_transit_accessibility(self, point: Point, transit_data: Dict) -> float:
        """Calculate transit accessibility score without requiring road data."""
        try:
            # Use only GRT and ION data
            grt_stops = transit_data['transit_network']['grt_stops_data']
            ion_stops = transit_data['transit_network']['ion_stops_data']
            
            # Convert to GeoDataFrame if necessary
            if not isinstance(grt_stops, gpd.GeoDataFrame):
                grt_stops = gpd.GeoDataFrame(
                    grt_stops,
                    geometry=gpd.points_from_xy(
                        grt_stops.longitude,
                        grt_stops.latitude
                    ),
                    crs=self.wgs84_crs
                )
            
            if not isinstance(ion_stops, gpd.GeoDataFrame):
                ion_stops = gpd.GeoDataFrame(
                    ion_stops,
                    geometry=gpd.points_from_xy(
                        ion_stops.longitude,
                        ion_stops.latitude
                    ),
                    crs=self.wgs84_crs
                )
            
            # Calculate distances
            grt_dist = grt_stops.geometry.distance(point).min() / 1000  # km
            ion_dist = ion_stops.geometry.distance(point).min() / 1000  # km
            
            # Score based on distance
            grt_score = max(0, 1 - (grt_dist / 0.8))  # 800m radius
            ion_score = max(0, 1 - (ion_dist / 1.2))  # 1.2km radius
            
            return 0.6 * grt_score + 0.4 * ion_score
            
        except Exception as e:
            logger.debug(f"Error calculating transit accessibility: {str(e)}")
            return 0.0
    
    def analyze_charging_infrastructure(self) -> Dict[str, Union[Dict, pd.DataFrame]]:
        """
        Analyze existing charging infrastructure and coverage.
        Uses previously saved data instead of fetching again.
        
        Returns:
            Dict containing charging infrastructure analysis
        """
        try:
            # Load saved charging station data
            stations_df = load_latest_file(DATA_PATHS['charging_stations'])
            
            # Convert to GeoDataFrame for spatial operations
            stations_gdf = gpd.GeoDataFrame(
                stations_df,
                geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude),
                crs="EPSG:4326"
            )
            
            # Convert to UTM for accurate measurements
            stations_utm = stations_gdf.to_crs('EPSG:32617')  # UTM Zone 17N
            
            # Basic statistics
            analysis = {
                'station_counts': {
                    'total_stations': len(stations_df),
                    'total_chargers': int(stations_df['num_chargers'].sum()),
                    'avg_chargers_per_station': float(stations_df['num_chargers'].mean()),
                    'stations_per_km2': len(stations_df) / self._get_cma_area()
                }
            }
            
            # Charger type distribution
            analysis['charger_types'] = (
                stations_df.groupby('charger_type')
                .agg({
                    'num_chargers': ['count', 'sum', 'mean']
                })
                .round(2)
            )
            
            # Load population data for coverage analysis
            population_data = load_latest_file(DATA_PATHS['population'], 'geojson')
            if population_data is not None:
                population_utm = population_data.to_crs('EPSG:32617')
                
                # Calculate accessibility metrics first
                accessibility = self._analyze_charging_accessibility(stations_utm, population_utm)
                analysis.update(accessibility)
                
                # Then calculate coverage areas for specific distances
                for radius in [0.5, 1.0, 5.0]:  # 500m, 1km, and 5km coverage
                    # Create service area buffers
                    buffer_distance = radius * 1000  # Convert km to meters
                    service_area = stations_utm.copy()
                    service_area.geometry = stations_utm.geometry.buffer(buffer_distance)
                    service_area = service_area.dissolve()  # Merge overlapping buffers
                    
                    # Calculate coverage
                    coverage = self._calculate_population_coverage(
                        service_area,
                        population_utm
                    )
                    analysis[f'coverage_{radius}km'] = coverage
            
            return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing charging infrastructure: {str(e)}")
            raise

    def _analyze_charging_accessibility(self, stations: gpd.GeoDataFrame, population: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Analyze accessibility of charging infrastructure."""
        try:
            # Filter population data to only use Region of Waterloo data
            population = population[population['data_source'] == 'Region of Waterloo'].copy()
            
            # Calculate distances for each population center to nearest station
            distances = []
            weights = []
            
            for idx, row in population.iterrows():
                dist = stations.geometry.distance(row.geometry.centroid).min() / 1000  # Convert to km
                pop = row['population']
                
                if pd.notna(dist) and pd.notna(pop) and pop > 0:
                    distances.append(dist)
                    weights.append(pop)
            
            if not distances:
                raise ValueError("No valid distances calculated")
                
            distances = np.array(distances)
            weights = np.array(weights)
            total_pop = weights.sum()
            
            # Calculate coverage at different distances
            coverage = {}
            distance_thresholds = [1, 2, 3, 5, 10]  # km
            
            for dist in distance_thresholds:
                mask = distances <= dist
                covered_pop = weights[mask].sum()
                coverage[f'within_{dist}km'] = (covered_pop / total_pop * 100)
            
            # Calculate smooth coverage curve
            # Use actual distances up to 10km
            max_dist = min(10, np.ceil(distances.max()))
            plot_distances = np.linspace(0, max_dist, 100)
            plot_coverages = []
            
            for d in plot_distances:
                mask = distances <= d
                covered_pop = weights[mask].sum()
                coverage_pct = (covered_pop / total_pop * 100)
                plot_coverages.append(coverage_pct)
            
            logger.info(f"Generated coverage curve from 0 to {max_dist:.1f}km")
            logger.info(f"Coverage range: {min(plot_coverages):.1f}% - {max(plot_coverages):.1f}%")
            
            return {
                'weighted_avg_distance_km': np.average(distances, weights=weights),
                'population_coverage': coverage,
                'distance_coverage': {
                    'distances': plot_distances.tolist(),
                    'coverage_percentages': plot_coverages,
                    'total_population': total_pop,
                    'max_distance': float(max_dist)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing charging accessibility: {str(e)}")
            return {
                'weighted_avg_distance_km': 0,
                'population_coverage': {f'within_{d}km': 0 for d in [1, 2, 3, 5, 10]},
                'distance_coverage': {
                    'distances': [],
                    'coverage_percentages': [],
                    'total_population': 0,
                    'max_distance': 0
                }
            }

    def _create_service_buffers(self, 
                              points: gpd.GeoDataFrame,
                              radii: List[float]
                              ) -> Dict[float, gpd.GeoDataFrame]:
        """
        Create service area buffers around points.
        
        Args:
            points: Points to buffer
            radii: List of buffer radii in kilometers
            
        Returns:
            Dict mapping radius to buffer GeoDataFrame
        """
        try:
            # Convert to UTM for accurate buffers
            points_utm = points.to_crs(self.utm_crs)
            
            buffers = {}
            for radius in radii:
                # Create buffer
                buffered = points_utm.copy()
                buffered.geometry = points_utm.geometry.buffer(radius * 1000)
                
                # Dissolve overlapping buffers
                dissolved = buffered.dissolve().reset_index()
                
                # Convert back to WGS84
                buffers[radius] = dissolved.to_crs(self.wgs84_crs)
            
            return buffers
            
        except Exception as e:
            logger.error(f"❌ Error creating service buffers: {str(e)}")
            raise

    def _calculate_distances_to_nearest(self,
                                     point: Point,
                                     targets: gpd.GeoDataFrame
                                     ) -> Dict[str, float]:
        """
        Calculate distances from point to nearest targets.
        
        Args:
            point: Origin point
            targets: GeoDataFrame of target points/geometries
            
        Returns:
            Dict containing distance metrics
        """
        try:
            # Convert to UTM for accurate distances
            point_utm = ops.transform(
                self.transformer.transform,
                point
            )
            targets_utm = targets.to_crs(self.utm_crs)
            
            # Calculate distances in meters
            distances = targets_utm.geometry.distance(point_utm)
            
            return {
                'min': float(distances.min()) / 1000,  # Convert to km
                'mean': float(distances.mean()) / 1000,
                'nearest_n': list(distances.nsmallest(3).values / 1000)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating distances: {str(e)}")
            return {'min': float('inf'), 'mean': float('inf'), 'nearest_n': []}

    def _distance_to_score(self, distance: float, max_dist: float) -> float:
        """
        Convert distance to accessibility score.
        
        Args:
            distance: Distance in kilometers
            max_dist: Maximum distance for non-zero score
            
        Returns:
            Float score between 0 and 1
        """
        if distance >= max_dist:
            return 0.0
        return max(0.0, 1.0 - (distance / max_dist))
        
    def _get_cma_area(self) -> float:
        """Get CMA area in square kilometers."""
        boundary = self.get_cma_boundary()
        return float(
            boundary.to_crs({'proj':'cea'}).geometry.area.sum() / 1_000_000
        )

    #
    # Electric Vehicle (EV) Data Methods
    #
    def process_ev_fsa_data(self) -> gpd.GeoDataFrame:
        try:
            with tqdm(total=6, desc="Processing EV FSA Data") as pbar:
                # Load FSA data
                fsa_data = load_latest_file(DATA_PATHS['ev_fsa'], 'csv')
                pbar.update(1)
                pbar.set_description("Loading FSA boundaries")
                
                # Load FSA boundaries

                boundary_file = self.get_boundaries_shapefile()
                if not boundary_file.exists():
                    raise FileNotFoundError(f"FSA boundary file not found at {boundary_file}")
                    
                fsa_boundaries = gpd.read_file(str(boundary_file))
                
                # Debug: print columns and sample FSA values
                print("\nBoundary File Columns:", fsa_boundaries.columns.tolist())
                print("\nSample FSA values:", fsa_boundaries['CFSAUID'].head())
                
                pbar.update(1)
                pbar.set_description("Filtering and cleaning data")
                
                # Filter to KW region FSAs and clean data
                kw_fsa_data = fsa_data[fsa_data['FSA'].isin(ALL_FSA_CODES)].copy()
                
                # Rename columns to be more code-friendly
                kw_fsa_data = kw_fsa_data.rename(columns={
                    'Total EV': 'total_ev',
                    'BEV': 'bev',
                    'PHEV': 'phev'
                })
                pbar.update(1)
                pbar.set_description("Calculating metrics")
                
                # Calculate initial metrics
                kw_total_evs = kw_fsa_data['total_ev'].sum()
                kw_fsa_data['ev_share'] = kw_fsa_data['total_ev'] / kw_total_evs * 100
                kw_fsa_data['bev_ratio'] = kw_fsa_data['bev'] / kw_fsa_data['total_ev'] * 100
                kw_fsa_data['phev_ratio'] = kw_fsa_data['phev'] / kw_fsa_data['total_ev'] * 100
                pbar.update(1)
                pbar.set_description("Processing spatial data")
                
                # Get FSA column from boundaries and ensure it matches our format
                fsa_boundaries['FSA'] = fsa_boundaries['CFSAUID']
                kw_boundaries = fsa_boundaries[fsa_boundaries['FSA'].isin(ALL_FSA_CODES)]
                
                # Debug: print FSA values before merge
                print("\nFSA values in boundary data:", kw_boundaries['FSA'].tolist())
                print("\nFSA values in EV data:", kw_fsa_data['FSA'].tolist())
                
                # Merge boundary and EV data
                fsa_gdf = kw_boundaries.merge(kw_fsa_data, on='FSA', how='left')
                pbar.update(1)
                pbar.set_description("Computing spatial metrics")
                
                # Calculate spatial metrics
                fsa_gdf['area_km2'] = fsa_gdf.to_crs({'proj':'cea'}).geometry.area / 1_000_000
                fsa_gdf['ev_density'] = fsa_gdf['total_ev'] / fsa_gdf['area_km2']
                
                # Debug: print final FSA values
                print("\nFinal FSA values:", fsa_gdf['FSA'].tolist())
                
                pbar.update(1)
                pbar.set_description("Processing complete!")
                
                return fsa_gdf
            
        except Exception as e:
            logger.error(f"Error processing FSA data: {str(e)}")
            raise

    def get_boundaries_shapefile(self) -> Path:
        url = "https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/files-fichiers/lfsa000b21a_e.zip"
        boundaries_path = DATA_PATHS['boundaries']
        zip_path = boundaries_path / 'lfsa000b21a_e.zip'
        shp_path = boundaries_path / 'boundaries.shp'

        # Download the zip file
        response = requests.get(url)
        with open(zip_path, 'wb') as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(boundaries_path)

        # Find the extracted folder
        extracted_folder = None
        for item in boundaries_path.iterdir():
            if item.is_dir():
                extracted_folder = item
                break

        # Move files to the parent directory and delete the folder
        if extracted_folder:
            for file in extracted_folder.iterdir():
                if file.suffix == '.shp':
                    file.rename(shp_path)
                else:
                    file.unlink()
            extracted_folder.rmdir()

        # Delete the zip file
        zip_path.unlink()

        return shp_path
    
    #
    # Data Integration and Quality Management Methods
    #
    
    def integrate_all_data(self) -> Dict[str, Union[gpd.GeoDataFrame, pd.DataFrame]]:
        """
        Integrate all data sources into a comprehensive dataset.
        
        Returns:
            Dict containing integrated datasets and metadata
        """
        try:
            logger.info("Starting comprehensive data integration...")
            
            integrated_data = {}
            
            # 1. Geographic Base Data
            integrated_data['boundary'] = self.get_cma_boundary()
            
            # 2. Census and Demographics
            population = self.get_population_data()
            
            # 3. OpenStreetMap Data
            osm_data = self._fetch_osm_population_data()

            # 4. Infrastructure Data
            roads = self.fetch_data('infrastructure', 'roads', 'geojson')
            grt_routes = self.fetch_data('transportation', 'grt_routes', 'geojson')
            grt_stops = self.fetch_data('transportation', 'grt_stops', 'geojson')
            ion_routes = self.fetch_data('transportation', 'ion_routes', 'geojson')
            ion_stops = self.fetch_data('transportation', 'ion_stops', 'geojson')
            
            # 5. Land Use Data
            business_areas = self.fetch_data('land_use', 'business_areas', 'geojson')
            downtown = self.fetch_data('land_use', 'downtown', 'geojson')
            parks = self.fetch_data('land_use', 'parks', 'geojson')
            
            # 6. Charging Infrastructure
            stations = self.fetch_charging_stations()
            stations_gdf = gpd.GeoDataFrame(
                stations,
                geometry=gpd.points_from_xy(stations.longitude, stations.latitude),
                crs=self.wgs84_crs
            )
            
            # Calculate integrated metrics
            for tract_idx, tract in population.iterrows():
                # Transit accessibility
                grt_dist = grt_stops.distance(tract.geometry.centroid).min() * 111
                ion_dist = ion_stops.distance(tract.geometry.centroid).min() * 111
                population.loc[tract_idx, 'transit_score'] = (
                    0.6 * self._distance_to_score(grt_dist, 0.8) +
                    0.4 * self._distance_to_score(ion_dist, 1.2)
                )
                
                # Charging accessibility
                station_dist = stations_gdf.distance(tract.geometry.centroid).min() * 111
                population.loc[tract_idx, 'charging_score'] = \
                    self._distance_to_score(station_dist, 2.0)
                
                # Land use mix
                business_dist = business_areas.distance(tract.geometry.centroid).min() * 111
                park_dist = parks.distance(tract.geometry.centroid).min() * 111
                population.loc[tract_idx, 'land_use_score'] = (
                    0.6 * self._distance_to_score(business_dist, 1.0) +
                    0.4 * self._distance_to_score(park_dist, 0.8)
                )
            
            # Calculate composite scores
            population['accessibility_score'] = (
                0.4 * population['transit_score'] +
                0.3 * population['charging_score'] +
                0.3 * population['land_use_score']
            )
            
            # Store all integrated data
            integrated_data.update({
                'population': population,
                'osm_features': osm_data,
                'transportation': {
                    'roads': roads,
                    'grt_routes': grt_routes,
                    'grt_stops': grt_stops,
                    'ion_routes': ion_routes,
                    'ion_stops': ion_stops
                },
                'land_use': {
                    'business_areas': business_areas,
                    'downtown': downtown,
                    'parks': parks
                },
                'charging': stations_gdf
            })
            
            # Add metadata
            integrated_data['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'cma_area_km2': self._get_cma_area(),
                'total_population': float(population['population'].sum()),
                'total_stations': len(stations),
                'total_chargers': int(stations['num_chargers'].sum())
            }
            
            # Validate integration
            self._validate_integrated_data(integrated_data)
            
            return integrated_data
            
        except Exception as e:
            logger.error(f"❌ Error during data integration: {str(e)}")
            raise

    def _validate_integrated_data(self, data: Dict) -> None:
        """
        Validate integrated dataset.
        
        Args:
            data: Dict of integrated data to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        validation_errors = []
        
        # Check required components
        required_keys = ['boundary', 'population', 'transportation', 
                        'land_use', 'charging', 'metadata']
        
        for key in required_keys:
            if key not in data:
                validation_errors.append(f"Missing required component: {key}")
        
        # Validate geographic alignment
        try:
            boundary = data['boundary'].geometry.iloc[0]
            
            # Check population data coverage
            pop_coverage = data['population'].geometry.union().area / boundary.area
            if pop_coverage < 0.95:  # Allow 5% discrepancy
                validation_errors.append(
                    f"Population data covers only {pop_coverage:.1%} of CMA"
                )
            
            # Check infrastructure coverage
            for infra_type, gdf in data['transportation'].items():
                if not gdf.geometry.intersects(boundary).any():
                    validation_errors.append(
                        f"No {infra_type} intersect with CMA boundary"
                    )
        except Exception as e:
            validation_errors.append(f"Geographic validation error: {str(e)}")
        
        # Validate data consistency
        try:
            # Check population numbers
            total_pop = data['population']['population'].sum()
            if not (600000 <= total_pop <= 800000):  # Expected range for KW
                validation_errors.append(
                    f"Unexpected total population: {total_pop:,.0f}"
                )
            
            # Check charging stations
            stations = data['charging']
            if len(stations) < 20:  # Minimum expected stations
                validation_errors.append(
                    f"Too few charging stations: {len(stations)}"
                )
        except Exception as e:
            validation_errors.append(f"Data consistency error: {str(e)}")
        
        # Check score ranges
        score_columns = ['transit_score', 'charging_score', 
                        'land_use_score', 'accessibility_score']
        
        for col in score_columns:
            scores = data['population'][col]
            if not (0 <= scores.min() <= scores.max() <= 1):
                validation_errors.append(
                    f"Invalid {col} range: {scores.min():.2f} - {scores.max():.2f}"
                )
        
        if validation_errors:
            raise DataValidationError(
                "Data integration validation failed:\n" +
                "\n".join(f"- {err}" for err in validation_errors)
            )
    
    def validate_data_quality(self) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive data quality checks.
        
        Returns:
            Dict containing quality metrics by category
        """
        quality_metrics = {
            'completeness': {},
            'consistency': {},
            'coverage': {}
        }
        
        try:
            # Get integrated data
            data = self.integrate_all_data()
            
            # Completeness checks
            for key, dataset in data.items():
                if isinstance(dataset, (pd.DataFrame, gpd.GeoDataFrame)):
                    quality_metrics['completeness'][key] = 1 - dataset.isnull().mean().mean()
            
            # Consistency checks
            quality_metrics['consistency'].update({
                'population_density_reasonable': self._check_density_consistency(
                    data['population']
                ),
                'station_distribution_reasonable': self._check_station_consistency(
                    data['charging']
                ),
                'infrastructure_coverage': self._check_infrastructure_consistency(
                    data['transportation']
                )
            })
            
            # Coverage checks
            boundary = data['boundary'].geometry.iloc[0]
            for key, dataset in data.items():
                if isinstance(dataset, gpd.GeoDataFrame):
                    coverage = dataset.geometry.union().area / boundary.area
                    quality_metrics['coverage'][key] = float(coverage)
            
            # Update timestamp
            quality_metrics['timestamp'] = datetime.now().isoformat()
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"❌ Error in data quality validation: {str(e)}")
            return quality_metrics
    
    def _check_density_consistency(self, population: gpd.GeoDataFrame) -> float:
        """Check if population density values are reasonable."""
        densities = population['population_density']
        
        # Check range
        in_range = ((0 <= densities) & (densities <= 50000)).mean()
        
        # Check distribution
        z_scores = np.abs((densities - densities.mean()) / densities.std())
        reasonable_distribution = (z_scores <= 3).mean()
        
        return float((in_range + reasonable_distribution) / 2)
    
    def _check_station_consistency(self, stations: gpd.GeoDataFrame) -> float:
        """Check if charging station distribution is reasonable."""
        if len(stations) == 0:
            return 0.0
        
        scores = []
        
        # Check station density
        area = self._get_cma_area()
        density = len(stations) / area
        scores.append(min(density / 0.5, 1.0))  # Expect at least 0.5 stations per km²
        
        # Check spatial distribution
        coords = stations[['geometry']].copy()
        coords['x'] = stations.geometry.x
        coords['y'] = stations.geometry.y
        
        distances = pdist(coords[['x', 'y']])
        cv = distances.std() / distances.mean()  # Coefficient of variation
        scores.append(max(0, 1 - cv))  # Lower CV indicates more uniform distribution
        
        return float(np.mean(scores))
    
    def _check_infrastructure_consistency(self, 
                                       infrastructure: Dict[str, gpd.GeoDataFrame]
                                       ) -> float:
        """Check if infrastructure data is internally consistent."""
        scores = []
        
        # Check if transit routes connect to stops
        if 'grt_routes' in infrastructure and 'grt_stops' in infrastructure:
            routes = infrastructure['grt_routes']
            stops = infrastructure['grt_stops']
            
            # Buffer routes slightly to account for geometric precision
            route_buffer = routes.geometry.buffer(0.001)
            stops_near_routes = stops.geometry.intersects(route_buffer.unary_union)
            scores.append(float(stops_near_routes.mean()))
        
        # Check road network connectivity
        if 'roads' in infrastructure:
            roads = infrastructure['roads']
            
            # Simple check for network gaps
            gaps = roads.geometry.union().boundary.difference(
                roads.geometry.boundary.union()
            )
            scores.append(1 - min(len(gaps), 10) / 10)
        
        return float(np.mean(scores)) if scores else 0.0

    def get_all_warnings(self) -> Dict[str, int]:
        """Get counts of all warnings encountered."""
        return self._warning_counts.copy()
    
    def clear_warnings(self):
        """Reset warning counters."""
        self._warning_counts = {
            'validation_warnings': 0,
            'coverage_warnings': 0,
            'consistency_warnings': 0
        }

    def calculate_enhancement_score(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate enhancement opportunity score for charging infrastructure.
        
        Args:
            row: Series containing location data
            
        Returns:
            Dict containing component scores and total score
        """
        scores = {}
        
        # EV adoption factor (35%)
        if 'ev_density' in row:
            ev_score = min(row['ev_density'] / self.ev_stats['ev_density'].max(), 1.0)
            scores['ev_adoption'] = ev_score
        else:
            scores['ev_adoption'] = 0.0
        
        # Current infrastructure assessment (25%)
        # Higher score for areas with many L2 chargers but few L3
        if 'num_chargers' in row and 'charger_type' in row:
            l2_ratio = (row['num_chargers'] * (row['charger_type'] == 'Level 2'))
            l3_ratio = (row['num_chargers'] * (row['charger_type'] == 'Level 3'))
            infrastructure_score = l2_ratio / (l3_ratio + 1)  # Add 1 to avoid division by zero
            scores['infrastructure'] = min(infrastructure_score, 1.0)
        else:
            scores['infrastructure'] = 0.0
        
        # Population density impact (20%)
        density_score = min(row['population_density'] / self.population_stats['density'].max(), 1.0)
        scores['density'] = density_score
        
        # Transit accessibility inverse (15%)
        transit_score = 1 - row.get('transit_score', 0)
        scores['transit'] = transit_score
        
        # Grid capacity and age factor (5%)
        if 'DWELL_PERIOD_2011_2015' in row and 'DWELL_PERIOD_2016_2021' in row:
            recent_units = row['DWELL_PERIOD_2011_2015'] + row['DWELL_PERIOD_2016_2021']
            total_units = sum(row[col] for col in self.housing_age_cols)
            recent_score = recent_units / total_units if total_units > 0 else 0
            scores['infrastructure_age'] = recent_score
        else:
            scores['infrastructure_age'] = 0.0
        
        # Calculate weighted total
        scores['total'] = (
            0.35 * scores['ev_adoption'] +
            0.25 * scores['infrastructure'] +
            0.20 * scores['density'] +
            0.15 * scores['transit'] +
            0.05 * scores['infrastructure_age']
        )
        
        return scores

    def analyze_enhancement_opportunities(self) -> pd.DataFrame:
        """
        Analyze opportunities for charging infrastructure enhancement.
        
        Returns:
            DataFrame with enhancement scores and recommendations
        """
        try:
            # Get integrated data
            integrated_data = self.integrate_all_data()
            
            # Calculate enhancement scores for each location
            enhancement_scores = []
            
            for idx, row in integrated_data['charging'].iterrows():
                scores = self.calculate_enhancement_score(row)
                
                enhancement_scores.append({
                    'station_id': idx,
                    'name': row['name'],
                    'latitude': row.geometry.y,
                    'longitude': row.geometry.x,
                    'current_type': row['charger_type'],
                    'num_chargers': row['num_chargers'],
                    'ev_adoption_score': scores['ev_adoption'],
                    'infrastructure_score': scores['infrastructure'],
                    'density_score': scores['density'],
                    'transit_score': scores['transit'],
                    'infrastructure_age_score': scores['infrastructure_age'],
                    'total_score': scores['total'],
                    'geometry': row.geometry
                })
            
            # Create GeoDataFrame with scores
            enhancement_gdf = gpd.GeoDataFrame(enhancement_scores)
            
            # Add recommendations based on scores
            enhancement_gdf['recommendation'] = enhancement_gdf.apply(
                lambda x: 'High Priority Upgrade' if x['total_score'] > 0.8
                else 'Consider Upgrade' if x['total_score'] > 0.6
                else 'Monitor' if x['total_score'] > 0.4
                else 'No Action',
                axis=1
            )
            
            return enhancement_gdf
            
        except Exception as e:
            logger.error(f"Error analyzing enhancement opportunities: {str(e)}")
            raise

    #
    # Optimization Preparation Methods
    #
    def prepare_optimization_data(self) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """Prepare data structures for optimization model."""
        try:
            print("\n📊 Preparing Optimization Model Data")
            print("=" * 50)
            
            # 1. Load Base Data
            print("\n🔄 OPT-PREP-STEP 1.1: LOADING BASE DATA...\n")
            logging.getLogger('data.data_manager').setLevel(logging.WARNING)
            population = self.get_population_data()
            
            print("\n📊 OPT-PREP-STEP 1.2: LOADING CHARGING STATIONS...\n")
            stations = self.fetch_charging_stations()

            # Substitute unknown charger types to Level 2
            print("\nSubstituting unknown charger types to Level 2...")
            stations.loc[stations['charger_type'] == 'Unknown', 'charger_type'] = 'Level 2'
            print(f"New charger type distribution:\n{stations['charger_type'].value_counts()}")
            
            print("\n🏢 OPT-PREP-STEP 1.3: LOADING POTENTIAL LOCATIONS...\n")
            potential_df = self.fetch_potential_locations()

            print("\n📊 OPT-SUMMARY 1: DATA LOADED")
            print("-" * 50)
            print(f"Population Areas: {len(population)}")
            print(f"Charging Stations: {len(stations)}")
            print(f"Potential Locations: {len(potential_df)}")

            print("\n" + "=" * 50 + "\n")

            # 2. Process Demand Points
            print("📍 OPT-PREP-STEP 2: PROCESSING DEMAND POINTS...")
            valid_pop = population[
                (population['data_source'] == 'Region of Waterloo') &
                population.geometry.notna() &
                population.geometry.is_valid &
                population['population'].notna() &
                (population['population'] > 0)
            ]
            
            # Convert to UTM for accurate calculations
            valid_pop_utm = valid_pop.to_crs("EPSG:32617")
            
            demand_points = pd.DataFrame({
                'point_id': range(len(valid_pop_utm)),
                'latitude': valid_pop_utm.geometry.centroid.y,
                'longitude': valid_pop_utm.geometry.centroid.x,
                'population': valid_pop_utm['population'],
                'weight': valid_pop_utm['population'] / valid_pop_utm['population'].sum()
            })
            
            # Convert back to WGS84
            demand_points_wgs84 = gpd.GeoDataFrame(
                demand_points,
                geometry=gpd.points_from_xy(
                    demand_points.longitude,
                    demand_points.latitude
                ),
                crs="EPSG:32617"
            ).to_crs("EPSG:4326")
            
            demand_points['latitude'] = demand_points_wgs84.geometry.y
            demand_points['longitude'] = demand_points_wgs84.geometry.x
            
            print(f"✓ Processed {len(demand_points)} demand points!")

            print("\n" + "=" * 50 + "\n")

            # 3. Process Potential Sites
            print("🎯 OPT-PREP-STEP 3: PROCESSING POTENTIAL SITES...")
            potential_sites = potential_df.copy()
            potential_sites['site_id'] = range(len(potential_df))

            # Calculate site scores efficiently using vectorized operations
            print("Calculating site accessibility scores...")
            sites_gdf = gpd.GeoDataFrame(
                potential_sites,
                geometry=gpd.points_from_xy(
                    potential_sites.longitude,
                    potential_sites.latitude
                ),
                crs="EPSG:4326"
            ).to_crs("EPSG:32617")
            
            scores = self.calculate_accessibility_scores_batch(sites_gdf)
            potential_sites['score'] = scores
            print(f"✓ Processed {len(potential_sites)} potential sites!")

            print("\n" + "=" * 50 + "\n")

            # 4. Calculate distance matrices
            print("📏 OPT-PREP-STEP 4: CALCULATING DISTANCE MATRICES...")

            # Create GeoDataFrames in UTM for accurate distances
            sites_gdf = gpd.GeoDataFrame(
                potential_sites,
                geometry=gpd.points_from_xy(
                    potential_sites.longitude,
                    potential_sites.latitude
                ),
                crs="EPSG:4326"
            ).to_crs("EPSG:32617")

            demand_gdf = gpd.GeoDataFrame(
                demand_points,
                geometry=gpd.points_from_xy(
                    demand_points.longitude,
                    demand_points.latitude
                ),
                crs="EPSG:4326"
            ).to_crs("EPSG:32617")

            stations_gdf = gpd.GeoDataFrame(
                stations,
                geometry=gpd.points_from_xy(
                    stations.longitude,
                    stations.latitude
                ),
                crs="EPSG:4326"
            ).to_crs("EPSG:32617")

            distances = np.zeros((len(sites_gdf), len(demand_gdf)))
            with tqdm(total=len(sites_gdf), desc="Site-to-Demand Distance") as pbar:
                for i, site in enumerate(sites_gdf.geometry):
                    for j, demand in enumerate(demand_gdf.geometry):
                        distances[i, j] = site.distance(demand) / 1000  # Convert to km
                    pbar.update(1)

            print(f"\nMatrix Dimensions:")
            print(f"- Distance matrix: {distances.shape}")
            print(f"- Number of stations: {len(stations_gdf)}")
            print(f"- Number of demand points: {len(demand_gdf)}")
            print(f"- Number of potential sites: {len(sites_gdf)}")

            # In prepare_optimization_data, after calculating distances
            print("\nDistance Statistics:")
            print(f"- Mean distance: {np.mean(distances):.2f} km")
            print(f"- Max distance: {np.max(distances):.2f} km")
            print(f"- Min distance: {np.min(distances):.2f} km")

            # Load config for coverage analysis
            config_base = PROJECT_ROOT / "configs" / "base.json"
            with open(config_base) as f:
                config = json.load(f)
            
            # Calculate theoretical maximum coverage
            l2_possible = np.any(distances <= config['coverage']['l2_radius'], axis=0)
            l3_possible = np.any(distances <= config['coverage']['l3_radius'], axis=0)
            print(f"\nTheoretical Coverage Possible (with unlimited budget):")
            print(f"- L2 coverage possible: {np.mean(l2_possible):.2%}")
            print(f"- L3 coverage possible: {np.mean(l3_possible):.2%}")
            
            return {
                'existing_stations': stations_gdf,
                'demand_points': demand_points,
                'potential_sites': potential_sites,
                'distance_matrix': distances
            }
                
        except Exception as e:
            logger.error(f"❌ Error preparing optimization data: {str(e)}")
            raise
    
    def __str__(self) -> str:
        """String representation of data manager state."""
        try:
            n_cached = len(self.cache_index.get('entries', {}))
            return (
                f"DataManager(\n"
                f"  cached_datasets={n_cached},\n"
                f"  cache_dir='{self.cache_dir}',\n"
                f"  warnings={sum(self._warning_counts.values())}\n"
                f")"
            )
        except:
            return "DataManager()"