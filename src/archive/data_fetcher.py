"""
Robust data fetcher for KWC-CMA region data with caching, error handling, and validation.
"""

import requests
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, Dict, Optional, Any, List
import logging
from ratelimit import limits, sleep_and_retry
import hashlib
import warnings
from shapely.geometry import shape, box
import pyproj

from ..data.endpoints import (
    get_endpoint_url, CENSUS_ENDPOINTS, INFRASTRUCTURE_ENDPOINTS,
    LAND_USE_ENDPOINTS, TRANSPORTATION_ENDPOINTS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetchError(Exception):
    """Custom exception for data fetching errors."""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class KWDataFetcher:
    """
    Fetches and manages data for the KWC-CMA region with caching and validation.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, 
                 cache_expiry: int = 7):
        """Initialize the data fetcher."""
        self.cache_dir = cache_dir or Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(days=cache_expiry)
        
        # Define default parameters for requests
        self.DEFAULT_PARAMS = {
            'outFields': '*',
            'where': '1=1',
            'returnGeometry': 'true'
        }
        
        # Initialize cache index
        self.cache_index_file = self.cache_dir / 'cache_index.json'
        self.cache_index = self._load_cache_index()
        
        # Set up coordinate reference system
        self.crs = 'EPSG:4326'  # WGS84
        
        # More precise region bounds for KWC-CMA
        self.region_bounds = box(-80.6247, 43.3127, -80.1013, 43.6896)
    
    def _load_cache_index(self) -> Dict:
        """Load or create cache index."""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index."""
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f)
    
    def _get_cache_key(self, endpoint_type: str, dataset: str, 
                       format: str, params: Dict) -> str:
        """Generate unique cache key."""
        # Create string to hash
        to_hash = f"{endpoint_type}_{dataset}_{format}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(to_hash.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_index:
            return False
        
        cache_time = datetime.fromisoformat(self.cache_index[cache_key]['timestamp'])
        return datetime.now() - cache_time < self.cache_expiry
    
    @sleep_and_retry
    @limits(calls=60, period=60)
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make request to ArcGIS REST API."""
        if params is None:
            params = {}
        
        # Ensure required parameters
        params.update({
            'outFields': '*',
            'where': '1=1',
            'returnGeometry': 'true'
        })
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for ArcGIS error
            if 'error' in data:
                raise DataFetchError(f"ArcGIS error: {data['error']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise DataFetchError(f"Failed to fetch data: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {str(e)}")
            raise DataFetchError(f"Invalid response format: {str(e)}")
    
    def _validate_geometry(self, gdf: gpd.GeoDataFrame) -> bool:
        """Validate geometry of fetched data."""
        try:
            # Check CRS
            if gdf.crs != self.crs:
                gdf.to_crs(self.crs, inplace=True)
            
            # Check for invalid geometries
            invalid_geoms = ~gdf.geometry.is_valid
            if invalid_geoms.any():
                logger.warning(f"Found {invalid_geoms.sum()} invalid geometries")
                gdf.geometry = gdf.geometry.buffer(0)  # Try to fix
            
            # For census data, allow geometries that intersect with bounds
            is_census = any(col for col in gdf.columns if 'CTUID' in col)
            if is_census:
                within_bounds = gdf.intersects(self.region_bounds)
            else:
                # For other data, require geometries to be within bounds
                within_bounds = gdf.within(self.region_bounds)
            
            out_of_bounds = ~within_bounds
            if out_of_bounds.any():
                if is_census:
                    logger.info(f"Found {out_of_bounds.sum()} geometries intersecting region bounds")
                else:
                    logger.warning(f"Found {out_of_bounds.sum()} geometries outside region bounds")
            
            return True
            
        except Exception as e:
            logger.error(f"Geometry validation failed: {str(e)}")
            return False
    
    def _validate_data(self, data: Union[pd.DataFrame, gpd.GeoDataFrame], 
                  dataset: str) -> bool:
        """Validate fetched data."""
        try:
            # Check for empty data
            if len(data) == 0:
                raise DataValidationError("Empty dataset")
            
            # Check for required columns based on dataset type (using transformed names)
            required_columns = {
                'census_housing': [
                    'CTUID', 
                    'TOTAL_DWELLINGS'
                ],
                'census_income': [
                    'CTUID',
                    'MEDIAN_INCOME'
                ],
                'census_labour': [
                    'CTUID',
                    'TOTAL_LABOUR_FORCE'
                ],
                'roads': ['ROAD_CLASS'],
                'trails': ['TRAIL_TYPE'],
                'parking': ['LOT_TYPE'],
                'buildings': ['BLDG_TYPE'],
                'parks': ['PARK_TYPE']
            }
            
            if dataset in required_columns:
                missing_cols = []
                for col in required_columns[dataset]:
                    if col not in data.columns:
                        missing_cols.append(col)
                
                if missing_cols:
                    raise DataValidationError(f"Missing required columns: {missing_cols}")
            
            # For spatial data, validate geometry
            if isinstance(data, gpd.GeoDataFrame):
                if not self._validate_geometry(data):
                    return False
            
            return True
                
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
        
    def _clean_coordinates(self, coordinates: List) -> List:
        """Clean coordinate arrays by removing extra dimensions and null values."""
        if not coordinates:
            return coordinates
        
        # Handle single coordinate
        if not isinstance(coordinates[0], (list, tuple)):
            # Take only first two elements if they exist and are not None
            return [c for c in coordinates[:2] if c is not None]
        
        # Handle list of coordinates
        return [self._clean_coordinates(coord) for coord in coordinates]

    def _process_response(self, response: Dict, format: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Process API response with coordinate cleaning."""
        try:
            if format == 'geojson':
                if 'features' in response:
                    # Clean up features and their geometries
                    valid_features = []
                    for feature in response['features']:
                        if feature.get('geometry') is not None:
                            # Clean up coordinates
                            coords = feature['geometry'].get('coordinates', [])
                            cleaned_coords = self._clean_coordinates(coords)
                            
                            # Create cleaned feature
                            cleaned_feature = {
                                'type': 'Feature',
                                'geometry': {
                                    'type': feature['geometry']['type'],
                                    'coordinates': cleaned_coords
                                },
                                'properties': feature.get('properties', {})
                            }
                            
                            if feature.get('id') is not None:
                                cleaned_feature['id'] = feature['id']
                            
                            valid_features.append(cleaned_feature)
                        else:
                            logger.warning(f"Skipping feature with missing geometry: {feature.get('properties', {})}")
                    
                    if not valid_features:
                        raise DataFetchError("No valid features found in response")
                    
                    try:
                        return gpd.GeoDataFrame.from_features(valid_features, crs=self.crs)
                    except Exception as e:
                        logger.error(f"Error creating GeoDataFrame: {str(e)}")
                        logger.debug(f"First feature example: {json.dumps(valid_features[0], indent=2)}")
                        raise DataFetchError("Failed to create GeoDataFrame")
                    
                elif 'geometries' in response:
                    # Alternative GeoJSON structure handling
                    features = []
                    for geom in response['geometries']:
                        if geom is not None:
                            features.append({
                                'type': 'Feature',
                                'geometry': geom,
                                'properties': {}
                            })
                    return gpd.GeoDataFrame.from_features(features, crs=self.crs)
                    
                elif 'rows' in response:
                    # ArcGIS REST API format handling
                    features = []
                    for row in response['rows']:
                        if row.get('geometry') is not None:
                            features.append({
                                'type': 'Feature',
                                'geometry': row['geometry'],
                                'properties': {k: v for k, v in row.items() if k != 'geometry'}
                            })
                    return gpd.GeoDataFrame.from_features(features, crs=self.crs)
                else:
                    raise DataFetchError(f"Unexpected response structure: {list(response.keys())}")
            else:
                # Handle JSON format
                if 'features' in response:
                    # Extract properties, handling missing geometries
                    records = []
                    for feature in response['features']:
                        if 'properties' in feature:
                            record = feature['properties'].copy()
                            if feature.get('geometry'):
                                record['geometry'] = feature['geometry']
                            records.append(record)
                    return pd.DataFrame(records)
                elif 'rows' in response:
                    return pd.DataFrame(response['rows'])
                else:
                    return pd.DataFrame(response)
                    
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            logger.debug(f"Response structure: {json.dumps(response, indent=2)}")
            raise DataFetchError(f"Failed to process response: {str(e)}")
        
    def fetch_grt_routes(self) -> gpd.GeoDataFrame:
        """
        Specialized method for fetching GRT routes.
        """
        endpoint_type = 'transportation'
        dataset = 'grt_routes'
        
        try:
            params = self.DEFAULT_PARAMS.copy()
            params['f'] = 'geojson'
            url = get_endpoint_url(endpoint_type, dataset, 'geojson')
            response = self._make_request(url, params)
            
            if 'features' not in response:
                raise DataFetchError("Invalid response format for GRT routes")
            
            # Process features with coordinate cleaning
            valid_features = []
            for feature in response['features']:
                if feature.get('geometry') and feature['geometry'].get('coordinates'):
                    coords = self._clean_coordinates(feature['geometry']['coordinates'])
                    if coords and all(len(c) == 2 for c in coords):
                        cleaned_feature = {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'LineString',
                                'coordinates': coords
                            },
                            'properties': feature.get('properties', {})
                        }
                        valid_features.append(cleaned_feature)
            
            if not valid_features:
                raise DataFetchError("No valid routes found in response")
            
            return gpd.GeoDataFrame.from_features(valid_features, crs=self.crs)
            
        except Exception as e:
            logger.error(f"Error fetching GRT routes: {str(e)}")
            raise DataFetchError(f"Failed to fetch GRT routes: {str(e)}")

    def _transform_census_data(self, data: Union[pd.DataFrame, gpd.GeoDataFrame], 
                         dataset: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Transform census data to standardized format."""
        try:
            if dataset.startswith('census_'):
                # Create standardized column names
                if dataset == 'census_housing':
                    mapping = {
                        'GEO_CODE': 'CTUID',
                        'TOT_DWELL_HHLDS_TENURE_25SAM': 'TOTAL_DWELLINGS',
                        'DWELL_OWNER': 'OWNED_DWELLINGS',
                        'DWELL_RENTER': 'RENTED_DWELLINGS'
                    }
                elif dataset == 'census_income':
                    mapping = {
                        'GEO_CODE': 'CTUID',
                        'TOT_INC_2020_DOLLARS_HHLDS_MED': 'MEDIAN_INCOME',
                        'TOT_INC_2020_HHLD_AVG': 'AVERAGE_INCOME'
                    }
                elif dataset == 'census_labour':
                    mapping = {
                        'GEO_CODE': 'CTUID',
                        'LABOUR_FORCE_15YR_25SAM': 'TOTAL_LABOUR_FORCE',
                        'LF_15YR_IN_THE_LABOUR_FORCE': 'ACTIVE_LABOUR_FORCE',
                        'LF_15YR_EMPLOYED': 'EMPLOYED',
                        'LF_15YR_UNEMPLOYED': 'UNEMPLOYED'
                    }
                
                # Rename existing columns
                existing_cols = {old: new for old, new in mapping.items() 
                            if old in data.columns}
                data = data.rename(columns=existing_cols)
                
                # Add calculated fields
                if dataset == 'census_income':
                    if 'TOT_INC_2020_HHLD_AVG' in data.columns:
                        data['TOTAL_INCOME'] = data['TOT_INC_2020_HHLD_AVG'] * data['TOTAL_PRIVATE_DWELL']
                
                logger.info(f"Transformed {dataset} data with columns: {list(data.columns)}")
                
            return data
            
        except Exception as e:
            logger.error(f"Error transforming {dataset} data: {str(e)}")
            return data
    
    def fetch_data(self, endpoint_type: str, dataset: str, 
               format: str = 'geojson', force_refresh: bool = False,
               validate: bool = True, debug: bool = False) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Fetch data from specified endpoint with caching and validation.
        """
        # Generate cache key
        params = {'outFields': '*', 'where': '1=1'}
        if format == 'geojson':
            params['f'] = 'geojson'
        else:
            params['f'] = 'json'
        
        cache_key = self._get_cache_key(endpoint_type, dataset, format, params)
        
        # Check cache unless force refresh
        if not force_refresh and self._is_cache_valid(cache_key):
            cache_file = self.cache_dir / f"{cache_key}.{format}"
            logger.info(f"Loading {dataset} from cache")
            
            if format == 'geojson':
                df = gpd.read_file(cache_file)
            else:
                df = pd.read_json(cache_file)
                
            # Transform cached data
            df = self._transform_census_data(df, f"{endpoint_type}_{dataset}")
            
            # Validate transformed data
            if validate and not self._validate_data(df, f"{endpoint_type}_{dataset}"):
                raise DataValidationError("Data validation failed")
                
            return df
        
        # Fetch new data
        logger.info(f"Fetching {dataset} data")
        url = get_endpoint_url(endpoint_type, dataset, format)
        
        try:
            # Make request with retries
            response = self._make_request(url, params)
            
            if debug:
                print(f"Response for {dataset}:")
                print(json.dumps(response, indent=2)[:500] + "...")  # Show first 500 chars

            # Process response based on format and structure
            df = self._process_response(response, format)

            # Log basic stats about the result
            logger.info(f"Processed {dataset}: {len(df)} records, {len(df.columns)} columns")
            if isinstance(df, gpd.GeoDataFrame):
                logger.info(f"Geometry types: {df.geometry.type.value_counts().to_dict()}")
            
            # Transform before validation
            df = self._transform_census_data(df, f"{endpoint_type}_{dataset}")
            
            # Validate after transformation
            if validate and not self._validate_data(df, f"{endpoint_type}_{dataset}"):
                raise DataValidationError("Data validation failed")
            
            # Cache the transformed and validated data
            cache_file = self.cache_dir / f"{cache_key}.{format}"
            if format == 'geojson':
                df.to_file(cache_file, driver='GeoJSON')
            else:
                df.to_json(cache_file)
            
            # Update cache index
            self.cache_index[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'endpoint_type': endpoint_type,
                'dataset': dataset,
                'format': format
            }
            self._save_cache_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {dataset}: {str(e)}")
            if debug:
                logger.exception("Detailed error information:")
            raise DataFetchError(f"Failed to fetch {dataset}: {str(e)}")
    
    def fetch_multiple(self, datasets: Dict[str, str], 
                      format: str = 'geojson') -> Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]:
        """
        Fetch multiple datasets.
        
        Args:
            datasets: Dictionary of {dataset_name: endpoint_type}
            format: Desired format
            
        Returns:
            Dictionary of {dataset_name: dataframe}
        """
        results = {}
        for name, endpoint_type in datasets.items():
            try:
                results[name] = self.fetch_data(endpoint_type, name, format)
            except Exception as e:
                logger.error(f"Error fetching {name}: {str(e)}")
                results[name] = None
        return results
    
    def clear_cache(self, dataset: Optional[str] = None):
        """Clear cache for specific dataset or all datasets."""
        if dataset:
            # Clear specific dataset
            keys_to_remove = []
            for key, info in self.cache_index.items():
                if info['dataset'] == dataset:
                    cache_file = self.cache_dir / f"{key}.{info['format']}"
                    if cache_file.exists():
                        cache_file.unlink()
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache_index[key]
        else:
            # Clear all cache
            for file in self.cache_dir.glob('*.*'):
                if file != self.cache_index_file:
                    file.unlink()
            self.cache_index.clear()
        
        self._save_cache_index()
    
    def get_cache_info(self) -> pd.DataFrame:
        """Get information about cached datasets."""
        cache_info = []
        for key, info in self.cache_index.items():
            cache_file = self.cache_dir / f"{key}.{info['format']}"
            if cache_file.exists():
                size = cache_file.stat().st_size / 1024  # KB
                cache_info.append({
                    'dataset': info['dataset'],
                    'type': info['endpoint_type'],
                    'format': info['format'],
                    'cached_at': info['timestamp'],
                    'size_kb': round(size, 2)
                })
        
        return pd.DataFrame(cache_info)

    def validate_all_cached(self) -> Dict[str, bool]:
        """Validate all cached datasets."""
        results = {}
        for key, info in self.cache_index.items():
            cache_file = self.cache_dir / f"{key}.{info['format']}"
            if cache_file.exists():
                try:
                    if info['format'] == 'geojson':
                        data = gpd.read_file(cache_file)
                    else:
                        data = pd.read_json(cache_file)
                    
                    results[info['dataset']] = self._validate_data(
                        data, 
                        f"{info['endpoint_type']}_{info['dataset']}"
                    )
                except Exception as e:
                    logger.error(f"Error validating {info['dataset']}: {str(e)}")
                    results[info['dataset']] = False
        
        return results