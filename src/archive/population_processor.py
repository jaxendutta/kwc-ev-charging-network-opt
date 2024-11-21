"""
Comprehensive population and infrastructure data processor for KWC-CMA region.
Integrates data from:
- Region of Waterloo Open Data Portal (primary source)
- Statistics Canada Census
- OpenStreetMap
- UN Data
- Data Commons
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import osmnx as ox
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from shapely.geometry import Point, Polygon, MultiPolygon, box
import os
from dotenv import load_dotenv
from functools import lru_cache
from src.data.utils import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass

class PopulationDataProcessor:
    """
    Processes and integrates population, demographic, and infrastructure data.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the data processor."""
        # Load environment variables
        load_dotenv()
        
        # Set up cache directory
        self.cache_dir = cache_dir or Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # ROW OpenData endpoints
        self.row_endpoints = {
            'base_url': "https://rowopendata-rmw.opendata.arcgis.com/datasets",
            'services': {
                'boundary': "259750cc5e8c44f78c56d27ac28b04ed_16",     # Regional boundary
                'census_tracts': "c933dc76e31d4c99a85aa11b58a1bdc4_0", # Census tracts
                'population': "db90ca1d018d4087970c109a72a71845_0",    # Population density
                'land_use': "1c051cb66c89432ab3fc7095d5899246_0",      # Land use
                'roads': "3d7629bf90e8400c9956c987e14c7b5d_0",         # Road network
                'employment': "8e07a4f30c5147eaa1d028c9c201d019_0",    # Employment areas
                'zoning': "6ad5e06795a24ba7a55b3bd6017e1497_0",        # Zoning
                'transit': "4d949c11263d4c4aac3d0d5266487c11_0",       # Transit routes
                'parks': "a8bf797ebf50463c9d2331694e0d0271_0"          # Parks and recreation
            }
        }
        
        # Other data source endpoints
        self.endpoints = {
            'statcan': {
                'base_url': "https://www12.statcan.gc.ca/rest/census-recensement/2021",
                'services': {
                    'population': "/population",
                    'dwellings': "/dwellings",
                    'age_distribution': "/age-distribution",
                    'income': "/income",
                    'employment': "/employment"
                }
            },
            'un_data': {
                'base_url': "https://data.un.org/ws/rest/data/",
                'population': "UNSD-MBS/POP_TABLE3_1"
            },
            'datacommons': {
                'base_url': "https://api.datacommons.org/v1",
                'services': {
                    'place': "/place",
                    'stats': "/stat/set"
                }
            }
        }
        
        # Region identifiers
        self.region_codes = {
            'kwc_cma_code': "541",        # ROW/StatCan CMA code
            'kwc_cma_uid': "35541",       # Alternative CMA ID
            'dc_place_id': "Q175735",     # Data Commons place ID
            'un_area_code': "124"         # UN area code for Canada
        }
        
        # Initialize cache
        self._cache = {}
        
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, retry_count: int = 3) -> Dict:
        """
        Make HTTP request with retry logic and error handling.
        """
        for attempt in range(retry_count):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == retry_count - 1:
                    logger.error(f"Request failed after {retry_count} attempts for {url}: {str(e)}")
                    raise DataSourceError(f"Failed to fetch data from {url}: {str(e)}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                continue

    def _get_row_feature_server_url(self, service: str) -> str:
        """
        Get Feature Server URL for ROW OpenData service.
        """
        service_id = self.row_endpoints['services'].get(service)
        if not service_id:
            raise ValueError(f"Unknown ROW service: {service}")
            
        return f"{self.row_endpoints['base_url']}/{service_id}/FeatureServer/0/query"

    @lru_cache(maxsize=None)
    def get_cma_boundary(self) -> gpd.GeoDataFrame:
        """
        Fetch KWC-CMA boundary from cache or Region of Waterloo Open Data.
        Returns a cleaned GeoDataFrame with the precise CMA boundary.
        """
        # Check cache first
        cache_file = self.cache_dir / 'kwc_boundary.geojson'
        
        if cache_file.exists():
            try:
                logger.info("Loading CMA boundary from cache...")
                gdf = gpd.read_file(cache_file)
                logger.info("Successfully loaded boundary from cache")
                return self._clean_boundary_data(gdf)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {str(e)}")
        
        # If not in cache or cache load failed, fetch from API
        url = "https://utility.arcgis.com/usrsvcs/servers/259750cc5e8c44f78c56d27ac28b04ed/rest/services/OpenData/OpenData/MapServer/16/query"
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson'
        }
        
        try:
            logger.info("Fetching KWC-CMA boundary from Region of Waterloo...")
            response = self._make_request(url, params)
            
            if not response.get('features'):
                raise DataSourceError("No boundary features found in response")
            
            # Create GeoDataFrame from features
            gdf = gpd.GeoDataFrame.from_features(response['features'], crs="EPSG:4326")
            
            # Clean the data
            gdf = self._clean_boundary_data(gdf)
            
            # Save to cache
            try:
                logger.info("Saving boundary to cache...")
                gdf.to_file(cache_file, driver='GeoJSON')
            except Exception as e:
                logger.warning(f"Failed to save to cache: {str(e)}")
            
            logger.info("Successfully retrieved KWC-CMA boundary")
            return gdf
            
        except Exception as e:
            logger.error(f"Error fetching CMA boundary: {str(e)}")
            raise DataSourceError("Unable to fetch CMA boundary from source")

    def _clean_boundary_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clean boundary data by removing redundant columns and standardizing names."""
        # Keep only essential columns
        essential_cols = ['geometry', 'ShortName', 'LongName']
        other_cols = [col for col in gdf.columns 
                    if col not in essential_cols 
                    and not any(x in col.lower() for x in ['shape', 'area', 'length', 'objectid'])]
        
        cols_to_keep = essential_cols + other_cols
        gdf = gdf[[col for col in cols_to_keep if col in gdf.columns]]
        
        return gdf

    def get_census_tracts(self) -> gpd.GeoDataFrame:
        """
        Fetch census tracts within KWC-CMA boundary from ROW Open Data.
        """
        # Use the correct URL from ROW Open Data for census tracts
        url = "https://utility.arcgis.com/usrsvcs/servers/c933dc76e31d4c99a85aa11b58a1bdc4/rest/services/OpenData/OpenData/MapServer/0/query"
        
        try:
            # First try to get from cache
            cache_file = self.cache_dir / 'census_tracts.geojson'
            
            if cache_file.exists():
                logger.info("Loading census tracts from cache...")
                return gpd.read_file(cache_file)
            
            logger.info("Fetching census tracts data...")
            
            # Get the boundary to use its extent
            boundary = self.get_cma_boundary()
            bbox = boundary.total_bounds
            
            params = {
                'where': '1=1',
                'outFields': '*',
                'outSR': '4326',
                'f': 'geojson',
                'geometryType': 'esriGeometryEnvelope',
                'spatialRel': 'esriSpatialRelIntersects',
                'returnGeometry': 'true',
                'geometry': {
                    'xmin': bbox[0],
                    'ymin': bbox[1],
                    'xmax': bbox[2],
                    'ymax': bbox[3]
                }
            }
            
            # Convert geometry to string
            params['geometry'] = json.dumps(params['geometry'])
            
            response = self._make_request(url, params)
            
            if not response or 'features' not in response:
                # Try alternative approach without spatial filter
                logger.info("Retrying with simplified request...")
                params = {
                    'where': '1=1',
                    'outFields': '*',
                    'outSR': '4326',
                    'f': 'geojson',
                    'returnGeometry': 'true'
                }
                response = self._make_request(url, params)
            
            if not response or 'features' not in response:
                raise DataSourceError("No census tract features found in response")
            
            # Create GeoDataFrame
            tracts = gpd.GeoDataFrame.from_features(response['features'], crs="EPSG:4326")
            
            # Clip to exact CMA boundary
            tracts_clipped = gpd.clip(tracts, boundary)
            
            # Cache the results
            tracts_clipped.to_file(cache_file, driver='GeoJSON')
            
            return tracts_clipped
            
        except Exception as e:
            logger.error(f"Error fetching census tracts: {str(e)}")
            raise DataSourceError(f"Failed to fetch census tracts: {str(e)}")

    def get_population_data(self) -> gpd.GeoDataFrame:
        """
        Fetch population density data from ROW OpenData.
        """
        # Use correct URL for population data
        url = "https://utility.arcgis.com/usrsvcs/servers/db90ca1d018d4087970c109a72a71845/rest/services/OpenData/OpenData/MapServer/0/query"
        
        try:
            # Check cache first
            cache_file = self.cache_dir / 'population_data.geojson'
            
            if cache_file.exists():
                logger.info("Loading population data from cache...")
                return gpd.read_file(cache_file)
            
            logger.info("Fetching population data...")
            
            params = {
                'where': '1=1',
                'outFields': '*',
                'outSR': '4326',
                'f': 'geojson',
                'returnGeometry': 'true'
            }
            
            response = self._make_request(url, params)
            
            if not response or 'features' not in response:
                raise DataSourceError("No population data features found in response")
            
            # Create GeoDataFrame
            population = gpd.GeoDataFrame.from_features(response['features'], crs="EPSG:4326")
            
            # Clip to CMA boundary
            boundary = self.get_cma_boundary()
            population_clipped = gpd.clip(population, boundary)
            
            # Cache the results
            population_clipped.to_file(cache_file, driver='GeoJSON')
            
            return population_clipped
            
        except Exception as e:
            logger.error(f"Error fetching population data: {str(e)}")
            raise DataSourceError(f"Failed to fetch population data: {str(e)}")

    def get_land_use(self) -> gpd.GeoDataFrame:
        """
        Fetch land use data from ROW OpenData.
        """
        url = self._get_row_feature_server_url('land_use')
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson'
        }
        
        data = self._make_request(url, params)
        return gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")

    def get_road_network(self) -> gpd.GeoDataFrame:
        """
        Fetch road network data from ROW OpenData.
        """
        url = self._get_row_feature_server_url('roads')
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson'
        }
        
        data = self._make_request(url, params)
        return gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")

    def get_employment_areas(self) -> gpd.GeoDataFrame:
        """
        Fetch employment areas data from ROW OpenData.
        """
        url = self._get_row_feature_server_url('employment')
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson'
        }
        
        data = self._make_request(url, params)
        return gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")

    def get_zoning(self) -> gpd.GeoDataFrame:
        """
        Fetch zoning data from ROW OpenData.
        """
        url = self._get_row_feature_server_url('zoning')
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson'
        }
        
        data = self._make_request(url, params)
        return gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")

    def get_transit_routes(self) -> gpd.GeoDataFrame:
        """
        Fetch transit routes data from ROW OpenData.
        """
        url = self._get_row_feature_server_url('transit')
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson'
        }
        
        data = self._make_request(url, params)
        return gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")

    def get_parks(self) -> gpd.GeoDataFrame:
        """
        Fetch parks and recreation data from ROW OpenData.
        """
        url = self._get_row_feature_server_url('parks')
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson'
        }
        
        data = self._make_request(url, params)
        return gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")
    
    def get_statcan_data(self) -> pd.DataFrame:
        """
        Fetch Statistics Canada census data for KWC-CMA.
        """
        data = {}
        params = {
            'dguid': f"2021S05{self.region_codes['kwc_cma_uid']}",
            'format': 'json'
        }
        
        for service, endpoint in self.endpoints['statcan']['services'].items():
            try:
                url = f"{self.endpoints['statcan']['base_url']}{endpoint}"
                response = self._make_request(url, params)
                data[service] = pd.DataFrame(response.get('data', []))
            except Exception as e:
                logger.error(f"Error fetching StatCan {service} data: {str(e)}")
                data[service] = pd.DataFrame()  # Empty DataFrame as fallback
        
        return self._process_statcan_data(data)

    def get_un_data(self) -> pd.DataFrame:
        """
        Fetch UN population data for trend validation.
        """
        url = f"{self.endpoints['un_data']['base_url']}{self.endpoints['un_data']['population']}"
        
        params = {
            'areaCodes': self.region_codes['un_area_code'],
            'format': 'json'
        }
        
        try:
            response = self._make_request(url, params)
            df = pd.DataFrame(response.get('data', []))
            
            # Process UN data
            if not df.empty:
                df['year'] = pd.to_datetime(df['year'], format='%Y')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.sort_values('year', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching UN data: {str(e)}")
            return pd.DataFrame()

    def get_datacommons_data(self) -> pd.DataFrame:
        """
        Fetch Data Commons population and demographic data.
        """
        headers = {'X-API-Key': os.getenv('DATACOMMONS_API_KEY')}
        
        try:
            # Fetch place data
            place_url = (f"{self.endpoints['datacommons']['base_url']}"
                        f"{self.endpoints['datacommons']['services']['place']}/"
                        f"{self.region_codes['dc_place_id']}")
            
            place_data = self._make_request(place_url, headers=headers)
            
            # Fetch statistical data
            stats_url = (f"{self.endpoints['datacommons']['base_url']}"
                        f"{self.endpoints['datacommons']['services']['stats']}")
            
            stats_params = {
                'place': self.region_codes['dc_place_id'],
                'stat_vars': [
                    'Count_Person',
                    'Count_Household',
                    'MedianAge_Person',
                    'MedianIncome_Household',
                    'Count_Vehicle',
                    'Count_Business_Establishment'
                ]
            }
            
            stats_data = self._make_request(stats_url, params=stats_params, headers=headers)
            
            return self._process_datacommons_data(place_data, stats_data)
            
        except Exception as e:
            logger.error(f"Error fetching Data Commons data: {str(e)}")
            return pd.DataFrame()

    def integrate_all_data(self) -> Dict[str, Union[gpd.GeoDataFrame, pd.DataFrame]]:
        """
        Integrate all data sources into a comprehensive dataset.
        """
        try:
            logger.info("Starting data integration process...")
            
            # Get all ROW data
            boundary = self.get_cma_boundary()
            census_tracts = self.get_census_tracts()
            population = self.get_population_data()
            land_use = self.get_land_use()
            roads = self.get_road_network()
            employment = self.get_employment_areas()
            zoning = self.get_zoning()
            transit = self.get_transit_routes()
            parks = self.get_parks()
            
            # Get external data
            statcan = self.get_statcan_data()
            un_data = self.get_un_data()
            dc_data = self.get_datacommons_data()
            
            # Merge census and population data
            base_data = census_tracts.merge(
                population,
                on='CTUID',
                how='left',
                suffixes=('', '_pop')
            )
            
            # Add land use metrics
            base_data = self._add_land_use_metrics(base_data, land_use)
            
            # Add accessibility metrics
            base_data = self._add_accessibility_metrics(
                base_data, roads, transit, parks
            )
            
            # Add employment and zoning information
            base_data = self._add_employment_metrics(
                base_data, employment, zoning
            )
            
            # Validate and reconcile population numbers
            base_data = self._reconcile_population_data(
                base_data, statcan, un_data, dc_data
            )
            
            logger.info("Data integration completed successfully")
            
            return {
                'integrated_data': base_data,
                'boundary': boundary,
                'roads': roads,
                'transit': transit,
                'parks': parks,
                'land_use': land_use,
                'employment': employment,
                'zoning': zoning,
                'external_data': {
                    'statcan': statcan,
                    'un_data': un_data,
                    'datacommons': dc_data
                }
            }
            
        except Exception as e:
            logger.error(f"Error in data integration: {str(e)}")
            raise

    def _add_land_use_metrics(self, base_data: gpd.GeoDataFrame,
                            land_use: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add land use metrics to base data.
        """
        # Calculate area percentages by land use type
        intersection = gpd.overlay(land_use, base_data[['geometry', 'CTUID']], 
                                 how='intersection')
        
        intersection['area'] = intersection.geometry.area
        
        # Calculate percentages by land use type for each census tract
        land_use_stats = (intersection.groupby(['CTUID', 'LAND_USE'])['area']
                         .sum()
                         .unstack(fill_value=0))
        
        # Calculate percentages
        total_area = land_use_stats.sum(axis=1)
        land_use_pct = land_use_stats.div(total_area, axis=0) * 100
        
        # Rename columns
        land_use_pct.columns = [f'pct_{col.lower()}_area' 
                              for col in land_use_pct.columns]
        
        # Merge back to base data
        return base_data.merge(land_use_pct, on='CTUID', how='left')

    def _add_accessibility_metrics(self, base_data: gpd.GeoDataFrame,
                                 roads: gpd.GeoDataFrame,
                                 transit: gpd.GeoDataFrame,
                                 parks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate accessibility metrics for each census tract.
        """
        # Road density
        base_data['road_density'] = base_data.apply(
            lambda row: self._calculate_road_density(row, roads),
            axis=1
        )
        
        # Transit accessibility
        base_data['transit_score'] = base_data.apply(
            lambda row: self._calculate_transit_score(row, transit),
            axis=1
        )
        
        # Park accessibility
        base_data['park_score'] = base_data.apply(
            lambda row: self._calculate_park_score(row, parks),
            axis=1
        )
        
        return base_data

    def _calculate_road_density(self, tract: gpd.GeoSeries, 
                              roads: gpd.GeoDataFrame) -> float:
        """Calculate road network density for a census tract."""
        try:
            # Clip roads to tract boundary
            tract_roads = gpd.clip(roads, tract.geometry)
            
            # Calculate total road length in kilometers
            road_length = tract_roads.geometry.length.sum() * 111  # Convert degrees to km
            
            # Calculate tract area in square kilometers
            tract_area = tract.geometry.area * (111 * 111)  # Convert degrees to km²
            
            return road_length / tract_area if tract_area > 0 else 0
            
        except Exception as e:
            logger.warning(f"Error calculating road density: {str(e)}")
            return 0

    def _calculate_transit_score(self, tract: gpd.GeoSeries,
                               transit: gpd.GeoDataFrame) -> float:
        """Calculate transit accessibility score for a census tract."""
        try:
            # Count transit routes within/intersecting tract
            intersecting_routes = transit[transit.intersects(tract.geometry)]
            
            # Calculate route density
            tract_area = tract.geometry.area * (111 * 111)  # Convert to km²
            route_density = len(intersecting_routes) / tract_area if tract_area > 0 else 0
            
            # Calculate stop density if stop data available
            if 'STOP_ID' in transit.columns:
                stops = transit[transit.geometry.type == 'Point']
                tract_stops = stops[stops.within(tract.geometry)]
                stop_density = len(tract_stops) / tract_area if tract_area > 0 else 0
                
                # Combine metrics
                return (0.6 * route_density + 0.4 * stop_density)
            
            return route_density
            
        except Exception as e:
            logger.warning(f"Error calculating transit score: {str(e)}")
            return 0

    def _calculate_park_score(self, tract: gpd.GeoSeries,
                            parks: gpd.GeoDataFrame) -> float:
        """Calculate park accessibility score for a census tract."""
        try:
            # Calculate park area within tract
            tract_parks = gpd.clip(parks, tract.geometry)
            park_area = tract_parks.geometry.area.sum() * (111 * 111)  # Convert to km²
            
            # Calculate tract area
            tract_area = tract.geometry.area * (111 * 111)
            
            # Calculate park percentage
            park_percentage = (park_area / tract_area * 100) if tract_area > 0 else 0
            
            # Count nearby parks (within 1km buffer)
            buffer = tract.geometry.buffer(1/111)  # 1km buffer
            nearby_parks = parks[parks.intersects(buffer)]
            park_count = len(nearby_parks)
            
            # Combine metrics
            return (0.7 * min(park_percentage, 100) + 0.3 * min(park_count * 10, 100)) / 100
            
        except Exception as e:
            logger.warning(f"Error calculating park score: {str(e)}")
            return 0

    def _add_employment_metrics(self, base_data: gpd.GeoDataFrame,
                              employment: gpd.GeoDataFrame,
                              zoning: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add employment and zoning metrics to base data."""
        try:
            # Calculate employment area percentages
            emp_intersection = gpd.overlay(employment, 
                                         base_data[['geometry', 'CTUID']], 
                                         how='intersection')
            
            emp_intersection['area'] = emp_intersection.geometry.area
            
            # Calculate employment area percentage
            emp_stats = emp_intersection.groupby('CTUID')['area'].sum()
            total_area = base_data.set_index('CTUID').geometry.area
            
            base_data['employment_area_pct'] = (
                base_data['CTUID'].map(emp_stats) / total_area * 100
            ).fillna(0)
            
            # Add zoning metrics
            zoning_intersection = gpd.overlay(zoning, 
                                            base_data[['geometry', 'CTUID']], 
                                            how='intersection')
            
            zoning_intersection['area'] = zoning_intersection.geometry.area
            
            # Calculate percentages by zoning type
            zoning_stats = (zoning_intersection.groupby(['CTUID', 'ZONE_TYPE'])['area']
                          .sum()
                          .unstack(fill_value=0))
            
            # Calculate percentages
            total_area = zoning_stats.sum(axis=1)
            zoning_pct = zoning_stats.div(total_area, axis=0) * 100
            
            # Rename columns
            zoning_pct.columns = [f'pct_{col.lower()}_zoning' 
                                for col in zoning_pct.columns]
            
            # Merge back to base data
            base_data = base_data.merge(zoning_pct, on='CTUID', how='left')
            
            return base_data
            
        except Exception as e:
            logger.error(f"Error adding employment metrics: {str(e)}")
            return base_data

    def generate_comprehensive_report(self, 
                                   data: Dict[str, Union[gpd.GeoDataFrame, pd.DataFrame]],
                                   save_file: bool = True,
                                   output_file: Optional[Path] = None) -> Dict:
        """
        Generate comprehensive report including all metrics and analyses.
        """
        try:
            integrated_data = data['integrated_data']
            
            report = {
                'summary_statistics': {
                    'total_population': float(integrated_data['POPULATION_final'].sum()),
                    'total_area_km2': float(integrated_data.geometry.area.sum() * 111 * 111),
                    'num_census_tracts': len(integrated_data),
                    'avg_population_density': float(
                        integrated_data['POPULATION_final'].sum() / 
                        (integrated_data.geometry.area.sum() * 111 * 111)
                    )
                },
                'land_use': {
                    'distribution': {
                        col: float(integrated_data[col].mean())
                        for col in integrated_data.columns
                        if col.startswith('pct_') and col.endswith('_area')
                    }
                },
                'accessibility': {
                    'avg_road_density': float(integrated_data['road_density'].mean()),
                    'avg_transit_score': float(integrated_data['transit_score'].mean()),
                    'avg_park_score': float(integrated_data['park_score'].mean())
                },
                'employment': {
                    'total_employment_area_pct': float(integrated_data['employment_area_pct'].mean()),
                    'zoning_distribution': {
                        col: float(integrated_data[col].mean())
                        for col in integrated_data.columns
                        if col.startswith('pct_') and col.endswith('_zoning')
                    }
                },
                'data_quality': {
                    'population_confidence': float(integrated_data['confidence_score'].mean()),
                    'missing_data_pct': float(integrated_data.isnull().mean().mean() * 100)
                },
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'data_sources': list(self.endpoints.keys()),
                    'region_codes': self.region_codes
                }
            }
            
            if save_file:
                report_file, report_timestamp = save_data(
                    report, 
                    'comprehensive_report', 
                    'json',
                    output_file=output_file
                )
                logger.info(f"Report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            raise

    def create_analysis_tables(self, 
                             data: Dict[str, Union[gpd.GeoDataFrame, pd.DataFrame]],
                             save_files: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Create analysis tables for various metrics.
        """
        integrated_data = data['integrated_data']
        
        try:
            tables = {}
            
            # Population density analysis
            tables['population_density'] = (
                integrated_data.groupby('CTUID')
                .agg({
                    'POPULATION_final': 'sum',
                    'geometry': lambda x: x.area.iloc[0] * 111 * 111  # km²
                })
                .assign(density=lambda df: df['POPULATION_final'] / df['geometry'])
                .sort_values('density', ascending=False)
            )

            # Land use analysis
            land_use_cols = [col for col in integrated_data.columns 
                           if col.startswith('pct_') and col.endswith('_area')]
            tables['land_use'] = integrated_data[['CTUID'] + land_use_cols].copy()
            
            # Accessibility analysis
            tables['accessibility'] = integrated_data[
                ['CTUID', 'road_density', 'transit_score', 'park_score']
            ].copy()
            
            # Employment and zoning analysis
            employment_cols = ['employment_area_pct'] + [
                col for col in integrated_data.columns 
                if col.startswith('pct_') and col.endswith('_zoning')
            ]
            tables['employment'] = integrated_data[['CTUID'] + employment_cols].copy()
            
            if save_files:
                timestamp = grab_time()
                for name, table in tables.items():
                    file_path = Path(f'data/analyzed/{name}_{timestamp}.csv')
                    table.to_csv(file_path)
                    logger.info(f"Saved {name} analysis to: {file_path}")
            
            return tables
            
        except Exception as e:
            logger.error(f"Error creating analysis tables: {str(e)}")
            raise

    def _process_statcan_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process and combine Statistics Canada datasets."""
        try:
            # Process population data
            population = data['population'].copy()
            population = population.rename(columns={
                'GEO_CODE': 'CTUID',
                'C1_COUNT_TOTAL': 'POPULATION'
            })

            # Process dwelling data
            dwellings = data['dwellings'].copy()
            dwellings = dwellings.rename(columns={
                'GEO_CODE': 'CTUID',
                'PRIVATE_DWELLINGS': 'TOTAL_DWELLINGS'
            })

            # Process age distribution
            age_dist = data['age_distribution'].copy()
            age_dist = age_dist.rename(columns={
                'GEO_CODE': 'CTUID'
            })

            # Process income data if available
            if 'income' in data and not data['income'].empty:
                income = data['income'].copy()
                income = income.rename(columns={
                    'GEO_CODE': 'CTUID',
                    'MEDIAN_TOTAL_INCOME': 'MEDIAN_INCOME'
                })
            else:
                income = pd.DataFrame()

            # Process employment data if available
            if 'employment' in data and not data['employment'].empty:
                employment = data['employment'].copy()
                employment = employment.rename(columns={
                    'GEO_CODE': 'CTUID',
                    'LABOUR_FORCE': 'TOTAL_LABOUR_FORCE'
                })
            else:
                employment = pd.DataFrame()

            # Combine all datasets
            combined = population.merge(
                dwellings[['CTUID', 'TOTAL_DWELLINGS']],
                on='CTUID',
                how='left'
            )

            if not income.empty:
                combined = combined.merge(
                    income[['CTUID', 'MEDIAN_INCOME']],
                    on='CTUID',
                    how='left'
                )

            if not employment.empty:
                combined = combined.merge(
                    employment[['CTUID', 'TOTAL_LABOUR_FORCE']],
                    on='CTUID',
                    how='left'
                )

            return combined

        except Exception as e:
            logger.error(f"Error processing StatCan data: {str(e)}")
            raise

    def _process_datacommons_data(self, place_data: Dict, 
                                stats_data: Dict) -> pd.DataFrame:
        """Process Data Commons data into usable format."""
        try:
            stats = []
            
            # Process statistical data
            for series in stats_data.get('data', []):
                stat_var = series.get('variable', '')
                values = series.get('values', [])
                
                if values:
                    latest_value = values[-1]  # Get most recent value
                    stats.append({
                        'variable': stat_var,
                        'value': latest_value.get('value'),
                        'date': latest_value.get('date')
                    })

            return pd.DataFrame(stats)

        except Exception as e:
            logger.error(f"Error processing Data Commons data: {str(e)}")
            return pd.DataFrame()

    def validate_data_quality(self, 
                            data: Dict[str, Union[gpd.GeoDataFrame, pd.DataFrame]]
                            ) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive data quality validation.
        """
        try:
            integrated_data = data['integrated_data']
            
            quality_metrics = {
                'completeness': {
                    'population_data': float(
                        (1 - integrated_data['POPULATION_final'].isna().mean()) * 100
                    ),
                    'land_use_data': float(
                        (1 - integrated_data.filter(regex='^pct_.*_area$')
                         .isna().mean().mean()) * 100
                    ),
                    'accessibility_data': float(
                        (1 - integrated_data[['road_density', 'transit_score', 
                                            'park_score']].isna().mean().mean()) * 100
                    )
                },
                'consistency': {
                    'population_confidence': float(
                        integrated_data['confidence_score'].mean() * 100
                    ),
                    'land_use_total': float(
                        abs(100 - integrated_data.filter(regex='^pct_.*_area$')
                            .sum(axis=1).mean())
                    )
                },
                'statistics': {
                    'total_census_tracts': len(integrated_data),
                    'total_population': float(integrated_data['POPULATION_final'].sum()),
                    'avg_population_density': float(
                        integrated_data['POPULATION_final'].sum() / 
                        (integrated_data.geometry.area.sum() * 111 * 111)
                    )
                }
            }
            
            return quality_metrics

        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            raise

    def _get_fallback_boundary(self) -> gpd.GeoDataFrame:
        """Create a fallback rectangular boundary for KWC-CMA region."""
        bounds = {
            'north': 43.6247,
            'south': 43.3127,
            'east': -80.1013,
            'west': -80.6247
        }
        
        polygon = box(bounds['west'], bounds['south'], 
                     bounds['east'], bounds['north'])
        
        return gpd.GeoDataFrame(
            {'geometry': [polygon]}, 
            crs="EPSG:4326"
        )

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        # Clear lru_cache for methods
        self.get_cma_boundary.cache_clear()

    def __str__(self) -> str:
        """String representation of the processor."""
        return (f"PopulationDataProcessor("
                f"region=KWC-CMA, "
                f"sources={len(self.endpoints)}, "
                f"cache_size={len(self._cache)})")

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"PopulationDataProcessor("
                f"cache_dir='{self.cache_dir}', "
                f"region_codes={self.region_codes})")