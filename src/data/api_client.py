"""Unified client for fetching data from various APIs for EV charging station optimization."""
import os
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from dotenv import load_dotenv
import osmnx as ox
from shapely.geometry import Point
from .constants import KW_BOUNDS, API_TIMEOUT, MAX_RESULTS
from .utils import grab_time, get_file_timestamp

class APIClient:
    """Unified client for data collection from various APIs."""
    
    def __init__(self):
        load_dotenv()
        # OpenChargeMap API key
        self.ocmap_api_key = os.getenv("OCMAP_API_KEY")
        if not self.ocmap_api_key:
            raise ValueError("OpenChargeMap API key not found in environment variables")
            
        # Base URLs
        self.ocmap_base_url = "https://api.openchargemap.io/v3"
        self.census_base_url = "https://www12.statcan.gc.ca/rest/census-recensement/2021/dp-pd/prof/details"

    def fetch_charging_stations(self) -> pd.DataFrame:
        """Fetch charging stations in KW region from OpenChargeMap."""
        print("Fetching charging station data...")
        
        params = {
            'key': self.ocmap_api_key,
            'maxresults': MAX_RESULTS,
            'latitude': (KW_BOUNDS['north'] + KW_BOUNDS['south']) / 2,
            'longitude': (KW_BOUNDS['east'] + KW_BOUNDS['west']) / 2,
            'distance': 10,
            'distanceunit': 'KM',
            'countrycode': 'CA'
        }
        
        response = requests.get(
            f"{self.ocmap_base_url}/poi", 
            params=params, 
            timeout=API_TIMEOUT
        )
        data = response.json()
        
        stations = []
        for poi in data:
            address_info = poi.get('AddressInfo', {})
            connections = poi.get('Connections', [])
            
            lat = address_info.get('Latitude')
            lon = address_info.get('Longitude')
            
            # Only include stations in KW region
            if (KW_BOUNDS['south'] <= lat <= KW_BOUNDS['north'] and
                KW_BOUNDS['west'] <= lon <= KW_BOUNDS['east']):
                
                operator = poi.get('OperatorInfo')
                stations.append({
                    'name': address_info.get('Title'),
                    'latitude': lat,
                    'longitude': lon,
                    'num_chargers': len(connections),
                    'charger_type': self._determine_charger_type(connections),
                    'operator': operator.get('Title', 'Unknown') if operator else 'Unknown',
                    'address': address_info.get('AddressLine1'),
                    'city': address_info.get('Town'),
                    'postal_code': address_info.get('Postcode'),
                    'usage_cost': self._extract_usage_cost(poi.get('UsageCost')),
                    'data_source': 'OpenChargeMap',
                    'location_type': 'charging_station'
                })
        
        df = pd.DataFrame(stations)
        print(f"Found {len(df)} stations in KW region")
        return df

    def fetch_potential_locations(self) -> pd.DataFrame:
        """Fetch potential locations from OpenStreetMap."""
        print("Fetching potential locations data...")
        
        # Define OSM tags for potential locations
        tags = {
            'amenity': ['parking', 'fuel'],
            'building': ['retail', 'commercial']
        }
        
        # Download POIs using OSMnx
        ox.settings.use_cache = False
        pois = ox.features_from_place(
            "Kitchener, Ontario, Canada", 
            tags=tags
        )
        
        # Convert to GeoDataFrame and process
        potential_locations_gdf = gpd.GeoDataFrame(pois)
        
        # Extract coordinates based on geometry type
        def get_coords(geom):
            """Extract representative coordinates from different geometry types."""
            if geom.geom_type == 'Point':
                return geom.y, geom.x
            elif geom.geom_type == 'Polygon':
                # Use centroid for polygons
                return geom.centroid.y, geom.centroid.x
            elif geom.geom_type == 'LineString':
                # Use midpoint for lines
                return geom.centroid.y, geom.centroid.x
            elif geom.geom_type == 'MultiPolygon':
                # Use centroid of largest polygon
                largest = max(geom.geoms, key=lambda x: x.area)
                return largest.centroid.y, largest.centroid.x
            else:
                # Default to centroid for any other type
                return geom.centroid.y, geom.centroid.x
        
        # Apply coordinate extraction
        coords = potential_locations_gdf.geometry.apply(get_coords)
        potential_locations_gdf['latitude'] = coords.apply(lambda x: x[0])
        potential_locations_gdf['longitude'] = coords.apply(lambda x: x[1])
        
        # Create unified schema matching charging stations
        processed_locations = []
        for _, row in potential_locations_gdf.iterrows():
            location_type = row.get('amenity') if pd.notnull(row.get('amenity')) else row.get('building')
            location_type = location_type if pd.notnull(location_type) else 'commercial'
            
            # Get area if available (for polygons)
            area = row.geometry.area if hasattr(row.geometry, 'area') else None
            
            processed_locations.append({
                'name': row.get('name', f"{location_type.title()} Location"),
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'address': row.get('addr:street', 'Unknown'),
                'city': row.get('addr:city', 'Unknown'),
                'postal_code': row.get('addr:postcode', 'Unknown'),
                'data_source': 'OpenStreetMap',
                'location_type': location_type,
                'area': area,
                'geometry_type': row.geometry.geom_type
            })
        
        df = pd.DataFrame(processed_locations)
        print(f"Found {len(df)} potential locations in KW region")
        print("\nGeometry types found:")
        print(df['geometry_type'].value_counts())
        return df

    def fetch_population_density(self) -> pd.DataFrame:
        """Fetch population density data from Statistics Canada Census API."""
        print("Fetching population density data...")
        
        # Dissemination area codes for KW region
        # Kitchener CSD: 3530013
        # Waterloo CSD: 3530016
        params = {
            'dguid': ['2021S0503530013', '2021S0503530016'],
            'topic': 1,  # Population topics
            'format': 'json'
        }
        
        try:
            response = requests.get(
                self.census_base_url,
                params=params,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Process census data
                population_data = []
                for area in data['data']:
                    # Extract geographic coordinates for the dissemination area
                    lat = float(area['geometry']['coordinates'][1])
                    lon = float(area['geometry']['coordinates'][0])
                    
                    # Only include points within KW bounds
                    if (KW_BOUNDS['south'] <= lat <= KW_BOUNDS['north'] and
                        KW_BOUNDS['west'] <= lon <= KW_BOUNDS['east']):
                        
                        population_data.append({
                            'latitude': lat,
                            'longitude': lon,
                            'population': int(area['population']),
                            'area_sq_km': float(area['area']),
                            'population_density': int(area['population']) / float(area['area'])
                        })
                
                return pd.DataFrame(population_data)
            else:
                print(f"Census API error: {response.status_code}")
                return self._get_sample_population_data()
                
        except Exception as e:
            print(f"Error fetching census data: {e}")
            return self._get_sample_population_data()
    
    def _determine_charger_type(self, connections: Optional[List[Dict]]) -> str:
        """Determine the highest level charger available."""
        if not connections:
            return 'Unknown'
        
        for conn in connections:
            level = conn.get('Level')
            if level and level.get('IsFastChargeCapable'):
                return 'Level 3'
        
        for conn in connections:
            level = conn.get('Level')
            if level and 'level 2' in level.get('Title', '').lower():
                return 'Level 2'
        
        return 'Level 1'
    
    def _extract_usage_cost(self, cost_info: Optional[str]) -> str:
        """Extract and clean usage cost information."""
        if not cost_info:
            return 'Unknown'
        
        cost = cost_info.replace('Payment required', '')
        cost = cost.replace('Payment Required', '')
        cost = cost.strip()
        
        return cost if cost else 'Unknown'
    
    def _get_sample_population_data(self) -> pd.DataFrame:
        """Generate sample population density data for testing."""
        print("Warning: Using sample population density data...")
        
        # Create grid of points covering KW region
        lats = np.linspace(KW_BOUNDS['south'], KW_BOUNDS['north'], 50)
        lons = np.linspace(KW_BOUNDS['west'], KW_BOUNDS['east'], 50)
        
        data = []
        for lat in lats:
            for lon in lons:
                # Higher density near city centers
                center_dist = np.sqrt(
                    (lat - 43.4516)**2 + (lon - (-80.4925))**2
                )
                density = max(0, 5000 * (1 - center_dist/0.1) + np.random.normal(0, 500))
                
                data.append({
                    'latitude': lat,
                    'longitude': lon,
                    'population_density': density
                })
        
        return pd.DataFrame(data)