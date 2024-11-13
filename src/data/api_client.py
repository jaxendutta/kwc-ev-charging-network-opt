"""Unified client for fetching data from various APIs for EV charging station optimization."""
import os
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from dotenv import load_dotenv
from datetime import datetime
import osmnx as ox
import pyproj
from shapely.geometry import Point
from tqdm import tqdm
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
        
        # Initialize coordinate transformer
        self.transformer = pyproj.Transformer.from_crs(
            'EPSG:32617',  # UTM Zone 17N
            'EPSG:4326',   # WGS84
            always_xy=True
        )

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

    def _convert_to_wgs84(self, geometry, from_crs='EPSG:32617') -> tuple:
        """
        Safely convert coordinates from UTM to WGS84.
        Returns (latitude, longitude) tuple.
        """
        try:
            # Get coordinates based on geometry type
            if geometry.geom_type == 'Point':
                x, y = geometry.x, geometry.y
            else:
                centroid = geometry.centroid
                x, y = centroid.x, centroid.y
            
            # Transform coordinates using pre-initialized transformer
            lon, lat = self.transformer.transform(x, y)
            
            return lat, lon
        except Exception as e:
            print(f"Error in coordinate conversion: {e}")
            return None, None

    def _process_geometry(self, geometry, mean_area=None, pop_per_unit=None, building_type=None) -> dict:
        """Process a geometry and return location data."""
        try:
            # Convert coordinates
            lat, lon = self._convert_to_wgs84(geometry)
            if lat is None or lon is None:
                return None
            
            # Calculate area in km²
            area_size = float(geometry.area) / 1_000_000
            
            # Calculate population if residential
            if pop_per_unit is not None and mean_area is not None:
                area_factor = area_size / (float(mean_area) / 1_000_000)
                multiplier = {
                    'apartments': 2.5,
                    'house': 0.8
                }.get(building_type, 1.0)
                est_pop = float(pop_per_unit * multiplier * min(max(area_factor, 0.5), 2.0))
            else:
                est_pop = None
                
            return {
                'latitude': lat,
                'longitude': lon,
                'area_sq_km': area_size,
                'estimated_population': est_pop
            }
        except Exception as e:
            print(f"Error processing geometry: {e}")
            return None

    def fetch_population_density(self) -> pd.DataFrame:
        """
        Fetch population density data using OSM API with verified estimates.
        Population estimates based on Statistics Canada 2021 Census.
        """
        print("Fetching population density data from OpenStreetMap...")
        
        try:
            import osmnx as ox
            import numpy as np
            import pyproj
            from tqdm import tqdm
            from datetime import datetime
            
            # Configure OSMnx
            ox.settings.use_cache = True
            ox.settings.requests_timeout = 180
            
            # Known statistics from 2021 Census
            KW_STATS = {
                'total_population': 715219,
                'households': 275940,
                'avg_household_size': 2.6
            }
            
            print("\n📊 Data Collection Summary")
            print("=" * 50)
            print(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
            
            print("\n📍 Data Sources and Versions:")
            print("• OpenStreetMap (OSM):")
            print(f"  - OSMnx Version: {ox.__version__}")
            print(f"  - PyProj Version: {pyproj.__version__}")
            
            cities = ["Kitchener, Ontario, Canada", "Waterloo, Ontario, Canada"]
            population_data = []
            total_buildings = 0
            total_amenities = 0
            
            print("\n🏘️ Processing cities...")
            for city in cities:
                city_name = city.split(',')[0]
                print(f"\n📍 Processing {city_name}...")
                
                try:
                    # Get city polygon
                    city_gdf = ox.geocode_to_gdf(city)
                    
                    # Get buildings within city
                    print(f"   Fetching residential buildings...")
                    buildings = ox.features.features_from_polygon(
                        city_gdf.geometry.iloc[0],
                        tags={'building': ['apartments', 'residential', 'house']}
                    )
                    
                    if not buildings.empty:
                        buildings = buildings.to_crs('EPSG:32617')
                        buildings_count = len(buildings)
                        total_buildings += buildings_count
                        print(f"   ✓ Found {buildings_count:,} buildings")
                        
                        city_pop = (KW_STATS['total_population'] * 
                                (0.6 if 'Kitchener' in city else 0.4))
                        pop_per_building = city_pop / buildings_count
                        mean_area = buildings.geometry.area.mean()
                        
                        print(f"   Processing building data...")
                        for idx, building in tqdm(buildings.iterrows(), 
                                            total=buildings_count,
                                            bar_format='{l_bar}{bar:30}{r_bar}'):
                            try:
                                # Convert coordinates
                                lat, lon = self._convert_to_wgs84(building.geometry)
                                if lat is None or lon is None:
                                    continue
                                    
                                building_type = building.get('building', 'house')
                                area_size = float(building.geometry.area) / 1_000_000
                                
                                est_pop = pop_per_building * {
                                    'apartments': 2.5,
                                    'house': 0.8
                                }.get(building_type, 1.0)
                                
                                area_factor = area_size / (float(mean_area) / 1_000_000)
                                est_pop = float(est_pop * min(max(area_factor, 0.5), 2.0))
                                
                                population_data.append({
                                    'latitude': lat,
                                    'longitude': lon,
                                    'population': est_pop,
                                    'area_sq_km': area_size,
                                    'population_density': est_pop / area_size if area_size > 0 else 0,
                                    'location_type': 'residential',
                                    'building_type': building_type,
                                    'city': city_name
                                })
                            except Exception as e:
                                continue
                    
                    # Get amenities
                    print(f"\n   Fetching amenities...")
                    amenities = ox.features.features_from_polygon(
                        city_gdf.geometry.iloc[0],
                        tags={'amenity': ['school', 'university', 'college', 'shopping_centre']}
                    )
                    
                    if not amenities.empty:
                        amenities = amenities.to_crs('EPSG:32617')
                        amenities_count = len(amenities)
                        total_amenities += amenities_count
                        print(f"   ✓ Found {amenities_count:,} amenities")
                        
                        pop_estimates = {
                            'University of Waterloo': 40000,
                            'Wilfrid Laurier': 20000,
                            'Conestoga': 25000,
                            'school': 1000,
                            'shopping_centre': 3000
                        }
                        
                        print(f"   Processing amenity data...")
                        for idx, amenity in tqdm(amenities.iterrows(),
                                            total=amenities_count,
                                            bar_format='{l_bar}{bar:30}{r_bar}'):
                            try:
                                # Convert coordinates
                                lat, lon = self._convert_to_wgs84(amenity.geometry)
                                if lat is None or lon is None:
                                    continue
                                
                                area_size = float(amenity.geometry.area) / 1_000_000
                                name = str(amenity.get('name', '')).lower()
                                amenity_type = amenity.get('amenity', 'other')
                                
                                est_pop = next(
                                    (pop for key, pop in pop_estimates.items() if key.lower() in name),
                                    pop_estimates.get(amenity_type, 500)
                                )
                                
                                population_data.append({
                                    'latitude': lat,
                                    'longitude': lon,
                                    'population': float(est_pop),
                                    'area_sq_km': area_size,
                                    'population_density': float(est_pop / area_size if area_size > 0 else 0),
                                    'location_type': 'amenity',
                                    'amenity_type': amenity_type,
                                    'name': amenity.get('name', 'Unknown'),
                                    'city': city_name
                                })
                            except Exception as e:
                                continue
                                
                except Exception as e:
                    print(f"❌ Error processing {city_name}: {e}")
                    continue
            
            print(f"\nFeatures Found:")
            print(f"  - Total Features: {total_buildings + total_amenities:,}")
            print(f"    ∟ {total_buildings:,} residential buildings")
            print(f"    ∟ {total_amenities:,} public amenities")
            
            if not population_data:
                print("No data retrieved from OSM")
                return self._get_sample_population_data_with_types()
            
            # Create DataFrame
            df = pd.DataFrame(population_data)
            
            print("\n📊 Regional Summary by City:")
            print("=" * 50)
            
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                res_data = city_data[city_data['location_type'] == 'residential']
                amen_data = city_data[city_data['location_type'] == 'amenity']
                
                print(f"\n{city}:")
                print(f"  Residential:")
                print(f"    • Buildings: {len(res_data):,}")
                print(f"    • Est. Population: {res_data['population'].sum():,.0f}")
                print(f"    • Total Area: {res_data['area_sq_km'].sum():,.2f} km²")
                
                print(f"  Amenities:")
                print(f"    • Locations: {len(amen_data):,}")
                print(f"    • Daily Population: {amen_data['population'].sum():,.0f}")
                print(f"    • Total Area: {amen_data['area_sq_km'].sum():,.2f} km²")
            
            return df
            
        except Exception as e:
            print(f"\n❌ Error fetching OSM population data: {e}")
            return self._get_sample_population_data_with_types()

    def _get_sample_population_data_with_types(self) -> pd.DataFrame:
        """
        Generate sample population density data with location types.
        """
        print("Generating sample population density data...")
        
        # Create grid of points
        lats = np.linspace(KW_BOUNDS['south'], KW_BOUNDS['north'], 50)
        lons = np.linspace(KW_BOUNDS['west'], KW_BOUNDS['east'], 50)
        
        data = []
        location_types = ['residential', 'building', 'amenity']
        weights = [0.6, 0.3, 0.1]  # Probability of each type
        
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
                    'population': density * 0.01,  # Scale to reasonable population
                    'area_sq_km': 0.01,
                    'population_density': density,
                    'location_type': np.random.choice(location_types, p=weights)
                })
        
        return pd.DataFrame(data)

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