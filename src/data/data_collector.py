from typing import Tuple, Dict, Optional
import os
import requests
import rasterio
import pandas as pd
import geopandas as gpd
from pathlib import Path
import osmnx as ox
from shapely.geometry import box, Point
import numpy as np
from dotenv import load_dotenv

class DataCollector:
    """Handles data collection for EV charging station optimization project."""
    
    def __init__(self, data_dir: Path):
        """Initialize the data collector."""
        load_dotenv()
        self.data_dir = data_dir
        self.census_api_key = os.getenv("CENSUS_API_KEY")
        self.grid_api_key = os.getenv("GRID_API_KEY")
        
    def get_population_density(self) -> gpd.GeoDataFrame:
        """
        Fetch population density data using Canadian census API.
        Returns data as a GeoDataFrame with population density per dissemination area.
        """
        # Census tract level population data for KW region
        base_url = "https://www12.statcan.gc.ca/rest/census-recensement/CR2021Geo"
        params = {
            'lang': 'E',
            'geos': 'DA',  # Dissemination Area
            'cpt': '35',   # Ontario
            'csd': ['3530013', '3530016'],  # Kitchener and Waterloo CSD codes
            'dataset': 'Population',
            'format': 'json'
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Process into GeoDataFrame
            gdf = gpd.GeoDataFrame(
                data['features'],
                geometry=[Point(x['properties']['LONGITUDE'], x['properties']['LATITUDE']) 
                         for x in data['features']]
            )
            
            # Calculate density
            gdf['density'] = gdf['Population'] / gdf['Area']
            
            # Save to file
            output_file = self.data_dir / 'population_density' / f'population_density_{pd.Timestamp.now().strftime("%Y%m%d")}.gpkg'
            gdf.to_file(output_file, driver='GPKG')
            
            return gdf
            
        except Exception as e:
            print(f"Error fetching population data: {e}")
            return self._generate_synthetic_population_data()
    
    def get_grid_capacity(self) -> pd.DataFrame:
        """
        Fetch grid capacity data from utility provider APIs.
        Falls back to synthetic data if API access isn't available.
        """
        try:
            # Try to fetch from utilities APIs
            # Note: Replace with actual API endpoints when available
            # For now, generate synthetic data
            return self._generate_synthetic_grid_data()
            
        except Exception as e:
            print(f"Error fetching grid data: {e}")
            return self._generate_synthetic_grid_data()
    
    def get_traffic_data(self) -> pd.DataFrame:
        """
        Fetch traffic flow data using OpenStreetMap and region traffic APIs.
        """
        # Get road network
        G = ox.graph_from_place('Kitchener-Waterloo, Ontario, Canada', network_type='drive')
        
        # Convert to GeoDataFrame
        gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        
        # Add synthetic traffic data
        gdf_edges['traffic_flow'] = np.random.normal(1000, 300, len(gdf_edges))
        gdf_edges['peak_hours_multiplier'] = np.random.uniform(1.5, 2.5, len(gdf_edges))
        
        # Save to file
        output_file = self.data_dir / 'traffic' / f'traffic_data_{pd.Timestamp.now().strftime("%Y%m%d")}.gpkg'
        gdf_edges.to_file(output_file, driver='GPKG')
        
        return gdf_edges
    
    def _generate_synthetic_population_data(self) -> gpd.GeoDataFrame:
        """Generate synthetic population density data for testing."""
        # Create grid of points covering KW region
        bounds = box(*ox.geocode_to_gdf('Kitchener-Waterloo, Ontario, Canada').total_bounds)
        grid_points = []
        
        # Generate 100x100 grid
        for x in np.linspace(bounds.bounds[0], bounds.bounds[2], 100):
            for y in np.linspace(bounds.bounds[1], bounds.bounds[3], 100):
                grid_points.append(Point(x, y))
        
        # Create GeoDataFrame with synthetic population density
        gdf = gpd.GeoDataFrame(geometry=grid_points)
        gdf['density'] = np.random.normal(5000, 1500, len(gdf))  # people per km²
        gdf['density'] = gdf['density'].clip(lower=0)
        
        return gdf
    
    def _generate_synthetic_grid_data(self) -> pd.DataFrame:
        """Generate synthetic grid capacity data for testing."""
        # Load existing and potential locations
        locations_df = pd.read_csv(
            self.data_dir / 'processed' / 'all_locations_processed.csv'
        )
        
        # Generate synthetic grid capacity data
        grid_data = pd.DataFrame({
            'latitude': locations_df['latitude'],
            'longitude': locations_df['longitude'],
            'available_capacity_kw': np.random.normal(500, 150, len(locations_df)),
            'peak_load_kw': np.random.normal(300, 100, len(locations_df)),
            'upgrade_cost_per_kw': np.random.uniform(100, 300, len(locations_df))
        })
        
        # Clip to reasonable values
        grid_data['available_capacity_kw'] = grid_data['available_capacity_kw'].clip(lower=0)
        grid_data['peak_load_kw'] = grid_data['peak_load_kw'].clip(lower=0)
        grid_data['upgrade_cost_per_kw'] = grid_data['upgrade_cost_per_kw'].clip(lower=50)
        
        return grid_data

    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect all required data sets."""
        return {
            'population': self.get_population_density(),
            'grid': self.get_grid_capacity(),
            'traffic': self.get_traffic_data()
        }