"""
Model builder for EV charging station facility location optimization.
Prepares and structures data for use with Gurobi.
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

class EVFacilityLocationModel:
    """Builds and manages data for EV charging station facility location optimization."""
    
    def __init__(self):
        """Initialize the model builder."""
        self.potential_sites = None
        self.demand_points = None
        self.existing_stations = None
        self.distances = None
        self.population = None
        self.costs = None
        self.coverage_radius = 2.0  # km
        
        # Model parameters
        self.params = {
            'max_stations': None,  # Maximum number of new stations
            'budget': None,        # Total budget constraint
            'min_coverage': 0.8,   # Minimum population coverage (80%)
            'max_distance': 2.0,   # Maximum service distance (km)
            'fixed_cost': 50000,   # Fixed cost per station
            'variable_cost': 1000  # Variable cost per covered person
        }
    
    def add_potential_sites(self, sites_df: pd.DataFrame) -> None:
        """
        Add potential site locations with their attributes.
        
        Args:
            sites_df: DataFrame with columns:
                - site_id: unique identifier
                - latitude, longitude: coordinates
                - score: site suitability score
                - type: location type
                - installation_cost: estimated cost
        """
        required_cols = ['site_id', 'latitude', 'longitude', 'score', 'type', 'installation_cost']
        if not all(col in sites_df.columns for col in required_cols):
            raise ValueError(f"sites_df must contain columns: {required_cols}")
            
        self.potential_sites = sites_df.copy()
        print(f"Added {len(sites_df)} potential sites")
    
    def add_demand_points(self, demand_df: pd.DataFrame) -> None:
        """
        Add demand points with population and weights.
        
        Args:
            demand_df: DataFrame with columns:
                - point_id: unique identifier
                - latitude, longitude: coordinates
                - population: number of people
                - weight: importance weight
        """
        required_cols = ['point_id', 'latitude', 'longitude', 'population', 'weight']
        if not all(col in demand_df.columns for col in required_cols):
            raise ValueError(f"demand_df must contain columns: {required_cols}")
            
        self.demand_points = demand_df.copy()
        print(f"Added {len(demand_df)} demand points")
    
    def add_existing_stations(self, stations_df: pd.DataFrame) -> None:
        """
        Add existing charging stations.
        
        Args:
            stations_df: DataFrame with columns:
                - station_id: unique identifier
                - latitude, longitude: coordinates
                - capacity: number of charging ports
        """
        required_cols = ['station_id', 'latitude', 'longitude', 'capacity']
        if not all(col in stations_df.columns for col in required_cols):
            raise ValueError(f"stations_df must contain columns: {required_cols}")
            
        self.existing_stations = stations_df.copy()
        print(f"Added {len(stations_df)} existing stations")
    
    def calculate_distances(self) -> None:
        """Calculate distance matrices between all points."""
        if self.potential_sites is None or self.demand_points is None:
            raise ValueError("Must add potential sites and demand points first")
        
        # Calculate distances between demand points and potential sites
        site_coords = self.potential_sites[['latitude', 'longitude']].values
        demand_coords = self.demand_points[['latitude', 'longitude']].values
        
        # Use broadcasting for efficient distance calculation
        lat1, lon1 = site_coords[:, 0][:, None], site_coords[:, 1][:, None]
        lat2, lon2 = demand_coords[:, 0], demand_coords[:, 1]
        
        # Haversine formula for distances in km
        R = 6371  # Earth's radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        self.distances = R * c
        
        print(f"Calculated distances between {len(self.potential_sites)} sites and {len(self.demand_points)} demand points")
    
    def prepare_model_data(self) -> Dict:
        """
        Prepare data structures for Gurobi model.
        
        Returns:
            Dictionary containing:
                - sites: list of potential site IDs
                - demands: list of demand point IDs
                - distances: matrix of distances
                - populations: dict of populations by demand point
                - costs: dict of costs by site
                - coverage_matrix: binary matrix of coverage
        """
        if self.distances is None:
            self.calculate_distances()
        
        # Create coverage matrix
        coverage_matrix = (self.distances <= self.coverage_radius).astype(int)
        
        model_data = {
            'sites': self.potential_sites['site_id'].tolist(),
            'demands': self.demand_points['point_id'].tolist(),
            'distances': self.distances,
            'populations': dict(zip(
                self.demand_points['point_id'],
                self.demand_points['population']
            )),
            'weights': dict(zip(
                self.demand_points['point_id'],
                self.demand_points['weight']
            )),
            'costs': dict(zip(
                self.potential_sites['site_id'],
                self.potential_sites['installation_cost']
            )),
            'coverage_matrix': coverage_matrix,
            'params': self.params
        }
        
        if self.existing_stations is not None:
            model_data['existing_stations'] = self.existing_stations['station_id'].tolist()
            
            # Calculate existing coverage
            station_coords = self.existing_stations[['latitude', 'longitude']].values
            demand_coords = self.demand_points[['latitude', 'longitude']].values
            
            lat1, lon1 = station_coords[:, 0][:, None], station_coords[:, 1][:, None]
            lat2, lon2 = demand_coords[:, 0], demand_coords[:, 1]
            
            R = 6371
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            existing_distances = R * c
            
            model_data['existing_coverage'] = (existing_distances <= self.coverage_radius).astype(int)
        
        return model_data
    
    def set_parameters(self, **kwargs) -> None:
        """
        Set model parameters.
        
        Args:
            **kwargs: Parameter values to update
                - max_stations: Maximum number of new stations
                - budget: Total budget constraint
                - min_coverage: Minimum population coverage (default: 0.8)
                - max_distance: Maximum service distance in km (default: 2.0)
                - fixed_cost: Fixed cost per station (default: 50000)
                - variable_cost: Variable cost per covered person (default: 1000)
        """
        for param, value in kwargs.items():
            if param in self.params:
                self.params[param] = value
            else:
                raise ValueError(f"Unknown parameter: {param}")
        
        print("Updated parameters:")
        for param, value in self.params.items():
            print(f"- {param}: {value}")
    
    def validate_data(self) -> bool:
        """
        Validate that all necessary data is present and consistent.
        
        Returns:
            bool: True if all data is valid
        """
        checks = []
        
        # Check required data is present
        checks.append(self.potential_sites is not None)
        checks.append(self.demand_points is not None)
        checks.append(self.distances is not None)
        
        # Check parameter validity
        checks.append(self.params['max_stations'] is not None)
        checks.append(self.params['budget'] is not None)
        checks.append(self.params['min_coverage'] > 0 and self.params['min_coverage'] <= 1)
        checks.append(self.params['max_distance'] > 0)
        
        # Check dimensional consistency
        if self.distances is not None:
            checks.append(self.distances.shape == (
                len(self.potential_sites),
                len(self.demand_points)
            ))
        
        valid = all(checks)
        if not valid:
            print("Data validation failed. Please ensure all required data is present and consistent.")
        
        return valid
    
    def __str__(self) -> str:
        """String representation of model state."""
        components = []
        if self.potential_sites is not None:
            components.append(f"Potential Sites: {len(self.potential_sites)}")
        if self.demand_points is not None:
            components.append(f"Demand Points: {len(self.demand_points)}")
        if self.existing_stations is not None:
            components.append(f"Existing Stations: {len(self.existing_stations)}")
        
        return "EVFacilityLocationModel(\n  " + "\n  ".join(components) + "\n)"