"""
Network optimization model for EV charging infrastructure enhancement.
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Union

class EVNetworkOptimizer:
    """Network enhancement optimization model."""
    
    def __init__(self, input_data: Dict[str, Any]):
        """
        Initialize optimizer with input data.
        
        Args:
            input_data: Dictionary containing:
                - demand_points: Population centers with demands
                - potential_sites: Candidate locations for upgrades
                - distance_matrix: Distances between points
                - existing_coverage: Current coverage matrix
                - constraints: Optimization constraints
        """
        self.model = gp.Model("ev_network_enhancement")
        self.logger = logging.getLogger("model.network_optimizer")
        
        # Store input data
        self.demand_points = input_data['demand_points']
        self.potential_sites = input_data['potential_sites']
        self.distances = input_data['distance_matrix']
        self.existing_coverage = input_data['existing_coverage']
        self.constraints = input_data.get('constraints', {
            'budget': 2000000,
            'min_coverage': 0.8,
            'max_l3_distance': 5.0,
            'max_stations_per_area': 3
        })
        
        # Initialize variable storage
        self.variables = {}
        
    def _create_variables(self):
        """Create optimization variables."""
        # Upgrade decisions (binary) for each potential site
        self.variables['upgrade'] = self.model.addVars(
            len(self.potential_sites),
            vtype=GRB.BINARY,
            name="upgrade"
        )
        
        # Additional ports at each site (integer)
        self.variables['new_ports'] = self.model.addVars(
            len(self.potential_sites),
            vtype=GRB.INTEGER,
            lb=0,
            ub=4,  # Maximum 4 new ports per site
            name="new_ports"
        )
        
        # Coverage variables for demand points
        self.variables['coverage'] = self.model.addVars(
            len(self.demand_points),
            vtype=GRB.BINARY,
            name="coverage"
        )
    
    def _add_constraints(self):
        """Add model constraints."""
        try:
            # Print dimensions for debugging
            self.logger.info(f"Dimensions:")
            self.logger.info(f"- Demand points: {len(self.demand_points)}")
            self.logger.info(f"- Potential sites: {len(self.potential_sites)}")
            self.logger.info(f"- Distance matrix: {self.distances.shape}")
            self.logger.info(f"- Existing coverage: {self.existing_coverage.shape}")

            # Budget constraint
            upgrade_costs = gp.quicksum(
                self.variables['upgrade'][i] * 50000  # $50k per upgrade
                for i in range(len(self.potential_sites))
            )
            
            new_port_costs = gp.quicksum(
                self.variables['new_ports'][i] * 5000  # $5k per new port
                for i in range(len(self.potential_sites))
            )
            
            self.model.addConstr(
                upgrade_costs + new_port_costs <= self.constraints['budget'],
                name="budget"
            )
            
            # Coverage constraints
            for i in range(len(self.demand_points)):
                # Get valid indices for current coverage radius
                nearby_sites = [
                    j for j in range(len(self.potential_sites))
                    if self.distances[j, i] <= 5.0  # 5km coverage radius
                ]
                
                existing_coverage_sum = sum(
                    self.existing_coverage[i, j]
                    for j in range(self.existing_coverage.shape[1])
                )
                
                self.model.addConstr(
                    self.variables['coverage'][i] <= 
                    existing_coverage_sum +
                    gp.quicksum(self.variables['upgrade'][j] for j in nearby_sites),
                    name=f"coverage_{i}"
                )
            
            # Minimum coverage requirement
            self.model.addConstr(
                gp.quicksum(
                    self.variables['coverage'][i] * self.demand_points['weight'].iloc[i]
                    for i in range(len(self.demand_points))
                ) >= self.constraints['min_coverage'],
                name="min_coverage"
            )
            
            # Maximum stations per area constraint
            if 'max_stations_per_area' in self.constraints:
                for area in range(len(self.demand_points)):
                    nearby_sites = [
                        j for j in range(len(self.potential_sites))
                        if self.distances[j, area] <= 2.0  # 2km radius for area constraints
                    ]
                    
                    self.model.addConstr(
                        gp.quicksum(self.variables['upgrade'][j] for j in nearby_sites) <= 
                        self.constraints['max_stations_per_area'],
                        name=f"max_stations_area_{area}"
                    )
            
        except Exception as e:
            self.logger.error(f"Error in constraints: {str(e)}")
            raise
        
    def _set_objective(self):
        """Set the optimization objective."""
        # Coverage benefit
        coverage_benefit = gp.quicksum(
            self.variables['coverage'][i] * self.demand_points['weight'].iloc[i]
            for i in range(len(self.demand_points))
        )
        
        # Cost component
        cost_penalty = (
            gp.quicksum(self.variables['upgrade']) * 0.1 +
            gp.quicksum(self.variables['new_ports']) * 0.05
        ) / self.constraints['budget']
        
        self.model.setObjective(
            coverage_benefit - cost_penalty,
            GRB.MAXIMIZE
        )
    
    def optimize(self, progress_callback=None):
        """Run optimization."""
        try:
            # Setup
            self._create_variables()
            if progress_callback:
                progress_callback(20)
            
            self._add_constraints()
            if progress_callback:
                progress_callback(40)
            
            self._set_objective()
            if progress_callback:
                progress_callback(60)
            
            # Optimize
            self.model.optimize()
            if progress_callback:
                progress_callback(80)
            
            # Process results
            if self.model.Status == GRB.OPTIMAL:
                solution = self._process_solution()
                if progress_callback:
                    progress_callback(20)
                return solution
            else:
                return {
                    'status': 'not_optimal',
                    'message': f"Status: {self.model.Status}"
                }
                
        except Exception as e:
            self.logger.error(f"General error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _process_solution(self):
        """Process optimization results."""
        solution = {
            'status': 'optimal',
            'objective_value': self.model.ObjVal,
            'upgrades': [
                i for i in range(len(self.potential_sites))
                if self.variables['upgrade'][i].X > 0.5
            ],
            'new_ports': {
                i: int(self.variables['new_ports'][i].X)
                for i in range(len(self.potential_sites))
                if self.variables['new_ports'][i].X > 0
            },
            'coverage': {
                'population': sum(
                    self.variables['coverage'][i].X * self.demand_points['weight'].iloc[i]
                    for i in range(len(self.demand_points))
                ),
                'l3': len(self.variables['upgrade']) / len(self.potential_sites)
            },
            'costs': {
                'upgrade_costs': sum(
                    self.variables['upgrade'][i].X * 50000
                    for i in range(len(self.potential_sites))
                ),
                'port_costs': sum(
                    self.variables['new_ports'][i].X * 5000
                    for i in range(len(self.potential_sites))
                )
            }
        }
        
        solution['costs']['total_cost'] = (
            solution['costs']['upgrade_costs'] +
            solution['costs']['port_costs']
        )
        
        return solution