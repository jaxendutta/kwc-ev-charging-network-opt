"""
Network optimization model for EV charging infrastructure enhancement.
Handles coverage calculations for both L2 and L3 chargers, station upgrades, 
and infrastructure costs with resale values.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime
from haversine import haversine

from src.data.utils import *
from src.model.utils import *

class EVNetworkOptimizer:
    """
    Network enhancement optimization model with port retention strategy.
    
    Implements a mixed-integer linear programming model for optimizing EV charging 
    infrastructure while maintaining service accessibility through strategic L2 port
    retention during upgrades.

    Key Features:
        - Multi-objective optimization (coverage and cost)
        - L2 port retention strategy
        - Grid capacity constraints
        - Budget optimization with resale value
        - Phased implementation planning
    """
    
    def __init__(self, input_data: Dict[str, Any], config: Dict[str, Any]):
        """Initialize optimizer with input data and configuration."""
        self.logger = logging.getLogger("model.network_optimizer")
        self.logger.setLevel(logging.INFO)
        self.model = gp.Model("ev_network_enhancement")

        # Store input data with validation
        if not all(k in input_data for k in ['demand_points', 'potential_sites', 'distance_matrix', 'existing_stations']):
            raise ValueError("Missing required input data")
            
        self.demand_points = input_data['demand_points'].copy()
        self.potential_sites = input_data['potential_sites'].copy()
        self.distances = input_data['distance_matrix'].copy()
        self.stations_df = input_data['existing_stations'].copy()
        
        # Process existing stations
        self.existing_l2 = self.stations_df[self.stations_df['charger_type'] == 'Level 2'].copy()
        self.existing_l3 = self.stations_df[self.stations_df['charger_type'] == 'Level 3'].copy()
        
        # Store key statistics
        self.n_demand_points = len(self.demand_points)
        self.n_potential_sites = len(self.potential_sites)
        self.n_existing_stations = len(self.stations_df)
        self.n_existing_l2 = len(self.existing_l2)
        self.n_existing_l3 = len(self.existing_l3)
        
        # Fill any null values with defaults
        self.demand_points = self.demand_points.fillna(0)
        self.potential_sites = self.potential_sites.fillna(0)
        self.stations_df = self.stations_df.fillna(0)

        # Load configuration parameters
        self.costs = config['costs']
        self.coverage = config['coverage']
        self.infrastructure = config['infrastructure']
        self.budget = config['budget']['default']
        self.weights = config['weights']

        # Calculate initial coverage
        self.initial_coverage = self._calculate_initial_coverage()
        
        # Initialize variables storage
        self.variables = {}
    
    # ----------------
    # Main Public Methods
    # ----------------

    def optimize(self):
        """Run optimization with full solution processing."""
        try:
            print("\n1. Setting up optimization model...")
            
            self.model.setParam('OutputFlag', 0)  # Suppress Gurobi output
            self.model.Params.DualReductions = 0
            self.model.Params.PreDual = 0
            self.model.Params.Method = 2
            self.model.Params.BarHomogeneous = 1
            print("✓ Model parameters configured")
            
            self._create_variables()
            print("✓ Decision variables created")
            
            self._add_constraints()
            print("✓ Constraints added")

            self._set_objective()
            print("✓ Objective function set")
            
            print("\n2. Starting optimization...")
            self.model.optimize()
            
            if self.model.Status == GRB.OPTIMAL:
                print("\n3. Processing optimal solution...")
                
                # Get decision variable values
                upgrades = [i for i in range(len(self.existing_l2)) 
                          if i < len(self.variables['upgrade']) 
                          and self.variables['upgrade'][i].X > 0.5]
                print(f"✓ Found {len(upgrades)} L2 -> L3 upgrades!")
                
                new_l2_stations = [i for i in range(len(self.potential_sites))
                                 if i < len(self.variables['new_l2_station'])
                                 and self.variables['new_l2_station'][i].X > 0.5]
                print(f"✓ Found {len(new_l2_stations)} new L2 stations!")

                new_l3_stations = [i for i in range(len(self.potential_sites))
                                 if i < len(self.variables['new_l3_station'])
                                 and self.variables['new_l3_station'][i].X > 0.5]
                print(f"✓ Found {len(new_l3_stations)} new L3 stations!")
                
                # Create base solution structure
                base_solution = {
                    'upgrades': upgrades,
                    'new_stations': {
                        'l2': new_l2_stations,
                        'l3': new_l3_stations
                    }
                }
                
                # Calculate costs and coverage
                costs = self._calculate_solution_costs(base_solution)
                coverage = self._calculate_coverage_metrics()
                
                print("\n4. Creating detailed solution...")
                
                # Update stations with detailed information
                detailed_stations = self.update_station_statuses(base_solution)
                print("✓ Station statuses updated")

                solution = {
                    'status': 'optimal',
                    'objective_value': float(self.model.ObjVal),
                    'upgrades': upgrades,
                    'new_stations': base_solution['new_stations'],
                    'stations': detailed_stations,
                    'costs': costs,
                    'coverage': coverage,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_stations': len(self.stations_df),
                        'initial_l2_stations': self.n_existing_l2,
                        'initial_l3_stations': self.n_existing_l3,
                        'upgrade_count': len(upgrades),
                        'new_l2_count': len(new_l2_stations),
                        'new_l3_count': len(new_l3_stations)
                    }
                }
                
                print("✓ Solution creation complete!")
                return solution
                
        except Exception as e:
            print(f"\nOptimization error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    # ----------------
    # Model Building Methods
    # ----------------
    
    def _create_variables(self):
        """Create optimization variables."""
        # Station type decisions
        self.variables = {
            # New station variables for all potential sites
            'new_l2_station': self.model.addVars(
                self.n_potential_sites, vtype=GRB.BINARY, name="new_l2_station"
            ),
            'new_l3_station': self.model.addVars(
                self.n_potential_sites, vtype=GRB.BINARY, name="new_l3_station"
            ),
            # Upgrade variables ONLY for existing L2 stations
            'upgrade': self.model.addVars(
                self.n_existing_l2, vtype=GRB.BINARY, name="upgrade"
            ),
            # Coverage variables
            'coverage_l2': self.model.addVars(
                self.n_demand_points, vtype=GRB.BINARY, name="coverage_l2"
            ),
            'coverage_l3': self.model.addVars(
                self.n_demand_points, vtype=GRB.BINARY, name="coverage_l3"
            )
        }

    def _add_constraints(self):
        """Add all model constraints."""
        self._add_station_type_constraints()
        self._add_budget_constraints()
        self._add_coverage_constraints()
        self._add_grid_constraints()

    def _set_objective(self):
        """Set the optimization objective."""
        # Coverage benefit components
        l3_coverage = gp.quicksum(
            self.variables['coverage_l3'][i] * 
            self.demand_points['weight'].iloc[i]
            for i in range(len(self.demand_points))
        )
        
        l2_coverage = gp.quicksum(
            self.variables['coverage_l2'][i] * 
            self.demand_points['weight'].iloc[i]
            for i in range(len(self.demand_points))
        )
        
        # Cost component from budget constraints
        total_cost = self._calculate_total_cost_expr()
        
        # Multi-objective function
        self.model.setObjective(
            self.weights['l3_coverage'] * l3_coverage +
            self.weights['l2_coverage'] * l2_coverage -
            self.weights['cost'] * total_cost,
            GRB.MAXIMIZE
        )
    
    # ----------------
    # Constraint Methods
    # ----------------

    def _add_station_type_constraints(self):
        """Add logical constraints for infrastructure decisions."""
        for i in range(self.n_potential_sites):
            # Can't have both L2 and L3 stations at same location
            self.model.addConstr(
                self.variables['new_l2_station'][i] + 
                self.variables['new_l3_station'][i] <= 1,
                name=f"station_type_{i}"
            )

    def _add_budget_constraints(self):
        """Add budget constraints including costs and resale revenue."""
        total_cost = self._calculate_total_cost_expr()
        
        # Net cost must be within budget
        self.model.addConstr(
            total_cost <= self.budget,
            name="budget"
        )

    def _calculate_total_cost_expr(self):
        """
        Calculate total cost expression for optimization.

        Includes:
        - New station costs (L2 and L3)
        - Upgrade costs with port retention strategy
        - Equipment resale revenue
        - Port retention costs

        The method implements a sophisticated L2 port retention strategy during upgrades:
        - Keeps minimum required L2 ports for accessibility
        - Sells excess ports based on L3 space requirements
        - Optimizes between retention and upgrade costs
        """

        # New station costs
        new_l2_costs = gp.quicksum(
            self.variables['new_l2_station'][i] * 
            (self.costs['l2_station'] + self.infrastructure['min_ports_per_l2'] * self.costs['l2_port'])
            for i in range(self.n_potential_sites)
        )
        
        new_l3_costs = gp.quicksum(
            self.variables['new_l3_station'][i] * 
            (self.costs['l3_station'] + self.infrastructure['min_ports_per_l3'] * self.costs['l3_port'])
            for i in range(self.n_potential_sites)
        )
        
        # Upgrade costs - use existing L2 station indices directly
        upgrade_costs = gp.quicksum(
            self.variables['upgrade'][i] * (
                self.costs['l3_station'] # New L3 station cost
                + self.infrastructure['min_ports_per_l3'] * self.costs['l3_port']   # New L3 ports
                - self.costs['l2_station'] * self.costs['resale_factor']            # L2 station resale
                - self._calculate_port_sale_count(self.existing_l2.iloc[i]) * self.costs['l2_port'] * self.costs['resale_factor']  # L2 ports resale
            )
            for i in range(self.n_existing_l2)
        )
        
        return new_l2_costs + new_l3_costs + upgrade_costs

    def _add_coverage_constraints(self):
        """Add coverage constraints considering both station types and their ports."""
        try:
            # Get initial coverage
            initial_coverage = self.initial_coverage
            
            # Process each demand point
            for i in range(self.n_demand_points):
                # Initial coverage contribution
                if initial_coverage['l2'][i] > 0:
                    self.model.addConstr(
                        self.variables['coverage_l2'][i] >= initial_coverage['l2'][i],
                        name=f"initial_l2_coverage_{i}"
                    )
                
                if initial_coverage['l3'][i] > 0:
                    # L3 stations contribute to both L2 and L3 coverage
                    self.model.addConstr(
                        self.variables['coverage_l2'][i] >= initial_coverage['l3'][i],
                        name=f"initial_l3_to_l2_coverage_{i}"
                    )
                    self.model.addConstr(
                        self.variables['coverage_l3'][i] >= initial_coverage['l3'][i],
                        name=f"initial_l3_coverage_{i}"
                    )
                
                # New coverage from potential sites
                nearby_l2 = [
                    j for j in range(self.n_potential_sites)
                    if self.distances[j, i] <= self.coverage['l2_radius']
                ]
                
                if nearby_l2:
                    # L2 coverage from both L2 and L3 stations
                    self.model.addConstr(
                        self.variables['coverage_l2'][i] <= (
                            initial_coverage['l2'][i] +  # Initial L2 coverage
                            initial_coverage['l3'][i] +  # Initial L3 coverage
                            gp.quicksum(
                                self.variables['new_l2_station'][j] +  # New L2 stations
                                self.variables['new_l3_station'][j]    # New L3 stations
                                for j in nearby_l2
                            )
                        ),
                        name=f"new_l2_coverage_{i}"
                    )
                
                # L3 coverage from new L3 stations and upgrades
                nearby_l3 = [
                    j for j in range(self.n_potential_sites)
                    if self.distances[j, i] <= self.coverage['l3_radius']
                ]
                
                if nearby_l3:
                    existing_l2_nearby = [
                        j for j in range(len(self.existing_l2))
                        if haversine(
                            (self.demand_points.iloc[i]['latitude'], 
                            self.demand_points.iloc[i]['longitude']),
                            (self.existing_l2.iloc[j]['latitude'],
                            self.existing_l2.iloc[j]['longitude'])
                        ) <= self.coverage['l3_radius']
                    ]
                    
                    self.model.addConstr(
                        self.variables['coverage_l3'][i] <= (
                            initial_coverage['l3'][i] +  # Initial L3 coverage
                            gp.quicksum(self.variables['new_l3_station'][j] 
                                    for j in nearby_l3) +  # New L3 stations
                            gp.quicksum(self.variables['upgrade'][j] 
                                    for j in existing_l2_nearby)  # Upgrades to L3
                        ),
                        name=f"l3_coverage_{i}"
                    )
            
            # Minimum coverage requirements
            self.model.addConstr(
                gp.quicksum(
                    self.variables['coverage_l2'][i] * 
                    self.demand_points['weight'].iloc[i]
                    for i in range(self.n_demand_points)
                ) >= self.coverage['min_coverage_l2'],
                name="min_l2_coverage"
            )
            
            self.model.addConstr(
                gp.quicksum(
                    self.variables['coverage_l3'][i] * 
                    self.demand_points['weight'].iloc[i]
                    for i in range(self.n_demand_points)
                ) >= self.coverage['min_coverage_l3'],
                name="min_l3_coverage"
            )
            
        except Exception as e:
            self.logger.error(f"Error adding coverage constraints: {str(e)}")
            raise

    def _add_grid_constraints(self):
        """Add power grid capacity constraints based on fixed port counts."""
        for i in range(self.n_potential_sites):
            # Calculate power demand for new stations and upgrades
            l2_power = (
                self.variables['new_l2_station'][i] * 
                self.infrastructure['min_ports_per_l2'] * 
                self.infrastructure['l2_power']
            )
            
            l3_power = (
                self.variables['new_l3_station'][i] * 
                self.infrastructure['min_ports_per_l3'] * 
                self.infrastructure['l3_power']
            )
            
            # Total power must not exceed grid capacity
            self.model.addConstr(
                l2_power + l3_power <= self.infrastructure['grid_capacity'],
                name=f"new_grid_capacity_{i}"
            )
        
        for i in range(self.n_existing_l2):
            # Calculate power demand for upgrades
            l3_power = (
                self.variables['upgrade'][i] * 
                (self.infrastructure['min_ports_per_l3'] * self.infrastructure['l3_power']
                + self.existing_l2.iloc[i]['num_chargers'] * self.infrastructure['l2_power'])
            )
            
            # Total power must not exceed grid capacity
            self.model.addConstr(
                l3_power <= self.infrastructure['grid_capacity'],
                name=f"upgrade_grid_capacity_{i}"
            )

    # ----------------
    # Calculation and Support Methods
    # ----------------
    def _calculate_initial_coverage(self) -> Dict[str, np.ndarray]:
        """
        Calculate initial coverage matrix considering both L2 and L3 ports.
        
        Accounts for:
        - Existing L2 and L3 station coverage
        - Port-based service capacity
        - Population distribution
        - Retained L2 port contribution at upgraded stations
        
        Returns:
            Dict with coverage arrays for both L2 and L3
        """

        try:
            # Initialize coverage arrays
            coverage = {
                'l2': np.zeros(self.n_demand_points),
                'l3': np.zeros(self.n_demand_points)
            }
            
            # Process each demand point
            for idx in range(self.n_demand_points):
                demand_point = self.demand_points.iloc[idx]
                demand_coords = (
                    self._safe_float(demand_point['latitude']),
                    self._safe_float(demand_point['longitude'])
                )
                
                # Skip invalid demand points
                if any(x == 0.0 for x in demand_coords):
                    continue
                
                # Check existing L2 stations for L2 coverage
                for _, station in self.existing_l2.iterrows():
                    station_coords = (
                        self._safe_float(station['latitude']),
                        self._safe_float(station['longitude'])
                    )
                    
                    # Skip invalid station coordinates
                    if any(x == 0.0 for x in station_coords):
                        continue
                        
                    dist = haversine(demand_coords, station_coords)
                    if dist <= self.coverage['l2_radius']:
                        coverage['l2'][idx] = 1
                        break  # Point is covered by L2
                
                # Check existing L3 stations for both L2 and L3 coverage
                for _, station in self.existing_l3.iterrows():
                    station_coords = (
                        self._safe_float(station['latitude']),
                        self._safe_float(station['longitude'])
                    )
                    
                    # Skip invalid station coordinates
                    if any(x == 0.0 for x in station_coords):
                        continue
                    
                    dist = haversine(demand_coords, station_coords)
                    
                    # L3 stations provide both L2 and L3 coverage within respective radii
                    if dist <= self.coverage['l2_radius']:
                        coverage['l2'][idx] = 1
                    if dist <= self.coverage['l3_radius']:
                        coverage['l3'][idx] = 1
                    if coverage['l2'][idx] and coverage['l3'][idx]:
                        break  # Point has both types of coverage
            
            return coverage
            
        except Exception as e:
            self.logger.error(f"Error calculating initial coverage: {str(e)}")
            return {'l2': np.zeros(self.n_demand_points), 
                    'l3': np.zeros(self.n_demand_points)}

    def _calculate_coverage_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate coverage metrics considering port-based coverage."""
        try:
            # Get initial coverage
            initial_l2 = self._calculate_coverage_percentage(self.initial_coverage['l2'])
            initial_l3 = self._calculate_coverage_percentage(self.initial_coverage['l3'])

            # Get final coverage from model variables
            final_l2_matrix = np.array([
                self.variables['coverage_l2'][i].X 
                for i in range(self.n_demand_points)
            ])
            final_l3_matrix = np.array([
                self.variables['coverage_l3'][i].X 
                for i in range(self.n_demand_points)
            ])

            final_l2 = self._calculate_coverage_percentage(final_l2_matrix)
            final_l3 = self._calculate_coverage_percentage(final_l3_matrix)
            
            return {
                'initial': {
                    'l2_coverage': float(initial_l2),
                    'l3_coverage': float(initial_l3)
                },
                'final': {
                    'l2_coverage': float(final_l2),
                    'l3_coverage': float(final_l3)
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error calculating coverage metrics: {str(e)}")
            return {
                'initial': {'l2_coverage': 0.0, 'l3_coverage': 0.0},
                'final': {'l2_coverage': 0.0, 'l3_coverage': 0.0}
            }
    
    def _calculate_coverage_percentage(self, coverage_matrix: np.ndarray) -> float:
        """Calculate weighted coverage percentage from coverage matrix."""
        if coverage_matrix is None or len(self.demand_points) == 0:
            return 0.0
            
        try:
            weighted_sum = sum(
                self._safe_float(coverage_matrix[i], 0.0) * 
                self._safe_float(self.demand_points['weight'].iloc[i], 0.0)
                for i in range(len(self.demand_points))
            )
            total_weight = sum(
                self._safe_float(self.demand_points['weight'].iloc[i], 0.0)
                for i in range(len(self.demand_points))
            )
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error in coverage calculation: {str(e)}")
            return 0.0

    def _calculate_port_sale_count(self, station: pd.Series) -> int:
        """Calculate number of L2 ports to sell based on L3 space requirements."""
        try:
            # Calculate L2 ports to sell
            l2_ports_sold = min(
                max(int(station['num_chargers']) - self.infrastructure['min_ports_per_l2'], 0),
                self.infrastructure['min_ports_per_l3']
            ) if pd.notna(station['num_chargers']) else 0
            return l2_ports_sold
            
        except Exception as e:
            self.logger.error(f"Error calculating port sale count: {str(e)}")
            return 0
    
    def _calculate_solution_costs(self, solution: Dict) -> Dict[str, Any]:
        """Calculate detailed cost breakdown for solution with exact integer arithmetic."""
        try:
            # Initialize counts and resale tracking
            l2_resale_ports = 0  # Track total L2 ports being resold from upgrades
            
            # For existing stations being upgraded, calculate total L2 ports being replaced
            for upgrade_idx in solution['upgrades']:
                if upgrade_idx < len(self.existing_l2):
                    station = self.existing_l2.iloc[upgrade_idx]
                    l2_resale_ports += self._calculate_port_sale_count(station)
            
            # Calculate new infrastructure costs
            costs = {
                'new_infrastructure': {
                    'l2_stations': {
                        'count': len(solution['new_stations']['l2']),
                        'cost': len(solution['new_stations']['l2']) * (
                            self.costs['l2_station'] + 
                            self.infrastructure['min_ports_per_l2'] * self.costs['l2_port']
                        )
                    },
                    'l3_stations_new': {
                        'count': len(solution['new_stations']['l3']),
                        'cost': len(solution['new_stations']['l3']) * (
                            self.costs['l3_station'] + 
                            self.infrastructure['min_ports_per_l3'] * self.costs['l3_port']
                        )
                    },
                    'l3_stations_upgrade': {
                        'count': len(solution['upgrades']),
                        'cost': len(solution['upgrades']) * (
                            self.costs['l3_station'] + 
                            self.infrastructure['min_ports_per_l3'] * self.costs['l3_port']
                        )
                    }
                },
                'resale_revenue': {
                    'l2_stations': {
                        'count': len(solution['upgrades']),
                        'revenue': len(solution['upgrades']) * 
                                self.costs['l2_station'] * 
                                self.costs['resale_factor']
                    },
                    'l2_ports': {
                        'count': l2_resale_ports,
                        'revenue': l2_resale_ports * 
                                self.costs['l2_port'] * 
                                self.costs['resale_factor']
                    }
                }
            }
            
            # Calculate summary totals
            total_purchase = sum(item['cost'] for item in costs['new_infrastructure'].values())
            total_revenue = sum(item['revenue'] for item in costs['resale_revenue'].values())
            
            costs['summary'] = {
                'total_purchase': total_purchase,
                'total_revenue': total_revenue,
                'net_cost': total_purchase - total_revenue
            }
            
            return costs
            
        except Exception as e:
            self.logger.error(f"Error calculating costs: {str(e)}")
            raise

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
    
    # ----------------
    # Station Status Update Methods
    # ----------------
    
    def update_station_statuses(self, solution: Dict) -> dict:
        """
        Update station statuses with rich metadata and port tracking.

        Handles:
        - Existing station status tracking
        - Upgrade processing with port retention
        - New station addition
        - Port count management
        - Infrastructure status documentation

        The method maintains detailed port counts considering:
        - Initial port configuration
        - L2 port retention at upgrades
        - Final port distribution
        - Total capacity changes
        """

        stations = self.stations_df.copy()
        
        # Initialize status for all stations
        stations['status'] = 'Existing L2'
        stations.loc[stations['charger_type'] == 'Level 3', 'status'] = 'Existing L3'
        
        detailed_stations = {
            'existing': [],
            'upgrades': [],
            'new': []
        }
        
        # 1. Process all existing stations, skipping those that will be upgraded
        upgrade_indices = [self.existing_l2.iloc[i].name for i in solution['upgrades']]
        for idx, station in stations.iterrows():
            try:
                # Skip stations that will be upgraded
                if station['charger_type'] == 'Level 2' and idx in upgrade_indices:
                    continue
                    
                # Get the actual number of ports, defaulting to 0 if unknown
                num_chargers = int(station['num_chargers']) if pd.notna(station['num_chargers']) else 0
                
                # Initialize both initial and final port counts
                initial_ports = {
                    'level_2': num_chargers if station['charger_type'] == 'Level 2' else 0,
                    'level_3': num_chargers if station['charger_type'] == 'Level 3' else 0,
                    'total': num_chargers
                }
                final_ports = initial_ports.copy()  # Start with same as initial
                
                existing_station = {
                    'id': f'existing_{idx}',
                    'name': station.get('name', f'Station {idx}'),
                    'location': {
                    'latitude': float(station['latitude']),
                    'longitude': float(station['longitude']),
                    'address': str(station.get('address', 'Unknown')),
                    'city': str(station.get('city', 'Unknown')),
                    'postal_code': str(station.get('postal_code', '')),
                    'province': 'Ontario',
                    },
                    'charging': {
                    'status': station['status'],
                    'charger_type': station['charger_type'],
                    'ports': {
                        'initial': initial_ports,
                        'final': final_ports
                    },
                    'power_output': self._get_power_output(station['charger_type']),
                    'usage_cost': str(station.get('usage_cost', 'Unknown'))
                    },
                    'operator': str(station.get('operator', 'Unknown'))
                }
                
                detailed_stations['existing'].append(existing_station)
                
            except Exception as e:
                self.logger.error(f"Error processing existing station {idx}: {str(e)}")
                continue

        # 2. Process upgrades to create detailed upgrade records
        for upgrade_idx in solution['upgrades']:
            try:
                # Get the original station that's being upgraded
                original_station = self.existing_l2.iloc[upgrade_idx]
                current_ports = int(original_station['num_chargers']) if pd.notna(original_station['num_chargers']) else 0
                
                # Calculate ports to keep and sell
                l2_ports_sold = self._calculate_port_sale_count(original_station)
                l2_ports_kept = current_ports - l2_ports_sold
                
                upgrade_data = {
                    'id': f'upgrade_{upgrade_idx}',
                    'name': f"Upgraded L3 Station",
                    'location': {
                        'latitude': float(original_station['latitude']),
                        'longitude': float(original_station['longitude']),
                        'type': str(original_station.get('location_type', 'Unknown')),
                        'address': str(original_station.get('address', 'Unknown')),
                        'city': str(original_station.get('city', 'Unknown')),
                        'postal_code': str(original_station.get('postal_code', '')),
                        'province': 'Ontario',
                    },
                    'charging': {
                        'status': 'Upgrade L2 -> L3',
                        'charger_type': 'Level 3',
                        'ports': {
                            'initial': {
                                'level_2': current_ports,
                                'level_3': 0,
                                'total': current_ports
                            },
                            'final': {
                                'level_2': l2_ports_kept,  # Keep minimum L2 ports
                                'level_3': self.infrastructure['min_ports_per_l3'],
                                'total': l2_ports_kept + self.infrastructure['min_ports_per_l3']
                            }
                        },
                        'power_output': self._get_power_output('Level 3')
                    },
                    'implementation': {
                        'estimated_timeline': 'Phase 1',
                        'grid_requirements': {
                            'power_increase': float(self._calculate_power_increase(
                                'Level 2', 'Level 3', 
                                self.infrastructure['min_ports_per_l3']
                            )),
                            'voltage_requirement': '480V'
                        },
                        'estimated_installation_cost': float(self._calculate_upgrade_cost(upgrade_idx))
                    }
                }
                detailed_stations['upgrades'].append(upgrade_data)
                    
            except Exception as e:
                self.logger.error(f"Error processing upgrade {upgrade_idx}: {str(e)}")
                continue

        # 3. Process new stations (Rest of the function remains the same)
        for site_type in ['l2', 'l3']:
            for site_idx in solution['new_stations'].get(site_type, []):
                try:
                    site = self.potential_sites.iloc[site_idx]
                    
                    # Set parameters based on station type
                    charger_type = 'Level 2' if site_type == 'l2' else 'Level 3'
                    min_ports = (self.infrastructure['min_ports_per_l2'] if site_type == 'l2' 
                                else self.infrastructure['min_ports_per_l3'])
                    
                    new_station = {
                        'id': f'new_{site_type.upper()}_{site_idx}',
                        'name': f"New {site_type.upper()} Station",
                        'location': {
                            'latitude': float(site['latitude']),
                            'longitude': float(site['longitude']),
                            'type': str(site.get('location_type', 'Unknown')),
                            'address': str(site.get('address', 'Location TBD')),
                            'city': str(site.get('city', 'Unknown')),
                            'postal_code': str(site.get('postal_code', '')),
                            'province': 'Ontario',
                            'score': float(site.get('score', 0)),
                        },
                        'charging': {
                            'status': f'New {site_type.upper()}',
                            'charger_type': charger_type,
                            'ports': {
                                'initial': {
                                    'level_2': 0,
                                    'level_3': 0,
                                    'total': 0
                                },
                                'final': {
                                    'level_2': min_ports if site_type == 'l2' else 0,
                                    'level_3': min_ports if site_type == 'l3' else 0,
                                    'total': min_ports
                                }
                            },
                            'power_output': self._get_power_output(charger_type)
                        },
                        'implementation': {
                            'estimated_timeline': 'Phase 2' if site_type == 'l2' else 'Phase 1',
                            'site_preparation': self._get_site_preparation_requirements(site),
                            'grid_requirements': {
                                'power_requirement': float(self._calculate_power_requirement(
                                    charger_type, min_ports
                                )),
                                'voltage_requirement': '240V' if site_type == 'l2' else '480V'
                            },
                            'estimated_installation_cost': float(
                                self._calculate_installation_cost(charger_type)
                            )
                        }
                    }
                    detailed_stations['new'].append(new_station)
                        
                except Exception as e:
                    self.logger.error(f"Error processing new {site_type} station {site_idx}: {str(e)}")
                    continue
        
        return detailed_stations
    
    # ----------------
    # Utility Methods
    # ----------------

    def _get_power_output(self, charger_type: str) -> Dict[str, Any]:
        """Get power output specifications for charger type."""
        if charger_type == 'Level 2' or charger_type == 'L2' or charger_type == 'l2':
            return {
                'kw': 19.2,  # Maximum L2 power
                'voltage': 240,
                'amperage': 80,
                'charge_rate': '~25-30 miles per hour'
            }
        elif charger_type == 'Level 3' or charger_type == 'L3' or charger_type == 'l3':
            return {
                'kw': 350,  # Maximum DC fast charging power
                'voltage': 480,
                'amperage': 729,  # 350kW at 480V
                'charge_rate': '~3-20 miles per minute'
            }
        return {'kw': 0, 'voltage': 0, 'amperage': 0, 'charge_rate': 'Unknown'}

    def _calculate_power_increase(self, current_type: str, new_type: str, new_ports: int) -> float:
        """Calculate power increase needed for upgrade."""
        # Get current and new power requirements per port
        current_power = self._get_power_output(current_type)['kw']
        new_power = self._get_power_output(new_type)['kw']
        
        # Calculate total power increase
        return (new_power * new_ports) - current_power

    def _calculate_power_requirement(self, charger_type: str, num_ports: int) -> float:
        """
        Calculate total power requirement for a station.

        Considers:
        - Charger type power requirements
        - Number of ports (both retained and new)
        - Total capacity needs

        Args:
            charger_type: Type of charger ('Level 2' or 'Level 3')
            num_ports: Number of ports to calculate for

        Returns:
            float: Total power requirement in kW
        """

        power_per_port = self._get_power_output(charger_type)['kw']
        return power_per_port * num_ports
    
    def _get_site_preparation_requirements(self, site: pd.Series) -> Dict[str, Any]:
        """
        Determine site preparation requirements including port modifications.
        
        Assesses:
        - Existing port infrastructure
        - Required modifications for retention/upgrade
        - Grid capacity needs
        - Installation requirements
        - Space utilization considerations
        
        Args:
            site: Site data series
            
        Returns:
            Dict of preparation requirements
        """

        base_requirements = {
            'parking_modifications': 'Required',
            'signage': 'Required',
            'lighting': 'Required'
        }

        case = site.get('location_type', '').lower()
        if ('fuel' in case or 'food' in case or 'automotive' in case or 
            'services' in case or 'vacant' in case or 'entertainment' in case or 
            'commercial' in case):
            base_requirements.update({
                'surface_work': 'Extensive',
                'existing_electrical': 'New Installation Required'
            })
        elif 'supermarket' in case or 'retail' in case:
            base_requirements.update({
                'surface_work': 'Moderate',
                'existing_electrical': 'May Need Upgrade'
            })
        elif 'charging' in case or 'parking' in case:
            base_requirements.update({
                'surface_work': 'Minimal',
                'existing_electrical': 'Likely Available'
            })
        
        return base_requirements

    def _calculate_upgrade_cost(self, upgrade_idx: int) -> float:
        """
        Calculate the total cost for upgrading a site.

        Includes:
        - New L3 infrastructure costs
        - L2 port retention costs
        - Resale value of excess L2 ports
        - Installation and modification costs

        Args:
            upgrade_idx: Index of station being upgraded

        Returns:
            float: Total upgrade cost considering port retention
        """

        try:
            # Get current L2 ports from the station being upgraded
            station = self.existing_l2.iloc[upgrade_idx]
            current_l2_ports = int(station['num_chargers']) if pd.notna(station['num_chargers']) else 0
            
            # Calculate new costs
            new_l3_cost = (
                self.costs['l3_station'] +  # Base station cost
                self.infrastructure['min_ports_per_l3'] * self.costs['l3_port']  # New L3 ports
            )
            
            # Calculate resale value of existing equipment
            resale_value = (
                self.costs['l2_station'] +  # Base station
                current_l2_ports * self.costs['l2_port']  # Existing ports
            ) * self.costs['resale_factor']
            
            return new_l3_cost - resale_value
            
        except Exception as e:
            self.logger.error(f"Error calculating upgrade cost: {str(e)}")
            return 0.0

    def _calculate_installation_cost(self, charger_type: str) -> float:
        """Calculate installation cost for new stations."""
        try:
            if charger_type == 'Level 2':
                return (
                    self.costs['l2_station'] + 
                    self.infrastructure['min_ports_per_l2'] * self.costs['l2_port']
                )
            elif charger_type == 'Level 3':
                return (
                    self.costs['l3_station'] + 
                    self.infrastructure['min_ports_per_l3'] * self.costs['l3_port']
                )
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating installation cost: {str(e)}")
            return 0.0

    # ----------------
    # Sensitivity Analysis Methods
    # ----------------
    
    def perform_sensitivity_analysis(self, display: bool = False) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on key model parameters and constraints.
        
        Analyzes:
        - Impact of port retention requirements
        - Coverage vs cost tradeoffs
        - Grid capacity utilization
        - Budget constraint effects
        - Implementation phasing sensitivities
        
        Args:
            display: Whether to display results
            
        Returns:
            Dict containing sensitivity analysis results
        """
        
        if self.model.Status != GRB.OPTIMAL:
            raise ValueError("Model must be solved optimally before performing sensitivity analysis.")
        
        sensitivity_results = {
            'constraints': self._analyze_constraints_via_slack_and_utilization(),
            'variables': self._analyze_variable_reduced_costs(),
            'insights': self._generate_insights(),
        }

        if display:
            get_sensitivity_results_summary(sensitivity_results, display=True)

        return sensitivity_results

    def _analyze_constraints_via_slack_and_utilization(self) -> Dict[str, Any]:
        """Analyze constraints using slacks and utilization."""
        key_constraints = {
            'Budget': self.model.getConstrByName('budget'),
            'L2 Coverage': self.model.getConstrByName('min_l2_coverage'),
            'L3 Coverage': self.model.getConstrByName('min_l3_coverage'),
        }

        results = {}
        for name, constraint in key_constraints.items():
            if constraint:
                slack = constraint.Slack
                rhs = constraint.RHS
                # Calculate utilization properly - should never exceed 100%
                if rhs != 0:
                    utilization = ((rhs - slack) / rhs) * 100
                else:
                    utilization = 100.0 if slack == 0 else 0.0
                
                results[name] = {
                    'slack': float(slack),
                    'rhs': float(rhs),
                    'utilization': float(utilization),
                    'status': "Binding" if abs(slack) < 1e-6 else "Non-Binding"
                }
        return results

    def _analyze_variable_reduced_costs(self) -> Dict[str, Any]:
        """Analyze variables that affect the solution most significantly."""
        variable_analysis = []
        
        # Look at all variables with non-zero values or significant reduced costs
        for var in self.model.getVars():
            if var.X > 0.5:  # For binary variables, check if they're selected
                try:
                    rc = var.RC if hasattr(var, 'RC') else None
                    # For MILPs, RC might not be available for integer variables
                    variable_analysis.append({
                        'variable': var.VarName,
                        'value': float(var.X),
                        'reduced_cost': float(rc) if rc is not None else None,
                        'type': 'Binary' if var.VType == GRB.BINARY else 'Integer' if var.VType == GRB.INTEGER else 'Continuous'
                    })
                except Exception as e:
                    self.logger.warning(f"Could not get reduced cost for {var.VarName}: {str(e)}")

        return variable_analysis

    def _generate_insights(self) -> List[str]:
        """Generate actionable insights based on analysis."""
        insights = []
        
        # Budget constraint analysis
        budget_stats = self._analyze_constraints_via_slack_and_utilization().get('Budget', {})
        if budget_stats.get('utilization', 0) > 95:
            insights.append(f"Budget is highly constrained ({budget_stats['utilization']:.4f}% utilized); consider increasing it.")
        elif budget_stats.get('utilization', 0) < 70:
            insights.append(f"Budget is underutilized ({budget_stats['utilization']:.4f}%); consider optimizing allocation or reducing budget planning.")

        # Coverage analysis
        l2_stats = self._analyze_constraints_via_slack_and_utilization().get('L2 Coverage', {})
        l3_stats = self._analyze_constraints_via_slack_and_utilization().get('L3 Coverage', {})
        
        if l2_stats and l3_stats:
            l2_util = l2_stats.get('utilization', 0)
            l3_util = l3_stats.get('utilization', 0)
            
            if abs(l2_util - l3_util) > 20:
                insights.append(
                    f"Large disparity between L2 ({l2_util:.1f}%) and L3 ({l3_util:.1f}%) "
                    f"coverage - consider rebalancing network."
                )
            
            if l2_stats.get('status') == "Binding":
                insights.append(f"L2 coverage constraint is binding at {l2_util:.1f}%")
            if l3_stats.get('status') == "Binding":
                insights.append(f"L3 coverage constraint is binding at {l3_util:.1f}%")
        
        return insights
    
    # ----------------
    # Program Documentation
    # ----------------

    def get_program_doc(self) -> Dict[str, Any]:
        """
        Return comprehensive documentation of the optimization program.

        Includes:
        - Data summary and key parameters
        - Decision variables and constraints
        - Logical constraints and relationships

        Returns:
            Dict containing program
        """

        return {
            "data_summary": {
                "network_data": {
                    "demand_points": {
                        "size": len(self.demand_points),
                        "description": "Population demand points\nWeighted by:\n• EV Adoption (35%)\n• Infrastructure Quality (25%)\n• Population Density (20%)\n• Transit Access (15%)\n• Infrastructure Age (5%)"
                    },
                    "existing_l2_stations": {
                        "size": len(self.existing_l2),
                        "description": "Current Level 2 charging infrastructure"
                    },
                    "existing_l3_stations": {
                        "size": len(self.existing_l3),
                        "description": "Current Level 3 charging infrastructure"
                    },
                    "potential_locations": {
                        "size": len(self.potential_sites),
                        "description": "Candidate sites for new charging stations"
                    }
                },
                "cost_parameters": {
                    "l2_station": {
                        "size": self.costs['l2_station'],
                        "description": "Base cost for new L2 station"
                    },
                    "l3_station": {
                        "size": self.costs['l3_station'],
                        "description": "Base cost for new L3 station"
                    },
                    "l2_port": {
                        "size": self.costs['l2_port'],
                        "description": "Cost per L2 charging port"
                    },
                    "l3_port": {
                        "size": self.costs['l3_port'],
                        "description": "Cost per L3 charging port"
                    },
                    "resale_factor": {
                        "size": self.costs['resale_factor'],
                        "description": "Resale value factor for equipment"
                    }
                },
                "infrastructure_parameters": {
                    "min_ports_l2": {
                        "size": self.infrastructure['min_ports_per_l2'],
                        "description": "Minimum L2 ports for new stations and retention"
                    },
                    "min_ports_l3": {
                        "size": self.infrastructure['min_ports_per_l3'],
                        "description": "Minimum L3 ports required"
                    },
                    "l2_power": {
                        "size": self.infrastructure['l2_power'],
                        "description": "L2 charger power draw (kW)"
                    },
                    "l3_power": {
                        "size": self.infrastructure['l3_power'],
                        "description": "L3 charger power draw (kW)"
                    },
                    "grid_capacity": {
                        "size": self.infrastructure['grid_capacity'],
                        "description": "Maximum grid capacity per site (kW)"
                    }
                },
                "coverage_parameters": {
                    "l2_radius": {
                        "size": self.coverage['l2_radius'],
                        "description": "L2 coverage radius (km)"
                    },
                    "l3_radius": {
                        "size": self.coverage['l3_radius'],
                        "description": "L3 coverage radius (km)"
                    },
                    "min_l2_coverage": {
                        "size": self.coverage['min_coverage_l2'],
                        "description": "Minimum L2 population coverage"
                    },
                    "min_l3_coverage": {
                        "size": self.coverage['min_coverage_l3'],
                        "description": "Minimum L3 population coverage"
                    }
                }
            },
            
            "decision_variables": {
                "new_l2_stations": {
                    "type": "Binary",
                    "dimension": len(self.potential_sites),
                    "description": "1 if new L2 station placed at site i"
                },
                "new_l3_stations": {
                    "type": "Binary",
                    "dimension": len(self.potential_sites),
                    "description": "1 if new L3 station placed at site i"
                },
                "upgrades": {
                    "type": "Binary",
                    "dimension": len(self.existing_l2),
                    "description": "1 if existing L2 station at site i upgraded to L3"
                },
                "coverage_l2": {
                    "type": "Binary",
                    "dimension": len(self.demand_points),
                    "description": "1 if demand point i is covered by L2 station"
                },
                "coverage_l3": {
                    "type": "Binary",
                    "dimension": len(self.demand_points),
                    "description": "1 if demand point i is covered by L3 station"
                }
            },
            
            "constraints": {
                "budget": {
                    "type": "Linear",
                    "bound": self.budget,
                    "description": f"Total cost must not exceed ${self.budget:,}"
                },
                "l2_coverage": {
                    "type": "Linear",
                    "bound": self.coverage['min_coverage_l2'],
                    "description": f"At least {self.coverage['min_coverage_l2']*100}% population within {self.coverage['l2_radius']}km of L2"
                },
                "l3_coverage": {
                    "type": "Linear",
                    "bound": self.coverage['min_coverage_l3'],
                    "description": f"At least {self.coverage['min_coverage_l3']*100}% population within {self.coverage['l3_radius']}km of L3"
                },
                "grid_capacity": {
                    "type": "Linear",
                    "bound": self.infrastructure['grid_capacity'],
                    "description": f"Maximum {self.infrastructure['grid_capacity']}kW power demand per site"
                },
                'logical': [
                    "Cannot have both L2 and L3 stations at same site",
                    "Only existing L2 stations can be upgraded to L3",
                    "Can't add new station where there's an upgrade",
                    f"Must keep minimum {self.infrastructure['min_ports_per_l2']} L2 ports at upgraded stations"
                ]
            },
            
            "objective": {
                "type": "Multi-objective Linear",
                "components": {
                    "l3_coverage": {
                        "weight": self.weights['l3_coverage'],
                        "description": "Maximize L3 charging coverage"
                    },
                    "l2_coverage": {
                        "weight": self.weights['l2_coverage'],
                        "description": "Maximize L2 charging coverage" 
                    },
                    "cost": {
                        "weight": self.weights['cost'],
                        "description": "Minimize total infrastructure cost"
                    }
                }
            }
        }