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
from tabulate import tabulate

from src.data.utils import *

class EVNetworkOptimizer:
    """Network enhancement optimization model."""
    
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
        
        # Fill any null values with defaults
        self.demand_points = self.demand_points.fillna(0)
        self.potential_sites = self.potential_sites.fillna(0)
        self.stations_df = self.stations_df.fillna(0)
        
        # Store key statistics
        self.n_demand_points = len(self.demand_points)
        self.n_potential_sites = len(self.potential_sites)
        self.n_existing_stations = len(self.stations_df)
        self.n_existing_l2 = len(self.stations_df[self.stations_df['charger_type'] == 'Level 2'])
        self.n_existing_l3 = len(self.stations_df[self.stations_df['charger_type'] == 'Level 3'])

        # Load configuration parameters
        self.costs = config['costs']
        self.infrastructure = config['infrastructure']
        self.coverage = config['coverage']
        self.weights = config['weights']
        self.budget = config['budget']
        self.requirements = config['requirements']
        
        # Initialize constraints from configuration
        self.constraints = input_data.get('constraints', {
            'budget': self.budget['default'],
            'min_coverage_l2': self.coverage['min_coverage_l2'],
            'min_coverage_l3': self.coverage['min_coverage_l3'],
            'max_l3_distance': self.coverage['l3_radius'],
            'max_stations_per_area': self.infrastructure['max_new_ports']
        })

        # Calculate initial coverage
        self.initial_l2_coverage = self._calculate_initial_coverage('Level 2', self.coverage['l2_radius'])
        self.initial_l3_coverage = self._calculate_initial_coverage('Level 3', self.coverage['l3_radius'])

        # Track existing infrastructure
        self.current_ports = np.zeros(len(self.potential_sites))
        self.current_types = np.zeros(len(self.potential_sites))  # 0=none, 1=L2, 2=L3
        
        # Initialize variables storage
        self.variables = {}
    
    # ----------------
    # Main Public Methods
    # ----------------

    def optimize(self) -> Dict[str, Any]:
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
                upgrades = [i for i in range(len(self.potential_sites)) 
                          if self.variables['upgrade'][i].X > 0.01]
                print(f"✓ Found {len(upgrades)} upgrades")
                
                new_l2_stations = [i for i in range(len(self.potential_sites))
                                 if self.variables['new_l2_station'][i].X > 0.01]
                new_l3_stations = [i for i in range(len(self.potential_sites))
                                 if self.variables['new_l3_station'][i].X > 0.01]
                print(f"✓ Found {len(new_l2_stations)} new L2 stations and {len(new_l3_stations)} new L3 stations")
                
                # Get port allocations
                l2_ports = {i: int(round(self.variables['new_l2_ports'][i].X))
                           for i in range(len(self.potential_sites))
                           if self.variables['new_l2_ports'][i].X > 0.01}
                l3_ports = {i: int(round(self.variables['new_l3_ports'][i].X))
                           for i in range(len(self.potential_sites))
                           if self.variables['new_l3_ports'][i].X > 0.01}
                print(f"✓ Processed {len(l2_ports)} L2 ports and {len(l3_ports)} L3 ports")
                
                # Calculate costs and coverage
                costs = self._calculate_solution_costs()
                coverage = self._calculate_coverage_metrics()
                
                print("\n4. Creating solution structure...")
                base_solution = {
                        'upgrades': upgrades,
                        'new_stations': {
                            'l2': new_l2_stations,
                            'l3': new_l3_stations
                        },
                        'new_ports': {
                            'l2': l2_ports,
                            'l3': l3_ports
                        },
                        'costs': costs,
                        'coverage': coverage
                    }

                # Update stations with detailed information
                stations_df = self.stations_df.copy()
                updated_stations, detailed_stations = self.update_station_statuses(
                    stations_df, base_solution
                )
                print("✓ Station statuses updated")

                solution = {
                    'status': 'optimal',
                    'objective_value': float(self.model.ObjVal),
                    'upgrades': upgrades,
                    'new_stations': base_solution['new_stations'],
                    'new_ports': base_solution['new_ports'],
                    'stations': detailed_stations,
                    'costs': costs,
                    'coverage': coverage,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_stations': len(self.stations_df),
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

    def perform_sensitivity_analysis(self) -> Dict[str, Any]:
        """Perform simplified sensitivity analysis focusing on key insights."""
        if self.model.Status != GRB.OPTIMAL:
            raise ValueError("Model must be solved optimally before performing sensitivity analysis")
            
        sensitivity_results = {
            'constraints': {},
            'insights': []
        }
        
        # Only analyze key constraints
        key_constraints = {
            'Budget': self.model.getConstrByName('budget'),
            'L2 Coverage': self.model.getConstrByName('min_l2_coverage'),
            'L3 Coverage': self.model.getConstrByName('min_l3_coverage')
        }
        
        # Analyze each key constraint
        for name, constr in key_constraints.items():
            if constr:
                slack = constr.Slack
                rhs = constr.RHS
                
                sensitivity_results['constraints'][name] = {
                    'slack': float(slack),
                    'rhs': float(rhs),
                    'utilization': float((rhs - slack) / rhs * 100 if rhs != 0 else 0)
                }
                
                # Generate meaningful insights
                if abs(slack) < 1e-6:  # Binding constraint
                    if 'Budget' in name:
                        sensitivity_results['insights'].append(
                            f"Budget constraint is binding - additional funding of ${rhs:,.2f} could improve the solution"
                        )
                    elif 'Coverage' in name:
                        achieved = ((rhs - slack) / rhs * 100) if rhs != 0 else 0
                        required = rhs * 100
                        sensitivity_results['insights'].append(
                            f"{name} requirement is binding - achieved {achieved:.1f}% vs required {required:.1f}%"
                        )
                else:
                    if 'Budget' in name:
                        remaining = float(slack)
                        sensitivity_results['insights'].append(
                            f"Solution uses ${rhs-slack:,.2f} of ${rhs:,.2f} budget (${remaining:,.2f} remaining)"
                        )
                    elif 'Coverage' in name:
                        achieved = ((rhs - slack) / rhs * 100) if rhs != 0 else 0
                        required = rhs * 100
                        sensitivity_results['insights'].append(
                            f"{name}: Achieved {achieved:.1f}% coverage (required {required:.1f}%)"
                        )
        
        return sensitivity_results
    
    def get_program_documentation(self) -> Dict[str, Any]:
        """Return comprehensive documentation of the optimization program."""
        return {
            "data_summary": {
                "demand_points": len(self.demand_points),
                "existing_stations": {
                    "total": len(self.stations_df),
                    "l2_stations": len(self.stations_df[self.stations_df['charger_type'] == 'Level 2']),
                    "l3_stations": len(self.stations_df[self.stations_df['charger_type'] == 'Level 3'])
                },
                "potential_sites": len(self.potential_sites)
            },
            
            "decision_variables": {
                "new_l2_stations": {
                    "type": "Binary",
                    "dimension": len(self.potential_sites),
                    "description": "1 if new L2 station placed at site i, 0 otherwise"
                },
                "new_l3_stations": {
                    "type": "Binary",
                    "dimension": len(self.potential_sites),
                    "description": "1 if new L3 station placed at site i, 0 otherwise"
                },
                "upgrades": {
                    "type": "Binary",
                    "dimension": len(self.potential_sites),
                    "description": "1 if L2 station at site i is upgraded to L3, 0 otherwise"
                },
                "new_l2_ports": {
                    "type": "Integer",
                    "dimension": len(self.potential_sites),
                    "bounds": [0, self.infrastructure['max_new_ports']],
                    "description": "Number of new L2 ports added at site i"
                },
                "new_l3_ports": {
                    "type": "Integer",
                    "dimension": len(self.potential_sites),
                    "bounds": [0, self.infrastructure['max_new_ports']],
                    "description": "Number of new L3 ports added at site i"
                }
            },
            
            "constraints": {
                "budget": {
                    "type": "Linear",
                    "bound": self.budget['default'],
                    "description": "Total cost must not exceed budget"
                },
                "coverage": {
                    "l2_coverage": {
                        "type": "Linear",
                        "bound": self.coverage['min_coverage_l2'],
                        "description": f"At least {self.coverage['min_coverage_l2']*100}% population coverage within {self.coverage['l2_radius']}km for L2"
                    },
                    "l3_coverage": {
                        "type": "Linear",
                        "bound": self.coverage['min_coverage_l3'],
                        "description": f"At least {self.coverage['min_coverage_l3']*100}% population coverage within {self.coverage['l3_radius']}km for L3"
                    }
                },
                "infrastructure": {
                    "grid_capacity": {
                        "type": "Linear",
                        "bound": self.infrastructure['grid_capacity'],
                        "description": "Total power demand at each site must not exceed grid capacity"
                    },
                    "min_l3_ports": {
                        "type": "Linear",
                        "bound": self.requirements['min_ports_per_l3'],
                        "description": "Minimum number of ports required for L3 stations"
                    }
                },
                "logical": [
                    "Cannot have both L2 and L3 stations at same location",
                    "L2 ports can only be added to L2 or L3 stations",
                    "L3 ports can only be added to L3 stations",
                    "New stations must have at least one port"
                ]
            },
            
            "objective": {
                "type": "Multi-objective Linear",
                "components": {
                    "l3_coverage": {
                        "weight": self.weights['l3_coverage'],
                        "description": "Maximize population coverage by L3 chargers"
                    },
                    "l2_coverage": {
                        "weight": self.weights['l2_coverage'],
                        "description": "Maximize population coverage by L2 chargers"
                    },
                    "cost": {
                        "weight": self.weights['cost'],
                        "description": "Minimize total cost"
                    }
                }
            },
            
            "parameters": {
                "costs": self.costs,
                "coverage_radii": {
                    "l2_radius": self.coverage['l2_radius'],
                    "l3_radius": self.coverage['l3_radius']
                },
                "infrastructure": {
                    "grid_capacity": self.infrastructure['grid_capacity'],
                    "max_new_ports": self.infrastructure['max_new_ports'],
                    "l2_power": self.infrastructure['l2_power'],
                    "l3_power": self.infrastructure['l3_power']
                }
            }
        }
    
    # ----------------
    # Model Building Methods
    # ----------------
    
    def _create_variables(self):
        """Create optimization variables."""
        n_sites = len(self.potential_sites)
        n_points = len(self.demand_points)
        
        # Decision variables for stations
        self.variables = {
            # Station type decisions
            'new_l2_station': self.model.addVars(
                n_sites, vtype=GRB.BINARY, name="new_l2_station"
            ),
            'new_l3_station': self.model.addVars(
                n_sites, vtype=GRB.BINARY, name="new_l3_station"
            ),
            'upgrade': self.model.addVars(
                n_sites, vtype=GRB.BINARY, name="upgrade"
            ),
            
            # Port decisions
            'new_l2_ports': self.model.addVars(
                n_sites, vtype=GRB.INTEGER, lb=0,
                ub=self.infrastructure['max_new_ports'],
                name="new_l2_ports"
            ),
            'new_l3_ports': self.model.addVars(
                n_sites, vtype=GRB.INTEGER, lb=0,
                ub=self.infrastructure['max_new_ports'],
                name="new_l3_ports"
            ),
            
            # Coverage variables
            'coverage_l2': self.model.addVars(
                n_points, vtype=GRB.BINARY, name="coverage_l2"
            ),
            'coverage_l3': self.model.addVars(
                n_points, vtype=GRB.BINARY, name="coverage_l3"
            ),
            
            # Cost tracking variables
            'upgrade_cost': self.model.addVars(
                n_sites, lb=0, name="upgrade_cost"
            )
        }
     
    def _add_constraints(self):
        """Add all model constraints."""
        self._add_budget_constraints()
        self._add_coverage_constraints()
        self._add_logical_constraints()
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
        
        # Cost component - use variables directly instead of solution costs
        total_cost = gp.quicksum(
            self.variables['new_l2_station'][i] * self.costs['l2_station'] +
            self.variables['new_l2_ports'][i] * self.costs['l2_port'] +
            self.variables['upgrade_cost'][i]
            for i in range(len(self.potential_sites))
        )
        
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

    def _add_budget_constraints(self):
        """Add budget constraints including costs and resale revenue."""
        # Calculate total costs
        new_l2_station_costs = gp.quicksum(
            self.variables['new_l2_station'][i] * self.costs['l2_station']
            for i in range(len(self.potential_sites))
        )
        
        new_l2_port_costs = gp.quicksum(
            self.variables['new_l2_ports'][i] * self.costs['l2_port']
            for i in range(len(self.potential_sites))
        )
        
        new_l3_station_costs = gp.quicksum(
            self.variables['new_l3_station'][i] * self.costs['l3_station']
            for i in range(len(self.potential_sites))
        )
        
        l3_upgrade_costs = gp.quicksum(
            self.variables['upgrade'][i] * self.costs['l3_station']
            for i in range(len(self.potential_sites))
        )
        
        new_l3_port_costs = gp.quicksum(
            self.variables['new_l3_ports'][i] * self.costs['l3_port']
            for i in range(len(self.potential_sites))
        )
        
        # Calculate resale revenue
        resale_revenue = gp.quicksum(
            self.variables['upgrade'][i] * (
                self.costs['l2_station'] * self.costs['resale_factor'] +
                self.current_ports[i] * self.costs['l2_port'] * self.costs['resale_factor']
            )
            for i in range(len(self.potential_sites))
        )
        
        # Net cost must be within budget
        total_cost = (
            new_l2_station_costs + new_l2_port_costs +
            new_l3_station_costs + l3_upgrade_costs + new_l3_port_costs
        )
        
        net_cost = total_cost - resale_revenue
        
        self.model.addConstr(
            net_cost <= self.constraints['budget'],
            name="budget"
        )

    def _add_coverage_constraints(self):
        """Add coverage-related constraints."""
        # Add initial coverage directly for both L2 and L3
        for i in range(len(self.demand_points)):
            # Initial L2 coverage contributes to L2 coverage directly
            self.model.addConstr(
                self.variables['coverage_l2'][i] >= self.initial_l2_coverage[i],
                name=f"initial_l2_coverage_{i}"
            )
            
            # Initial L3 coverage contributes to both L2 and L3 coverage
            if self.initial_l3_coverage[i] > 0:
                self.model.addConstr(
                    self.variables['coverage_l2'][i] >= self.initial_l3_coverage[i],
                    name=f"initial_l3_to_l2_coverage_{i}"
                )
                self.model.addConstr(
                    self.variables['coverage_l3'][i] >= self.initial_l3_coverage[i],
                    name=f"initial_l3_coverage_{i}"
                )
        
        # Add coverage from new stations
        for i in range(len(self.demand_points)):
            # Get points within L2 coverage radius
            nearby_l2 = [
                j for j in range(len(self.potential_sites))
                if self.distances[j, i] <= self.coverage['l2_radius']
            ]
            
            if nearby_l2:  # Only add constraint if there are nearby potential sites
                self.model.addConstr(
                    self.variables['coverage_l2'][i] <= self.initial_l2_coverage[i] + 
                    self.initial_l3_coverage[i] +
                    gp.quicksum(self.variables['new_l2_station'][j] +
                               self.variables['new_l3_station'][j]
                               for j in nearby_l2),
                    name=f"l2_coverage_{i}"
                )
            
            # Get points within L3 coverage radius
            nearby_l3 = [
                j for j in range(len(self.potential_sites))
                if self.distances[j, i] <= self.coverage['l3_radius']
            ]
            
            if nearby_l3:  # Only add constraint if there are nearby potential sites
                self.model.addConstr(
                    self.variables['coverage_l3'][i] <= self.initial_l3_coverage[i] +
                    gp.quicksum(self.variables['new_l3_station'][j] +
                               self.variables['upgrade'][j]
                               for j in nearby_l3),
                    name=f"l3_coverage_{i}"
                )
            
            # Minimum Coverage Requirements
        self.model.addConstr(
            gp.quicksum(self.variables['coverage_l2'][i] * 
                       self.demand_points['weight'].iloc[i]
                       for i in range(len(self.demand_points))) >= 
            self.constraints['min_coverage_l2'],
            name="min_l2_coverage"
        )
        
        self.model.addConstr(
            gp.quicksum(self.variables['coverage_l3'][i] * 
                       self.demand_points['weight'].iloc[i]
                       for i in range(len(self.demand_points))) >= 
            self.constraints['min_coverage_l3'],
            name="min_l3_coverage"
        )
    
    def _add_logical_constraints(self):
        """Add logical constraints for infrastructure decisions."""
        for i in range(len(self.potential_sites)):
            # 1. Can't have both L2 and L3 stations at same location
            self.model.addConstr(
                self.variables['new_l2_station'][i] + 
                self.variables['new_l3_station'][i] + 
                self.variables['upgrade'][i] <= 1,
                name=f"station_type_{i}"
            )
            
            # 2. L2 ports can only be added if there's a station (L2 or L3)
            self.model.addConstr(
                self.variables['new_l2_ports'][i] <= 
                (self.variables['new_l2_station'][i] + 
                self.variables['new_l3_station'][i] + 
                self.variables['upgrade'][i]) * 
                self.infrastructure['max_new_ports'],
                name=f"l2_ports_require_station_{i}"
            )
            
            # 3. L3 ports can ONLY be added to L3 stations
            self.model.addConstr(
                self.variables['new_l3_ports'][i] <= 
                (self.variables['new_l3_station'][i] + 
                self.variables['upgrade'][i]) * 
                self.infrastructure['max_new_ports'],
                name=f"l3_ports_require_l3_station_{i}"
            )
            
            # 4. For existing L2 stations being upgraded to L3
            if self.current_ports[i] > 0:  # If this is an existing L2 station
                # When upgrading:
                # - Sell L2 station (get resale value)
                # - Buy L3 station
                # - Keep or sell existing L2 ports
                # - Add new L3 ports if needed
                upgrade_cost = (
                    self.costs['l3_station'] +  # Cost of new L3 station
                    self.variables['new_l3_ports'][i] * self.costs['l3_port'] -  # Cost of new L3 ports
                    (self.costs['l2_station'] * self.costs['resale_factor']) -  # Revenue from selling L2 station
                    (self.current_ports[i] * self.costs['l2_port'] * self.costs['resale_factor'])  # Revenue from selling L2 ports
                )
                
                self.model.addConstr(
                    self.variables['upgrade'][i] * upgrade_cost == self.variables['upgrade_cost'][i],
                    name=f"upgrade_cost_{i}"
                )
            
            # 5. New stations must have at least one port
            self.model.addConstr(
                self.variables['new_l2_ports'][i] >= self.variables['new_l2_station'][i],
                name=f"min_l2_ports_new_station_{i}"
            )
            
            # 6. New L3 stations (including upgrades) must have minimum L3 ports
            self.model.addConstr(
                self.variables['new_l3_ports'][i] >= 
                (self.variables['new_l3_station'][i] + self.variables['upgrade'][i]) * 
                self.requirements['min_ports_per_l3'],
                name=f"min_l3_ports_{i}"
            )

    def _add_grid_constraints(self):
        """Add power grid capacity constraints."""
        for i in range(len(self.potential_sites)):
            # Calculate power demand based on configuration constants
            l2_power = (
                self.variables['new_l2_ports'][i] * 
                self.infrastructure['l2_power'] +
                (1 - self.variables['upgrade'][i]) * 
                self.current_ports[i] * self.infrastructure['l2_power']
            )
            
            l3_power = (
                self.variables['new_l3_ports'][i] * 
                self.infrastructure['l3_power'] +
                self.variables['upgrade'][i] * 
                self.current_ports[i] * self.infrastructure['l3_power']
            )
            
            # Total power must not exceed grid capacity
            self.model.addConstr(
                l2_power + l3_power <= self.infrastructure['grid_capacity'],
                name=f"grid_capacity_{i}"
            )

    # ----------------
    # Calculation Methods
    # ----------------
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _calculate_solution_costs(self):
        """Calculate detailed cost breakdown for solution with exact integer arithmetic."""
        try:
            if not hasattr(self.model, 'Status') or self.model.Status != GRB.OPTIMAL:
                raise ValueError("Model not optimally solved")

            # Initialize counts with proper range checking
            n_sites = len(self.potential_sites)
            
            # New L2 Infrastructure - use integer counting
            new_l2_stations = sum(1 for i in range(n_sites) 
                                if i < len(self.variables['new_l2_station']) 
                                and round(self.variables['new_l2_station'][i].X) == 1)
            
            new_l2_ports = sum(int(round(self.variables['new_l2_ports'][i].X))
                            for i in range(n_sites)
                            if i < len(self.variables['new_l2_ports']))
            
            # New L3 stations (direct placements)
            new_l3_stations = sum(1 for i in range(n_sites)
                                if i < len(self.variables['new_l3_station'])
                                and round(self.variables['new_l3_station'][i].X) == 1)
            
            # Upgrades (L2 to L3 conversions)
            upgrades = sum(1 for i in range(n_sites)
                        if i < len(self.variables['upgrade'])
                        and round(self.variables['upgrade'][i].X) == 1)
            
            # New L3 ports
            new_l3_ports = sum(int(round(self.variables['new_l3_ports'][i].X))
                            for i in range(n_sites)
                            if i < len(self.variables['new_l3_ports']))

            # Calculate costs using integer multiplication
            costs = {
                'new_infrastructure': {
                    'l2_stations': {
                        'count': new_l2_stations,
                        'cost': new_l2_stations * self.costs['l2_station']
                    },
                    'l2_ports': {
                        'count': new_l2_ports,
                        'cost': new_l2_ports * self.costs['l2_port']
                    },
                    'l3_stations_new': {
                        'count': new_l3_stations,
                        'cost': new_l3_stations * self.costs['l3_station']
                    },
                    'l3_stations_upgrade': {
                        'count': upgrades,
                        'cost': upgrades * self.costs['l3_station']
                    },
                    'l3_ports': {
                        'count': new_l3_ports,
                        'cost': new_l3_ports * self.costs['l3_port']
                    }
                }
            }
            
            # Calculate resale revenue
            resale_l2_stations = upgrades  # One L2 station sold per upgrade
            resale_l2_ports = 0
            for i in range(n_sites):
                if (i < len(self.variables['upgrade']) and 
                    round(self.variables['upgrade'][i].X) == 1):
                    resale_l2_ports += int(self.current_ports[i])
            
            costs['resale_revenue'] = {
                'l2_stations': {
                    'count': resale_l2_stations,
                    'revenue': resale_l2_stations * self.costs['l2_station'] * self.costs['resale_factor']
                },
                'l2_ports': {
                    'count': resale_l2_ports,
                    'revenue': resale_l2_ports * self.costs['l2_port'] * self.costs['resale_factor']
                }
            }
            
            # Calculate summary
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

    def _calculate_coverage_metrics(self):
        """Calculate coverage metrics for current solution."""
        try:                
            # Calculate initial coverage from model variables
            initial_l2 = self._calculate_coverage_percentage(self.initial_l2_coverage)
            initial_l3 = self._calculate_coverage_percentage(self.initial_l3_coverage)

            # Calculate final coverage from model variables
            final_l2_matrix = np.array([self.variables['coverage_l2'][i].X for i in range(len(self.demand_points))])
            final_l3_matrix = np.array([self.variables['coverage_l3'][i].X for i in range(len(self.demand_points))])

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
            raise
    
    def _calculate_coverage_percentage(self, coverage_matrix: np.ndarray) -> float:
        """Calculate coverage percentage from coverage matrix with null safety."""
        if coverage_matrix is None or len(self.demand_points) == 0:
            return 0.0
            
        try:
            return sum(self._safe_float(coverage_matrix[i], 0.0) * 
                    self._safe_float(self.demand_points['weight'].iloc[i], 0.0)
                    for i in range(len(self.demand_points)))
        except Exception as e:
            self.logger.error(f"Error in coverage calculation: {str(e)}")
            return 0.0
    
    def _calculate_initial_coverage(self, charger_type: str, radius_km: float) -> np.ndarray:
        """Calculate initial coverage matrix for a given charger type with null safety."""
        try:
            coverage = np.zeros(len(self.demand_points))
            type_stations = self.stations_df[self.stations_df['charger_type'] == charger_type]
            
            if len(type_stations) == 0:
                return coverage
                
            for idx in range(len(self.demand_points)):
                demand = self.demand_points.iloc[idx]
                demand_lat = self._safe_float(demand['latitude'])
                demand_lon = self._safe_float(demand['longitude'])
                
                for _, station in type_stations.iterrows():
                    station_lat = self._safe_float(station['latitude'])
                    station_lon = self._safe_float(station['longitude'])
                    
                    if any(x == 0.0 for x in [demand_lat, demand_lon, station_lat, station_lon]):
                        continue
                        
                    dist = haversine(
                        (demand_lat, demand_lon),
                        (station_lat, station_lon)
                    )
                    
                    if dist <= radius_km:
                        coverage[idx] = 1
                        break

            return coverage
            
        except Exception as e:
            self.logger.error(f"Error calculating initial coverage: {str(e)}")
            return np.zeros(len(self.demand_points))
    
    def update_station_statuses(self, stations_df: pd.DataFrame, solution: Dict) -> tuple[pd.DataFrame, dict]:
        """Update station statuses with rich metadata and error handling."""
        stations = stations_df.copy()
        stations['status'] = 'Existing L2'
        mask = stations['charger_type'].str.contains('Level 3', na=False)
        stations.loc[mask, 'status'] = 'Existing L3'
        
        detailed_stations = {
            'existing': [],
            'upgrades': [],
            'new': []
        }
        
        # Process existing stations
        for idx, station in stations.iterrows():
            try:
                num_chargers = int(station['num_chargers']) if pd.notna(station['num_chargers']) else 0
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
                        'current_ports': num_chargers,
                        'num_ports': num_chargers,
                        'power_output': self._get_power_output(station['charger_type']),
                        'usage_cost': str(station.get('usage_cost', 'Unknown'))
                    },
                    'operator': str(station.get('operator', 'Unknown'))
                }
                detailed_stations['existing'].append(existing_station)
            except Exception as e:
                self.logger.error(f"Error processing existing station {idx}: {str(e)}")

        # Process upgrades
        if solution.get('upgrades'):
            for upgrade_idx in solution['upgrades']:
                try:
                    if upgrade_idx < len(self.potential_sites):
                        site = self.potential_sites.iloc[upgrade_idx]
                        num_l3_ports = solution.get('new_ports', {}).get('l3', {}).get(upgrade_idx, 2)
                        current_ports = int(self.current_ports[upgrade_idx]) if hasattr(self, 'current_ports') else 0
                        
                        upgrade_data = {
                            'id': f'upgrade_{upgrade_idx}',
                            'name': f"Upgraded L3 Station",
                            'location': {
                                'latitude': float(site['latitude']),
                                'longitude': float(site['longitude']),
                                'type': str(site.get('location_type', 'Unknown')),
                                'address': str(site.get('address', 'Location TBD')),
                                'city': str(site['city']),
                                'postal_code': str(site.get('postal_code', '')),
                                'province': 'Ontario',
                                'score': float(site.get('score', 0)),
                            },
                            'charging': {
                                'status': 'Upgrade to L3',
                                'charger_type': 'Level 3',
                                'current_ports': current_ports,
                                'new_ports': int(num_l3_ports),  # Add this explicitly
                                'num_ports': int(num_l3_ports),
                                'power_output': self._get_power_output('Level 3')
                            },
                            'implementation': {
                                'estimated_timeline': 'Phase 1',
                                'grid_requirements': {
                                    'power_increase': float(self._calculate_power_increase(
                                        'Level 2', 'Level 3', num_l3_ports
                                    )),
                                    'voltage_requirement': '480V'
                                },
                                'estimated_upgrade_cost': float(self._calculate_upgrade_cost(
                                    upgrade_idx, solution
                                ))
                            }
                        }
                        detailed_stations['upgrades'].append(upgrade_data)
                except Exception as e:
                    self.logger.error(f"Error processing upgrade {upgrade_idx}: {str(e)}")

         # Process new stations
        if 'new_stations' in solution:
            # Process L2 stations
            for idx in solution['new_stations'].get('l2', []):
                try:
                    site = self.potential_sites.iloc[idx]
                    num_ports = solution.get('new_ports', {}).get('l2', {}).get(str(idx), 1)
                    
                    new_station = {
                        'id': f'new_L2_{idx}',
                        'name': f"New L2 Station",
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
                            'status': 'New L2',
                            'charger_type': 'Level 2',
                            'current_ports': 0,
                            'new_ports': int(num_ports),
                            'num_ports': int(num_ports),
                            'power_output': self._get_power_output('Level 2')
                        },
                        'implementation': {
                            'estimated_timeline': 'Phase 2',
                            'site_preparation': self._get_site_preparation_requirements(site),
                            'grid_requirements': {
                                'power_requirement': float(self._calculate_power_requirement('Level 2', num_ports)),
                                'voltage_requirement': '240V'
                            },
                            'estimated_installation_cost': float(self._calculate_installation_cost('Level 2', num_ports))
                        }
                    }
                    detailed_stations['new'].append(new_station)
                except Exception as e:
                    self.logger.error(f"Error processing new L2 station {idx}: {str(e)}")

            # Process L3 stations
            for idx in solution['new_stations'].get('l3', []):
                try:
                    site = self.potential_sites.iloc[idx]
                    num_ports = solution.get('new_ports', {}).get('l3', {}).get(str(idx), 2)
                    
                    new_station = {
                        'id': f'new_L3_{idx}',
                        'name': f"New L3 Station",
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
                            'status': 'New L3',
                            'charger_type': 'Level 3',
                            'current_ports': 0,
                            'new_ports': int(num_ports),
                            'num_ports': int(num_ports),
                            'power_output': self._get_power_output('Level 3')
                        },
                        'implementation': {
                            'estimated_timeline': 'Phase 1',
                            'site_preparation': self._get_site_preparation_requirements(site),
                            'grid_requirements': {
                                'power_requirement': float(self._calculate_power_requirement('Level 3', num_ports)),
                                'voltage_requirement': '480V'
                            },
                            'estimated_installation_cost': float(self._calculate_installation_cost('Level 3', num_ports))
                        }
                    }
                    detailed_stations['new'].append(new_station)
                except Exception as e:
                    self.logger.error(f"Error processing new L3 station {idx}: {str(e)}")

        return stations, detailed_stations

    # Helper methods for the enriched data
    def _calculate_power_increase(self, current_type: str, new_type: str, new_ports: int) -> float:
        """Calculate power increase needed for upgrade."""
        # Get current and new power requirements per port
        current_power = self._get_power_output(current_type)['kw']
        new_power = self._get_power_output(new_type)['kw']
        
        # Calculate total power increase
        return (new_power * new_ports) - current_power

    def _calculate_upgrade_cost(self, site_idx: int, solution: Dict) -> float:
        """Calculate the total cost for upgrading a site."""
        # Base cost for L3 upgrade
        base_cost = self.costs['l3_station'] - self.costs['l2_station'] * self.costs['resale_factor']
        
        # New ports cost
        new_ports = solution['new_ports']['l3'].get(site_idx, 0)
        port_cost = new_ports * self.costs['l3_port']
        
        return base_cost + port_cost

    # Updated power output method with more accurate values
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

    # Updated power requirement calculation
    def _calculate_power_requirement(self, charger_type: str, num_ports: int) -> float:
        """Calculate total power requirement."""
        power_per_port = self._get_power_output(charger_type)['kw']
        return power_per_port * num_ports
    
    def _get_site_preparation_requirements(self, site: pd.Series) -> Dict[str, Any]:
        """Determine site preparation requirements based on location type."""
        base_requirements = {
            'parking_modifications': 'Required',
            'signage': 'Required',
            'lighting': 'Required'
        }

        # Cases:
        # - Parking lot: Minimal work, likely existing electrical
        # - Retail: Moderate work, may need electrical upgrade
        # - Commercial: Extensive work, new installation required
        # - Fuel: Extensive work, new installation required
        # - Supermarket: Moderate work, may need electrical upgrade
        # - Food: Extensive work, new installation required
        # - Automotive: Extensive work, new installation required
        # - Charging Station: Minimal work, likely existing electrical
        # - Services: Extensive work, new installation required
        # - Vacant: Extensive work, new installation required
        # - Entertainment: Extensive work, new installation required
        
        case = site.get('location_type', '').lower()
        if 'fuel' in case or 'food' in case or 'automotive' in case or 'services' in case or 'vacant' in case or 'entertainment' in case or 'commercial' in case:
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

    def _calculate_installation_cost(self, charger_type: str, num_ports: int) -> float:
        """Estimate installation cost based on charger type and number of ports."""
        if charger_type == 'Level 2':
            return self.costs['l2_station'] + num_ports * self.costs['l2_port']
        elif charger_type == 'Level 3':
            return self.costs['l3_station'] + num_ports * self.costs['l3_port']

    # ----------------
    # Display Methods
    # ----------------

    def display_sensitivity_results(self, sensitivity_results: Dict[str, Any]) -> None:
        """Display formatted sensitivity analysis results.

        Args:
            sensitivity_results: Dictionary containing sensitivity analysis data
        """
        print(make_header("Sensitivity Analysis".upper(), "="))
        
        # Print insights
        print("\nKey Insights:")
        print("-" * 50)
        for i, insight in enumerate(sensitivity_results['insights'], 1):
            print(f"{i}. {insight}")

        # Print constraint analysis
        print(make_header("Constraint Analysis", "-"))
        print("\nConstraint Analysis:")
        headers = ["Constraint", "Required", "Achieved", "Utilization"]
        constraint_data = []

        for name, values in sensitivity_results['constraints'].items():
            rhs = values['rhs']
            achieved = rhs - values['slack']
            utilization = values['utilization']
            
            if 'Budget' in name:
                required = f"${rhs:,.2f}"
                achieved = f"${achieved:,.2f}"
            else:
                required = f"{rhs*100:.1f}%"
                achieved = f"{(rhs - values['slack'])*100:.1f}%"
            
            constraint_data.append([
                name,
                required,
                achieved,
                f"{utilization:.1f}%"
            ])

        print(tabulate(constraint_data, headers=headers, tablefmt="grid"))

    # ----------------
    # Implementation Plan Methods
    # ----------------
    def get_implementation_plan(self, solution: dict) -> pd.DataFrame:
        """Display implementation plan with detailed breakdown."""
        
        print(make_header("Implementation Plan".upper(), "="))

        print(make_header("Phase Details", "-"))
        print("Phase 1: L2 to L3 Upgrades")
        print("   • Convert existing L2 stations to L3")
        print("   • Install new L3 ports")
        print("   • Upgrade electrical infrastructure")
        
        print("\nPhase 2: New L3 Stations")
        print("   • Install new L3 stations")
        print("   • Add L3 charging ports")
        print("   • Implement grid connections")
        
        print("\nPhase 3: New L2 Network")
        print("   • Install new L2 stations")
        print("   • Add L2 charging ports")
        print("   • Complete network coverage")
        
        print(make_header("Detailed Phase Analysis", "-"))

        # Initialize phase stats
        phases = {
            1: {'actions': 0, 'stations': 0, 'ports': 0, 'cost': 0.0},
            2: {'actions': 0, 'stations': 0, 'ports': 0, 'cost': 0.0},
            3: {'actions': 0, 'stations': 0, 'ports': 0, 'cost': 0.0}
        }
        
        # Phase 1: L2 to L3 Upgrades
        for station in solution['stations']['upgrades']:
            phase = phases[1]
            phase['actions'] += 1 + station['charging']['new_ports']
            phase['stations'] += 1
            phase['ports'] += station['charging']['new_ports']
            phase['cost'] += station['implementation']['estimated_upgrade_cost']
        
        for station in solution['stations']['new']:
            # Phase 2: New L3 Stations
            if station['charging']['charger_type'] == 'Level 3':
                phase = phases[2]
                phase['actions'] += 1 + station['charging']['num_ports']
                phase['stations'] += 1
                phase['ports'] += station['charging']['num_ports']
                phase['cost'] += station['implementation']['estimated_installation_cost']
        
            # Phase 3: New L2 Stations
            if station['charging']['charger_type'] == 'Level 2':
                phase = phases[3]
                phase['actions'] += 1 + station['charging']['num_ports']
                phase['stations'] += 1
                phase['ports'] += station['charging']['num_ports']
                phase['cost'] += station['implementation']['estimated_installation_cost']

        # Create DataFrame with correct index alignment
        phase_summary = pd.DataFrame([phases[i] for i in range(1, 4)],
                                columns=['actions', 'stations', 'ports', 'cost'])
        phase_summary.index = range(1, 4)
        phase_summary.index.name = 'Phase'

        # Rename columns
        phase_summary.columns = ['Actions', 'Stations Modified', 'Ports Added', 'Estimated Cost']

        # Create formatters for each column
        formatters = {
            'Actions': '{:,.0f}',
            'Stations Modified': '{:,.0f}',
            'Ports Added': '{:,.0f}',
            'Estimated Cost': '${:,.2f}'
        }

        # Format each column
        for col, formatter in formatters.items():
            phase_summary[col] = phase_summary[col].apply(lambda x: formatter.format(x))

        # Print the table
        print(tabulate(phase_summary, headers='keys', tablefmt='grid', showindex=True,
                       colalign=('center', 'center', 'center', 'center', 'decimal')))
        
        return phase_summary