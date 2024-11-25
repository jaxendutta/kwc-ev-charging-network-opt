"""
Test suite for EVNetworkOptimizer.
"""

import pytest
import pandas as pd
import numpy as np
from src.model.network_optimizer import EVNetworkOptimizer

# Sample test data
@pytest.fixture
def sample_input_data():
    return {
        'stations': [
            {
                'station_id': i,
                'latitude': 43.4516 + np.random.normal(0, 0.01),
                'longitude': -80.4925 + np.random.normal(0, 0.01),
                'charger_type': 'Level 2' if i < 8 else 'Level 3',
                'num_chargers': np.random.randint(1, 5),
                'can_upgrade': i < 8
            }
            for i in range(10)
        ],
        'coverage': {
            'population_areas': [
                {
                    'latitude': 43.4516 + np.random.normal(0, 0.02),
                    'longitude': -80.4925 + np.random.normal(0, 0.02),
                    'population': np.random.randint(1000, 5000),
                    'ev_density': np.random.uniform(0.01, 0.05)
                }
                for _ in range(5)
            ],
            'target_l3': 0.8,
            'max_l3_distance': 5.0
        },
        'capacity': [
            {
                'cell_id': i,
                'available_capacity': 1000,
                'stations': list(range(max(0, i-2), min(10, i+3)))
            }
            for i in range(5)
        ],
        'costs': {
            'l2_to_l3_upgrade': 50000,
            'new_port_l2': 5000,
            'new_port_l3': 15000,
            'operating_l2': 2000,
            'operating_l3': 5000
        },
        'constraints': {
            'budget': 1000000,
            'target_coverage': 0.8,
            'max_l3_distance': 5.0
        }
    }

def test_optimizer_initialization(sample_input_data):
    """Test optimizer initialization."""
    optimizer = EVNetworkOptimizer(sample_input_data)
    assert optimizer.data == sample_input_data
    assert optimizer.model is not None

def test_station_sets(sample_input_data):
    """Test station set preparation."""
    optimizer = EVNetworkOptimizer(sample_input_data)
    station_sets = optimizer._prepare_station_sets()
    
    assert len(station_sets['all_stations']) == 10
    assert len(station_sets['upgradeable']) == 8
    assert len(station_sets['l3_stations']) == 2
    assert len(station_sets['l2_stations']) == 8

def test_variable_creation(sample_input_data):
    """Test optimization variable creation."""
    optimizer = EVNetworkOptimizer(sample_input_data)
    optimizer._create_variables()
    
    # Check variable counts
    assert len(optimizer.variables['upgrade']) == 8  # Upgradeable stations
    assert len(optimizer.variables['new_ports']) == 10  # All stations
    assert len(optimizer.variables['l3_coverage']) == 5  # Population areas

def test_basic_constraints(sample_input_data):
    """Test basic constraint creation."""
    optimizer = EVNetworkOptimizer(sample_input_data)
    optimizer._create_variables()
    optimizer._add_basic_constraints()
    
    # Check model has constraints
    assert optimizer.model.NumConstrs > 0
    
    # Test budget constraint
    budget = sample_input_data['constraints']['budget']
    cost_vars = optimizer.model.getVars()
    assert any(var.VarName.startswith("budget") for var in optimizer.model.getConstrs())

def test_coverage_constraints(sample_input_data):
    """Test coverage constraint creation."""
    optimizer = EVNetworkOptimizer(sample_input_data)
    optimizer._create_variables()
    optimizer._add_coverage_constraints()
    
    # Check coverage constraints exist
    constrs = optimizer.model.getConstrs()
    assert any(var.ConstrName.startswith("l3_coverage") for var in constrs)
    assert any(var.ConstrName.startswith("min_l3_coverage") for var in constrs)

def test_objective_function(sample_input_data):
    """Test objective function creation."""
    optimizer = EVNetworkOptimizer(sample_input_data)
    optimizer._create_variables()
    optimizer._set_objective_function()
    
    # Check objective is set
    assert optimizer.model.ModelSense == GRB.MAXIMIZE
    assert optimizer.model.NumObj > 0

def test_complete_optimization(sample_input_data):
    """Test complete optimization process."""
    optimizer = EVNetworkOptimizer(sample_input_data)
    solution = optimizer.optimize()
    
    # Check solution structure
    assert solution['status'] == 'optimal'
    assert 'objective_value' in solution
    assert 'upgrades' in solution
    assert 'new_ports' in solution
    assert 'costs' in solution
    
    # Check solution feasibility
    total_cost = solution['costs']['total_cost']
    assert total_cost <= sample_input_data['constraints']['budget']
    
    # Check coverage improvement
    assert solution['final_coverage']['l3_coverage'] >= sample_input_data['coverage']['target_l3']

def test_sensitivity_analysis(sample_input_data):
    """Test sensitivity analysis functionality."""
    optimizer = EVNetworkOptimizer(sample_input_data)
    
    param_ranges = {
        'budget': [800000, 1000000, 1200000],
        'target_coverage': [0.75, 0.80, 0.85],
        'max_l3_distance': [4.0, 5.0, 6.0]
    }
    
    results = optimizer.perform_sensitivity_analysis(param_ranges)
    
    # Check results structure
    assert 'parameter_sensitivity' in results
    assert 'scenario_analysis' in results
    assert 'statistics' in results
    
    # Check parameter variations
    for param in param_ranges:
        assert param in results['parameter_sensitivity']
        assert len(results['parameter_sensitivity'][param]) == len(param_ranges[param])

def test_infeasible_case(sample_input_data):
    """Test handling of infeasible cases."""
    # Make problem infeasible by setting impossible constraints
    sample_input_data['constraints']['budget'] = 1000  # Too low budget
    sample_input_data['constraints']['target_coverage'] = 1.0  # Impossible coverage
    
    optimizer = EVNetworkOptimizer(sample_input_data)
    solution = optimizer.optimize()
    
    assert solution['status'] != 'optimal'
    assert 'message' in solution

def test_edge_cases(sample_input_data):
    """Test edge cases and boundary conditions."""
    # Test with no upgradeable stations
    for station in sample_input_data['stations']:
        station['can_upgrade'] = False
    
    optimizer = EVNetworkOptimizer(sample_input_data)
    solution = optimizer.optimize()
    
    assert solution['status'] == 'optimal'
    assert len(solution['upgrades']) == 0
    
    # Test with no budget
    sample_input_data['constraints']['budget'] = 0
    optimizer = EVNetworkOptimizer(sample_input_data)
    solution = optimizer.optimize()
    
    assert solution['status'] != 'optimal'

def test_data_validation(sample_input_data):
    """Test input data validation."""
    # Test missing required fields
    incomplete_data = sample_input_data.copy()
    del incomplete_data['stations']
    
    with pytest.raises(ValueError):
        EVNetworkOptimizer(incomplete_data)
    
    # Test invalid data types
    invalid_data = sample_input_data.copy()
    invalid_data['stations'] = "not a list"
    
    with pytest.raises(TypeError):
        EVNetworkOptimizer(invalid_data)

# Integration Tests
def test_end_to_end_workflow():
    """Test complete workflow from data loading to result visualization."""
    from src.data.data_manager import DataManager
    from src.visualization.optimization_viz import plot_optimization_results, create_results_map
    
    # Load real data
    data_mgr = DataManager()
    input_data = data_mgr.prepare_optimization_inputs()
    
    # Run optimization
    optimizer = EVNetworkOptimizer(input_data)
    solution = optimizer.optimize()
    
    # Create visualizations
    stations = pd.DataFrame(input_data['stations'])
    fig = plot_optimization_results(solution, stations)
    assert fig is not None
    
    m = create_results_map(solution, stations)
    assert m is not None

if __name__ == '__main__':
    pytest.main([__file__])