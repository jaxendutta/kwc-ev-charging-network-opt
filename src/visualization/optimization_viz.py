"""
Visualization utilities for optimization results.
"""

import folium
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def plot_optimization_results(solution: Dict[str, Any], 
                            stations: pd.DataFrame,
                            save_path: str = None):
    """
    Create comprehensive visualization of optimization results.
    
    Args:
        solution: Optimization solution dictionary
        stations: DataFrame of station information
        save_path: Optional path to save plots
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Upgrade Decisions
    stations['upgrade_status'] = 'No Change'
    stations.loc[solution['upgrades'], 'upgrade_status'] = 'Upgrade to L3'
    stations.loc[stations['charger_type'] == 'Level 3', 'upgrade_status'] = 'Existing L3'
    
    sns.countplot(data=stations, x='upgrade_status', ax=ax1)
    ax1.set_title('Station Upgrade Decisions')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Cost Breakdown
    costs = pd.Series(solution['costs'])
    costs.plot(kind='bar', ax=ax2)
    ax2.set_title('Cost Breakdown')
    ax2.set_ylabel('Cost ($)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Coverage Improvement
    coverage_data = pd.DataFrame({
        'Stage': ['Before', 'After'],
        'L3 Coverage': [
            solution['initial_coverage']['l3_coverage'],
            solution['final_coverage']['l3_coverage']
        ]
    })
    
    sns.barplot(data=coverage_data, x='Stage', y='L3 Coverage', ax=ax3)
    ax3.set_title('L3 Coverage Improvement')
    ax3.set_ylabel('Population Coverage (%)')
    
    # 4. New Ports Distribution
    new_ports = pd.Series(solution['new_ports']).value_counts()
    new_ports.plot(kind='bar', ax=ax4)
    ax4.set_title('Distribution of New Ports')
    ax4.set_xlabel('Number of New Ports')
    ax4.set_ylabel('Count of Stations')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        
    return fig

def create_results_map(solution: Dict[str, Any],
                      stations: pd.DataFrame,
                      save_path: str = None) -> folium.Map:
    """
    Create interactive map visualization of optimization results.
    
    Args:
        solution: Optimization solution dictionary
        stations: DataFrame of station information
        save_path: Optional path to save HTML map
    
    Returns:
        folium.Map object
    """
    # Create base map centered on KW region
    m = folium.Map(
        location=[43.4516, -80.4925],
        zoom_start=12,
        tiles='cartodbpositron'
    )
    
    # Add stations with different colors based on decisions
    for idx, station in stations.iterrows():
        if idx in solution['upgrades']:
            color = 'red'
            status = 'Upgrade to L3'
        elif station['charger_type'] == 'Level 3':
            color = 'purple'
            status = 'Existing L3'
        else:
            color = 'blue'
            status = 'Remain L2'
            
        # Add new ports information
        new_ports = solution['new_ports'].get(idx, 0)
        
        folium.CircleMarker(
            location=[station['latitude'], station['longitude']],
            radius=8,
            color=color,
            fill=True,
            popup=f"""
                <div style="font-family: Arial; max-width: 200px;">
                    <b>Station {idx}</b><br>
                    Status: {status}<br>
                    Current Type: {station['charger_type']}<br>
                    Current Ports: {station['num_chargers']}<br>
                    New Ports: +{new_ports}
                </div>
            """
        ).add_to(m)
    
    # Add coverage layer
    if 'coverage_areas' in solution:
        coverage_layer = folium.FeatureGroup(name='L3 Coverage')
        for area in solution['coverage_areas']:
            folium.Circle(
                location=[area['latitude'], area['longitude']],
                radius=5000,  # 5km radius
                color='green',
                fill=True,
                fillOpacity=0.1
            ).add_to(coverage_layer)
        coverage_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    if save_path:
        m.save(save_path)
        
    return m

def plot_sensitivity_results(sensitivity_results: Dict[str, Any],
                           save_path: str = None):
    """
    Create visualization of sensitivity analysis results.
    
    Args:
        sensitivity_results: Results from sensitivity analysis
        save_path: Optional path to save plots
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Parameter Sensitivity
    for param, results in sensitivity_results['parameter_sensitivity'].items():
        df = pd.DataFrame(results)
        ax1.plot(
            df['param_value'], 
            df['objective_value'],
            'o-',
            label=param
        )
    
    ax1.set_title('Parameter Sensitivity')
    ax1.set_xlabel('Parameter Value')
    ax1.set_ylabel('Objective Value')
    ax1.legend()
    
    # 2. Scenario Comparison
    scenarios = pd.DataFrame(sensitivity_results['scenario_analysis']).T
    scenarios[['objective_value', 'l3_coverage', 'total_cost']].plot(
        kind='bar',
        ax=ax2
    )
    ax2.set_title('Scenario Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Statistics Summary
    stats = pd.DataFrame(sensitivity_results['statistics']).T
    stats[['mean', 'std']].plot(kind='bar', ax=ax3)
    ax3.set_title('Parameter Statistics')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Parameter Range Impact
    ranges = pd.DataFrame({
        param: stats.loc[param, 'range']
        for param in stats.index
    }, index=['range']).T
    ranges.plot(kind='bar', ax=ax4)
    ax4.set_title('Parameter Range Impact')
    ax4.set_ylabel('Objective Value Range')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        
    return fig