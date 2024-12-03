"""
Visualization utilities for optimization results.
"""

import folium
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
from pathlib import Path

from src.visualization.map_viz import *
from src.data.utils import *

def plot_optimization_results(solution: Dict[str, Any],
                            save_path: Optional[str] = None) -> plt.Figure:
    """Create comprehensive visualization of optimization results."""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Station Status Distribution
    station_counts = {
        'Initial L2': sum(1 for s in solution['stations']['existing'] 
                         if s['charging']['charger_type'] == 'Level 2'),
        'Initial L3': sum(1 for s in solution['stations']['existing']
                         if s['charging']['charger_type'] == 'Level 3'),
        'Upgrades': len(solution['stations']['upgrades']),
        'New L2': sum(1 for s in solution['stations']['new'] 
                     if s['charging']['charger_type'] == 'Level 2'),
        'New L3': sum(1 for s in solution['stations']['new']
                     if s['charging']['charger_type'] == 'Level 3')
    }
    
    ax1.bar(station_counts.keys(), station_counts.values(), color=['blue', 'purple', 'red', 'green', 'orange'])
    ax1.set_title('Infrastructure Changes')
    ax1.set_ylabel('Number of Stations')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Cost Breakdown
    if 'costs' in solution:
        costs = solution['costs']
        if 'new_infrastructure' in costs:
            cost_data = {
                'New L2 Stations': costs['new_infrastructure']['l2_stations']['cost'],
                'New L2 Ports': costs['new_infrastructure']['l2_ports']['cost'],
                'L3 Stations (new)': costs['new_infrastructure']['l3_stations_new']['cost'],
                'L3 Stations (upgrades)': costs['new_infrastructure']['l3_stations_upgrade']['cost'],
                'New L3 Ports': costs['new_infrastructure']['l3_ports']['cost']
            }
            labels = list(cost_data.keys())
            values = list(cost_data.values())
            
            ax2.bar(labels, values, color='skyblue')
            ax2.set_title('Infrastructure Costs')
            ax2.set_ylabel('Cost ($)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            if 'summary' in costs:
                net_cost = costs['summary']['net_cost']
                ax2.text(0.5, 0.95, f'Net Cost: ${net_cost:,.2f}',
                        transform=ax2.transAxes, ha='center')
    
    # 3. Coverage Improvement
    if 'coverage' in solution:
        stages = ['Initial', 'Final']
        l2_coverage = [
            solution['coverage']['initial']['l2_coverage'] * 100,
            solution['coverage']['final']['l2_coverage'] * 100
        ]
        l3_coverage = [
            solution['coverage']['initial']['l3_coverage'] * 100,
            solution['coverage']['final']['l3_coverage'] * 100
        ]
        
        x = range(len(stages))
        width = 0.35
        ax3.bar([i - width/2 for i in x], l2_coverage, width, 
                label='L2 Coverage', color='blue', alpha=0.6)
        ax3.bar([i + width/2 for i in x], l3_coverage, width, 
                label='L3 Coverage', color='red', alpha=0.6)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(stages)
        ax3.set_title('Coverage Improvement')
        ax3.set_ylabel('Population Coverage (%)')
        ax3.legend()
    
    # 4. Port Distribution
    total_ports = {
        'Initial L2': sum(s['charging']['current_ports'] for s in solution['stations']['existing']
                         if s['charging']['charger_type'] == 'Level 2'),
        'Initial L3': sum(s['charging']['current_ports'] for s in solution['stations']['existing']
                         if s['charging']['charger_type'] == 'Level 3'),
        'New L2': sum(s['charging']['num_ports'] for s in solution['stations']['new']
                     if s['charging']['charger_type'] == 'Level 2'),
        'New L3': sum(s['charging']['num_ports'] for s in solution['stations']['new']
                     if s['charging']['charger_type'] == 'Level 3') +
                  sum(u['charging']['num_ports'] for u in solution['stations']['upgrades'])
    }
            
    ax4.bar(total_ports.keys(), total_ports.values(), color='lightgreen')
    ax4.set_title('Port Distribution')
    ax4.set_ylabel('Number of Ports')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Use isinstance() to properly check save_path type
    if isinstance(save_path, (str, Path)) and save_path:
        plt.savefig(save_path)
        
    plt.close(fig)
    return fig

def create_results_map(solution: Dict[str, Any],
                      config: Dict[str, Any],
                      save_path: Optional[str] = None) -> folium.Map:
    """Create interactive map visualization of optimization results."""
    # Create base map
    m = create_kwc_map("Enhanced Charging Network Plan", kwc=True)
    
    # Add population density heatmap
    population_data = load_latest_file(DATA_PATHS['population'], 'geojson')
    m = map_population_density(m, population_data, style='heatmap')
    
    # Define colors for each type
    status_colors = {
        'Existing L2': 'blue',
        'Existing L3': 'purple',
        'Upgrade to L3': 'red',
        'New L2': 'green',
        'New L3': 'orange',
        # Add lowercase variants
        'New l2': 'green',
        'New l3': 'orange'
    }

    # Card colours
    styles = {
        "colors": {
            "background": "#dadada",
            "header": "#2c3e50",
            "coordinates": "#5c5757"
        },
        "padding": "2.5px 10px",
        "margin": "5px 0",
        "gap": "10px",
        "border_radius": "5px",
        "max_width": "350px",
        "font_family": "Arial"
    }
    
    # Create feature groups for different station types (using uppercase versions)
    layers = {
        'Existing L2': folium.FeatureGroup(name='Existing L2 Stations'),
        'Existing L3': folium.FeatureGroup(name='Existing L3 Stations'),
        'Upgrade to L3': folium.FeatureGroup(name='Upgrade to L3 Stations'),
        'New L2': folium.FeatureGroup(name='New L2 Stations'),
        'New L3': folium.FeatureGroup(name='New L3 Stations')
    }
    
    # Initialize counts (using uppercase versions)
    counts = {
        'Existing L2': 0,
        'Existing L3': 0,
        'Upgrade to L3': 0,
        'New L2': 0,
        'New L3': 0
    }

    # expand_level: Helper function to take in a text string and replace l2 or L2 with Level 2 and l3 or L3 with Level 3
    def expand_level(text: str) -> str:
        level_map = {
            'l2': 'Level 2',
            'L2': 'Level 2',
            'l3': 'Level 3',
            'L3': 'Level 3'
        }

        for k, v in level_map.items():
            text = text.replace(k, v)

        return text

    # Helper function to standardize status
    def standardize_status(status: str) -> str:
        """Standardize status string to handle case variations."""
        status = status.replace('l2', 'L2').replace('l3', 'L3')
        return status
    
    # Add coverage visualization
    coverage_l2 = folium.FeatureGroup(name=f'L2 Coverage ({config["coverage"]["l2_radius"]}km)')
    coverage_l3 = folium.FeatureGroup(name=f'L3 Coverage ({config["coverage"]["l3_radius"]}km)')
    
    # Process existing stations
    for station in solution['stations']['existing']:
        loc = station['location']
        status = station['charging']['status']
        counts[status] += 1
        
        # Add coverage circle
        coverage_radius = config['coverage']['l2_radius'] if station['charging']['charger_type'] == 'Level 2' else config['coverage']['l3_radius']
        coverage_color = 'blue' if station['charging']['charger_type'] == 'Level 2' else 'red'
        
        folium.Circle(
            location=[loc['latitude'], loc['longitude']],
            radius=coverage_radius * 1000,
            color=coverage_color,
            fill=True,
            opacity=0.1,
            fillOpacity=0.05
        ).add_to(coverage_l2 if station['charging']['charger_type'] == 'Level 2' else coverage_l3)
        
        # Enhanced popup content
        power_output = station['charging']['power_output']
        popup_html = f"""
            <div style="font-family: {styles['font_family']}; max-width: {styles['max_width']}; display: flex; flex-direction: column; gap: {styles['gap']};">
            <h4 style="color: {styles['colors']['header']};">{expand_level(station['name'])}</h4>
            <div style="background-color: {styles['colors']['background']}; padding: {styles['padding']}; border-radius: {styles['border_radius']};">
                <p style="margin: {styles['margin']};"><b>Status:</b> {expand_level(status)}</p>
                <p style="margin: {styles['margin']};"><b>Type:</b> {expand_level(station['charging']['charger_type'])}</p>
                <p style="margin: {styles['margin']};"><b>Ports:</b> {station['charging']['current_ports']}</p>
                <p style="margin: {styles['margin']};"><b>Operator:</b> {station['operator']}</p>
            </div>
            <div style="background-color: {styles['colors']['background']}; padding: {styles['padding']}; border-radius: {styles['border_radius']};">
                <p style="margin: {styles['margin']};"><b>Power Output:</b> {power_output['kw']} kW</p>
                <p style="margin: {styles['margin']};"><b>Voltage:</b> {power_output['voltage']}V</p>
                <p style="margin: {styles['margin']};"><b>Charge Rate:</b> {power_output['charge_rate']}</p>
                <p style="margin: {styles['margin']};"><b>Usage Cost:</b> {station['charging'].get('usage_cost', 'Unknown')}</p>
                
            </div>
            <div>
                <p style="margin: {styles['margin']};">{loc['address']}</p>
                <p style="margin: {styles['margin']};">{loc['city']} {loc['postal_code']}</p>
                <p style="margin: {styles['margin']}; color: {styles['colors']['coordinates']};">({loc['latitude']}, {loc['longitude']})</p>
            </div>
            </div>
        """
        
        folium.CircleMarker(
            location=[loc['latitude'], loc['longitude']],
            radius=8,
            color=status_colors[status],
            fill=True,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{expand_level(station['name'] + ' (' + status)})"
        ).add_to(layers[status])
    
    # Process upgrades
    for station in solution['stations']['upgrades']:
        loc = station['location']
        status = 'Upgrade to L3'
        counts[status] += 1
        
        # Add L3 coverage circle
        folium.Circle(
            location=[loc['latitude'], loc['longitude']],
            radius=config['coverage']['l3_radius'] * 1000,
            color='red',
            fill=True,
            opacity=0.1,
            fillOpacity=0.05
        ).add_to(coverage_l3)
        
        # Enhanced popup for upgrades
        power_increase = station['implementation']['grid_requirements']['power_increase']
        popup_html = f"""
            <div style="font-family: {styles['font_family']}; max-width: {styles['max_width']}; display: flex; flex-direction: column; gap: {styles['gap']};">
                <h4 style="color: {styles['colors']['header']}; margin-bottom: {styles['margin']};">{expand_level(station['name'])} (Upgrade)</h4>
                <div style="background-color: {styles['colors']['background']}; padding: {styles['padding']}; border-radius: {styles['border_radius']};">
                    <p style="margin: {styles['margin']};"><b>Current Status:</b> Level 2 → Level 3</p>
                    <p style="margin: {styles['margin']};"><b>Current Ports:</b> {station['charging']['current_ports']}</p>
                    <p style="margin: {styles['margin']};"><b>New L3 Ports:</b> {station['charging']['new_ports']}</p>
                </div>
                <div style="background-color: {styles['colors']['background']}; padding: {styles['padding']}; border-radius: {styles['border_radius']};">
                    <p style="margin: {styles['margin']};"><b>Current Power:</b> {station['charging']['power_output']['kw']} kW</p>
                    <p style="margin: {styles['margin']};"><b>New Power:</b> {station['charging']['power_output']['kw']} kW</p>
                    <p style="margin: {styles['margin']};"><b>Power Increase:</b> {power_increase:.1f} kW</p>
                </div>
                <div style="background-color: {styles['colors']['background']}; padding: {styles['padding']}; border-radius: {styles['border_radius']};">
                    <p style="margin: {styles['margin']};"><b>Timeline:</b> {station['implementation']['estimated_timeline']}</p>
                    <p style="margin: {styles['margin']};"><b>Grid Requirements:</b></p>
                    <ul style="margin: {styles['margin']}; padding-left: 20px;">
                    <li>New Voltage: {station['charging']['power_output']['voltage']}V</li>
                    <li>Estimated Cost: ${station['implementation']['estimated_upgrade_cost']:,.2f}</li>
                    </ul>
                </div>
                <div>
                    <p style="margin: {styles['margin']};">{loc['address']}</p>
                    <p style="margin: {styles['margin']};">{loc['city']} {loc['postal_code']}</p>
                    <p style="margin: {styles['margin']}; color: {styles['colors']['coordinates']};">({loc['latitude']}, {loc['longitude']})</p>
                </div>
            </div>
        """
        
        folium.CircleMarker(
            location=[loc['latitude'], loc['longitude']],
            radius=8,
            color=status_colors[status],
            fill=True,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"Upgrade: {expand_level(station['name'])}"
        ).add_to(layers[status])
    
    # Process new stations
    for station in solution['stations']['new']:
        loc = station['location']
        # Standardize the status to handle case variations
        status = station['charging']['status']
        counts[status] += 1
        
        # Add coverage circle
        coverage_radius = config['coverage']['l2_radius'] if station['charging']['charger_type'] == 'Level 2' else config['coverage']['l3_radius']
        coverage_color = 'blue' if station['charging']['charger_type'] == 'Level 2' else 'red'
        
        folium.Circle(
            location=[loc['latitude'], loc['longitude']],
            radius=coverage_radius * 1000,
            color=coverage_color,
            fill=True,
            opacity=0.1,
            fillOpacity=0.05
        ).add_to(coverage_l2 if station['charging']['charger_type'] == 'Level 2' else coverage_l3)
        
        # Enhanced popup for new stations
        site_prep = station['implementation']['site_preparation']
        grid_reqs = station['implementation']['grid_requirements']
        popup_html = f"""
            <div style="font-family: {styles['font_family']}; max-width: {styles['max_width']}; display: flex; flex-direction: column; gap: {styles['gap']};">
                <h4 style="color: {styles['colors']['header']};">{expand_level(station['name'])}</h4>
                <div style="background-color: {styles['colors']['background']}; padding: {styles['padding']}; border-radius: {styles['border_radius']};">
                    <p style="margin: {styles['margin']};"><b>Type:</b> New {expand_level(station['charging']['charger_type'])}</p>
                    <p style="margin: {styles['margin']};"><b>Planned Ports:</b> {station['charging']['num_ports']}</p>
                    <p style="margin: {styles['margin']};"><b>Location Type:</b> {station['location']['type'].capitalize()}</p>
                </div>
                <div style="background-color: {styles['colors']['background']}; padding: {styles['padding']}; border-radius: {styles['border_radius']};">
                    <p style="margin: {styles['margin']};"><b>Power Output:</b> {station['charging']['power_output']['kw']} kW</p>
                    <p style="margin: {styles['margin']};"><b>Power Requirement:</b> {grid_reqs['power_requirement']:.1f} kW</p>
                    <p style="margin: {styles['margin']};"><b>Voltage:</b> {grid_reqs['voltage_requirement']}</p>
                </div>
                <div style="background-color: {styles['colors']['background']}; padding: {styles['padding']}; border-radius: {styles['border_radius']};">
                    <p style="margin: {styles['margin']};"><b>Timeline:</b> {station['implementation']['estimated_timeline']}</p>
                    <p style="margin: {styles['margin']};"><b>Site Preparation:</b></p>
                    <ul style="margin: {styles['margin']}; padding-left: 20px;">
                    {' '.join(f'<li>{k}: {v}</li>' for k, v in site_prep.items())}
                    </ul>
                    <p style="margin: {styles['margin']};"><b>Installation Cost:</b> ${station['implementation']['estimated_installation_cost']:,.2f}</p>
                </div>
                <div>
                    <p style="margin: {styles['margin']};">{loc['address']}</p>
                    <p style="margin: {styles['margin']};">{loc['city']} {loc['postal_code']}</p>
                    <p style="margin: {styles['margin']}; color: {styles['colors']['coordinates']};">({loc['latitude']}, {loc['longitude']})</p>
                </div>
            </div>
        """
        
        folium.CircleMarker(
            location=[loc['latitude'], loc['longitude']],
            radius=8,
            color=status_colors[status],
            fill=True,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{expand_level(station['name'])}"
        ).add_to(layers[status])
    
    # Add all layers to map in proper order
    coverage_l2.add_to(m)
    coverage_l3.add_to(m)
    for layer in layers.values():
        layer.add_to(m)
    
    # Add legend
    add_legend_to_map(m, status_colors, counts)
    
    if isinstance(save_path, (str, Path)) and save_path:
        m.save(save_path)
        
    return m

def plot_sensitivity_results(sensitivity_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of sensitivity analysis results.
    
    Args:
        sensitivity_results: Dictionary containing sensitivity analysis data
        
    Returns:
        matplotlib Figure object with sensitivity analysis plots
    """
    plt.style.use('seaborn-v0_8-colorblind')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Optimization Sensitivity Analysis', fontsize=14, y=1.05)

    # 1. Constraint utilization plot
    names = []
    utilizations = []
    colors = []
    for name, values in sensitivity_results['constraints'].items():
        names.append(name)
        utilization = values['utilization']
        utilizations.append(utilization)
        # Color based on utilization level
        if utilization >= 99.9:  # Binding
            colors.append('#ff9999')  # Red
        elif utilization >= 90:
            colors.append('#ffcc99')  # Orange
        else:
            colors.append('#99ff99')  # Green

    # Create horizontal bar chart
    bars = ax1.barh(names, utilizations, color=colors)
    ax1.set_title('Constraint Utilization')
    ax1.set_xlabel('Utilization (%)')
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis='x')

    # Add percentage labels on bars
    for bar in bars:
        width = bar.get_width()
        ax1.text(min(width + 1, 98), 
                bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%',
                va='center')

    # 2. Budget breakdown if available
    if 'Budget' in sensitivity_results['constraints']:
        budget_data = sensitivity_results['constraints']['Budget']
        used = budget_data['rhs'] - budget_data['slack']
        remaining = budget_data['slack']
        total = budget_data['rhs']
        
        # Create pie chart
        sizes = [used, remaining]
        labels = [f'Used\n(${used:,.0f})', f'Remaining\n(${remaining:,.0f})']
        colors = ['#ff9999', '#99ff99'] if remaining < 0.1 * total else ['#99ff99', '#e6e6e6']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, labeldistance=1.1, pctdistance=0.8)
        ax2.set_title(f'Budget Utilization\n(Total: ${total:,.0f})')

    plt.tight_layout()

    if isinstance(save_path, (str, Path)) and save_path:
        plt.savefig(save_path)

    plt.close(fig)
    return fig

def plot_implementation_plan(plan: Dict, save_path: Optional[str] = None) -> plt.Figure:
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Convert string formatted numbers back to numeric
    phase_data = plan.copy()
    phase_data['Stations Modified'] = phase_data['Stations Modified'].apply(lambda x: float(x.replace(',', '')))
    phase_data['Ports Added'] = phase_data['Ports Added'].apply(lambda x: float(x.replace(',', '')))
    phase_data['Actions'] = phase_data['Actions'].apply(lambda x: float(x.replace(',', '')))
    phase_data['Estimated Cost'] = phase_data['Estimated Cost'].apply(lambda x: float(x.replace('$', '').replace(',', '')))

    # Plot 1: Infrastructure Changes
    x = np.arange(3)  # For 3 phases
    width = 0.25  # Width of bars

    # Create bars
    bars1 = ax1.bar(x - width, phase_data['Actions'], width, label='Total Actions', color='#2ecc71')
    bars2 = ax1.bar(x, phase_data['Stations Modified'], width, label='Stations Modified', color='#3498db')
    bars3 = ax1.bar(x + width, phase_data['Ports Added'], width, label='Ports Added', color='#e74c3c')

    # Customize first plot
    ax1.set_ylabel('Count')
    ax1.set_title('Implementation Actions by Phase')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Phase 1\nL2 to L3 Upgrades', 'Phase 2\nNew L3 Stations', 'Phase 3\nNew L2 Network'])
    ax1.legend()

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Plot 2: Cost Analysis
    costs = phase_data['Estimated Cost'] / 1_000_000  # Convert to millions
    bars4 = ax2.bar(x, costs, width=0.5, color=['#2ecc71', '#3498db', '#e74c3c'])

    # Customize second plot
    ax2.set_ylabel('Cost (Million $)')
    ax2.set_title('Estimated Cost by Phase')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Phase 1\nL2 to L3 Upgrades', 'Phase 2\nNew L3 Stations', 'Phase 3\nNew L2 Network'])

    # Add value labels on cost bars
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'${height:.1f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Add a grid to both plots
    ax1.grid(True, alpha=0.3, axis='y')
    ax2.grid(True, alpha=0.3, axis='y')

    # Adjust layout
    plt.tight_layout()

    if isinstance(save_path, (str, Path)) and save_path:
        plt.savefig(save_path)

    plt.close(fig)
    return fig