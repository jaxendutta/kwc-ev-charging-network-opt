"""
Visualization utilities for optimization results.
"""

import folium
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path

from src.visualization.map_viz import *
from src.data.utils import *

def plot_optimization_results(solution: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive visualization of optimization results.
    
    Generates four plots:
    1. Infrastructure Changes:
       - Initial vs final station distribution
       - Upgrade counts with retained ports
       - New station additions
       
    2. Charging Port Distribution:
       - L2 ports (including retained)
       - L3 ports
       - Port distribution changes
       
    3. Coverage Improvement:
       - L2 and L3 coverage progress
       - Population coverage metrics
       - Service accessibility analysis
       
    4. Infrastructure Costs:
       - New station investments
       - Upgrade costs with port retention
       - Equipment resale revenue
       
    Args:
        solution: Optimization solution dictionary
        save_path: Optional path to save visualization
        
    Returns:
        matplotlib Figure object
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Station Status Distribution - Use the new station structure
    station_counts = {
        'Existing L2': sum(1 for s in solution['stations']['existing'] 
                         if s['charging']['ports']['initial']['level_2'] > 0 and s not in solution['stations']['upgrades']),
        'Existing L3': sum(1 for s in solution['stations']['existing']
                         if s['charging']['ports']['initial']['level_3'] > 0 and s not in solution['stations']['upgrades']),
        'Upgrades': len(solution['stations']['upgrades']),
        'New L2': sum(1 for s in solution['stations']['new'] 
                     if s['charging']['ports']['final']['level_2'] > 0),
        'New L3': sum(1 for s in solution['stations']['new']
                     if s['charging']['ports']['final']['level_3'] > 0)
    }
    
    # Plot station distribution
    bars1 = ax1.bar(
        list(station_counts.keys()),
        list(station_counts.values()),
        color=['#3498db', '#9b59b6', '#e74c3c', '#2ecc71', '#f1c40f']
    )
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom'
        )
    
    ax1.set_title('Infrastructure Changes', fontsize=14, pad=20)
    ax1.set_ylabel('Number of Stations')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Port Distribution Over Time
    port_data = {
        'Initial': {
            'L2': sum(s['charging']['ports']['initial']['level_2'] for s in solution['stations']['existing']),
            'L3': sum(s['charging']['ports']['initial']['level_3'] for s in solution['stations']['existing'])
        },
        'Final': {
            'L2': (
                sum(s['charging']['ports']['final']['level_2'] 
                    for s in solution['stations']['existing'] 
                    if s not in solution['stations']['upgrades']) +
                sum(s['charging']['ports']['final']['level_2'] 
                    for s in solution['stations']['upgrades']) +
                sum(s['charging']['ports']['final']['level_2'] 
                    for s in solution['stations']['new'])
            ),
            'L3': (
                sum(s['charging']['ports']['final']['level_3'] 
                    for s in solution['stations']['existing']
                    if s not in solution['stations']['upgrades']) +
                sum(s['charging']['ports']['final']['level_3'] 
                    for s in solution['stations']['upgrades']) +
                sum(s['charging']['ports']['final']['level_3'] 
                    for s in solution['stations']['new'])
            )
        }
    }
    
    # Create grouped bar chart for ports
    x = np.arange(2)
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, [port_data['Initial']['L2'], port_data['Final']['L2']], 
                     width, label='L2 Ports', color='#3498db')
    rects2 = ax2.bar(x + width/2, [port_data['Initial']['L3'], port_data['Final']['L3']], 
                     width, label='L3 Ports', color='#e74c3c')

    # Add value labels
    def autolabel(axis, rects):
        for rect in rects:
            height = rect.get_height()
            axis.text(rect.get_x() + rect.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

    autolabel(ax2, rects1)
    autolabel(ax2, rects2)
    
    ax2.set_title('Charging Port Distribution', fontsize=14, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Initial', 'Final'])
    ax2.legend()

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
        
        x = np.arange(len(stages))
        width = 0.35
        
        rects3 = ax3.bar(x - width/2, l2_coverage, width, 
                        label='L2 Coverage', color='#3498db', alpha=0.7)
        rects4 = ax3.bar(x + width/2, l3_coverage, width,
                        label='L3 Coverage', color='#e74c3c', alpha=0.7)
        
        autolabel(ax3, rects3)
        autolabel(ax3, rects4)
        
        ax3.set_title('Coverage Improvement', fontsize=14, pad=20)
        ax3.set_ylabel('Population Coverage (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stages)
        ax3.legend()

    # 4. Cost Analysis
    if 'costs' in solution:
        costs = solution['costs']
        cost_categories = {
            'New L2\nStations': costs['new_infrastructure']['l2_stations']['cost'],
            'New L3\nStations': costs['new_infrastructure']['l3_stations_new']['cost'],
            'L3\nUpgrades': costs['new_infrastructure']['l3_stations_upgrade']['cost']
        }
        
        bars4 = ax4.bar(
            list(cost_categories.keys()),
            list(cost_categories.values()),
            color=['#2ecc71', '#e74c3c', '#f1c40f']
        )
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'${height:,.0f}',
                ha='center',
                va='bottom'
            )
        
        ax4.set_title('Infrastructure Costs', fontsize=14, pad=20)
        ax4.set_ylabel('Cost ($)')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, y: f'${x:,.0f}'))

    plt.tight_layout()
    
    if isinstance(save_path, (str, Path)) and save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.close(fig)    
    return fig

def create_results_map(solution: Dict[str, Any],
                      config: Dict[str, Any],
                      save_path: Optional[str] = None) -> folium.Map:
    """
    Create interactive map visualization of optimization results.
    
    Features:
    - Existing station visualization
    - Upgrade locations with port retention details
    - New station placements
    - Coverage areas for both L2 and L3
    - Pop-up information showing:
        * Station details
        * Port configurations
        * Implementation timeline
        * Technical specifications
    
    Color coding:
    - Blue: Existing L2 stations
    - Purple: Existing L3 stations
    - Red: L2 to L3 upgrades with retained ports
    - Green: New L2 stations
    - Orange: New L3 stations
    
    Args:
        solution: Optimization solution
        config: Configuration parameters
        save_path: Optional path to save map
        
    Returns:
        Folium map object
    """
    
    # Create base map
    m = create_kwc_map("Enhanced Charging Network Plan", kwc=True)
    
    # Add population density heatmap
    population_data = load_latest_file(DATA_PATHS['population'], 'geojson')
    m = map_population_density(m, population_data, style='heatmap')
    
    # Define colors for each type
    status_colors = {
        'Existing L2': 'blue',
        'Existing L3': 'purple',
        'Upgrade L2 -> L3': 'red',
        'New L2': 'green',
        'New L3': 'orange'
    }
    
    # Create feature groups for different station types
    layers = {status: folium.FeatureGroup(name=str(status)+'Stations')
             for status in status_colors.keys()}
    
    # Initialize counts
    counts = {status: 0 for status in status_colors.keys()}

    # Add coverage visualization
    coverage_l2 = folium.FeatureGroup(name=f"L2 Coverage ({config['coverage']['l2_radius']}km)")
    coverage_l3 = folium.FeatureGroup(name=f"L3 Coverage ({config['coverage']['l3_radius']}km)")

    def ensure_float_string(value):
        """Ensure float values are properly stringified."""
        if isinstance(value, float):
            return f"{value:.8f}"
        return str(value)
    
    def create_station_popup(station_data: Dict[str, Any], 
                           status: str, 
                           implementation: Optional[Dict] = None) -> str:
        """
        Create rich HTML popup content for station visualization.
        
        Displays:
        - Station identification
        - Port configuration details:
            * Initial counts
            * Retained L2 ports
            * New L3 ports
            * Final configuration
        - Implementation details
        - Technical specifications
        
        Args:
            station: Station data
            implementation: Implementation details
            
        Returns:
            Formatted HTML string
        """
        
        # Style settings for popups
        styles = {
            "width": "300px",
            "colors": {
                "background": "#f8f9fa",
                "header": "#2c3e50",
                "subheader": "#34495e",
                "text": "#2c3e50",
                "highlight": "#3498db",
                "muted": "#95a5a6"
            },
            "fonts": {
                "main": "Arial, sans-serif",
                "size": {
                    "normal": "13px",
                    "small": "11px"
                }
            },
            "spacing": {
                "section": "12px",
                "item": "4px",
                "gap": "8px",
                "padding": "8px",
                "margin": "5px 0",
                "border-radius": "4px",
                "header-bottom": "5px"
            }
        }

        loc = station_data['location']

        station = {
            'name': str(station_data.get('name', '')),
            'status': str(status),
            'location': {
                'latitude': str(float(loc['latitude'])),
                'longitude': str(float(loc['longitude'])),
                'address': str(loc.get('address', '')),
                'city': str(loc.get('city', '')),
                'postal_code': str(loc.get('postal_code', ''))
            },
            'power': {
                'kw': str(float(station_data['charging']['power_output']['kw'])),
                'voltage': str(int(station_data['charging']['power_output']['voltage'])),
                'charge_rate': str(station_data['charging']['power_output']['charge_rate'])
            },
            'charging': {
                'charger_type': str(station_data['charging']['charger_type']),
                'status': str(status),
                'ports': {
                    'initial': {
                        'level_2': station_data['charging']['ports']['initial']['level_2'],
                        'level_3': station_data['charging']['ports']['initial']['level_3']
                    },
                    'final': {
                        'level_2': station_data['charging']['ports']['final']['level_2'],
                        'level_3': station_data['charging']['ports']['final']['level_3']
                    }
                }
            }
        }

        power = station['power']
        ports = station['charging']['ports']
        # If station['operator'] is not available, set operator to 'TBD'
        operator = station.get('operator', '[TBD]')

        # Ensure all numeric values are properly stringified
        lat = ensure_float_string(loc['latitude'])
        lon = ensure_float_string(loc['longitude'])
        
        popup_html = f"""
        <div style="font-family: {styles['fonts']['main']}; max-width: {styles['width']}; gap: {styles['spacing']['gap']};">
            <h4 style="color: {styles['colors']['header']}; margin-bottom: {styles['spacing']['section']};">
                {station['name']}
            </h4>

            <div style="background: {styles['colors']['background']}; padding: {styles['spacing']['padding']}; margin: {styles['spacing']['margin']}; border-radius: {styles['spacing']['border-radius']};">
                <h5 style="color: {styles['colors']['subheader']}; bottom: {styles['spacing']['header-bottom']};">Operated by <b>{operator}</b></h5>
            </div>
            
            <!-- Status Section -->
            <div style="background: {styles['colors']['background']}; padding: {styles['spacing']['padding']}; margin: {styles['spacing']['margin']}; border-radius: {styles['spacing']['border-radius']};">
                <h5 style="color: {styles['colors']['subheader']}; bottom: {styles['spacing']['header-bottom']};">Status</h5>
                <p style="margin: {styles['spacing']['item']} 0;">
                    <b>Type:</b> {station['charging']['charger_type']}
                </p>
                <p style="margin: {styles['spacing']['item']} 0;">
                    <b>Status:</b> {status}
                </p>
            </div>

            <!-- Ports Section -->
            <div style="background: {styles['colors']['background']}; padding: {styles['spacing']['padding']}; margin: {styles['spacing']['margin']}; border-radius: {styles['spacing']['border-radius']};">
                <h5 style="color: {styles['colors']['subheader']}; bottom: {styles['spacing']['header-bottom']};">Charging Ports</h5>
                <table style="width: 100%; border-collapse: collapse; font-size: {styles['fonts']['size']['small']};">
                    <tr style="border-bottom: 1px solid {styles['colors']['muted']};">
                        <th style="text-align: left; padding: 4px;">Type</th>
                        <th style="text-align: center; padding: 4px;">Initial</th>
                        <th style="text-align: center; padding: 4px;">Final</th>
                        <th style="text-align: center; padding: 4px;">Change</th>
                    </tr>
                    <tr>
                        <td>Level 2</td>
                        <td style="text-align: center;">{str(ports['initial']['level_2'])}</td>
                        <td style="text-align: center;">{str(ports['final']['level_2'])}</td>
                        <td style="text-align: center;">{str(f"{ports['final']['level_2'] - ports['initial']['level_2']:+d}")}</td>
                    </tr>
                    <tr>
                        <td>Level 3</td>
                        <td style="text-align: center;">{str(ports['initial']['level_3'])}</td>
                        <td style="text-align: center;">{str(ports['final']['level_3'])}</td>
                        <td style="text-align: center;">{str(f"{ports['final']['level_3'] - ports['initial']['level_3']:+d}")}</td>
                    </tr>
                </table>
            </div>

            <!-- Technical Details -->
            <div style="background: {styles['colors']['background']}; padding: {styles['spacing']['padding']}; margin: {styles['spacing']['margin']}; border-radius: {styles['spacing']['border-radius']};">
                <h5 style="color: {styles['colors']['subheader']}; bottom: {styles['spacing']['header-bottom']};">Technical Details</h5>
                <p style="margin: {styles['spacing']['item']} 0;">
                    <b>Power Output:</b> {power['kw']} kW
                </p>
                <p style="margin: {styles['spacing']['item']} 0;">
                    <b>Voltage:</b> {power['voltage']}V
                </p>
                <p style="margin: {styles['spacing']['item']} 0;">
                    <b>Charge Rate:</b> {power['charge_rate']}
                </p>
            </div>
        """
        
        # Add implementation details if available
        if implementation:
            popup_html += f"""
            <div style="background: {styles['colors']['background']}; padding: {styles['spacing']['padding']}; margin: {styles['spacing']['margin']}; border-radius: {styles['spacing']['border-radius']};">
                <h5 style="color: {styles['colors']['subheader']}; bottom: {styles['spacing']['header-bottom']};">Implementation</h5>
                <p style="margin: {styles['spacing']['item']} 0;">
                    <b>Timeline:</b> {implementation.get('estimated_timeline', 'TBD')}
                </p>
                <p style="margin: {styles['spacing']['item']} 0;">
                    <b>Cost:</b> ${implementation.get('estimated_installation_cost', 0):,.2f}
                </p>
                {
                    f"<p style='margin: {styles['spacing']['item']} 0;'><b>Grid Requirement:</b> {implementation['grid_requirements'].get('power_requirement', 0):,.1f} kW</p>"
                    if 'grid_requirements' in implementation else ""
                }
            </div>
            """
        
        address = loc.get('address', '')
        postal_code = loc.get('postal_code', '')
        
        location_section = ""
        if pd.notna(address) or pd.notna(postal_code):
            location_section = f"""
            <!-- Location Section -->
            <div style="background: {styles['colors']['background']}; padding: {styles['spacing']['padding']}; margin: {styles['spacing']['margin']}; border-radius: {styles['spacing']['border-radius']};">
            <h5 style="color: {styles['colors']['subheader']}; bottom: {styles['spacing']['header-bottom']};">Location</h5>
            {f'<p style="margin: {styles['spacing']['item']} 0;">{address}</p>' if address != 'nan' else ''}
            <p style="margin: {styles['spacing']['item']} 0;">
            {loc.get('city', 'Unknown City')}{f', {postal_code}' if postal_code != 'nan' else ''}
            </p>
            <p style="margin: {styles['spacing']['item']} 0; color: {styles['colors']['muted']};">
            ({lat}, {lon})
            </p>
            </div>
            """
        
        popup_html += location_section
        
        return popup_html

    # Process existing stations
    for station in solution['stations']['existing']:
        loc = station['location']
        status = str(station['charging']['status'])
        counts[status] += 1
        
        # Add coverage circle
        coverage_radius = float(config['coverage']['l2_radius'] if station['charging']['charger_type'] == 'Level 2' else config['coverage']['l3_radius'])
        coverage_color = str(status_colors[status])
        
        folium.Circle(
            location=[float(loc['latitude']), float(loc['longitude'])],
            radius=coverage_radius * 1000,  # Convert to meters
            color=coverage_color,
            fill=True,
            opacity=0.1,
            fillOpacity=0.05,
            popup=None
        ).add_to(coverage_l2 if station['charging']['charger_type'] == 'Level 2' else coverage_l3)
        
        # Add station marker
        folium.CircleMarker(
            location=[float(loc['latitude']), float(loc['longitude'])],
            radius=8,
            color=status_colors[status],
            fill=True,
            popup=folium.Popup(
                create_station_popup(station, status),
                max_width=400
            ),
            tooltip=str(f"{station['name']} ({status})")
        ).add_to(layers[status])

    # Process upgrades
    for station in solution['stations']['upgrades']:
        loc = station['location']
        status = 'Upgrade L2 -> L3'
        counts[status] += 1
        
        # Add coverage circle
        coverage_radius = float(config['coverage']['l3_radius'])
        coverage_color = str(status_colors[status])
        
        folium.Circle(
            location=[float(loc['latitude']), float(loc['longitude'])],
            radius=coverage_radius * 1000,
            color=coverage_color,
            fill=True,
            opacity=0.1,
            fillOpacity=0.05,
            popup=None
        ).add_to(coverage_l3)
        
        # Add station marker
        folium.CircleMarker(
            location=[float(loc['latitude']), float(loc['longitude'])],
            radius=8,
            color=str(status_colors[status]),
            fill=True,
            popup=folium.Popup(
                create_station_popup(station, status),
                max_width=400
            ),
            tooltip=str(f"{station['name']} ({status})")
        ).add_to(layers[status])

    # Process new stations
    for station in solution['stations']['new']:
        loc = station['location']
        status = str(station['charging']['status'])
        counts[status] += 1
        
        # Add coverage circle
        coverage_radius = float(config['coverage']['l2_radius'] if station['charging']['charger_type'] == 'Level 2' else config['coverage']['l3_radius'])
        coverage_color = str(status_colors[status])
        
        folium.Circle(
            location=[float(loc['latitude']), float(loc['longitude'])],
            radius=coverage_radius * 1000,
            color=coverage_color,
            fill=True,
            opacity=0.1,
            fillOpacity=0.05,
            popup=None
        ).add_to(coverage_l2 if station['charging']['charger_type'] == 'Level 2' else coverage_l3)
        
        # Add station marker
        folium.CircleMarker(
            location=[float(loc['latitude']), float(loc['longitude'])],
            radius=8,
            color=str(status_colors[status]),
            fill=True,
            popup=folium.Popup(
                create_station_popup(station, status),
                max_width=400
            ),
            tooltip=str(f"{station['name']} ({status})")
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

def plot_sensitivity_analysis(sensitivity_results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize sensitivity results focusing on constraint utilization and slack.
    """
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Constraint Sensitivity Analysis', fontsize=14)

    constraints = list(sensitivity_results['constraints'].keys())
    utilizations = [c['utilization'] for c in sensitivity_results['constraints'].values()]
    slacks = [c['slack'] for c in sensitivity_results['constraints'].values()]

    bars = ax.barh(constraints, utilizations, color='skyblue', label="Utilization (%)")
    ax.set_xlabel('Utilization (%)')
    ax.set_title('Constraint Sensitivity: Utilization and Slack')

    for bar, slack in zip(bars, slacks):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"Slack: {slack:.2f}", va='center', fontsize=10, color='red')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.close(fig)
    return fig

def plot_implementation_plan(plan: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of implementation plan phases.
    
    Shows:
    1. Phase-wise breakdown of:
       - Station modifications
       - Port changes including retention
       - Total actions required
       
    2. Cost distribution showing:
       - New infrastructure investments
       - Upgrade costs with retention
       - Revenue from port resale
       
    Color coding:
    - Green: Total actions
    - Blue: Station modifications
    - Red: Port changes
    
    Args:
        plan: Implementation plan data
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Convert list of dicts to DataFrame for plotting
    phase_data = pd.DataFrame(plan)
    
    # Convert string formatted numbers back to numeric
    numeric_columns = {
        'Stations Modified': lambda x: float(x.replace(',', '')),
        'Ports Added': lambda x: float(x.replace(',', '')),
        'Ports Removed': lambda x: float(x.replace(',', '')),
        'Total Actions': lambda x: float(x.replace(',', '')),
        'Estimated Cost': lambda x: float(x.replace('$', '').replace(',', ''))
    }

    for col, converter in numeric_columns.items():
        phase_data[col] = phase_data[col].apply(converter)

    # Plot 1: Implementation Actions
    x = np.arange(3)  # For 3 phases
    width = 0.15  # Width of bars

    # Create bars for each metric
    bars1 = ax1.bar(x - 1.5 * width, phase_data['Total Actions'], width, 
                   label='Total Actions', color='#2ecc71')
    bars2 = ax1.bar(x - 0.5 * width, phase_data['Stations Modified'], width,
                   label='Stations Modified', color='#3498db')
    bars3 = ax1.bar(x + 0.5 * width, phase_data['Ports Added'], width,
                   label='Ports Added', color='#e74c3c')
    bars4 = ax1.bar(x + 1.5 * width, phase_data['Ports Removed'], width,
                   label='Ports Removed', color='#f1c40f')

    # Customize first plot
    ax1.set_ylabel('Count')
    ax1.set_title('Implementation Actions by Phase')
    ax1.set_xticks(x)
    ax1.set_xticklabels([
        'Phase 1\nL2 to L3 Upgrades',
        'Phase 2\nNew L3 Stations',
        'Phase 3\nNew L2 Network'
    ])
    ax1.legend()

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom'
            )

    # Plot 2: Cost Analysis
    costs = phase_data['Estimated Cost'] / 1_000_000  # Convert to millions
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars5 = ax2.bar(x, costs, width=0.5, color=colors)

    # Customize second plot
    ax2.set_ylabel('Cost (Million $)')
    ax2.set_title('Estimated Cost by Phase')
    ax2.set_xticks(x)
    ax2.set_xticklabels([
        'Phase 1\nL2 to L3 Upgrades',
        'Phase 2\nNew L3 Stations',
        'Phase 3\nNew L2 Network'
    ])

    # Add value labels on cost bars
    for bar in bars5:
        height = bar.get_height()
        ax2.annotate(
            f'${height:.1f}M',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom'
        )

    # Add grids
    ax1.grid(True, alpha=0.3, axis='y')
    ax2.grid(True, alpha=0.3, axis='y')

    # Adjust layout
    plt.tight_layout()

    if isinstance(save_path, (str, Path)) and save_path:
        plt.savefig(save_path)

    plt.close(fig)
    return fig