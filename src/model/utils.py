from typing import Any, Dict, List, Tuple, Union
import pandas as pd
from tabulate import tabulate

from src.data.utils import *

# Add these color constants at the top of the file
COLORS = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'CYAN': '\033[96m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def colorize(text: str, color: str) -> str:
    """Add color to text if color is valid."""
    return f"{COLORS.get(color, '')}{text}{COLORS['ENDC']}"

def dict_print(d: Dict[str, Any], indent: int = 0) -> List[str]:
    """Recursively format dictionary into list of lines with proper indentation."""
    output = []
    for key, value in d.items():
        # Format key name nicely
        key_name = str(key).replace('_', ' ').title()
        prefix = '  ' * indent + '• ' if indent > 0 else '- '
        
        if isinstance(value, dict):
            output.append(f"{prefix}{key_name}:")
            output.extend(dict_print(value, indent + 1))
        else:
            # Handle different value types appropriately
            if isinstance(value, (int, float)):
                formatted_value = f"{value:,}" if value > 1000 else str(value)
            elif isinstance(value, str):
                formatted_value = value.capitalize()
            elif isinstance(value, list):
                formatted_value = ', '.join(str(x).capitalize() for x in value)
            else:
                formatted_value = str(value)
                
            output.append(f"{prefix}{key_name}: {formatted_value}")
            
    return output

def format_newline(text: List[str]) -> str:
    """Format text with newlines for better readability."""
    return '\n'.join(text) + '\n'

def save_results(data, data_type, file_type, output_dir: Path) -> Path:
    """Save program documentation in both JSON and human-readable formats."""
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Save JSON data: config, program, solution, sensitivity
    if file_type == 'json':
        output_file = output_dir / f'{data_type}.json'
        with open(output_dir / f'{data_type}.json', 'w') as f:
            json.dump(data, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
    
    # Save plots (in the form of PNG images)
    elif file_type == 'png':
        output_file = output_dir / f'{data_type}.png'
        data.savefig(output_file, bbox_inches='tight')
    
    # Save maps (in the form of HTML files)
    elif file_type == 'map':
        output_file = output_dir / f'{data_type}.html'
        data.save(output_file)

    # Save text data: program, solution, sensitivity
    elif file_type == 'txt':
        output_file = output_dir / f'{data_type}.txt'
        with open(output_file, 'w') as f:
            if isinstance(data, list):
                f.write("\n".join(data))
            else:
                f.write(data)
    
    return output_file

def get_program_doc_summary(doc: Dict[str, Any], display: Optional[bool] = False) -> str:
    """
    Return comprehensive documentation of the optimization program.
    
    Documents:
    - Data structure and parameters
    - Port retention strategy
    - Decision variables including port retention
    - Constraints including minimum port requirements
    - Multi-objective optimization approach
    - Implementation considerations
    
    Returns:
        Dict containing complete program documentation
    """
    
    doc_summary = []
    
    # Title
    title = "EV CHARGING NETWORK ENHANCEMENT OPTIMIZATION PROGRAM"
    doc_summary.append(make_header(title, "="))
    
    # 1. Data Summary with subsections
    doc_summary.append(make_header("1. DATA SUMMARY", "-"))
    
    # Network Data
    network_rows = []
    for name, info in doc['data_summary']['network_data'].items():
        if name == 'demand_points':
            # Special formatting for demand points with proper line breaks
            description = (
                "Population demand points\n"
                "Weighted by:\n"
                "  • EV Adoption (35%)\n"
                "  • Infrastructure Quality (25%)\n"
                "  • Population Density (20%)\n"
                "  • Transit Access (15%)\n"
                "  • Infrastructure Age (5%)"
            )
        else:
            description = info['description']
            
        network_rows.append([
            name.replace('_', ' ').title(),
            f"{info['size']:,}",
            description
        ])
    
    network_table = tabulate(
        network_rows,
        headers=['Component', 'Size', 'Description'],
        tablefmt='grid',
        maxcolwidths=[15, 10, OUTPUT_WIDTH - (16 + 11 + 2)],
        colalign=('center', 'center', 'left'),
        stralign='left',
        disable_numparse=True
    )
    doc_summary.extend(["\nNetwork Infrastructure:", network_table, ""])

    # Cost Parameters
    cost_rows = []
    for name, info in doc['data_summary']['cost_parameters'].items():
        cost_rows.append([
            name.replace('_', ' ').title(),
            f"${info['size']:,}" if name != 'resale_factor' else f"{info['size']:.2f}",
            info['description']
        ])
    
    cost_table = tabulate(
        cost_rows,
        headers=['Parameter', 'Value', 'Description'],
        tablefmt='grid',
        maxcolwidths=[15, 12, OUTPUT_WIDTH - (16 + 13 + 2)],
        colalign=('center', 'center', 'left')
    )
    doc_summary.extend(["\nCost Parameters:", cost_table, ""])

    # Technical Parameters
    tech_rows = []
    for section in ['infrastructure_parameters', 'coverage_parameters']:
        for name, info in doc['data_summary'][section].items():
            tech_rows.append([
                name.replace('_', ' ').title(),
                f"{info['size']:,.2f}" if isinstance(info['size'], float) else f"{info['size']:,}",
                info['description']
            ])
    
    tech_table = tabulate(
        tech_rows,
        headers=['Parameter', 'Value', 'Description'],
        tablefmt='grid',
        maxcolwidths=[15, 10, OUTPUT_WIDTH - (16 + 11 + 2)],
        colalign=('center', 'center', 'left')
    )
    doc_summary.extend(["\nTechnical Parameters:", tech_table, ""])
    
    # 2. Decision Variables
    doc_summary.append(make_header("2. DECISION VARIABLES", "-"))
    
    var_rows = []
    for var_name, var_info in doc['decision_variables'].items():
        if var_info['type'] == 'Binary':
                bounds = "[0, 1]"
        elif 'bounds' in var_info:
            bounds = f"[{var_info['bounds'][0]}, {var_info['bounds'][1]}]"
        else:
            bounds = "N/A"
        var_rows.append([
            var_name.replace('_', ' ').title(),
            var_info['type'],
            f"{var_info['dimension']:,}",
            bounds,
            var_info['description']
        ])
    
    var_table = tabulate(
        var_rows,
        headers=['Variable', 'Type', 'Size', 'Bounds', 'Description'],
        tablefmt='grid',
        maxcolwidths=[14, 9, 8, 10, OUTPUT_WIDTH - (15 + 10 + 9 + 11 + 2)],
        colalign=('center', 'center', 'center', 'center', 'left')
    )
    doc_summary.extend([var_table, ""])
    
    # 3. Constraints
    doc_summary.append(make_header("3. CONSTRAINTS", "-"))
    
    constraint_rows = []
    for name, info in doc['constraints'].items():
        if name != 'logical':
            bound_str = f"{info['bound']:,.0f}" if isinstance(info['bound'], (int, float)) and info['bound'] > 999 else str(info['bound'])
            constraint_rows.append([
                name.replace('_', ' ').title(),
                info['type'],
                bound_str,
                info['description']
            ])
    
    constraint_table = tabulate(
        constraint_rows,
        headers=['Constraint', 'Type', 'Bound', 'Description'],
        tablefmt='grid',
        maxcolwidths=[15, 8, 11, OUTPUT_WIDTH - (16 + 9 + 12 + 2)],
        colalign=('center', 'center', 'center', 'left')
    )
    doc_summary.extend([constraint_table, ""])
    
    # Logical constraints with bullet points
    if 'logical' in doc['constraints']:
        doc_summary.append("Logical Constraints:")
        for constraint in doc['constraints']['logical']:
            doc_summary.append(f"• {constraint.capitalize()}")
    doc_summary.append("")
    
    # 4. Objective Function
    doc_summary.append(make_header("4. MULTI-OBJECTIVE FUNCTION", "-"))
    
    obj_rows = []
    for component, info in doc['objective']['components'].items():
        weight_str = f"{info['weight']:.7f}"
        obj_rows.append([
            component.replace('_', ' ').title(),
            weight_str,
            info['description']
        ])
    
    obj_table = tabulate(
        obj_rows,
        headers=['Component', 'Weight', 'Description'],
        tablefmt='grid',
        maxcolwidths=[12, 12, OUTPUT_WIDTH - (13 + 13 + 2)],
        colalign=('center', 'right', 'left'),
        disable_numparse=True  # Important: Prevent number parsing
    )
    doc_summary.extend([obj_table])

    doc_summary = format_newline(doc_summary)

    if display:
        print(doc_summary)
    
    return doc_summary

def get_solution_summary(solution: Dict[str, Any],
                        scenario: Optional[str] = None,
                        display: Optional[bool] = False) -> str:
    """
    Generate comprehensive solution summary with detailed port analysis.
    
    Provides a complete overview of the optimization solution including:
    - Coverage improvements for both L2 and L3
    - Infrastructure changes with port retention details
    - Financial analysis including retained port costs
    - Implementation phasing
    
    Args:
        solution: Complete optimization solution
        scenario: Optional scenario name
        display: Whether to print summary
        
    Returns:
        str: Formatted solution summary
    """
    
    summary = []

    # Results Summary Header
    summary.append(make_header("EV Charging Network Enhancement Optimization Results".upper(), "="))
    summary.append("")

    # Basic Info - right-aligned values with consistent width
    width_label = 30

    summary.append(f"{'Date:':<{width_label}}{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):>{width_label}}".center(OUTPUT_WIDTH))
    summary.append(f"{'Scenario:':<{width_label}}{(scenario or 'Base'):>{width_label}}".center(OUTPUT_WIDTH))
    summary.append(f"{'Status:':<{width_label}}{solution['status']:>{width_label}}".center(OUTPUT_WIDTH))
    summary.append(f"{'Objective Value:':<{width_label}}{solution['objective_value']:>{width_label}.2f}".center(OUTPUT_WIDTH))

    # Coverage Summary
    if 'coverage' in solution:
        summary.append(make_header("Coverage Updates".upper(), "-"))

        # Create coverage comparison table
        headers = ["Coverage Type", "Initial", "Final", "Change"]
        coverage_data = []
        
        # L2 Coverage
        initial_l2 = solution['coverage']['initial']['l2_coverage'] * 100
        final_l2 = solution['coverage']['final']['l2_coverage'] * 100
        change_l2 = final_l2 - initial_l2
        
        coverage_data.append([
            "Level 2",
            f"{initial_l2:.2f}%",
            f"{final_l2:.2f}%",
            f"{change_l2:+.2f}%"
        ])

        # L3 Coverage
        initial_l3 = solution['coverage']['initial']['l3_coverage'] * 100
        final_l3 = solution['coverage']['final']['l3_coverage'] * 100
        change_l3 = final_l3 - initial_l3
        
        coverage_data.append([
            "Level 3",
            f"{initial_l3:.2f}%",
            f"{final_l3:.2f}%",
            f"{change_l3:+.2f}%"
        ])

        summary.append(tabulate(
            coverage_data,
            headers=headers,
            tablefmt="grid",
            colalign=("left", "right", "right", "right")
        ))

    # Infrastructure Changes
    summary.append(make_header("Infrastructure Changes".upper(), "-"))
    
    # Calculations and Metrics
    remaining_l2_stations = [s for s in solution['stations']['existing'] if s['charging']['charger_type'] == 'Level 2']
    new_l2_stations = [s for s in solution['stations']['new'] if s['charging']['status'] == 'New L2']
    new_l3_stations = [s for s in solution['stations']['new'] if s['charging']['charger_type'] == 'Level 3']
    l2_to_l3_stations = solution['stations']['upgrades']

    initial_l3_stations = [s for s in solution['stations']['existing'] if s['charging']['charger_type'] == 'Level 3']
    initial_l2_stations = remaining_l2_stations + l2_to_l3_stations

    final_l2_stations = remaining_l2_stations + new_l2_stations
    final_l3_stations = initial_l3_stations + new_l3_stations + l2_to_l3_stations

    initial_l2_ports = sum(s['charging']['ports']['initial']['level_2'] for s in initial_l2_stations)
    initial_l3_ports = sum(s['charging']['ports']['initial']['level_3'] for s in initial_l3_stations)

    l2_ports_new = sum(s['charging']['ports']['final']['level_2'] for s in new_l2_stations)
    l2_ports_sold = sum((s['charging']['ports']['initial']['level_2'] - s['charging']['ports']['final']['level_2'])
                         for s in solution['stations']['upgrades'])
    
    l3_ports_new = sum(s['charging']['ports']['final']['level_3'] for s in solution['stations']['new'])
    l3_ports_upgrade = sum(s['charging']['ports']['final']['level_3'] for s in solution['stations']['upgrades'])

    final_l2_ports = initial_l2_ports + l2_ports_new - l2_ports_sold
    final_l3_ports = initial_l3_ports + l3_ports_new + l3_ports_upgrade

    # Create infrastructure summary table
    headers = ["Category", "Initial", "Final", "Change"]
    infrastructure_data = [
        [   
            
            'L2 Stations',                          # Name
            len([s for s in initial_l2_stations]),  # Initial
            len([s for s in final_l2_stations]),    # Final
            f'{(len([s for s in final_l2_stations]) - len([s for s in initial_l2_stations])):+.0f}' # Change
        ],
        [   
            'L3 Stations',                          # Name
            len([s for s in initial_l3_stations]),  # Initial
            len([s for s in final_l3_stations]),    # Final
            f'{(len([s for s in final_l3_stations]) - len([s for s in initial_l3_stations])):+.0f}' # Change
        ],
        [   
            "L2 Ports",             # Name
            initial_l2_ports,       # Initial 
            final_l2_ports,         # Final
            f'{(final_l2_ports - initial_l2_ports):+.0f}'   # Change
        ],
        [
            "L3 Ports",             # Name
            initial_l3_ports,       # Initial 
            final_l3_ports,         # Final
            f'{(final_l3_ports - initial_l3_ports):+.0f}'   # Change
        ]
    ]

    summary.append(tabulate(
        infrastructure_data,
        headers=headers,
        tablefmt="grid",
        colalign=("left", "center", "center", "center")
    ))

    # Financial Summary
    summary.append(make_header("Financial Summary".upper(), "-"))
    
    costs = solution['costs']
    
    # New Infrastructure Costs
    summary.append("\n1. New Infrastructure Costs")
    infra_headers = ["Type", "Stations", "Chargers", "Cost ($)"]
    infra_data = [
        [
            "Level 2 (New)",                                                # Type
            costs['new_infrastructure']['l2_stations']['count'],            # Stations
            l2_ports_new,                                                   # Chargers
            f"{costs['new_infrastructure']['l2_stations']['cost']:,.2f}"    # Cost
        ],
        [
            "Level 3 (New)",                                                    # Type
            costs['new_infrastructure']['l3_stations_new']['count'],            # Stations
            l3_ports_new,                                                        # Chargers
            f"{costs['new_infrastructure']['l3_stations_new']['cost']:,.2f}"    # Cost
        ],
        [
            "Level 3 (Upgrades)",                                                   # Type
            costs['new_infrastructure']['l3_stations_upgrade']['count'],            # Stations
            l3_ports_upgrade,                                                       # Chargers
            f"{costs['new_infrastructure']['l3_stations_upgrade']['cost']:,.2f}"    # Cost
        ]
    ]
    
    summary.append(tabulate(
        infra_data,
        headers=infra_headers,
        tablefmt="grid",
        colalign=("left", "center", "center", "decimal")
    ))

    # Resale Revenue
    summary.append("\n2. Resale Revenue")
    resale_headers = ["Type", "Count", "Revenue ($)"]
    resale_data = [
        [
            "L2 Stations",                                              # Type
            costs['resale_revenue']['l2_stations']['count'],            # Count
            f"{costs['resale_revenue']['l2_stations']['revenue']:,.2f}" # Revenue
        ],
        [
            "L2 Ports",                                                 # Type
            costs['resale_revenue']['l2_ports']['count'],               # Count
            f"{costs['resale_revenue']['l2_ports']['revenue']:,.2f}"    # Revenue
        ]
    ]
    
    summary.append(tabulate(
        resale_data,
        headers=resale_headers,
        tablefmt="grid",
        colalign=("left", "center", "decimal")
    ))

    # Total Financial Impact
    summary.append(make_header("Total Financial Impact", "-"))
    
    # Format financial summary with consistent spacing
    purchase = costs['summary']['total_purchase']
    revenue = costs['summary']['total_revenue']
    net_cost = costs['summary']['net_cost']
    
    # Get maximum value width for clean alignment
    max_value = max(purchase, revenue, net_cost)
    value_width = len(f"{max_value:,.2f}")
    
    summary.append(f"{'Total Purchase:':<{width_label}}  $ {purchase:>{value_width},.2f}".center(OUTPUT_WIDTH))
    summary.append(f"{'Total Revenue:':<{width_label}}  $ {revenue:>{value_width},.2f}".center(OUTPUT_WIDTH))
    summary.append(f"{'Net Cost:':<{width_label}}  $ {net_cost:>{value_width},.2f}".center(OUTPUT_WIDTH))

    summary = format_newline(summary)

    if display:
        print(summary)
    
    return summary

def get_sensitivity_results_summary(sensitivity_results: Dict[str, Any], display: bool = False) -> str:
    """Generate sensitivity analysis summary with improved insights."""
    summary = [make_header("SENSITIVITY ANALYSIS SUMMARY", "=")]

    # Constraint Analysis Summary
    summary.append(make_header("Constraint Analysis", "-"))
    constraint_headers = ["Constraint", "Utilization", "Slack", "RHS", "Status"]
    constraint_data = [
        [
            name,
            f"{values['utilization']:.6f}%",
            f"{values['slack']:+,.6f}",
            f"{values['rhs']:,.2f}",
            values['status']
        ]
        for name, values in sensitivity_results['constraints'].items()
    ]
    summary.append(tabulate(constraint_data, 
                            headers=constraint_headers, 
                            tablefmt="grid",
                            colalign=("left", "decimal", "decimal", "decimal", "center")))

    # Variable Analysis Summary
    summary.append(make_header("Variable Analysis", "-"))
    variable_headers = ["Variable", "Value", "Type"]
    
    # Filter significant variables
    significant_vars = [
        var for var in sensitivity_results['variables']
        if var['value'] > 0.5  # For binary variables
    ]
    
    variabbles_shown = 5
    variable_data = [
        [
            var['variable'],
            f"{var['value']:.2f}",
            var['type']
        ]
        for var in significant_vars[:variabbles_shown]  # Show top most significant variables
    ]

    if variable_data:
        summary.append(tabulate(variable_data, headers=variable_headers, tablefmt="grid"))
        if len(significant_vars) > variabbles_shown:
            summary.append(f"\n(Showing top {variabbles_shown} of {len(significant_vars)} significant variables)")
    else:
        summary.append("No significant variables to display.")

    # Insights Section
    if sensitivity_results['insights']:
        summary.append(make_header("Key Insights", "-"))
        for idx, insight in enumerate(sensitivity_results['insights'], 1):
            summary.append(f"{idx}. {insight}")

    summary = format_newline(summary)

    if display:
        print(summary)
    
    return summary

def get_implementation_plan(solution: Dict[str, Any], display: Optional[bool] = False) -> Tuple[str, List[Dict]]:
    """
    Generate detailed implementation plan with port transition strategy.
    
    Creates phased implementation plan considering:
    - L2 port retention during upgrades
    - New station deployment
    - Grid capacity modifications
    - Service continuity maintenance
    - Cost-effective transition timing
    
    Args:
        solution: Optimization solution
        display: Whether to display plan
        
    Returns:
        Tuple of implementation summary and phase details
    """
    try:
        # Initialize phase data
        phases = []
        
        # Phase 1: L2 to L3 Upgrades
        phase1_stations = solution['stations']['upgrades']
        phase1 = get_phase_data(1, phase1_stations)
        phases.append(phase1)

        # Phase 2: New L3 Stations
        phase2_stations = [s for s in solution['stations']['new'] 
                         if s['charging']['charger_type'] == 'Level 3']
        phase2 = get_phase_data(2, phase2_stations)
        phases.append(phase2)

        # Phase 3: New L2 Network
        phase3_stations = [s for s in solution['stations']['new'] 
                         if s['charging']['charger_type'] == 'Level 2']
        phase3 = get_phase_data(3, phase3_stations)
        phases.append(phase3)

        # Format numbers for display
        for phase in phases:
            phase['Stations Modified'] = f"{phase['Stations Modified']:,}"
            phase['Ports Added'] = f"{phase['Ports Added']:,}"
            phase['Ports Removed'] = f"{phase['Ports Removed']:,}"
            phase['Total Actions'] = f"{phase['Total Actions']:,}"
            phase['Estimated Cost'] = f"{phase['Estimated Cost']:,.2f}"

        # Create implementation summary text
        summary = []
        summary.append(make_header("Implementation Plan".upper(), "="))
        summary.append(make_header("Phase Details", "-"))
        
        # Phase descriptions
        phase_details = {
            1: ("L2 to L3 Upgrades", [
                "Convert existing L2 stations to L3",
                "Install new L3 ports",
                "Upgrade electrical infrastructure"
            ]),
            2: ("New L3 Stations", [
                "Install new L3 stations",
                "Add L3 charging ports",
                "Implement grid connections"
            ]),
            3: ("New L2 Network", [
                "Install new L2 stations",
                "Add L2 charging ports",
                "Complete network coverage"
            ])
        }

        # Add phase descriptions
        for phase_num, (name, tasks) in phase_details.items():
            summary.append(f"\nPhase {phase_num}: {name}")
            for task in tasks:
                summary.append(f"   • {task}")

        # Add phase details table
        phase_df = pd.DataFrame(phases)
        summary.append(make_header("Detailed Phase Analysis", "-"))
        
        # Modify headers to include new line characters
        headers = {
            'Phase': 'Phase',
            'Stations Modified': 'Stations\nModified',
            'Ports Added': 'Ports\nAdded',
            'Ports Removed': 'Ports\nRemoved',
            'Total Actions': 'Total\nActions',
            'Estimated Cost': 'Estimated\nCost ($)'
        }
        
        summary.append(tabulate(
            phase_df.rename(columns=headers),
            headers='keys',
            tablefmt='grid',
            showindex=False,
            colalign=('center', 'center', 'center', 'center', 'center', 'decimal')
        ))

        # Join summary into single string
        summary = format_newline(summary)

        if display:
            print(summary)

        return summary, phases
        
    except Exception as e:
        raise RuntimeError(f"Error creating implementation plan: {str(e)}")

def get_phase_data(phase: int, stations: Dict[str, Any]) -> dict:
    """
    Get implementation plan details for a given phase.

    Computes:
    - Total stations modified
    - Ports added and removed
    - Total actions required
    - Estimated cost of implementation

    Args:
        phase: Phase number
        stations: List of stations for the phase

    Returns:
        Dict containing phase details
    """
    station_changes = len(stations)
    total_changes = 0
    ports_added = 0
    ports_removed = 0
    estimated_cost = 0

    for station in stations:
        """Get the changes in ports for a given station."""
        initial_l2_ports = station['charging']['ports']['initial']['level_2']
        initial_l3_ports = station['charging']['ports']['initial']['level_3']
        final_l2_ports = station['charging']['ports']['final']['level_2']
        final_l3_ports = station['charging']['ports']['final']['level_3']
        
        l2_change = abs(final_l2_ports - initial_l2_ports)
        l3_change = abs(final_l3_ports - initial_l3_ports)

        total_changes += l2_change + l3_change

        ports_added += max(final_l2_ports - initial_l2_ports, 0) + max(final_l3_ports - initial_l3_ports, 0)
        ports_removed += max(initial_l2_ports - final_l2_ports, 0) + max(initial_l3_ports - final_l3_ports, 0)

        estimated_cost += station['implementation']['estimated_installation_cost']
    
    return {
        'Phase': phase,
        'Stations Modified': station_changes,
        'Ports Added': ports_added,
        'Ports Removed': ports_removed,
        'Total Actions': station_changes + total_changes,
        'Estimated Cost': estimated_cost
    }