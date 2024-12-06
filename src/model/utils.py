from typing import Any, Dict, List
import pandas as pd
from tabulate import tabulate, PRESERVE_WHITESPACE
# Set PRESERVE_WHITESPACE to True
PRESERVE_WHITESPACE = True

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
    return'\n'.join(text)

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
    """Format optimization program documentation with compact, readable tables."""
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
                        config: Dict[str, Any],
                        scenario: Optional[str] = None,
                        display: Optional[bool] = False) -> List[str]:
    summary = []

    # Results Summary
    summary.append(make_header("EV Charging Network Enhancement Optimization Results".upper(), "="))
    summary.append("")

    # Right-align values with consistent width
    width_label = 30
    width_value = 30

    summary.append(f"{'Date:':<{width_label}}{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):>{width_value}}")
    summary.append(f"{'Scenario:':<{width_label}}{(scenario or 'Base'):>{width_value}}")
    summary.append(f"{'Status:':<{width_label}}{solution['status']:>{width_value}}")
    summary.append(f"{'Objective Value:':<{width_label}}{solution['objective_value']:>{width_value}.2f}")

    if 'coverage' in solution:
        summary.append(make_header("Coverage Updates".upper(), "-"))

        initial_l3_coverage = solution['coverage']['initial']['l3_coverage']
        final_l3_coverage = solution['coverage']['final']['l3_coverage']
        initial_l2_coverage = solution['coverage']['initial']['l2_coverage']
        final_l2_coverage = solution['coverage']['final']['l2_coverage']

        headers = ["Station Level", "Radius", "Initial", "Achieved"]
        coverage_data = [
            ["Level 3", f"{int(config['coverage']['l3_radius'] * 1000)}m", color_text(f"{(initial_l3_coverage * 100):.2f}%", 33), color_text(f"{(final_l3_coverage * 100):.2f}%", 32)],
            ["Level 2", f"{int(config['coverage']['l2_radius'] * 1000)}m", color_text(f"{(initial_l2_coverage * 100):.2f}%", 33), color_text(f"{(final_l2_coverage * 100):.2f}%", 32)]
        ]

        summary.append(tabulate(
            coverage_data,
            headers=headers,
            tablefmt="grid",
            stralign="center"
        ))

    summary.append(make_header("Infrastructure Changes".upper(), "-"))

    summary.append("\n1. New Installations\n")

    headers = ["Infrastructure Type", "Count", "Cost ($)"]
    new_infrastructure_data = [
        ["L2 Stations", str(solution['costs']['new_infrastructure']['l2_stations']['count']), 
        f"{solution['costs']['new_infrastructure']['l2_stations']['cost']:,.2f}"],
        ["L2 Ports", str(solution['costs']['new_infrastructure']['l2_ports']['count']), 
        f"{solution['costs']['new_infrastructure']['l2_ports']['cost']:,.2f}"],
        ["New L3 Stations", str(solution['costs']['new_infrastructure']['l3_stations_new']['count']), 
        f"{solution['costs']['new_infrastructure']['l3_stations_new']['cost']:,.2f}"],
        ["L2 -> L3 Stations", str(solution['costs']['new_infrastructure']['l3_stations_upgrade']['count']), 
        f"{solution['costs']['new_infrastructure']['l3_stations_upgrade']['cost']:,.2f}"],
        ["L3 Ports", str(solution['costs']['new_infrastructure']['l3_ports']['count']), 
        f"{solution['costs']['new_infrastructure']['l3_ports']['cost']:,.2f}"]
    ]

    summary.append(tabulate(
        new_infrastructure_data, 
        headers=headers, 
        tablefmt="grid", 
        colalign=("left", "center", "decimal")))

    summary.append("\n2. Resale Revenue\n")

    headers = ["Infrastructure Type", "Count", "Revenue ($)"]        
    resale_revenue_data = [
        ["L2 Stations", str(solution['costs']['resale_revenue']['l2_stations']['count']), 
        f"{solution['costs']['resale_revenue']['l2_stations']['revenue']:,.2f}"],
        ["L2 Ports", str(solution['costs']['resale_revenue']['l2_ports']['count']), 
        f"{solution['costs']['resale_revenue']['l2_ports']['revenue']:,.2f}"]
    ]

    summary.append(tabulate(
        resale_revenue_data,
        headers=headers,
        tablefmt="grid",
        colalign=("left", "center", "decimal"),
    ))

    summary.append(make_header("Total Financial Summary".upper(), "-"))

    # Format financial summary with consistent spacing
    purchase = solution['costs']['summary']['total_purchase']
    revenue = solution['costs']['summary']['total_revenue']
    net_cost = solution['costs']['summary']['net_cost']

    # Get maximum value width for clean alignment
    max_value = max(purchase, revenue, net_cost)
    value_width = len(f"{max_value:,.2f}")

    summary.append(f"{'Total Purchase:':<20}  \033[91m$ {purchase:>{value_width},.2f}\033[0m")   # Red
    summary.append(f"{'Total Revenue:':<20}  \033[92m$ {revenue:>{value_width},.2f}\033[0m")     # Green
    summary.append(f"{'Net Cost:':<20}  \033[93m$ {net_cost:>{value_width},.2f}\033[0m")         # Yellow

    summary = format_newline(summary)

    if display:
        print(summary)
    
    return summary

def get_sensitivity_results_summary(sensitivity_results: Dict[str, Any], display: Optional[bool] = False) -> None:
    """Format sensitivity analysis results with insights and detailed breakdown."""
    sensitivity_summary = []
    
    # Print header
    sensitivity_summary.append(make_header("Sensitivity Analysis".upper(), "="))

    # Print insights
    sensitivity_summary.append("\nKey Insights:")
    sensitivity_summary.append("-" * 50)
    for i, insight in enumerate(sensitivity_results['insights'], 1):
        sensitivity_summary.append(f"{i}. {insight}")

    # Print constraint analysis
    sensitivity_summary.append(make_header("Constraint Analysis", "-"))
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

    sensitivity_summary.append(tabulate(constraint_data, headers=headers, tablefmt="grid"))
    
    sensitivity_summary = format_newline(sensitivity_summary)

    if display:
        print(sensitivity_summary)
    
    return sensitivity_summary

def get_implementation_plan(solution: dict, display: Optional[bool] = False) -> List:
    """Display implementation plan with detailed breakdown."""

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
    implementation_plan = pd.DataFrame([phases[i] for i in range(1, 4)],
                            columns=['actions', 'stations', 'ports', 'cost'])
    implementation_plan.index = range(1, 4)
    implementation_plan.index.name = 'Phase'

    # Rename columns
    implementation_plan.columns = ['Actions', 'Stations Modified', 'Ports Added', 'Estimated Cost']

    # Create formatters for each column
    formatters = {
        'Actions': '{:,.0f}',
        'Stations Modified': '{:,.0f}',
        'Ports Added': '{:,.0f}',
        'Estimated Cost': '${:,.2f}'
    }

    # Format each column
    for col, formatter in formatters.items():
        implementation_plan[col] = implementation_plan[col].apply(lambda x: formatter.format(x))
    
    implementation_summary = []
    implementation_summary.append(make_header("Implementation Plan".upper(), "="))
    implementation_summary.append(make_header("Phase Details", "-"))
    implementation_summary.append("Phase 1: L2 to L3 Upgrades")
    implementation_summary.append("   • Convert existing L2 stations to L3")
    implementation_summary.append("   • Install new L3 ports")
    implementation_summary.append("   • Upgrade electrical infrastructure")

    implementation_summary.append("\nPhase 2: New L3 Stations")
    implementation_summary.append("   • Install new L3 stations")
    implementation_summary.append("   • Add L3 charging ports")
    implementation_summary.append("   • Implement grid connections")

    implementation_summary.append("\nPhase 3: New L2 Network")
    implementation_summary.append("   • Install new L2 stations")
    implementation_summary.append("   • Add L2 charging ports")
    implementation_summary.append("   • Complete network coverage")
    
    implementation_summary.append(make_header("Detailed Phase Analysis", "-"))

    # Print the table
    implementation_summary.append(tabulate(implementation_plan, headers='keys', tablefmt='grid', showindex=True,
                        colalign=('center', 'center', 'center', 'center', 'decimal')))

    implementation_summary = format_newline(implementation_summary)

    if display:
        print(implementation_summary)
    
    return implementation_summary, implementation_plan.to_dict(orient='records')