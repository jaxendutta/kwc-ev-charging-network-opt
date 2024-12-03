"""
Run optimization script for EV charging network enhancement.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import time

from data.data_manager import DataManager
from data.utils import *
from model.network_optimizer import EVNetworkOptimizer
from visualization.optimization_viz import *
from data.constants import RESULTS_DIR, DATA_PATHS
from tabulate import tabulate

# Configure logging - suppress Gurobi messages
logging.getLogger('gurobipy').setLevel(logging.ERROR)

def setup_logging(log_file: str = None):
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def format_program_documentation(doc: Dict[str, Any]) -> str:
    """Format program documentation into human-readable text."""
    summary = []
    summary.append(make_header("EV Charging Network Enhancement Optimization Program".upper(), '='))
    
    # Data Summary
    summary.append(make_header("1. DATA SUMMARY", "-"))
    summary.append(f"\n- Demand Points: {doc['data_summary']['demand_points']}")
    summary.append(f"\n - Existing Stations:")
    summary.append(f"    - Total: {doc['data_summary']['existing_stations']['total']}")
    summary.append(f"    - L2: {doc['data_summary']['existing_stations']['l2_stations']}")
    summary.append(f"    - L3: {doc['data_summary']['existing_stations']['l3_stations']}")
    summary.append(f"\n - Potential Sites: {doc['data_summary']['potential_sites']}")
    
    # Decision Variables
    summary.append(make_header("2. DECISION VARIABLES", "-"))
    for var_name, var_info in doc['decision_variables'].items():
        summary.append(f"\n - {var_name}:")
        summary.append(f"    - Type: {var_info['type']}")
        summary.append(f"    - Size: {var_info['dimension']}")
        if 'bounds' in var_info:
            summary.append(f"    - Bounds: {var_info['bounds']}")
        summary.append(f"    - Description: {var_info['description']}")
    
    # Constraints
    summary.append(make_header("3. CONSTRAINTS", "-"))
    summary.append(f"\n - Budget: ${doc['constraints']['budget']['bound']:,.2f}")
    summary.append(f"\n -  Coverage Requirements:")
    summary.append(f"    - L2: {doc['constraints']['coverage']['l2_coverage']['bound']*100}% within {doc['parameters']['coverage_radii']['l2_radius']}km")
    summary.append(f"    - L3: {doc['constraints']['coverage']['l3_coverage']['bound']*100}% within {doc['parameters']['coverage_radii']['l3_radius']}km")
    summary.append(f"\n - Infrastructure:")
    summary.append(f"    - Grid Capacity: {doc['constraints']['infrastructure']['grid_capacity']['bound']} kW")
    summary.append(f"    - Minimum L3 Ports: {doc['constraints']['infrastructure']['min_l3_ports']['bound']}")
    summary.append(f"\n - Logical Constraints:")
    for constraint in doc['constraints']['logical']:
        summary.append(f"    - {constraint}")
    
    # Objective Function
    summary.append(make_header("4. OBJECTIVE FUNCTION", "-"))
    summary.append("\nMaximize:")
    for component, info in doc['objective']['components'].items():
        summary.append(f"  {info['weight']} * {info['description']}")
    
    return '\n'.join(summary)

def save_to_results(data, data_type, output_dir: Path) -> Path:
    """Save program documentation in both JSON and human-readable formats."""
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    if data_type == 'program':
        # Save JSON version
        output_file = output_dir / f'{data_type}.json'
        with open(output_dir / f'{data_type}.json', 'w') as f:
            json.dump(data, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
        
        # Save human-readable version
        summary = format_program_documentation(data)
        with open(output_dir / 'program.txt', 'w') as f:
            f.write(summary)

    elif data_type == 'solution':
        # Save JSON version
        output_file = output_dir / f'{data_type}.json'
        with open(output_dir / f'{data_type}.json', 'w') as f:
            json.dump(data, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
    
    return output_file

def run_optimization(scenario: Optional[str] = None, output_dir: str = None):
    """
    Run the EV charging network enhancement optimization.
    
    Args:
        scenario: Name of scenario to run (e.g., 'aggressive', 'conservative')
        output_dir: Optional output directory for results
        
    Returns:
        Dict containing optimization results
    """
    # Setup and configuration loading remain the same
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_DIR / Path(output_dir or f'results_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    base_config = "configs/base.json"
    scenario_config = f"configs/scenarios/{scenario}.json" if scenario else None
    config = load_config(base_config, scenario_config)
    
    # Save used configuration
    with open(output_dir / 'config_used.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[ 🛈 Running {(scenario or 'base').capitalize()} Scenario ]")
    
    setup_logging(output_dir / 'optimization.log')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    try:        
        # Data Preparation
        data_mgr = DataManager()
        input_data = data_mgr.prepare_optimization_data()
        print(make_header("✅ Optimization Data Preparation Complete!", "="))
        
        # Add constraints from config
        input_data['constraints'] = {
            'budget': config['budget']['default'],
            'min_coverage_l2': config['coverage']['min_coverage_l2'],
            'min_coverage_l3': config['coverage']['min_coverage_l3'],
            'max_l3_distance': config['coverage']['l3_radius'],
            'max_stations_per_area': config['infrastructure']['max_new_ports']
        }
        
        divider = "\n" + 70 * "=" + "\n"

        # Initialize Optimizer
        print("🤖 Initializing Optimizer...")
        optimizer = EVNetworkOptimizer(input_data, config)

        print(divider)
        # Save program documentation
        print("📝 Documenting Optimization Program...")
        program_doc = optimizer.get_program_documentation()
        program_doc_file = save_to_results(program_doc, 'program', output_dir)
        print(f"✓ Program documentation saved to {program_doc_file}")
        print(divider)

        # Run Optimization
        print("✨ Running Optimization Model...")
        solution = optimizer.optimize()
        
        if solution['status'] != 'optimal':
            raise RuntimeError(f"Optimization failed: {solution.get('message', 'Unknown error')}")
        
        # Save Results
        print("\n💾 Saving Optimization Results...")
        solution_file = save_to_results(solution, 'solution', output_dir)
        print(f"✓ Solution saved to {solution_file}")
        print(divider)
        
        # Perform sensitivity analysis if enabled
        if config.get('run_sensitivity', True):
            print("📊 Running Sensitivity Analysis...")
            sensitivity_results = optimizer.perform_sensitivity_analysis()
            
            # Save sensitivity results
            sensitivity_path = output_dir / 'sensitivity.json'
            with open(sensitivity_path, 'w') as f:
                json.dump(sensitivity_results, f, indent=2)
            
            # Create and save visualization with controlled parameters
            sens_fig = plot_sensitivity_results(sensitivity_results)
            sens_fig.savefig(
                output_dir / 'sensitivity_analysis.png',
                bbox_inches='tight'
            )
            print("✓ Sensitivity analysis complete!")
        
        print(divider)
        
        # Create visualizations
        print("📈 Creating Visualizations...\n")
            
        stations_df = load_latest_file(DATA_PATHS['charging_stations'])
        solution_mapped = solution.copy()
        
        # Results visualization
        fig = plot_optimization_results(solution_mapped, stations_df)
        fig.savefig(output_dir / 'optimization_results.png', dpi=300, bbox_inches='tight')
        
        # Coverage map
        m = create_results_map(solution, config, stations_df)
        m.save(output_dir / 'coverage_map.html')
        
        print(f"💾 Saving Visualizations...")
        print(f"✓ Visualizations saved to {output_dir}")

        # Print Program Documentation
        with open(output_dir / 'program.txt', 'r') as f:
            print(f.read())
        
        # Results Summary
        print(make_header("EV Charging Network Enhancement Optimization Results".upper(), "="))
        print()
        
        # Right-align values with consistent width
        width_label = 30
        width_value = 30

        print(f"{'Date:':<{width_label}}{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):>{width_value}}")
        print(f"{'Scenario:':<{width_label}}{(scenario or 'Base'):>{width_value}}")
        print(f"{'Status:':<{width_label}}{solution['status']:>{width_value}}")
        print(f"{'Objective Value:':<{width_label}}{solution['objective_value']:>{width_value}.2f}")

        if 'coverage' in solution:
            print(make_header("Coverage Updates".upper(), "-"))
            
            initial_l3_coverage = solution['coverage']['initial']['l3_coverage']
            final_l3_coverage = solution['coverage']['final']['l3_coverage']
            initial_l2_coverage = solution['coverage']['initial']['l2_coverage']
            final_l2_coverage = solution['coverage']['final']['l2_coverage']

            headers = ["Station Level", "Radius", "Initial", "Achieved"]
            coverage_data = [
                ["Level 3", f"{int(config['coverage']['l3_radius'] * 1000)}m", color_text(f"{(initial_l3_coverage * 100):.2f}%", 33), color_text(f"{(final_l3_coverage * 100):.2f}%", 32)],
                ["Level 2", f"{int(config['coverage']['l2_radius'] * 1000)}m", color_text(f"{(initial_l2_coverage * 100):.2f}%", 33), color_text(f"{(final_l2_coverage * 100):.2f}%", 32)]
            ]

            print(tabulate(
                coverage_data,
                headers=headers,
                tablefmt="grid",
                stralign="center"
            ))

        print(make_header("Infrastructure Changes".upper(), "-"))

        print("\n1. New Installations\n")
        
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

        print(tabulate(
            new_infrastructure_data, 
            headers=headers, 
            tablefmt="grid", 
            colalign=("left", "center", "decimal")))

        print("\n2. Resale Revenue\n")
        
        headers = ["Infrastructure Type", "Count", "Revenue ($)"]        
        resale_revenue_data = [
            ["L2 Stations", str(solution['costs']['resale_revenue']['l2_stations']['count']), 
            f"{solution['costs']['resale_revenue']['l2_stations']['revenue']:,.2f}"],
            ["L2 Ports", str(solution['costs']['resale_revenue']['l2_ports']['count']), 
            f"{solution['costs']['resale_revenue']['l2_ports']['revenue']:,.2f}"]
        ]

        print(tabulate(
            resale_revenue_data,
            headers=headers,
            tablefmt="grid",
            colalign=("left", "center", "decimal"),
        ))

        print(make_header("Total Financial Summary".upper(), "-"))
        
        # Format financial summary with consistent spacing
        purchase = solution['costs']['summary']['total_purchase']
        revenue = solution['costs']['summary']['total_revenue']
        net_cost = solution['costs']['summary']['net_cost']

        # Get maximum value width for clean alignment
        max_value = max(purchase, revenue, net_cost)
        value_width = len(f"{max_value:,.2f}")

        print(f"{'Total Purchase:':<20}  \033[91m$ {purchase:>{value_width},.2f}\033[0m")   # Red
        print(f"{'Total Revenue:':<20}  \033[92m$ {revenue:>{value_width},.2f}\033[0m")     # Green
        print(f"{'Net Cost:':<20}  \033[93m$ {net_cost:>{value_width},.2f}\033[0m")       # Yellow

        if config.get('run_sensitivity', True):
            optimizer.display_sensitivity_results(sensitivity_results)

        optimizer.get_implementation_plan(solution)
    
        print(make_header("Optimization Complete".upper(), "="))
    
        print("\n💾 We saved your optimization results for you!")
        print(f"Find them here: {output_dir}\n")
        
        return {
            'status': 'success',
            'output_dir': str(output_dir),
            'solution': solution
        }
        
    except Exception as e:
        logger.error(f"❌ Error in optimization: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run EV charging network enhancement optimization'
    )
    parser.add_argument('--scenario', type=str, 
                       help='Scenario to run (e.g., aggressive, conservative)')
    parser.add_argument('--output', type=str, 
                       help='Output directory')
    
    args = parser.parse_args()
    start_time = time.time()
    run_optimization(args.scenario, args.output)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        print(f"[Operation concluded in {int(hours)}h {int(minutes)}m {seconds:.2f}s]\n")
    elif minutes > 0:
        print(f"[Operation concluded in {int(minutes)}m {seconds:.2f}s]\n")
    else:
        print(f"[Operation concluded in {seconds:.2f}s]\n")