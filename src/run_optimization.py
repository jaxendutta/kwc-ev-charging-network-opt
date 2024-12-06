"""
Run optimization script for EV charging network enhancement.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import time
from tabulate import tabulate

from data.data_manager import DataManager
from data.utils import *
from model.utils import *
from model.network_optimizer import EVNetworkOptimizer
from visualization.optimization_viz import *
from data.constants import RESULTS_DIR, DATA_PATHS

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
    save_results(config, 'config', 'json', output_dir)
    
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
        program_doc = optimizer.get_program_doc()

        print("\n💾 Saving Program Documentation...")

        program_doc_file = save_results(program_doc, 'program', 'json', output_dir)
        print(f"✓ Program documentation saved to {program_doc_file}")

        program_doc_summary = get_program_doc_summary(program_doc)
        program_doc_summary_file = save_results(program_doc_summary, 'program', 'txt', output_dir)
        print(f"✓ Formatted program documentation saved to {program_doc_summary_file}")

        print(divider)

        # Run Optimization
        print("✨ Running Optimization Model...")
        solution = optimizer.optimize()
        
        if solution['status'] != 'optimal':
            raise RuntimeError(f"Optimization failed: {solution.get('message', 'Unknown error')}")
        
        # Save Results
        print("\n💾 Saving Optimization Results...")
        solution_file = save_results(solution, 'solution', 'json', output_dir)
        print(f"✓ Solution saved to {solution_file}")

        solution_summary = get_solution_summary(solution, config, scenario)
        solution_summary_file = save_results(solution_summary, 'solution_summary', 'txt', output_dir)
        print(f"✓ Solution summary saved to {solution_summary_file}")
        print(divider)

        # Create visualizations
        print("📈 Creating Visualizations...")
            
        stations_df = load_latest_file(DATA_PATHS['charging_stations'])
        solution_mapped = solution.copy()
        
        # Results visualization
        fig = plot_optimization_results(solution_mapped, stations_df)
        print("✓ Optimization results plots created!")

        # Coverage map
        m = create_results_map(solution, config, stations_df)
        print("✓ Coverage map created!")

        # Save visualizations
        print("\n💾 Saving Visualizations...")

        fig_file = save_results(fig, 'solution_plots', 'png', output_dir)
        print(f"✓ Optimization results plots saved to {fig_file}")
        
        map_file = save_results(m, 'coverage_map', 'map', output_dir)
        print(f"✓ Coverage map saved to {map_file}")

        print(divider)
        
        # Perform sensitivity analysis if enabled
        if config.get('run_sensitivity', True):
            print("\n📊 Running Sensitivity Analysis...")
            
            sensitivity_analysis = optimizer.perform_sensitivity_analysis()
            print("✓ Sensitivity analysis complete!")
            
            sensitivity_analysis_plot = plot_sensitivity_analysis(sensitivity_analysis)
            print("✓ Sensitivity analysis plots created!")

            sensitivity_results_summary = get_sensitivity_results_summary(sensitivity_analysis)
            print("✓ Sensitivity analysis results summary created!")

            print("\n💾 Saving Sensitivity Analysis Results...")
            
            sensitivity_results_file = save_results(sensitivity_analysis, 'sensitivity', 'json', output_dir)
            print(f"✓ Sensitivity analysis results saved to {sensitivity_results_file}")

            sensitivity_plot_file = save_results(sensitivity_analysis_plot, 'sensitivity_plot', 'png', output_dir)
            print(f"✓ Sensitivity analysis plots saved to {sensitivity_plot_file}")

            sensitivity_results_summary_file = save_results(sensitivity_results_summary, 'sensitivity_summary', 'txt', output_dir)
            print(f"✓ Sensitivity analysis results summary saved to {sensitivity_results_summary_file}")

        print(divider)
        
        # Implementation Plan
        print("🚧 Creating Implementation Plan...")
        implementation_plan_summary, implementation_plan = get_implementation_plan(solution)
        print("✓ Implementation Plan created!")

        implementation_plot = plot_implementation_plan(implementation_plan)
        print("✓ Implementation Plan plots created!")

        print("\n💾 Saving Implementation Plan...")

        implementation_plan_file = save_results(implementation_plan, 'implementation_plan_details', 'json', output_dir)
        print(f"✓ Implementation Plan saved to {implementation_plan_file}")

        implementation_plan_summary_file = save_results(implementation_plan_summary, 'implementation_plan', 'txt', output_dir)
        print(f"✓ Implementation Plan Summary to {implementation_plan_summary_file}")
               
        implementation_plot_file = save_results(implementation_plot, 'implementation_plan', 'png', output_dir)
        print(f"✓ Implementation Plan plots saved to {implementation_plot_file}")

        # Print Program Documentation
        print(program_doc_summary)
        
        # Print Solution / Results Summary
        print(solution_summary)

        # Print Sensitivity Analysis Summary
        if config.get('run_sensitivity', True):
            print(sensitivity_results_summary)

        # Print Implementation Plan
        print(implementation_plan_summary)
    
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