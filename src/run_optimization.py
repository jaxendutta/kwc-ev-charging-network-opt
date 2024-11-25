"""
Run optimization script for EV charging network enhancement.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from data.data_manager import DataManager
from model.network_optimizer import EVNetworkOptimizer
from visualization.optimization_viz import (
    plot_optimization_results,
    create_results_map,
    plot_sensitivity_results
)
from data.constants import RESULTS_DIR

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

def run_optimization(config_path: str = None, output_dir: str = None):
    """
    Run the EV charging network enhancement optimization.
    
    Args:
        config_path: Optional path to configuration file
        output_dir: Optional output directory for results
        
    Returns:
        Dict containing:
            - status: 'success' or 'error'
            - output_dir: Path to results directory
            - solution: Optimization solution if successful
            - message: Error message if failed
    """
    # Setup
    print(f"Please be patient, this may take a minute to start...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_DIR / Path(output_dir or f'results_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir / 'optimization.log')
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if config_path:
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {
                'run_sensitivity': True,
                'budget': 2000000,  # $2M default budget
                'min_coverage': 0.8,  # 80% minimum coverage
                'max_l3_distance': 5.0,  # 5km between L3
                'grid_capacity': 350  # kW per site
            }
        
        # Data Preparation
        logger.info("Preparing optimization data...")
        data_mgr = DataManager()
        input_data = data_mgr.prepare_optimization_data()
        logger.info("Data preparation complete.")
        
        # Optimization
        logger.info("Initializing optimizer...")
        optimizer = EVNetworkOptimizer(input_data)
        
        logger.info("Running optimization...")
        with tqdm(total=100, desc="Network Enhancement Optimization") as pbar:
            solution = optimizer.optimize(
                progress_callback=lambda x: pbar.update(x)
            )
        
        if solution['status'] != 'optimal':
            raise RuntimeError(f"Optimization failed: {solution.get('message', 'Unknown error')}")
        
        # Save Results
        solution_path = output_dir / 'solution.json'
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        logger.info(f"Solution saved to {solution_path}")
        
        # Sensitivity Analysis
        if config.get('run_sensitivity', True):
            logger.info("Running sensitivity analysis...")
            param_ranges = {
                'budget': [
                    config['budget'] * 0.8,
                    config['budget'],
                    config['budget'] * 1.2
                ],
                'min_coverage': [0.75, 0.80, 0.85],
                'max_l3_distance': [4.0, 5.0, 6.0],
                'grid_capacity': [250, 350, 450]
            }
            
            sensitivity_results = optimizer.perform_sensitivity_analysis(param_ranges)
            sensitivity_path = output_dir / 'sensitivity.json'
            with open(sensitivity_path, 'w') as f:
                json.dump(sensitivity_results, f, indent=2)
            logger.info(f"Sensitivity analysis saved to {sensitivity_path}")
        
        # Visualization
        logger.info("Creating visualizations...")
        stations_df = pd.DataFrame(input_data['stations'])
        
        # Results visualization
        fig = plot_optimization_results(solution, stations_df)
        fig.savefig(output_dir / 'optimization_results.png', dpi=300, bbox_inches='tight')
        
        # Coverage map
        m = create_results_map(solution, stations_df)
        m.save(output_dir / 'coverage_map.html')
        
        # Sensitivity plots
        if config.get('run_sensitivity', True):
            sens_fig = plot_sensitivity_results(sensitivity_results)
            sens_fig.savefig(output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        
        logger.info(f"All visualizations saved to {output_dir}")
        
        # Results Summary
        print("\n=== Network Enhancement Results ===")
        print("=" * 40)
        print(f"Status: {solution['status']}")
        print(f"Objective Value: {solution['objective_value']:,.2f}")
        print(f"\nEnhancements:")
        print(f"- L2 → L3 Upgrades: {len(solution['upgrades'])}")
        print(f"- New Ports Added: {sum(solution['new_ports'].values())}")
        print(f"\nCoverage:")
        print(f"- Population Coverage: {solution['coverage']['population']:,.1%}")
        print(f"- L3 Coverage: {solution['coverage']['l3']:,.1%}")
        print(f"\nCosts:")
        print(f"- Total Cost: ${solution['costs']['total_cost']:,.2f}")
        print(f"- Upgrade Costs: ${solution['costs']['upgrade_costs']:,.2f}")
        print(f"- Port Addition Costs: ${solution['costs']['port_costs']:,.2f}")
        print(f"\nResults Directory: {output_dir}")
        
        return {
            'status': 'success',
            'output_dir': str(output_dir),
            'solution': solution
        }
        
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run EV charging network enhancement optimization'
    )
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    run_optimization(args.config, args.output)