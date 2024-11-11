#!/usr/bin/env python3
"""
verify_setup.py - Verifies the setup for the KW EV Charging Station Optimization project.
"""

import sys
import os
import warnings
import platform
from pathlib import Path
import pandas as pd
import numpy as np
from importlib.metadata import version, PackageNotFoundError

# Get project root 
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def check_python_version():
    """Check if Python version meets requirements."""
    required_version = (3, 12, 7)
    current_version = sys.version_info[:3]
    
    print(f"Checking Python version...")
    print(f"Current version: {'.'.join(map(str, current_version))}")
    print(f"Required version: {'.'.join(map(str, required_version))}")
    
    if current_version >= required_version:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python version too old")
        return False

def check_gurobi_license():
    """Check if Gurobi is properly licensed."""
    print("\nChecking Gurobi license...")
    try:
        import gurobipy as gp
        m = gp.Model()
        m.dispose()
        print("✓ Gurobi license OK")
        return True
    except Exception as e:
        print(f"✗ Gurobi license issue: {str(e)}")
        print("Please ensure you have a valid Gurobi license")
        return False

def check_dependencies():
    """Check if all required packages are installed and install if missing."""
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"\n✗ requirements.txt not found at {requirements_file}")
        return False
    
    print("\nChecking package dependencies...")
    with open(requirements_file) as f:
        requirements = f.read().splitlines()
    
    all_satisfied = True
    missing_packages = []
    
    for requirement in requirements:
        if requirement.strip() and not requirement.startswith('#'):
            pkg_requirement = requirement.strip()
            pkg_name = pkg_requirement.split('>=')[0].strip()
            try:
                __import__(pkg_name)
                print(f"✓ {pkg_name} OK")
            except ImportError:
                print(f"✗ {pkg_name} not installed")
                missing_packages.append(pkg_requirement)
                all_satisfied = False
    
    if missing_packages:
        print("\nAttempting to install missing packages...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✓ Successfully installed missing packages")
            all_satisfied = True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing packages: {str(e)}")
            print("Please try installing manually using:")
            print(f"pip install -r {requirements_file}")
            all_satisfied = False
    
    return all_satisfied

def check_directory_structure():
    """Check if the required directory structure exists."""
    print("\nChecking directory structure...")
    required_dirs = [
        PROJECT_ROOT / 'data/raw',
        PROJECT_ROOT / 'data/processed',
        PROJECT_ROOT / 'notebooks',
        PROJECT_ROOT / 'src/models',
        PROJECT_ROOT / 'src/data',
        PROJECT_ROOT / 'src/visualization'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists():
            print(f"✓ {directory.relative_to(PROJECT_ROOT)} exists")
        else:
            print(f"✗ {directory.relative_to(PROJECT_ROOT)} missing")
            all_exist = False
            try:
                directory.mkdir(parents=True)
                print(f"  Created {directory.relative_to(PROJECT_ROOT)}")
            except Exception as e:
                print(f"  Error creating {directory.relative_to(PROJECT_ROOT)}: {str(e)}")
    
    return all_exist

def check_data_files():
    """Check if necessary data files exist."""
    print("\nChecking data files...")
    required_files = [
        PROJECT_ROOT / 'notebooks/01_data_exploration.ipynb',
        PROJECT_ROOT / 'notebooks/02_model_development.ipynb',
        PROJECT_ROOT / 'src/models/facility_location.py',
        PROJECT_ROOT / 'src/data/load_data.py'
    ]
    
    all_exist = True
    for file in required_files:
        if file.exists():
            print(f"✓ {file.relative_to(PROJECT_ROOT)} exists")
        else:
            print(f"✗ {file.relative_to(PROJECT_ROOT)} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("KW EV Charging Station Optimization - Setup Verification")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_gurobi_license(),
        check_dependencies(),
        check_directory_structure(),
        check_data_files()
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("\n✅ All checks passed! Setup is complete.")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()