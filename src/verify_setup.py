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
from data.constants import *

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
                pkg_version = version(pkg_name)
                print(f"✓ {pkg_name} (version {pkg_version})")
            except PackageNotFoundError:
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
    print("\nChecking data files directory structure...")
    
    all_exist = True
    for directory_name, directory in DATA_PATHS.items():
        if directory.exists():
            print(f"✓ {directory_name}")
        else:
            print(f"✗ {directory} missing")
            all_exist = False
            try:
                directory.mkdir(parents=True)
                print(f"  Created {directory}")
            except Exception as e:
                print(f"  Error creating {directory}: {str(e)}")
    
    return all_exist

def check_source_code_files():
    """Check if necessary source code files exist."""
    print("\nChecking source code files...")
        
    all_exist = True
    for file in SOURCE_CODE_FILES:
        if file.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist

def check_api_keys():
    """Check if required API keys are set as environment variables."""
    print("\nChecking API keys...")
    required_api_keys = [
        'OCMAP_API_KEY'
    ]
    
    all_set = True
    for api_key in required_api_keys:
        if os.getenv(api_key):
            print(f"✓ {api_key} is set")
        else:
            print(f"✗ {api_key} is not set")
            all_set = False
    
    return all_set

def check_openstreetmap_api():
    """Checks if the OpenStreetMap API is accessible."""
    print("\nChecking OpenStreetMap API connectivity...")
    try:
        import osmnx as ox
        ox.settings.use_cache=False
        ox.geocode('Kitchener, Ontario, Canada')
        print("✓ OpenStreetMap API is accessible")
        return True
    except Exception as e:
        print(f"✗ OpenStreetMap API error: {str(e)}")
        return False

def check_openchargemap_api():
    """Check if the OpenChargeMap API is accessible."""
    print("\nChecking OpenChargeMap API connectivity...")
    try:
        from archive.api_client import APIClient
        client = APIClient()
        response = client.fetch_charging_stations(limit=1)
        if response and 'status' in response and response['status'] == 'success':
            print("✓ OpenChargeMap API is accessible")
            return True
        else:
            print("✗ OpenChargeMap API returned an unexpected response")
            return False
    except Exception as e:
        print(f"✗ OpenChargeMap API error: {str(e)}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("KW EV Charging Station Optimization - Setup Verification")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_gurobi_license(),
        check_api_keys(),
        check_openstreetmap_api(),
        check_openchargemap_api(),
        check_dependencies(),
        check_directory_structure(),
        check_source_code_files()
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