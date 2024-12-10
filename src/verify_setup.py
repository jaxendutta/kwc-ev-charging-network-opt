"""
verify_setup.py - Verifies the setup for the KW EV Charging Station Optimization project.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

# Get project root without importing from package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_WIDTH = 70

def make_header(text: str, char: str, width: int = OUTPUT_WIDTH) -> str:
    """Create a header with a given text and width."""
    return f"\n{char * width}\n{text:^{width}}\n{char * width}"

def check_virtual_env():
    """Check if running in a virtual environment that's in the project root."""
    # Get paths
    project_venv = PROJECT_ROOT / "venv"
    active_venv = Path(sys.prefix)
    base_prefix = Path(sys.base_prefix)

    # Debug information
    print("\n📦 Virtual Environment Check:")
    print(f"Project Virtual Environment path: {project_venv}")
    print(f"Active Environment path: {active_venv}")
    print(f"Base prefix path: {base_prefix}")
    
     # Check if project venv exists
    project_venv_exists = project_venv.exists()

    # Check if project venv is active
    try:
        project_venv_active = project_venv.resolve() == active_venv.resolve()
    except Exception:
        project_venv_active = False
    
    if not project_venv_exists:
        print("\n⚠️  WARNING: No virtual environment found in project root!")
        print("You must create a virtual environment specifically for this project.")
    elif not project_venv_active:
        print("\n⚠️  WARNING: Project's virtual environment is not activated!")
        print(f"Found project venv at: {project_venv}")
        print(f"Currently active environment: {active_venv}")
    
    if not project_venv_active:
        print("\n🚧 It is highly recommended to use a virtual environment for this project.")
        print("Installing packages globally can lead to version conflicts with other projects,")
        print("potentially causing crashes or unexpected behavior!")

        # Windows instructions
        if os.name == 'nt':
            print("\nTo set up a virtual environment in the project root on your system (Windows):")
            print(f"  cd {PROJECT_ROOT}")
            print("  python -m venv venv")
            print("  \\venv\\Scripts\\activate")
            
        # Linux/MacOS instructions
        else:
            print("\nTo set up a virtual environment in the project root on your system (Linux/MacOS):")
            print(f"  cd {PROJECT_ROOT}")
            print("  python3 -m venv venv")
            print("  source venv/bin/activate")

        while True:
            response = input("\nDo you want to proceed without a virtual environment anyway? [y/N]: ").lower()
            if response in ['n', '']:
                print("\nGood choice! Please create and activate the project's virtual environment and try again.\n")
                sys.exit(1)
            elif response == 'y':
                print("\n⚠️  FINAL WARNING: Proceeding without a virtual environment may affect other projects on this system!")
                print("By proceeding, you acknowledge that the author of this script/project is not liable")
                print("for any issues caused to other projects or the system itself.")
                
                response_final = input("\nAre you absolutely sure you want to proceed? [y/N]: ").lower()
                if response_final in ['n', '']:
                    print("\nGood choice! Please create and activate the project's virtual environment and try again.\n")
                    sys.exit(1)
                elif response_final == 'y':
                    print("\nProceeding without proper virtual environment setup...\n")
                    break
                print("Please enter 'y' to proceed or 'n' to exit (default: n)")
            else:
                print("Please enter 'y' to proceed or 'n' to exit (default: n)")

    return project_venv_active

def install_project_package():
    """Install the project package in editable mode."""
    print("\nInstalling project package...")
    try:
        # Run pip install and capture output
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Process output line by line
        show_line = True
        for line in process.stdout:
            line = line.strip()
            
            # Skip lines that start with "Requirement already satisfied"
            if line.startswith("Requirement already satisfied"):
                if show_line:
                    print("Satisfying external package requirements ... done")
                    show_line = False
                continue
                
            # Show all other lines
            if line:
                print(line)
        
        # Check return code
        process.wait()
        if process.returncode == 0:
            print("✓ Project package installed successfully")
            return True
        else:
            error = process.stderr.read()
            print(f"✗ Error installing project package: {error}")
            print("Please try installing manually using:")
            print(f"pip install -e {PROJECT_ROOT}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing project package: {str(e)}")
        print("Please try installing manually using:")
        print(f"pip install -e {PROJECT_ROOT}")
        return False

def check_python_version():
    """Check if Python version meets requirements."""
    required_version = (3, 12, 7)
    current_version = sys.version_info[:3]
    
    print(f"\nChecking Python version...")
    print(f"Current version: {'.'.join(map(str, current_version))}")
    print(f"Required version: {'.'.join(map(str, required_version))}\n")
    
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
        print("\n✓ Gurobi license OK")
        return True
    except Exception as e:
        print(f"\n✗ Gurobi license issue: {str(e)}")
        print("Please ensure you have a valid Gurobi license")
        return False

def parse_requirements(requirements_file: Path) -> dict:
    """Parse requirements.txt into a dictionary of categories and their packages."""
    categories = {}
    current_category = None
    
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):
                # New category
                current_category = line.lstrip('# ')
                categories[current_category] = []
            elif not line.startswith('#') and current_category:
                # Package specification
                categories[current_category].append(line)
    
    return categories

def check_dependencies():
    """Check if all required packages are installed and install if missing."""
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"\n✗ requirements.txt not found at {requirements_file}")
        return False
    
    print("\nChecking package dependencies...")
    
    # Parse requirements file into categories
    try:
        package_categories = parse_requirements(requirements_file)
    except Exception as e:
        print(f"✗ Error parsing requirements.txt: {str(e)}")
        return False
    
    missing_packages = []
    
    # Check each category
    for category, packages in package_categories.items():
        print(f"\n📦 {category}:")
        for package_req in packages:
            pkg_name = package_req.split('>=')[0].strip()
            try:
                pkg_version = version(pkg_name)
                print(f"  ✓ {package_req} (installed: {pkg_version})")
            except PackageNotFoundError:
                print(f"  ✗ {package_req}")
                missing_packages.append((category, package_req))
    
    if missing_packages:
        print(f"\n🚧 Installing {len(missing_packages)} missing packages...")
        print("-" * OUTPUT_WIDTH)

        # First, check if Halo is available or can be installed
        spinner = None
        try:
            import halo
            spinner = halo.Halo(spinner='dots')
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "halo>=0.0.31"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                import halo
                spinner = halo.Halo(spinner='dots')
            except:
                print("Note: Progress indicator (Halo) not available. Continuing with standard output.")
        
        try:
            pip_command = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
            total_to_install = len(missing_packages)
            
            for category in package_categories:
                category_packages = [pkg for cat, pkg in missing_packages if cat == category]
                if category_packages:
                    print(f"\n🔧 Installing missing {category} packages...")
                    for pkg in category_packages:
                        pkg_name = pkg.split('>=')[0]
                        if spinner:
                            spinner.start(f"Installing {pkg_name}")
                            
                        try:
                            subprocess.check_call(
                                pip_command + [pkg],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            installed_version = version(pkg_name)
                            if spinner:
                                spinner.succeed(f"Installed {pkg_name} {installed_version}")
                            else:
                                print(f"✓ Installed {pkg_name} {installed_version}")
                        except subprocess.CalledProcessError as e:
                            if spinner:
                                spinner.fail(f"Failed to install {pkg_name}")
                            else:
                                print(f"✗ Failed to install {pkg_name}")
                            return False
            
            print("\n✨ Successfully installed all missing packages!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error installing packages: {str(e)}")
            print("🔧 Please try installing manually using:")
            print(f"   pip install -r {requirements_file}")
            return False
    
    print("\n✨ All required packages are installed!")
    return True

def check_directory_structure():
    """Check if the required directory structure exists."""
    # Import here after project package is installed
    from data.constants import DATA_PATHS
    
    print("\nChecking data files directory structure...")
    
    all_exist = True
    for directory_name, directory in DATA_PATHS.items():
        if directory.exists():
            print(f"✓   Found: {directory_name}")
        else:
            try:
                directory.mkdir(parents=True)
                print(f"✓ Created: {directory}")
            except Exception as e:
                all_exist = False
                print(f"✗ Error creating missing {directory}: {str(e)}")
    
    return all_exist

def check_source_code_files():
    """Check if necessary source code files exist."""
    # Import here after project package is installed
    from data.constants import SOURCE_CODE_FILES
        
    all_exist = True
    for file_type in SOURCE_CODE_FILES:
        print(f"\n📄 Checking for {file_type} files:")
        for file in SOURCE_CODE_FILES[file_type]:
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

    from dotenv import load_dotenv
    load_dotenv()
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
        print("✓ OpenStreetMap API is accessible!")
        return True
    except Exception as e:
        print(f"✗ OpenStreetMap API error: {str(e)}")
        return False

def check_openchargemap_api():
    """Check if the OpenChargeMap API is accessible."""
    print("\nChecking OpenChargeMap API connectivity...")
    try:
        import requests
        api_key = os.getenv('OCMAP_API_KEY')
        if not api_key:
            print("✗ OpenChargeMap API key is not set. Please check out the README to learn how to set it.")
            return False
        url = "https://api.openchargemap.io/v3/poi/"
        params = {
            'key': api_key,
            'latitude': 43.4643,
            'longitude': -80.5204,
            'maxresults': 1
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("✓ OpenChargeMap API is accessible!")
            return True
        else:
            print(f"✗ OpenChargeMap API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ OpenChargeMap API error: {str(e)}")
        return False
    
def check_row_opendata_api():
    """Check if the Region of Waterloo Open Data API is accessible."""
    print("\nChecking Region of Waterloo Open Data API connectivity...")
    try:
        import requests
        url = "https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Little_Libraries/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson"
        response = requests.get(url)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if 'features' in data and len(data['features']) > 0:
                    print("✓ Region of Waterloo Open Data API is accessible!")
                    print(f"(🔖 Fun fact: There are currently {str(len(data['features']))} Little Libraries in the Region of Waterloo!)")
                    return True
                else:
                    print("✗ Region of Waterloo Open Data API returned an unexpected response")
                    return False
            except ValueError:
                print("✗ Region of Waterloo Open Data API returned non-JSON response")
                print(f"Response content: {response.text}")
                return False
        else:
            print(f"✗ Region of Waterloo Open Data API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Region of Waterloo Open Data API error: {str(e)}")
        return False

def main():
    """Run all verification checks."""
    print(make_header(f"\n🔍 KWC EV Charging Network Optimization - Setup Verification\n📂 Project Root: {PROJECT_ROOT}\n", "="))    
    
    # Check virtual environment first
    check_virtual_env()
    
    # Create a progress structure
    check_stages = [
        ("Environment", [
            ("Python Version", check_python_version),
            ("Dependencies", check_dependencies),
            ("Project Package", install_project_package)
        ]),
        ("Licensing", [
            ("Gurobi License", check_gurobi_license)
        ]),
        ("API Access", [
            ("API Keys", check_api_keys),
            ("OpenStreetMap API", check_openstreetmap_api),
            ("OpenChargeMap API", check_openchargemap_api),
            ("ROW OpenData API", check_row_opendata_api)
        ]),
        ("Project Structure", [
            ("Directory Structure", check_directory_structure),
            ("Source Code Files", check_source_code_files)
        ])
    ]
    
    all_passed = True
    
    # Run checks by stage
    for stage_name, checks in check_stages:
        print(make_header(f'📋 {stage_name} Checks'.upper(), '-'))
        
        stage_passed = True
        for check_name, check_func in checks:
            print(f"\n→ {check_name.upper()}")
            if not check_func():
                stage_passed = False
                all_passed = False
            print("\n" + "-" * OUTPUT_WIDTH)
                
        if stage_passed:
            print(f'✅ All {stage_name} Checks Passed'.center(OUTPUT_WIDTH))
        else:
            print(f'❌ Some {stage_name} Checks Failed'.center(OUTPUT_WIDTH))
    
    if all_passed:
        print(make_header("🎉 SETUP VERIFICATION COMPLETED SUCCESSFULLY", "="))
        return 0
    else:
        print(make_header("❌ Some checks failed. Please fix the issues above.", "="))
        return 1

if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        op_time_msg = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        op_time_msg = f"{int(minutes)}m {seconds:.2f}s"
    else:
        op_time_msg = f"{seconds:.2f}s"
    print(f'\n[ Operation concluded in {op_time_msg} ]\n')
    sys.exit(exit_code)