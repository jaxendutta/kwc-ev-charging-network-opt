import glob
from pathlib import Path
from datetime import datetime

# Get time right now in the format YYYY-MM-DD_HH-MM-SS
grab_time = lambda: datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Get the timestamp of a file
get_file_timestamp = lambda x: datetime.fromtimestamp(Path(x).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

# Flatten the timestamp to file name format
flatten_timestamp = lambda x: x.replace(' ', '_').replace(':', '-')

# Get latest file in a directory
def get_latest_csv(directory):
    directory = Path(directory)
    csv_files = sorted(directory.glob('*.csv'), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")
    
    latest_csv = csv_files[0]
    timestamp = get_file_timestamp(latest_csv)
    
    return latest_csv, timestamp