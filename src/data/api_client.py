"""Client for fetching data from OpenChargeMap API."""
import os
from typing import Dict, List, Optional
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

class OpenChargeMapClient:
    def __init__(self):
        # Load API key
        load_dotenv()
        self.api_key = os.getenv("OCMAP_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenChargeMap API key not found in environment variables")
        
        # KW region bounds
        self.kw_bounds = {
            'north': 43.5445,
            'south': 43.3839,
            'east': -80.4013,
            'west': -80.6247
        }
        
        self.base_url = "https://api.openchargemap.io/v3"

    def fetch_stations(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch charging stations in KW region."""
        cache_file = Path('data/raw/charging_stations.csv')

        # Create the directory if it doesn't exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not force_refresh and cache_file.exists():
            return pd.read_csv(cache_file)
        
        print("Fetching charging station data...")
        
        params = {
            'key': self.api_key,
            'maxresults': 100,
            'latitude': (self.kw_bounds['north'] + self.kw_bounds['south']) / 2,
            'longitude': (self.kw_bounds['east'] + self.kw_bounds['west']) / 2,
            'distance': 10,
            'distanceunit': 'KM',
            'countrycode': 'CA'
        }
        
        response = requests.get(f"{self.base_url}/poi", params=params)
        data = response.json()
        
        stations = []
        for poi in data:
            address_info = poi.get('AddressInfo', {})
            operator = poi.get('OperatorInfo', {})
            connections = poi.get('Connections', [])
            
            lat = address_info.get('Latitude')
            lon = address_info.get('Longitude')
            
            # Only include stations in KW region
            if (self.kw_bounds['south'] <= lat <= self.kw_bounds['north'] and
                self.kw_bounds['west'] <= lon <= self.kw_bounds['east']):
                
                operator = poi.get('OperatorInfo')
                stations.append({
                    'name': address_info.get('Title'),
                    'latitude': lat,
                    'longitude': lon,
                    'num_chargers': len(connections),
                    'charger_type': self._determine_charger_type(connections),
                    'operator': operator.get('Title', 'Unknown') if operator else 'Unknown',
                    'address': address_info.get('AddressLine1'),
                    'city': address_info.get('Town'),
                    'postal_code': address_info.get('Postcode'),
                    'usage_cost': self._extract_usage_cost(poi.get('UsageCost'))
                })
        
        df = pd.DataFrame(stations)
        df.to_csv(cache_file, index=False)
        print(f"Found {len(df)} stations in KW region")
        return df
    
    def _determine_charger_type(self, connections: Optional[List[Dict]]) -> str:
        """Determine the highest level charger available."""
        if not connections:
            return 'Unknown'
        
        for conn in connections:
            level = conn.get('Level')
            if level and level.get('IsFastChargeCapable'):
                return 'Level 3'
        
        for conn in connections:
            level = conn.get('Level')
            if level and 'level 2' in level.get('Title', '').lower():
                return 'Level 2'
        
        return 'Level 1'
    
    def _extract_usage_cost(self, cost_info: Optional[str]) -> str:
        """Extract and clean usage cost information."""
        if not cost_info:
            return 'Unknown'
        
        cost = cost_info.replace('Payment required', '')
        cost = cost.replace('Payment Required', '')
        cost = cost.strip()
        
        return cost if cost else 'Unknown'