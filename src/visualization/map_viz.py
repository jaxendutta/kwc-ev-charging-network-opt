"""Visualization utilities for EV charging station locations."""
import folium
import pandas as pd
from folium import Element
import geopandas as gpd

def create_kw_map():
    """Create a base map of KW region."""
    return folium.Map(
        location=[43.4516, -80.4925],  # Center of KW
        zoom_start=12,
        tiles='cartodbpositron'
    )

def plot_charging_stations(m: folium.Map, stations_df: pd.DataFrame) -> folium.Map:
    """Plot charging stations on map with detailed popups."""
    
    # Color mapping for charger types
    colors = {
        'Level 1': 'blue',
        'Level 2': 'green',
        'Level 3': 'red',
        'Unknown': 'gray'
    }
    
    for idx, row in stations_df.iterrows():
        # Create detailed popup content
        popup_content = f"""
        <div style="font-family: Arial; max-width: 300px;">
            <h4 style="margin-bottom: 5px;">{row.get('name', 'Unnamed Station')}</h4>
            <p style="margin: 2px 0;"><b>Type:</b> {row.get('charger_type', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Chargers:</b> {row.get('num_chargers', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Operator:</b> {row.get('operator', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Cost:</b> {row.get('usage_cost', 'Unknown')}</p>
            <hr style="margin: 5px 0;">
            <p style="margin: 2px 0;"><small>{row.get('address', '')}<br>
            {row.get('city', '')}, {row.get('postal_code', '')}</small></p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=colors.get(row.get('charger_type', 'Unknown'), 'gray'),
            fill=True,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(m)
    
    # Add legend with statistics
    stats_html = f"""
    <div style="position: fixed; 
                bottom: 50px; 
                left: 50px; 
                z-index: 1000; 
                background-color: white; 
                padding: 10px; 
                border: 2px solid grey; 
                border-radius: 5px;">
        <h4 style="margin: 0 0 10px 0;">Charging Stations</h4>
        <p style="margin: 2px 0;"><span style='color: blue;'>●</span> Level 1: {len(stations_df[stations_df['charger_type']=='Level 1'])}</p>
        <p style="margin: 2px 0;"><span style='color: green;'>●</span> Level 2: {len(stations_df[stations_df['charger_type']=='Level 2'])}</p>
        <p style="margin: 2px 0;"><span style='color: red;'>●</span> Level 3: {len(stations_df[stations_df['charger_type']=='Level 3'])}</p>
        <p style="margin: 2px 0;"><span style='color: gray;'>●</span> Unknown: {len(stations_df[stations_df['charger_type']=='Unknown'])}</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 2px 0;"><b>Total Stations:</b> {len(stations_df)}</p>
        <p style="margin: 2px 0;"><b>Total Chargers:</b> {stations_df['num_chargers'].sum()}</p>
    </div>
    """
    m.get_root().html.add_child(Element(stats_html))
    
    return m

def plot_potential_locations(m: folium.Map, locations_gdf: gpd.GeoDataFrame) -> folium.Map:
    """Plot potential locations on the map."""
    for idx, row in locations_gdf.iterrows():
        if row.geometry.geom_type == 'Point':
            popup_content = f"<b>{row.get('name', 'Unknown')}</b><br>Type: {row.get('location_type', 'Unknown')}"
            
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
    
    return m

def plot_heatmap(m: folium.Map, data_gdf: gpd.GeoDataFrame, 
                 value_column: str, legend_name: str, radius: int = 15) -> folium.Map:
    """Add a heatmap layer to the map."""
    locations = [[point.y, point.x] for point in data_gdf.geometry]
    values = data_gdf[value_column].tolist()
    
    folium.plugins.HeatMap(
        locations,
        weights=values,
        name=legend_name,
        radius=radius,
        max_zoom=13,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
    ).add_to(m)
    
    return m

def plot_traffic_flow(m: folium.Map, traffic_gdf: gpd.GeoDataFrame, 
                     value_column: str) -> folium.Map:
    """Add traffic flow visualization to the map."""
    def get_color(value):
        """Get color based on traffic value."""
        if value > 2000:
            return 'red'
        elif value > 1000:
            return 'orange'
        else:
            return 'green'
    
    def get_weight(value):
        """Get line weight based on traffic value."""
        return 2 + (value / 500)
    
    for idx, row in traffic_gdf.iterrows():
        folium.PolyLine(
            locations=[[coord[1], coord[0]] for coord in row.geometry.coords],
            color=get_color(row[value_column]),
            weight=get_weight(row[value_column]),
            popup=f"Traffic flow: {row[value_column]:,.0f}"
        ).add_to(m)
    
    return m