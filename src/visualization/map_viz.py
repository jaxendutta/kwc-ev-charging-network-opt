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

def add_legend_to_map(m: folium.Map, color_map: dict) -> folium.Map:
    """Add a legend to the map."""
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; 
                left: 50px; 
                z-index: 1000; 
                background-color: white; 
                padding: 10px; 
                border: 2px solid grey; 
                border-radius: 5px;">
    """
    
    for loc_type, color in color_map.items():
        legend_html += f"<p style='margin: 2px 0;'><span style='color: {color};'>●</span> {loc_type}</p>"
    
    legend_html += "</div>"
    m.get_root().html.add_child(Element(legend_html))
    
    return m

def folium_pop_up_html(row: pd.Series, charger: bool) -> str:
    """Create a popup HTML content for a charging station or potential location."""
    if charger:
        content = f"""
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
    else:
        content = f"""
        <div style="font-family: Arial; max-width: 300px;">
            <h4 style="margin-bottom: 5px;">{row.get('name', 'Unnamed Location')}</h4>
            <p style="margin: 2px 0;"><b>Type:</b> {row.get('location_type', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Address:</b> {row.get('address', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>City:</b> {row.get('city', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Postal Code:</b> {row.get('postal_code', 'Unknown')}</p>
            <hr style="margin: 5px 0;">
            <p style="margin: 2px 0;"><small>{row.get('description', '')}</small></p>
        </div>
        """
    return content

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
        popup_content = folium_pop_up_html(row, charger=True)       
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=colors.get(row.get('charger_type', 'Unknown'), 'gray'),
            fill=True,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(m)
    
    # Add legend with statistics
    # Use add_legend_to_map function
    m = add_legend_to_map(m, colors)

    return m

def plot_potential_locations(m: folium.Map, locations_gdf: gpd.GeoDataFrame, column=None) -> folium.Map:
    """Plots potential locations on the map, optionally color-coded by a column.

    Args:
        m: The Folium map object.
        locations_gdf: GeoDataFrame containing location data.
        column: Optional column name to use for color-coding.

    Returns:
        The updated Folium map object.
    """
    # List of predefined colors supported by Folium
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 
              'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    if column:
        location_types = locations_gdf[column].unique()
        color_map = {loc_type: colors[i % len(colors)] for i, loc_type in enumerate(location_types)}

        for idx, row in locations_gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                popup_content = folium_pop_up_html(row, charger=False)
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=color_map[row[column]], icon='info-sign')
                ).add_to(m)

        m = add_legend_to_map(m, color_map)

    else:
        for idx, row in locations_gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                popup_content = folium_pop_up_html(row, charger=False)
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