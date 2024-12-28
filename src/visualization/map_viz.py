"""Visualization utilities for EV charging station locations."""
from typing import Optional
from IPython.display import display
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import folium
from folium import Element
from folium.plugins import HeatMap
import branca.colormap as cm
import geopandas as gpd

from src.data.data_manager import DataManager
from src.data.constants import *

def draw_map(m: folium.Map):
    """Draw the map with final touches"""

    # Add layer control to the map
    folium.LayerControl().add_to(m)

    # Add fullscreen button to the map
    folium.plugins.Fullscreen().add_to(m)

    display(m)
    return m

def create_kwc_map(title: Optional[str] = None, kwc: Optional[bool] = False) -> folium.Map:
    """Create a base map of KW region."""

    # Generate a folium map centered around KW region
    m = folium.Map(
        location=[43.4516, -80.4925],  # Center of KW
        zoom_start=10,
        tiles=None
    )

    # Add OpenStreetMap tiles as an alternative
    folium.TileLayer('openstreetmap', name='Open Street Map').add_to(m)

    # Add CartoDB Positron tiles as the default
    folium.TileLayer('cartodbpositron', name='CartoDB Positron').add_to(m)
    
    # Add CMA boundary to the map
    if (kwc):
        # Get CMA boundary
        data_mgr = DataManager()
        boundary = data_mgr.get_cma_boundary()

        # Add boundary to the map
        folium.GeoJson(
            boundary.__geo_interface__,
            name='KWC-CMA Boundary',
            style_function=lambda x: {
                'color': 'red',
                'weight': 2,
                'fillColor': '#3186cc',
                'fillOpacity': 0.1
            }
        ).add_to(m)

    # Add title to the map with a specific ID for styling 
    if title: 
        title_html = f''' 
        <div id="map-title" style="position: fixed; 
                                justify-content: center; 
                                left: 50%; 
                                transform: translateX(-50%); 
                                z-index: 1000; 
                                background-color: white;
                                padding: 2px 10px; 
                                border-radius: 5px; 
                                font-size: 14px; 
                                text-align: center;">
            <b>{title}</b>
        </div> 
        ''' 
        
        m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def add_legend_to_map(m: folium.Map, color_map: dict, counts: dict) -> folium.Map:
    """Add a legend to the map."""
    legend_html = f"""
    <div id="map-legend" style="position: fixed; 
                                bottom: 15px; 
                                left: 15px; 
                                z-index: 1000; 
                                background-color: white; 
                                padding: 10px; 
                                border: 1px solid grey; 
                                border-radius: 5px;
                                font-size: 12px;">
    """
    
    # Sort types by counts and ensure all values are strings
    sorted_types = sorted(
        [(str(k), int(v)) for k, v in counts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    for loc_type, count in sorted_types:
        color = str(color_map.get(loc_type, 'black'))
        legend_html += f'<p style="margin: 2px 0;"><span style="color: {color};">●</span> {loc_type} <span style="color: grey;">({count})</span></p>'
    
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
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
        'Level 1': 'red',
        'Level 2': 'blue',
        'Level 3': 'green'
    }
    
    counts = stations_df['charger_type'].value_counts().to_dict()
    
    for charger_type, color in colors.items():
        if counts.get(charger_type, 0) > 0:
            layer = folium.FeatureGroup(name=f'{charger_type} Charging Stations')
            for _, row in stations_df[stations_df['charger_type'] == charger_type].iterrows():
                popup_content = folium_pop_up_html(row, charger=True)
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    icon=folium.Icon(color=color, icon='bolt', prefix='fa'),
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(layer)
            layer.add_to(m)
    
    # Handle stations with charger types other than Level 1, 2, or 3
    other_types = stations_df[~stations_df['charger_type'].isin(colors.keys())]
    other_charger_types = other_types['charger_type'].unique()
    free_colours = [col for col in FOLIUM_COLORS if col not in colors.values()]
    for i, charger_type in enumerate(other_charger_types):
        layer = folium.FeatureGroup(name=f'{charger_type} Charging Stations')
        color = free_colours[(i + 3) % len(free_colours)]
        for _, row in other_types[other_types['charger_type'] == charger_type].iterrows():
            popup_content = folium_pop_up_html(row, charger=True)
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.Icon(color=color, icon='bolt', prefix='fa'),
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(layer)
        layer.add_to(m)
        colors[charger_type] = color
        counts[charger_type] = len(other_types[other_types['charger_type'] == charger_type])

    # Add legend with statistics
    m = add_legend_to_map(m, colors, counts)

    return m

def plot_potential_locations(m: folium.Map, locations_gdf: gpd.GeoDataFrame) -> folium.Map:
    """Plots potential locations on the map, optionally color-coded by a column.

    Args:
        m: The Folium map object.
        locations_gdf: GeoDataFrame containing location data.

    Returns:
        The updated Folium map object.
    """
    column = 'location_type'
    location_types = locations_gdf[column].unique()
    color_map = {loc_type: FOLIUM_COLORS[i % len(FOLIUM_COLORS)] for i, loc_type in enumerate(location_types)}
    
    for loc_type in location_types:
        layer = folium.FeatureGroup(name=f'Potential {loc_type} Locations')
        for _, row in locations_gdf[locations_gdf[column] == loc_type].iterrows():
            popup_content = folium_pop_up_html(row, charger=False)
            if row.geometry.geom_type == 'Point':
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=color_map[loc_type], icon=icons.get(loc_type, 'info-sign'), prefix='fa')
                ).add_to(layer)
            elif row.geometry.geom_type == 'Polygon':
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    popup=folium.Popup(popup_content, max_width=300),
                    style_function=lambda _: {'color': color_map[loc_type]}
                ).add_to(layer)
        layer.add_to(m)
    
    # Add legend with statistics
    m = add_legend_to_map(m, color_map, locations_gdf[column].value_counts().to_dict())

    return m

def plot_transportation_network(m: folium.Map, transport_analysis: dict) -> folium.Map:
    """Plot transit routes and stops on the map."""
    
    # Add road network with categorized styling
    roads = transport_analysis['road_network']['data']
    grt_routes = transport_analysis['transit_network']['grt_routes_data']
    ion_routes = transport_analysis['transit_network']['ion_routes_data']
    grt_stops = transport_analysis['transit_network']['grt_stops_data']
    ion_stops = transport_analysis['transit_network']['ion_stops_data']
    
    def grt_stop_popup(stop):
        stop_name = stop.get('Street', '') or stop.get('CrossStreet', 'Unnamed Stop')
        if stop.get('Street') and stop.get('CrossStreet'):
            stop_name = f"{stop['Street']} / {stop['CrossStreet']}"
    
        popup_html = f"""
        <div style="font-family: Arial; max-width: 500px;">
            <h4 style="margin-bottom: 5px;">{stop_name}</h4>
            <p style="margin: 2px 0;"><b>Stop ID:</b> {stop.get('StopID', 'Unnamed Stop')}</p>
            <p style="margin: 2px 0;"><b>EasyGo:</b> {stop.get('EasyGo', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Municipality:</b> {stop.get('Municipality', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Status:</b> {stop.get('Status', 'Unknown')}</p>
            <p style="margin: 2px 0; color: grey;">({stop.geometry.y}, {stop.geometry.x})</p>
        </div>
        """
        return popup_html
    
    def ion_stop_popup(stop):
        popup_html = f"""
        <div style="font-family: Arial; max-width: 500px;">
            <h4 style="margin-bottom: 5px;">{stop.get('StopName', 'Unnamed Stop')}</h4>
            <p style="margin: 2px 0;"><b>Municipality:</b> {stop.get('Municipality', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Location:</b> {stop.get('StopLocation', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Status:</b> {stop.get('StopStatus', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Stage 1:</b> {stop.get('Stage1', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Stage 2:</b> {stop.get('Stage2', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Phase:</b> {stop.get('Phase', 'Unknown')}</p>
            <p style="margin: 2px 0;"><b>Direction:</b> {stop.get('StopDirection', 'Unknown')}</p>
            <p style="margin: 2px 0; color: grey;">({stop.geometry.y}, {stop.geometry.x})</p>
        </div>
        """
        return popup_html
    
    # Add all layers to map with proper styling
    for name, gdf in [
        ('Roads', roads),
        ('GRT Routes', grt_routes),
        ('ION Routes', ion_routes),
    ]:
        folium.GeoJson(
            gdf.__geo_interface__,
            name=name,
            style_function=lambda x, color=('gray' if name == 'Roads' else 'blue' if name == 'GRT Routes' else 'red'): {
                'color': color,
                'weight': 1 if name == 'Roads' else 2,
                'opacity': 0.5 if name == 'Roads' else 0.7
            }
        ).add_to(m)
    
    # Add stops with clustering
    grt_stops_group = folium.FeatureGroup(name='GRT Stops')
    for _, stop in grt_stops.iterrows():
        folium.CircleMarker(
            location=[stop.geometry.y, stop.geometry.x],
            radius=3,
            color='blue',
            fill=True,
            popup=grt_stop_popup(stop)
        ).add_to(grt_stops_group)
    grt_stops_group.add_to(m)
    
    ion_stops_group = folium.FeatureGroup(name='ION Stops')
    for _, stop in ion_stops.iterrows():
        folium.CircleMarker(
            location=[stop.geometry.y, stop.geometry.x],
            radius=5,
            color='red',
            fill=True,
            popup=ion_stop_popup(stop)
        ).add_to(ion_stops_group)
    ion_stops_group.add_to(m)
    
    return m

def plot_heatmap(m: folium.Map, data_gdf: gpd.GeoDataFrame, value_column: str, legend_name: str, radius: int = 15) -> folium.Map:
    """Add a heatmap layer to the map."""
    
    points = data_gdf[data_gdf.geometry.geom_type == 'Point']
    locations = [[point.y, point.x] for point in points.geometry]
    values = points[value_column].tolist()
    
    heatmap_layer = HeatMap(
        locations,
        weights=values,
        name=legend_name,
        radius=radius,
        max_zoom=13,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
    )
    
    heatmap_layer.add_to(m)
    
    return m

def map_population_density(m: folium.Map, 
                          population_data: gpd.GeoDataFrame,
                          style: str = 'heatmap') -> folium.Map:
    """
    Add population density visualization to map.
    
    Args:
        m: Folium map object
        population_data: GeoDataFrame with population data
        style: 'heatmap' or 'choropleth'
        
    Returns:
        Updated map with population density layer
    """
    # Filter for Region of Waterloo data
    population_data = population_data[population_data['data_source'] == 'Region of Waterloo']
    
    if style == 'choropleth':
        population_data['CTUID'] = population_data['CTUID'].astype(str)

        # Define the population density thresholds
        threshold_scale = [0, 100, 825, 1750, 3500, 7000, 14000]# Original choropleth implementation
        
        folium.Choropleth(
            geo_data=population_data.__geo_interface__,
            name='Population Density',
            data=population_data,
            columns=['CTUID', 'population_density'],
            key_on='feature.properties.CTUID',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            threshold_scale=threshold_scale,
            legend_name='Population Density (people/km²)'
        ).add_to(m)
    
    else:  # heatmap
        # Create locations and weights for heatmap
        locations = []
        weights = []
        
        # Use centroids of census tracts, weighted by population
        for idx, row in population_data.iterrows():
            if pd.notna(row['population']) and row['population'] > 0:
                centroid = row.geometry.centroid
                locations.append([centroid.y, centroid.x])
                weights.append(float(row['population_density']))
        
        # Add heatmap layer
        folium.plugins.HeatMap(
            locations,
            weights=weights,
            name='Population Density',
            min_opacity=0.3,
            max_zoom=13,
            radius=25,  # Adjust for smoother/sharper heatmap
            blur=15,    # Adjust for smoother/sharper heatmap
            gradient={
                '0.4': 'blue',
                '0.65': 'lime',
                '0.8': 'yellow',
                '1': 'red'
            }
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

def plot_density_clusters(
    m: folium.Map, 
    clean_data: gpd.GeoDataFrame, 
    clustering_labels: np.ndarray,
    show_hulls: bool = True
) -> folium.Map:
    """
    Add density cluster visualization to a Folium map.
    """    
    # Create a feature group for clusters
    clusters_group = folium.FeatureGroup(name='Population Clusters')

    # Format cluster labels and count points
    cluster_counts = {}
    for cid in sorted(set(clustering_labels)):
        label = "Noise Points" if cid == -1 else f"Cluster {cid}"
        count = len(clean_data[clean_data['density_cluster'] == cid])
        cluster_counts[label] = count

    # Assign colors to each label
    cluster_colors = {}
    for i, label in enumerate(cluster_counts.keys()):
        if label == "Noise Points":
            cluster_colors[label] = 'gray'
        else:
            cluster_colors[label] = FOLIUM_COLORS[i % len(FOLIUM_COLORS)]
            
    # Create mapping from cluster ID to color for plotting
    id_to_color = {
        -1: cluster_colors["Noise Points"]
    }
    id_to_color.update({
        int(label.split()[-1]): color
        for label, color in cluster_colors.items()
        if label != "Noise Points"
    })

    # Add each cluster with different colors
    for cluster_id in sorted(set(clustering_labels)):
        cluster_data = clean_data[clean_data['density_cluster'] == cluster_id]
        cluster_color = id_to_color[cluster_id]
        
        # Create convex hull for cluster visualization if requested
        if show_hulls and len(cluster_data) >= 3:
            points = np.column_stack((
                cluster_data.geometry.centroid.x,
                cluster_data.geometry.centroid.y
            ))
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])  # Close polygon
            
            # Convert to folium coordinates
            locations = [[y, x] for x, y in hull_points]
            
            # Add cluster polygon
            folium.Polygon(
                locations=locations,
                color=cluster_color,
                fill=True,
                weight=2,
                fillOpacity=0.2,
                popup=f"""
                <div style="font-family: Arial; max-width: 300px;">
                    <h4>{"Noise Points" if cluster_id == -1 else f"Cluster {cluster_id}"}</h4>
                    <p><b>Points:</b> {len(cluster_data)}</p>
                    <p><b>Population:</b> {cluster_data['population'].sum():,.0f}</p>
                    <p><b>Avg Density:</b> {cluster_data['population_density'].mean():,.1f}/km²</p>
                </div>
                """
            ).add_to(clusters_group)
        
        # Add cluster centroids
        centroid = cluster_data.geometry.unary_union.centroid
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=8,
            color=cluster_color,
            fill=True,
            popup=f"""
            <div style="font-family: Arial; max-width: 300px;">
                <h4>{"Noise Points" if cluster_id == -1 else f"Cluster {cluster_id}"}</h4>
                <p><b>Points:</b> {len(cluster_data)}</p>
                <p><b>Population:</b> {cluster_data['population'].sum():,.0f}</p>
                <p><b>Avg Density:</b> {cluster_data['population_density'].mean():,.1f}/km²</p>
            </div>
            """,
            tooltip=f"{'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'}"
        ).add_to(clusters_group)

    # Add cluster group to map
    clusters_group.add_to(m)

    # Add legend using existing function
    m = add_legend_to_map(m, cluster_colors, cluster_counts)

    return m

def plot_housing_patterns(m: folium.Map, population_data: gpd.GeoDataFrame) -> folium.Map:
    """Plot housing patterns including property values and high-value areas on the map."""
    
    # Add base choropleth of property values directly to the map
    folium.Choropleth(
        geo_data=population_data.__geo_interface__,
        name='Property Values',
        data=population_data,
        columns=['OBJECTID', 'HHLD_DWELL_VALUE_MED'],
        key_on='feature.properties.CTUID',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Median Property Value ($)'
    ).add_to(m)

    # Create a feature group for high-value areas
    high_value_areas_layer = folium.FeatureGroup(name='High-Value Areas')

    # Add high-value areas as a GeoJson layer
    high_value_areas = population_data[
        population_data['HHLD_DWELL_VALUE_MED'] > population_data['HHLD_DWELL_VALUE_MED'].quantile(0.9)
    ]

    folium.GeoJson(
        high_value_areas.__geo_interface__,
        name='High-Value Areas',
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 2,
            'fillOpacity': 0.5,
        },
        highlight_function=lambda x: {
            'weight': 3,
            'color': 'blue',
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['HHLD_DWELL_VALUE_MED', 'DWELL_OWNER', 'DWELL_RENTER'],
            aliases=['Median Value ($)', 'Owner-Occupied', 'Rental Units'],
            localize=True
        )
    ).add_to(high_value_areas_layer)

    # Add the high-value areas layer to the map
    high_value_areas_layer.add_to(m)

    # Create custom legend HTML for high-value areas
    color_map = {'High-Value Areas': 'blue'}
    counts = {'High-Value Areas': len(high_value_areas)}
    m = add_legend_to_map(m, color_map, counts)

    return m

import branca.colormap as cm

def plot_ev_density(m: folium.Map, ev_data: gpd.GeoDataFrame) -> folium.Map:
    """
    Add EV density layer to the map.
    
    Args:
        m: Base map to add the layer to
        ev_data: GeoDataFrame containing EV data with density information
        
    Returns:
        Updated map with EV density layer
    """
    # Create feature group for EV density
    ev_group = folium.FeatureGroup(name='EV Density')

    # Create color map for EV density
    colormap = cm.LinearColormap(
        colors=['#FFEDA0', '#FEB24C', '#FC4E2A', '#E31A1C'],
        vmin=ev_data['ev_density'].min(),
        vmax=ev_data['ev_density'].max(),
        caption='EVs per km²'
    )

    # Style function for GeoJSON layer
    def style_function(feature):
        density = feature['properties']['ev_density']
        return {
            'fillColor': colormap(density),
            'fillOpacity': 0.7,
            'color': 'black',
            'weight': 1
        }

    # Add the styled GeoJSON layer
    folium.GeoJson(
        ev_data,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['FSA', 'total_ev', 'bev', 'phev', 'ev_density', 'bev_ratio'],
            aliases=['FSA:', 'Total EVs:', 'BEVs:', 'PHEVs:', 'Density (EVs/km²):', 'BEV Ratio (%):'],
            localize=True,
            sticky=False,
            labels=True,
            style=('background-color: white; '
                   'color: #333333; '
                   'font-family: arial; '
                   'font-size: 12px; '
                   'padding: 10px;')
        )
    ).add_to(ev_group)

    # Add the colormap legend to the map
    m.add_child(colormap)

    # Add the EV density layer to the map
    m.add_child(ev_group)
    
    return m