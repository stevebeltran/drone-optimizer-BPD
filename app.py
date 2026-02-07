import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import shutil
import itertools

# --- PAGE CONFIG ---
st.set_page_config(page_title="Citywide Drone Optimizer", layout="wide")

# --- SPEED OPTIMIZATION: CACHING ---
@st.cache_data
def process_geo_data(shp_path, selection):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None: gdf.set_crs(epsg=4269, inplace=True)
    
    # Simplify geometry slightly to boost speed (0.0001 degrees)
    gdf['geometry'] = gdf['geometry'].simplify(0.0001, preserve_topology=True)
    
    name_col = 'DISTRICT' if 'DISTRICT' in gdf.columns else 'NAME'
    
    if selection == "SHOW ALL DISTRICTS":
        active_gdf = gdf.to_crs(epsg=4326)
        boundary = unary_union(active_gdf.geometry)
    else:
        active_gdf = gdf[gdf[name_col] == selection].to_crs(epsg=4326)
        boundary = active_gdf.iloc[0].geometry
        
    return gdf, active_gdf, boundary, name_col

# --- INITIALIZE ---
call_data, station_data, shape_components = None, None, []

with st.expander("📁 Upload Data Files", expanded=False):
    uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True)

STATION_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#800000", "#333333", "#000075"]

if uploaded_files:
    for f in uploaded_files:
        fname = f.name.lower()
        if fname == "calls.csv": call_data = f
        elif fname == "stations.csv": station_data = f
        elif any(fname.endswith(ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
            shape_components.append(f)

if call_data and station_data and len(shape_components) >= 3:
    if not os.path.exists("temp"): os.mkdir("temp")
    for f in shape_components:
        with open(os.path.join("temp", f.name), "wb") as buffer:
            buffer.write(f.getbuffer())
    
    try:
        shp_path = [os.path.join("temp", f.name) for f in shape_components if f.name.endswith('.shp')][0]
        
        # Determine available districts for the dropdown
        temp_gdf = gpd.read_file(shp_path)
        name_col_init = 'DISTRICT' if 'DISTRICT' in temp_gdf.columns else 'NAME'
        options = ["SHOW ALL DISTRICTS"] + sorted(temp_gdf[name_col_init].unique().tolist())
        
        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        selection = ctrl_col1.selectbox("📍 Jurisdiction Focus", options)

        # USE CACHED GEO PROCESSING
        gdf_all, active_gdf, city_boundary, name_col = process_geo_data(shp_path, selection)

        utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
        epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
        city_m = active_gdf.to_crs(epsg=epsg_code).unary_union
        
        df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
        df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
        calls_in_city = gdf_calls[gdf_calls.within(city_boundary)].to_crs(epsg=epsg_code)
        calls_in_city['point_idx'] = range(len(calls_in_city))
        
        # --- FAST ANALYSIS ---
        radius_m = 3218.69 # 2 miles
        station_metadata = []
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            # Fast Spatial Filter
            mask = calls_in_city.geometry.distance(s_pt_m) <= radius_m
            indices = set(calls_in_city[mask]['point_idx'])
            # Only clip geometry if absolutely necessary for land %
            clipped_buf = s_pt_m.buffer(radius_m).intersection(city_m)
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_m': clipped_buf, 'indices': indices
            })

        # --- OPTIMIZER ---
        st.sidebar.header("🎯 Optimizer")
        k = st.sidebar.slider("Stations", 1, len(station_metadata), min(5, len(station_metadata)))
        strategy = st.sidebar.radio("Strategy", ("Max Response Volume", "Max Geographic Equity"))

        combos = list(itertools.combinations(range(len(station_metadata)), k))
        if len(combos) > 600: combos = combos[:600] # Cap combos for speed
        
        best_call_combo, max_calls = None, -1
        best_geo_combo, max_area = None, -1
        
        for combo in combos:
            u_set = set().union(*(station_metadata[i]['indices'] for i in combo))
            if len(u_set) > max_calls: max_calls = len(u_set); best_call_combo = combo
            
            if strategy == "Max Geographic Equity":
                u_geo = unary_union([station_metadata[i]['clipped_m'] for i in combo])
                if u_geo.area > max_area: max_area = u_geo.area; best_geo_combo = combo
            
        default_sel = [station_metadata[i]['name'] for i in (best_call_combo if strategy == "Max Response Volume" else best_geo_combo)]
        active_names = ctrl_col2.multiselect("📡 Active Stations", options=df_stations_all['name'].tolist(), default=default_sel)

        # --- METRICS & MAP ---
        active_data = [s for s in station_metadata if s['name'] in active_names]
        active_indices = [s['indices'] for s in active_data]
        all_ids = set().union(*active_indices) if active_indices else set()
        cap_perc = (len(all_ids) / len(calls_in_city)) * 100 if len(calls_in_city) > 0 else 0

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Incidents", f"{len(calls_in_city):,}")
        m2.metric("Capacity %", f"{cap_perc:.1f}%")
        m3.metric("Uncovered", f"{len(calls_in_city) - len(all_ids):,}")

        fig = go.Figure()
        # Draw District Outlines (Simplified for Speed)
        for _, row in gdf_all.to_crs(epsg=4326).iterrows():
            geom = row.geometry
            p_list = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
            for p in p_list:
                bx, by = p.exterior.coords.xy
                fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#444", width=1), showlegend=False, hoverinfo='skip'))
        
        # Optimized Scatter (Sample 2000 for speed)
        sample = calls_in_city.to_crs(epsg=4326).sample(min(2000, len(calls_in_city)))
        fig.add_trace(go.Scattermap(lat=sample.geometry.y, lon=sample.geometry.x, mode='markers', marker=dict(size=4, color='#000080', opacity=0.3), name="Incidents"))
        
        def get_circle(lat, lon):
            angles = np.linspace(0, 2*np.pi, 60) # Fewer points for speed
            return lat + (2/69.172) * np.sin(angles), lon + (2/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)

        all_st_names = df_stations_all['name'].tolist()
        for s in active_data:
            color = STATION_COLORS[all_st_names.index(s['name']) % len(STATION_COLORS)]
            clats, clons = get_circle(s['lat'], s['lon'])
            fig.add_trace(go.Scattermap(lat=list(clats) + [None, s['lat']], lon=list(clons) + [None, s['lon']], mode='lines+markers', marker=dict(size=[0]*60 + [18], color=color), line=dict(color=color, width=4), name=s['name']))

        fig.update_layout(map_style="open-street-map", map_zoom=11, map_center={"lat": city_boundary.centroid.y, "lon": city_boundary.centroid.x}, margin={"r":0,"t":0,"l":0,"b":0}, height=700)
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"Error: {e}")
