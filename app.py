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
st.set_page_config(page_title="Strategic Drone Optimizer", layout="wide")
st.title("🛰️ Strategic Drone Optimizer")

# --- 1. INITIALIZE VARIABLES ---
# This prevents the "NameError" by ensuring variables exist even if empty
call_data, station_data, shape_components = None, None, []

# --- 2. UPLOAD SECTION ---
if 'files_ready' not in st.session_state:
    st.session_state['files_ready'] = False

with st.expander("📁 Upload Data Files", expanded=not st.session_state['files_ready']):
    uploaded_files = st.file_uploader("Drop all 6 files here (calls.csv, stations.csv, and 4 Shapefile components)", accept_multiple_files=True)

# Define high-contrast palette
STATION_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#800000", "#333333", "#000075"]

def get_circle_coords(lat, lon, r_mi=2):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

# --- 3. FILE ROUTING ---
if uploaded_files:
    for f in uploaded_files:
        fname = f.name.lower()
        if fname == "calls.csv": call_data = f
        elif fname == "stations.csv": station_data = f
        elif any(fname.endswith(ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
            shape_components.append(f)

# --- 4. MAIN ANALYSIS (Only runs if files are present) ---
if call_data and station_data and len(shape_components) >= 3:
    st.session_state['files_ready'] = True
    if os.path.exists("temp"): shutil.rmtree("temp")
    os.mkdir("temp")
    for f in shape_components:
        with open(os.path.join("temp", f.name), "wb") as buffer:
            buffer.write(f.getbuffer())
    
    try:
        # LOAD GEOGRAPHY
        shp_path = [os.path.join("temp", f.name) for f in shape_components if f.name.endswith('.shp')][0]
        gdf_all_districts = gpd.read_file(shp_path)
        if gdf_all_districts.crs is None: gdf_all_districts.set_crs(epsg=4269, inplace=True)
        
        name_col = 'DISTRICT' if 'DISTRICT' in gdf_all_districts.columns else 'NAME'
        options = ["SHOW ALL DISTRICTS"] + sorted(gdf_all_districts[name_col].unique().tolist())
        
        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        selection = ctrl_col1.selectbox("📍 Jurisdiction Focus", options)
        
        # Area Filtering
        if selection == "SHOW ALL DISTRICTS":
            active_gdf = gdf_all_districts.to_crs(epsg=4326)
            city_boundary = unary_union(active_gdf.geometry)
        else:
            active_gdf = gdf_all_districts[gdf_all_districts[name_col] == selection].to_crs(epsg=4326)
            city_boundary = active_gdf.iloc[0].geometry

        # Projection setup
        utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
        epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
        city_m = active_gdf.to_crs(epsg=epsg_code).unary_union
        
        # Load CSVs
        df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
        df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
        calls_in_city = gdf_calls[gdf_calls.within(city_boundary)].to_crs(epsg=epsg_code)
        calls_in_city['point_idx'] = range(len(calls_in_city))
        
        # Analysis
        radius_m = 3218.69 
        station_metadata = []
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            mask = calls_in_city.geometry.distance(s_pt_m) <= radius_m
            covered_indices = set(calls_in_city[mask]['point_idx'])
            clipped_buf = s_pt_m.buffer(radius_m).intersection(city_m)
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_m': clipped_buf, 'indices': covered_indices, 'count': len(covered_indices)
            })

        # OPTIMIZER
        st.sidebar.header("🎯 Optimizer Controls")
        k = st.sidebar.slider("Number of Stations", 1, len(station_metadata), min(2, len(station_metadata)))
        strategy = st.sidebar.radio("Strategy Mode", ("Max Response Volume", "Max Geographic Equity"))

        combos = list(itertools.combinations(range(len(station_metadata)), k))
        if len(combos) > 1000: combos = combos[:1000]
        
        best_call_combo, max_calls = -1, -1
        best_geo_combo, max_area = -1, -1
        
        for combo in combos:
            u_set = set().union(*(station_metadata[i]['indices'] for i in combo))
            if len(u_set) > max_calls: max_calls = len(u_set); best_call_combo = combo
            u_geo = unary_union([station_metadata[i]['clipped_m'] for i in combo])
            if u_geo.area > max_area: max_area = u_geo.area; best_geo_combo = combo
            
        best_call_names = [station_metadata[i]['name'] for i in best_call_combo]
        best_geo_names = [station_metadata[i]['name'] for i in best_geo_combo]
        default_sel = best_call_names if strategy == "Max Response Volume" else best_geo_names
        
        active_names = ctrl_col2.multiselect("📡 Active Stations", options=df_stations_all['name'].tolist(), default=default_sel)

        # METRICS
        active_data = [s for s in station_metadata if s['name'] in active_names]
        active_buffers = [s['clipped_m'] for s in active_data]
        active_indices = [s['indices'] for s in active_data]
        
        area_perc = (unary_union(active_buffers).area / city_m.area) * 100 if active_buffers else 0
        all_ids = set().union(*active_indices) if active_indices else set()
        cap_perc = (len(all_ids) / len(calls_in_city)) * 100 if len(calls_in_city) > 0 else 0
        uncovered = len(calls_in_city) - len(all_ids)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Incidents", f"{len(calls_in_city):,}")
        m2.metric("Capacity %", f"{cap_perc:.1f}%")
        m3.metric("Land Covered %", f"{area_perc:.1f}%")
        m4.metric("Uncovered", f"{uncovered:,}")

        # MAP
        fig = go.Figure()
        
        # Draw Districts
        for _, row in gdf_all_districts.to_crs(epsg=4326).iterrows():
            geom = row.geometry
            p_list = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
            for p in p_list:
                bx, by = p.exterior.coords.xy
                fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#555", width=1), showlegend=False))
        
        # Sample Calls
        sample = calls_in_city.to_crs(epsg=4326).sample(min(3000, len(calls_in_city)))
        fig.add_trace(go.Scattermap(lat=sample.geometry.y, lon=sample.geometry.x, mode='markers', marker=dict(size=4, color='#000080', opacity=0.3), name="Incidents"))
        
        # Rings
        all_st_names = df_stations_all['name'].tolist()
        for s in station_metadata:
            if s['name'] in active_names:
                color = STATION_COLORS[all_st_names.index(s['name']) % len(STATION_COLORS)]
                clats, clons = get_circle_coords(s['lat'], s['lon'])
                fig.add_trace(go.Scattermap(lat=list(clats) + [None, s['lat']], lon=list(clons) + [None, s['lon']], mode='lines+markers', marker=dict(size=[0]*len(clats) + [0, 18], color=color), line=dict(color=color, width=4), name=s['name']))

        fig.update_layout(map_style="open-street-map", map_zoom=11, map_center={"lat": city_boundary.centroid.y, "lon": city_boundary.centroid.x}, margin={"r":0,"t":0,"l":0,"b":0}, height=800)
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 System Ready. Please upload the 6 required files to begin analysis.")