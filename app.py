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
st.set_page_config(page_title="Drone Logistics Portal", layout="wide", page_icon="🛰️")

# --- 1. INITIALIZE SESSION STATE ---
if 'box_open' not in st.session_state:
    st.session_state.box_open = True

# --- 2. HEADER ---
if st.sidebar.button("🔄 Reset & Upload New Data"):
    st.session_state.box_open = True
    st.rerun()

st.title("🛰️ Strategic Drone Deployment Optimizer")
st.markdown("#### **Geospatial Operations Analysis Tool**")

# --- SPEED OPTIMIZATION: CACHING ---
@st.cache_data
def process_geo_data(shp_path, selection):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None: gdf.set_crs(epsg=4269, inplace=True)
    gdf['geometry'] = gdf['geometry'].simplify(0.0001, preserve_topology=True)
    name_col = 'DISTRICT' if 'DISTRICT' in gdf.columns else 'NAME'
    
    if selection == "SHOW ALL DISTRICTS":
        active_gdf = gdf.to_crs(epsg=4326)
        boundary = unary_union(active_gdf.geometry)
    else:
        active_gdf = gdf[gdf[name_col] == selection].to_crs(epsg=4326)
        boundary = active_gdf.iloc[0].geometry
    return gdf, active_gdf, boundary, name_col

# --- 3. DATA IMPORT ---
call_data, station_data, shape_components = None, None, []

with st.expander("📁 Secure Data Import", expanded=st.session_state.box_open):
    uploaded_files = st.file_uploader("Upload Incident CSVs and Shapefiles (6 files total)", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        fname = f.name.lower()
        if fname == "calls.csv": call_data = f
        elif fname == "stations.csv": station_data = f
        elif any(fname.endswith(ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
            shape_components.append(f)

    if call_data and station_data and len(shape_components) >= 4:
        if st.session_state.box_open:
            st.session_state.box_open = False
            st.rerun()

STATION_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#800000", "#333333", "#000075"]

# --- 4. MAIN ANALYSIS ENGINE ---
if call_data and station_data and len(shape_components) >= 3:
    if not os.path.exists("temp"): os.mkdir("temp")
    for f in shape_components:
        with open(os.path.join("temp", f.name), "wb") as buffer:
            buffer.write(f.getbuffer())
    
    try:
        shp_path = [os.path.join("temp", f.name) for f in shape_components if f.name.endswith('.shp')][0]
        temp_gdf = gpd.read_file(shp_path)
        name_col_init = 'DISTRICT' if 'DISTRICT' in temp_gdf.columns else 'NAME'
        options = ["SHOW ALL DISTRICTS"] + sorted(temp_gdf[name_col_init].unique().tolist())
        
        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        selection = ctrl_col1.selectbox("📍 Active Jurisdiction Focus", options)

        gdf_all, active_gdf, city_boundary, name_col = process_geo_data(shp_path, selection)
        utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
        epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
        city_m = active_gdf.to_crs(epsg=epsg_code).unary_union
        
        df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
        df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
        calls_in_city = gdf_calls[gdf_calls.within(city_boundary)].to_crs(epsg=epsg_code)
        calls_in_city['point_idx'] = range(len(calls_in_city))
        
        # PRE-CALC
        radius_m = 3218.69 
        station_metadata = []
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            mask = calls_in_city.geometry.distance(s_pt_m) <= radius_m
            indices = set(calls_in_city[mask]['point_idx'])
            clipped_buf = s_pt_m.buffer(radius_m).intersection(city_m)
            station_metadata.append({'name': row['name'], 'lat': row['lat'], 'lon': row['lon'], 'clipped_m': clipped_buf, 'indices': indices})

        # OPTIMIZER
        st.sidebar.header("🎯 Optimizer Controls")
        k = st.sidebar.slider("Drones to Deploy", 1, len(station_metadata), min(2, len(station_metadata)))
        strategy = st.sidebar.radio("Optimization Goal", ("Maximize Call Volume", "Maximize Land Equity"))

        combos = list(itertools.combinations(range(len(station_metadata)), k))
        if len(combos) > 500: combos = combos[:500] 
        
        best_call_combo, max_calls = None, -1
        best_geo_combo, max_area = -1, -1
        
        for combo in combos:
            u_set = set().union(*(station_metadata[i]['indices'] for i in combo))
            if len(u_set) > max_calls: max_calls = len(u_set); best_call_combo = combo
            if strategy == "Maximize Land Equity":
                u_geo = unary_union([station_metadata[i]['clipped_m'] for i in combo])
                if u_geo.area > max_area: max_area = u_geo.area; best_geo_combo = combo
            
        default_sel = [station_metadata[i]['name'] for i in (best_call_combo if strategy == "Maximize Call Volume" else (best_geo_combo if best_geo_combo != -1 else best_call_combo))]
        active_names = ctrl_col2.multiselect("📡 Current Drone List", options=df_stations_all['name'].tolist(), default=default_sel)

        # --- METRICS ---
        active_data = [s for s in station_metadata if s['name'] in active_names]
        active_indices = [s['indices'] for s in active_data]
        all_ids = set().union(*active_indices) if active_indices else set()
        cap_perc = (len(all_ids) / len(calls_in_city)) * 100 if len(calls_in_city) > 0 else 0
        total_union_geo = unary_union([s['clipped_m'] for s in active_data]) if active_data else None
        land_perc = (total_union_geo.area / city_m.area * 100) if total_union_geo else 0
        
        overlap_perc = 0.0
        if len(active_data) > 1:
            active_bufs = [s['clipped_m'] for s in active_data]
            inters = [active_bufs[i].intersection(active_bufs[j]) for i in range(len(active_bufs)) for j in range(i+1, len(active_bufs)) if not active_bufs[i].intersection(active_bufs[j]).is_empty]
            overlap_perc = (unary_union(inters).area / city_m.area * 100) if inters else 0.0

        # --- HEALTH SCORE ---
        norm_redundancy = min(overlap_perc / 39.0, 1.0) * 100
        health_score = (cap_perc * 0.50) + (land_perc * 0.25) + (norm_redundancy * 0.25)

        if health_score >= 85: h_color, h_label = "#28a745", "OPTIMAL"
        elif health_score >= 75: h_color, h_label = "#94c11f", "SUFFICIENT"
        elif health_score >= 55: h_color, h_label = "#ffc107", "MARGINAL"
        else: h_color, h_label = "#dc3545", "CRITICAL"

        # BANNER
        st.markdown(f"""
            <div style="background-color: {h_color}; padding: 10px; border-radius: 5px; color: white; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 1.2em; font-weight: bold;">Department Health Score: {health_score:.1f}%</span>
                <span style="font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 2px 10px; border-radius: 4px;">{h_label}</span>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Response Capacity", f"{cap_perc:.1f}%")
        m2.metric("Land Covered", f"{land_perc:.1f}%")
        m3.metric("Redundancy", f"{overlap_perc:.1f}%")
        m4.metric("Uncovered Calls", f"{len(calls_in_city) - len(all_ids):,}")

        # --- SIDEBAR SCORECARD ---
        st.sidebar.markdown("---")
        with st.sidebar.expander("📝 Tactical Scorecard", expanded=True):
            summary_text = f"""DRONE DEPLOYMENT ANALYSIS
---------------------------------
Jurisdiction: {selection}
Strategy: {strategy}
Drones Deployed: {len(active_names)}

PERFORMANCE METRICS:
- Health Score: {health_score:.1f}% ({h_label})
- Call Capacity: {cap_perc:.1f}%
- Land Coverage: {land_perc:.1f}%
- Redundancy: {overlap_perc:.1f}%

DEPLOYED LOCATIONS:
""" + "\n".join([f"✅ {name}" for name in active_names])
            st.text_area("Copy Report for Briefing:", summary_text, height=250)

        # --- THE MAP ---
        fig = go.Figure()
        
        # 1. District Lines
        for _, row in gdf_all.to_crs(epsg=4326).iterrows():
            geom = row.geometry
            p_list = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
            for p in p_list:
                bx, by = p.exterior.coords.xy
                fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#444", width=1), showlegend=False, hoverinfo='skip'))
        
        # 2. Incidents (NO HOVER)
        sample = calls_in_city.to_crs(epsg=4326).sample(min(2000, len(calls_in_city)))
        fig.add_trace(go.Scattermap(
            lat=sample.geometry.y, 
            lon=sample.geometry.x, 
            mode='markers', 
            marker=dict(size=4, color='#000080', opacity=0.3), 
            name="Incidents",
            hoverinfo='skip'
        ))
        
        # 3. Stations (Split into Ring and Center Dot)
        all_st_names = df_stations_all['name'].tolist()
        for s in active_data:
            color = STATION_COLORS[all_st_names.index(s['name']) % len(STATION_COLORS)]
            angles = np.linspace(0, 2*np.pi, 60)
            clats = s['lat'] + (2/69.172) * np.sin(angles)
            clons = s['lon'] + (2/(69.172 * np.cos(np.radians(s['lat'])))) * np.cos(angles)
            
            # A) The Ring (No Hover, No Legend Entry)
            fig.add_trace(go.Scattermap(
                lat=list(clats) + [clats[0]], # Close the loop
                lon=list(clons) + [clons[0]], 
                mode='lines', 
                line=dict(color=color, width=4.5), 
                hoverinfo='skip',
                showlegend=False
            ))
            
            # B) The Center Dot (Hover Name Enabled, Big Size)
            fig.add_trace(go.Scattermap(
                lat=[s['lat']], 
                lon=[s['lon']], 
                mode='markers', 
                marker=dict(size=12, color=color), # Increased Size
                name=s['name'],
                hoverinfo='name'
            ))

        fig.update_layout(map_style="open-street-map", map_zoom=11, map_center={"lat": city_boundary.centroid.y, "lon": city_boundary.centroid.x}, margin={"r":0,"t":0,"l":0,"b":0}, height=750)
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("System Ready: Please upload deployment files above to initialize session.")



