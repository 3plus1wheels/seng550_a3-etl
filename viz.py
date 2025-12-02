'''
Used for getting POSTGIS setup/connection in python:
https://medium.com/nerd-for-tech/geographic-data-visualization-using-geopandas-and-postgresql-7578965dedfe
Used for streamlit:

'''


import geopandas as gpd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import json
import pandas as pd
import streamlit as st
import pydeck as pdk


# from queries import acc_view, acc_district_query, acc_weather_query
# weather(date, min_temp_c, max_temp_c, total_precip_mm)
# traffic_incidents(start_dt, geometry)
# community_boundaries(name, geometry)

load_dotenv()
# Connecting to PostgreSQL database
host = os.getenv("PGHOST", "localhost")
port = os.getenv("PGPORT", "5432")
dbname = os.getenv("PGDB", "a3_db")
user = os.getenv("PGUSER", "postgres")
password = os.getenv("PGPASS", "mcfruity")

connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_string)


# Basic viz: just accidents on map. To test database connection and geopandas read
def plot_accidents():
    """ 
    Plot accidents from accident_geo_view
    """
    with engine.connect() as conn:
        acc_gdf = gpd.read_postgis("SELECT * FROM accident_geo_view;", conn, geom_col='geom')
        
        # Plot accidents
        ax = acc_gdf.plot(figsize=(10, 10), color='red', alpha=0.5, markersize=5)
        ax.set_title("Traffic Accidents")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return ax
    
# _______________Heavier data loading func with caching ______________________________________________
# By using @st.cache_data, we avoid reloading data on every interaction :D

@st.cache_data(ttl=300) # Performance improvement - cache for 5 minutes
def load_accidents_detailed():
    # Load accidents from materialized view
    query = """
        SELECT
            occurred_at,
            geom,
            district_name,
            weather_date,
            min_temp_c,
            max_temp_c,
            total_precip_mm,
            lon,
            lat
        FROM accident_geo_view;  -- ← Using materialized view now
    """
    gdf = gpd.read_postgis(query, engine, geom_col="geom")
    gdf["occurred_at"] = pd.to_datetime(gdf["occurred_at"])
    
    return gdf


@st.cache_data(ttl=300)
def load_districts_with_counts():
    """ Load districts with accident counts, with geometry."""
    query = """
        SELECT
            cb.name AS district_name,
            COUNT(ti.*) AS accident_count,
            cb.geometry AS geom
        FROM community_boundaries cb
        LEFT JOIN traffic_incidents ti
            ON ST_Contains(cb.geometry, ti.geometry)
        GROUP BY cb.name, cb.geometry;
    """
    gdf = gpd.read_postgis(query, engine, geom_col="geom")
    return gdf

@st.cache_data(ttl=300)
def load_daily_weather_accidents():
    """ Load daily weather and accident counts."""
    query = """
        SELECT
            w.date,
            COUNT(ti.*) AS accident_count,
            w.total_precip_mm,
            w.min_temp_c,
            w.max_temp_c
        FROM weather w
        LEFT JOIN traffic_incidents ti
            ON w.date = ti.start_dt::date
        GROUP BY w.date, w.total_precip_mm, w.min_temp_c, w.max_temp_c
        ORDER BY w.date;
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])
    return df

# ___________________ Data Loading ______________________________________________
# Load data once at start, with error handling
try:
    acc = load_accidents_detailed()
    districts = load_districts_with_counts()
    daily_stats = load_daily_weather_accidents()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if acc.empty:
    st.warning("No data found in accidents_weather_district_view")
    st.stop()

# ___________________ Streamlit App ______________________________________________

# SECTION 1: **FILTERING! SIDEBAR****
# Give some filters in sidebar

st.sidebar.header("Filters")

# Date range
min_date = acc["occurred_at"].min().date()
max_date = acc["occurred_at"].max().date()

date_range = st.sidebar.date_input(
    "Accident date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# District
district_options = sorted(acc["district_name"].dropna().unique().tolist())

# Add "Select All" button
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    if st.button("Select All"):
        st.session_state.selected_districts = district_options
with col2:
    if st.button("Clear All"):
        st.session_state.selected_districts = []

# Initialize session state if it doesn't exist
if "selected_districts" not in st.session_state:
    st.session_state.selected_districts = district_options

selected_districts = st.sidebar.multiselect(
    "District",
    options=district_options,
    default=st.session_state.selected_districts,
    key="district_multiselect"
)

# Update session state
st.session_state.selected_districts = selected_districts

# Precipitation
min_precip = float(acc["total_precip_mm"].min())
max_precip = float(acc["total_precip_mm"].max())

#lil slider for precip
precip_range = st.sidebar.slider(
    "Total precipitation (mm)",
    min_value=min_precip,
    max_value=max_precip,
    value=(min_precip, max_precip)
)

# "wet vs dry" toggle
wet_only = st.sidebar.checkbox("Wet conditions only (precipitation > 0)", value=False)

# Map visualization type
st.sidebar.subheader("Map Display")
show_choropleth = st.sidebar.checkbox("Show choropleth (by district)", value=False)


# PART 2: FILTER DATA BASED ON SIDEBAR INPUTS

# Apply filters to acc DataFrame
mask = (
    (acc["occurred_at"].dt.date >= date_range[0]) &
    (acc["occurred_at"].dt.date <= date_range[1])
)

if selected_districts:
    mask &= acc["district_name"].isin(selected_districts)

mask &= (
    (acc["total_precip_mm"] >= precip_range[0]) &
    (acc["total_precip_mm"] <= precip_range[1])
)

if wet_only:
    mask &= acc["total_precip_mm"] > 0


# PART 3 ? LAYOUT THE DASHBOARD
st.title("Accident & Weather Dashboard")
filt_acc = acc[mask]
col_map, col_summary = st.columns([2, 1])

# ____________________Map viz ___________________________
with col_map:
    if filt_acc.empty:
        st.info("No accidents match the selected filters.")
    else:
        # Calculate median lat/lon for centering map 
        mid_lat = filt_acc["lat"].median()
        mid_lon = filt_acc["lon"].median()
        
        if show_choropleth:
            # Choropleth map: color districts by accident count
            # Count accidents per district from filtered data
            district_counts = filt_acc.groupby("district_name").size().reset_index(name="accident_count")
            
            # Merge with district geometries
            choropleth_data = districts.merge(district_counts, on="district_name", how="left", suffixes=("", "_filtered"))
            choropleth_data["accident_count_filtered"] = choropleth_data["accident_count_filtered"].fillna(0)
            
            # Convert geometry to GeoJSON format for pydeck
            choropleth_data["coordinates"] = choropleth_data["geom"].apply(
                lambda x: json.loads(gpd.GeoSeries([x]).to_json())["features"][0]["geometry"]["coordinates"]
            )
            
            # Normalize accident counts for color scaling
            max_count = choropleth_data["accident_count_filtered"].max()
            if max_count > 0:
                choropleth_data["color_value"] = (choropleth_data["accident_count_filtered"] / max_count * 255).astype(int)
            else:
                choropleth_data["color_value"] = 0
            
            # Create color based on accident count (red gradient)
            choropleth_data["fill_color"] = choropleth_data["color_value"].apply(
                lambda x: [255, 255 - x, 255 - x, 140]  # Red gradient with transparency
            )
            
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/streets-v11',
                initial_view_state=pdk.ViewState(
                    latitude=mid_lat,
                    longitude=mid_lon,
                    zoom=10.5,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        'GeoJsonLayer',
                        data=choropleth_data,
                        opacity=0.6,
                        stroked=True,
                        filled=True,
                        extruded=False,
                        wireframe=True,
                        get_fill_color='fill_color',
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=2,
                        pickable=True,
                        auto_highlight=True,
                    ),
                ],
                tooltip={
                    "html": "<b>{district_name}</b><br/>Accidents: {accident_count_filtered}",
                    "style": {"backgroundColor": "steelblue", "color": "white"}
                }
            ))
            
            # Show color legend
            st.caption(f"Darker red = More accidents (Max: {int(max_count)} accidents)")
            
        else:
            # Scatter plot: individual accident points
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/dark-v10',
                initial_view_state=pdk.ViewState(
                    latitude=mid_lat,
                    longitude=mid_lon,
                    zoom=10.5,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=filt_acc,
                        get_position='[lon, lat]',
                        get_color='[200, 30, 0, 160]',
                        get_radius=50,
                        pickable=True,
                    ),
                ],
            ))

with col_summary:
    st.subheader("Summary Statistics")
    st.metric("Total Accidents", len(filt_acc))
    st.metric("Avg Temperature (°C)", f"{filt_acc['min_temp_c'].mean():.1f} - {filt_acc['max_temp_c'].mean():.1f}")
    st.metric("Avg Precipitation (mm)", f"{filt_acc['total_precip_mm'].mean():.2f}")
