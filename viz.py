'''
Used for getting POSTGIS setup/connection in python:
https://medium.com/nerd-for-tech/geographic-data-visualization-using-geopandas-and-postgresql-7578965dedfe
Used for streamlit:
https://medium.com/@verinamk/streamlit-for-beginners-build-your-first-dashboard-58b764a62a2d
'''


import geopandas as gpd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import pydeck as pdk


# from queries import acc_view, acc_district_query, acc_weather_query
# weather(date, min_temp_c, max_temp_c, total_precip_mm)
# traffic_incidents(start_dt, geometry)
# community_boundaries(name, geometry)

# Load .env file for local development (override=True to override system env vars)
# Use an explicit path so Streamlit's working directory won't affect finding the file
# ONLY load .env (not .env.example which may have localhost config)
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

# Build database connection string.
# New order of precedence (local-first):
# 1) .env `DATABASE_URL` (explicitly from repo .env) - local development convenience
# 2) Streamlit secrets: DATABASE_URL/DB_URL or PG* keys - cloud deployment
# 3) process environment PG* keys
env_path = Path(__file__).parent.joinpath('.env')
file_db_url = None
if env_path.exists():
    file_vals = dotenv_values(env_path)
    file_db_url = file_vals.get('DATABASE_URL')

connection_string = None

# 1) Prefer Streamlit secrets when present (cloud authoritative)
if hasattr(st, "secrets") and st.secrets and len(st.secrets) > 0:
    secrets = st.secrets
    connection_string = (
        secrets.get("DATABASE_URL")
        or secrets.get("DB_URL")
    )
    if not connection_string:
        for v in secrets.values():
            if isinstance(v, str) and v.startswith("postgres") and "@" in v:
                connection_string = v
                break
    if not connection_string and "PGHOST" in secrets:
        host = secrets.get("PGHOST")
        port = str(secrets.get("PGPORT", "5432"))
        dbname = secrets.get("PGDB")
        user = secrets.get("PGUSER")
        password = secrets.get("PGPASSWORD")
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# 2) repo .env DATABASE_URL (local development)
if not connection_string and file_db_url:
    connection_string = file_db_url

# 3) process environment (fallback)
if not connection_string:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDB", "a3_db")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "")
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# Final fallback: read .env file directly if the process env still points to localhost
if connection_string and ("localhost" in connection_string or "127.0.0.1" in connection_string):
    # Try to read the repo .env values directly (in case system env overrides are set)
    env_path = Path(__file__).parent.joinpath('.env')
    if env_path.exists():
        dotvals = dotenv_values(env_path)
        # Prefer DATABASE_URL from .env
        file_db_url = dotvals.get('DATABASE_URL')
        if file_db_url:
            connection_string = file_db_url
        else:
            fh = dotvals.get('PGHOST')
            if fh and fh != 'localhost':
                host = fh
                port = dotvals.get('PGPORT', port)
                dbname = dotvals.get('PGDB', dbname)
                user = dotvals.get('PGUSER', user)
                password = dotvals.get('PGPASSWORD', password)
                connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# Ensure SSL for cloud hosts when not explicitly set
if "@" in connection_string and "localhost" not in connection_string and "127.0.0.1" not in connection_string:
    if "sslmode=" not in connection_string:
        if "?" in connection_string:
            connection_string += "&sslmode=require"
        else:
            connection_string += "?sslmode=require"

# Sanitize connection string (remove leading 'psql ', surrounding quotes)
cs = connection_string.strip()
if cs.startswith("psql "):
    cs = cs[len("psql "):].strip()
if (cs.startswith("'") and cs.endswith("'")) or (cs.startswith('"') and cs.endswith('"')):
    cs = cs[1:-1]

def _mask_pw(s: str) -> str:
    try:
        proto, rest = s.split('://', 1)
        creds, hostpart = rest.split('@', 1)
        if ':' in creds:
            user, pw = creds.split(':', 1)
            return f"{proto}://{user}:***@{hostpart}"
    except Exception:
        return s
masked = _mask_pw(cs)

# Show which connection string source is being used (masked)
st.info(f"Using DB connection: {masked}")

try:
    # If connecting to localhost, don't force SSL connect args
    if "localhost" in cs or "127.0.0.1" in cs:
        engine = create_engine(cs)
    else:
        # For cloud DBs (Neon/Supabase) ensure SSL mode is passed to the DBAPI
        engine = create_engine(cs, connect_args={"sslmode": "require"})
except Exception as e:
    st.error(f"Failed to create DB engine from connection string: {masked}\n\nError: {e}")
    st.stop()


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
    # switched from loading materialized view to a regular view in a fact table for up-to-date data
    query = """
        SELECT
            occurred_date,
            accident_geom,
            district_name,
            min_temp_c,
            max_temp_c,
            total_precip_mm,
            accident_lon,
            accident_lat
        FROM accident_facts;  -- ← Using materialized view now
    """
    # Read using actual geom column name from the table
    gdf = gpd.read_postgis(query, engine, geom_col="accident_geom")

    # Normalize column names so the rest of the app can keep using the original names
    gdf = gdf.rename(columns={
        "occurred_date": "occurred_at",
        "accident_geom": "geom",
        "accident_lon": "lon",
        "accident_lat": "lat",
    })

    # Ensure geometry column is set correctly after rename
    gdf = gdf.set_geometry("geom")

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
