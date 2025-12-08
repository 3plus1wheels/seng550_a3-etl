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


from queries import acc_view, acc_district_query, acc_weather_query
# weather(date, min_temp_c, max_temp_c, total_precip_mm)
# traffic_incidents(start_dt, geometry)
# community_boundaries(name, geometry)

load_dotenv()
# Connecting to PostgreSQL database
# Use Streamlit secrets in production, .env in development
if "PGHOST" in st.secrets:
    host = st.secrets["PGHOST"]
    port = int(st.secrets["PGPORT"])
    dbname = st.secrets["PGDB"]
    user = st.secrets["PGUSER"]
    password = st.secrets["PGPASSWORD"]
else:
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
    """ Load accidents with weather and district info from view."""
    query = """
        SELECT
            start_dt,
            accident_geom,
            district_name,
            weather_date,
            min_temp_c,
            max_temp_c,
            total_precip_mm
        FROM accidents_weather_district_view;
    """
    # Read into GeoDataFrame
    gdf = gpd.read_postgis(query, engine, geom_col="accident_geom")
    # Extract lon/lat (for pydeck shtuff)
    gdf["lon"] = gdf["accident_geom"].x
    gdf["lat"] = gdf["accident_geom"].y
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
    st.error(f"{e}")
    st.stop()

if acc.empty:
    st.warning("No data found in accidents_detailed ")
    st.stop()

# ___________________ Streamlit App ______________________________________________

# SECTION 1: **FILTERING! SIDEBAR****
# Give some filters in sidebar

st.sidebar.header("Filters")

# Date range
min_date = acc["start_dt"].min().date()
max_date = acc["start_dt"].max().date()

date_range = st.sidebar.date_input(
    "Accident date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# District
district_options = sorted(acc["district_name"].dropna().unique().tolist())
selected_districts = st.sidebar.multiselect(
    "District",
    options=district_options,
    default=district_options
)

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

# Simple “wet vs dry” toggle
wet_only = st.sidebar.checkbox("Wet conditions only (precipitation > 0)", value=False)


# PART 2: FILTER DATA BASED ON SIDEBAR INPUTS

# Apply filters to acc DataFrame
mask = (
    (acc["start_dt"].dt.date >= date_range[0]) &
    (acc["start_dt"].dt.date <= date_range[1])
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
        st.info("No accidents match the selected filters.") # info or warning?
    else:
        # Calculate median lat/lon for centering map 
        mid_lat = filt_acc["lat"].median()
        mid_lon = filt_acc["lon"].median()
        
        
       
    
   