# etl.py
import os
from meteostat import Point, Daily, Hourly
from datetime import datetime
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import requests
import json
from sqlalchemy import create_engine, text
from queries import create_acc_fact_table, update_acc_fact_data

load_dotenv()

# API endpoints
API_TRAFFIC = os.getenv("API_TRAFFIC_ENDPOINT")
API_BORDERS = os.getenv("API_BORDERS_ENDPOINT")

# PostgreSQL credentials
PGHOST = os.getenv("PGHOST")
PGPORT = int(os.getenv("PGPORT"))
PGDB = os.getenv("PGDB")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")

# Set up SQLAlchemy engine to load data into PostgreSQL
def get_db_engine(database_name=None):
    """Create SQLAlchemy engine for PostgreSQL"""
    db_name = database_name or PGDB
    connection_string = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{db_name}"
    return create_engine(connection_string)

def create_database_if_not_exists(db_name):
    """Create PostgreSQL database if it doesn't exist"""
    try:
        # Connect to default postgres database to create new database
        admin_engine = get_db_engine("postgres")
        
        with admin_engine.connect() as conn:
            # Set autocommit mode for database creation
            conn.execute(text("COMMIT"))
            
            # Check if database exists
            result = conn.execute(text(f"""
                SELECT 1 FROM pg_database WHERE datname = '{db_name}'
            """))
            
            if not result.fetchone():
                # Database doesn't exist, create it
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                print(f"Database '{db_name}' created successfully")
            else:
                print(f"Database '{db_name}' already exists")
                
        admin_engine.dispose()
        
    except Exception as e:
        print(f"Error creating database: {e}")
        raise

# Fetch data from API
def fetch_data_from_api(endpoint, limit=50000):
    try:
        print(f"Fetching data from {endpoint}...")

        params = {"$limit": limit}
        
        response = requests.get(endpoint, params=params)
        
        data = response.json()
        # Handle GeoJSON format
        if isinstance(data, dict) and "features" in data:
            # Extract features from GeoJSON
            features = data["features"]
            print(f"Successfully fetched \n")
            return features
        else:
            print(f"Successfully fetched \n")
            return data
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

# Fetch weather data using Meteostat
def fetch_weather_data():    
    calgary = Point(51.0501, -114.0853, 1042) # Calgary coordinates
    start = datetime(2018, 5, 5) # Start date
    end = datetime(2025, 11, 30) # End date

    weather_data = Daily(calgary, start, end)
    weather_data = weather_data.fetch()
    weather = weather_data.reset_index()    # Convert index to column

    # All columns
    weather.columns = ['date', 'avg_temp_c', 'min_temp_c', 'max_temp_c', 
                       'total_precip_mm', 'snow_depth_cm', 'wind_speed', 
                       'peak_gust', 'wind_dir', 'atm_pressure', 'sunshine']

    # Select relevant columns
    weather_clean = weather[['date', 'min_temp_c', 'max_temp_c', 
                             'total_precip_mm']].copy()
    
    print(f"Fetched {len(weather_clean)} weather records\n")
    return weather_clean

# Convert GeoJSON geometry to WKT (Well-Known Text) for PostGIS
def geojson_to_wkt(geom):
    if not geom:
        return None
    
    geom_type = geom.get('type')
    coords = geom.get('coordinates')
    
    if geom_type == 'Point':
        # point: [lon, lat]
        return f"POINT({coords[0]} {coords[1]})"
    
    elif geom_type == 'MultiPolygon':
        # multipolygon: [[[[lon, lat], [lon, lat], ...]]]
        polygons = []
        for polygon in coords:
            rings = []
            for ring in polygon:
                points = ', '.join([f"{pt[0]} {pt[1]}" for pt in ring])
                rings.append(f"({points})")
            polygons.append(f"({', '.join(rings)})")
        return f"MULTIPOLYGON({', '.join(polygons)})"
    
    return None

# Convert JSON/GeoJSON data to pandas DataFrame with PostGIS geometries
def json_to_dataframe(data):
    if not data:
        return pd.DataFrame()
    
    records = []
    for item in data:
        if 'properties' in item:
            # GeoJSON format - extract properties
            record = item['properties'].copy()
            if 'geometry' in item and item['geometry']:
                # Convert GeoJSON geometry to WKT for PostGIS
                record['geometry'] = geojson_to_wkt(item['geometry'])
        else:
            # Regular JSON
            record = item.copy()
            # Convert any dict/list fields to JSON strings
            for key, value in record.items():
                if isinstance(value, (dict, list)):
                    # Convert multipolygon and other complex fields to WKT
                    if key == 'multipolygon' and isinstance(value, dict):
                        record[key] = geojson_to_wkt(value)
                    else:
                        record[key] = json.dumps(value)
        records.append(record)
    
    df = pd.DataFrame(records)
    return df


def load_to_postgres(df, table_name, engine, if_exists='replace', has_geometry=False):
    """Load DataFrame to PostgreSQL table with optional PostGIS geometry handling"""
    try:
        # Drop dependent materialized views first if replacing tables
        if if_exists == 'replace':
            with engine.connect() as conn:
                if table_name in ['traffic_incidents', 'community_boundaries', 'weather']:
                    try:
                        conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS accident_geo_view CASCADE;"))
                        conn.commit()
                        print(f"Dropped dependent views for {table_name}")
                    except Exception as e:
                        print(f"Note: {e}")
        
        # Load data to PostgreSQL
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        
        if has_geometry and 'geometry' in df.columns:
            with engine.connect() as conn:
                # Try to use PostGIS if available, fallback to text
                try:
                    # Test if PostGIS is available
                    conn.execute(text("SELECT PostGIS_Version();"))
                    
                    # PostGIS is available - convert to proper geometry type
                    conn.execute(text(f"""
                        ALTER TABLE {table_name} 
                        ADD COLUMN geom geometry(GEOMETRY, 4326);
                    """))
                    
                    conn.execute(text(f"""
                        UPDATE {table_name} 
                        SET geom = ST_GeomFromText(geometry, 4326)
                        WHERE geometry IS NOT NULL;
                    """))
                    
                    conn.execute(text(f"ALTER TABLE {table_name} DROP COLUMN geometry;"))
                    conn.execute(text(f"ALTER TABLE {table_name} RENAME COLUMN geom TO geometry;"))
                    
                    print(f"PostGIS geometry columns created")
                    
                except Exception:
                    # PostGIS not available - add geometry type info as comment
                    conn.execute(text(f"""
                        COMMENT ON COLUMN {table_name}.geometry IS 'WKT geometry data (SRID: 4326)';
                    """))
                    print(f"Geometry stored as WKT text (PostGIS unavailable)")
                
                conn.commit()
        
        print(f"Successfully saved to '{table_name}' table")
                        
        # Verify the data was saved
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            print(f"{count} rows verified\n")
            
    except Exception as e:
        print(f"Error loading to {table_name}: {e}\n")


def create_indexes(engine):
    # Create spatial and temporal indexes to improve speed
    # Comment this function out and create_indexes() call in main to compare speed
    
    with engine.connect() as conn:
        # Spatial indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_traffic_geom
                ON traffic_incidents USING GIST (geometry);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_boundaries_geom
                ON community_boundaries USING GIST (geometry);
        """))
        
        # Temporal indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_traffic_date
                ON traffic_incidents (start_dt);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_weather_date
                ON weather (date);
        """))
        
        conn.commit()

def create_materialized_view(engine):
    # Create materialized view for faster dashboard queries
    
    with engine.connect() as conn:
        # Drop if exists
        conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS accident_geo_view;"))
        
        # Create materialized view
        conn.execute(text("""
            CREATE MATERIALIZED VIEW accident_geo_view AS
            SELECT
                ti.start_dt AS occurred_at,
                ti.geometry AS geom,
                cb.name AS district_name,
                w.date AS weather_date,
                w.min_temp_c,
                w.max_temp_c,
                w.total_precip_mm,
                ST_X(ti.geometry) AS lon,
                ST_Y(ti.geometry) AS lat
            FROM traffic_incidents ti
            LEFT JOIN community_boundaries cb
                ON ST_Contains(cb.geometry, ti.geometry)
            LEFT JOIN weather w
                ON w.date::date = ti.start_dt::date;
        """))
        
        # Create index on materialized view
        conn.execute(text("""
            CREATE INDEX idx_accident_geo_view_geom 
                ON accident_geo_view USING GIST (geom);
        """))
        
        conn.commit()
        
        # Show row count
        result = conn.execute(text("SELECT COUNT(*) FROM accident_geo_view;"))
        count = result.scalar()

def create_accident_analysis_table(engine):
    #Create denormalized table for FAST accident analysis (will not be deleted on reruns)
    
    with engine.connect() as conn:
        conn.execute(text(create_acc_fact_table))
        conn.commit()
        
        # Show row count
        result = conn.execute(text("SELECT COUNT(*) FROM accident_facts;"))
        count = result.scalar()
        print(f"Accident analysis table created with {count} records\n")
        
def update_accident_analysis_table(engine):
    #Update denormalized table for SPEEDY accident analysis. This upserts existing table.
    
    with engine.connect() as conn:
        conn.execute(text(update_acc_fact_data))
        conn.commit()
        
        # Show row count
        result = conn.execute(text("SELECT COUNT(*) FROM accident_facts;"))
        count = result.scalar()
        print(f"Accident analysis table updated with {count} records\n")
        


def main():
    # Create the PostGIS database
    create_database_if_not_exists("a3_db")
    
    # Connect to the new database
    engine = get_db_engine("a3_db")    
    
    # Enable PostGIS extension
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.commit()
            print("PostGIS extension enabled\n")
    except Exception as e:
        print(f"PostGIS warning: {e}\n")

    
    # ========== 1. WEATHER DATA ============================

    weather_df = fetch_weather_data()
    load_to_postgres(weather_df, 'weather', engine)
    
    # ========== 2. TRAFFIC INCIDENTS ======================
    traffic_data = fetch_data_from_api(API_TRAFFIC, limit=50000)
    if traffic_data:
        traffic_df = json_to_dataframe(traffic_data)
        # Available: ['count', 'latitude', 'description', 'incident_info', 'start_dt', 'modified_dt', 'longitude', 'id', 'quadrant', 'geometry']
        traffic_clean = traffic_df[['start_dt','geometry', 'modified_dt', 'id']]
        
        load_to_postgres(traffic_clean, 'traffic_incidents', engine, has_geometry=True)
    else:
        print("No traffic data fetched\n")
        
    # ========== 3. COMMUNITY BOUNDARIES ====================
    borders_data = fetch_data_from_api(API_BORDERS, limit=50000)
    if borders_data:
        borders_df = json_to_dataframe(borders_data)
        # Available: ['comm_structure', 'class', 'comm_code', 'name', 'sector', 'srg', 'class_code', 'created_dt', 'modified_dt', 'geometry']
        borders_clean = borders_df[['name', 'geometry']]
        
        load_to_postgres(borders_clean, 'community_boundaries', engine, has_geometry=True)
    else:
        print("No borders data fetched\n")

    create_indexes(engine) # Actually implements the indexes
    create_materialized_view(engine)
    
    # ========== 4. CREATE ACCIDENT ANALYSIS TABLE (denormalized table) ====================
    create_accident_analysis_table(engine)
    update_accident_analysis_table(engine)

if __name__ == "__main__":
    main()
