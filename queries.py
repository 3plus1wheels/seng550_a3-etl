"""
SQL queries for analyzing traffic accidents with spatial and temporal joins.
Includes:


Author: Jamie MacDonald
"""
# Tables:
# weather(date, min_temp_c, max_temp_c, total_precip_mm)
# traffic_incidents(start_dt, geometry)
# community_boundaries(name, geometry)




# ONLY ONE WEATHER BASE IN TABLE?




# Indexing
spatial_indexes = """
CREATE INDEX IF NOT EXISTS idx_traffic_incidents_geom
    ON traffic_incidents
    USING GIST (geometry);


CREATE INDEX IF NOT EXISTS idx_community_boundaries_geom
    ON community_boundaries
    USING GIST (geometry);
"""


temporal_indexes = """
CREATE INDEX IF NOT EXISTS idx_weather_date
    ON weather (date);


CREATE INDEX IF NOT EXISTS idx_traffic_incidents_start_dt
    ON traffic_incidents (start_dt);
"""




# View for less joins
create_mega_view = """
CREATE OR REPLACE VIEW accidents_weather_district_view AS
SELECT
    ti.start_dt AS occurred_at,
    ti.geometry AS accident_geom,
    cb.name AS district_name,
    w.date AS weather_date,
    w.min_temp_c,
    w.max_temp_c,
    w.total_precip_mm
FROM traffic_incidents ti
LEFT JOIN community_boundaries cb
    ON ST_Contains(cb.geometry, ti.geometry)
LEFT JOIN weather w
    ON w.date = ti.start_dt::date;
"""




# Queries
# Accidents per community district
acc_district_query = """
SELECT
    cb.name AS district_name,
    COUNT(ti.*) AS accident_count
FROM community_boundaries cb
LEFT JOIN traffic_incidents ti
    ON ST_Contains(cb.geometry, ti.geometry)
GROUP BY cb.name, cb.geometry;
"""


# Incidents per day and precipitation
acc_day_precip_query = """
SELECT
    w.date,
    COUNT(ti.*) AS accident_count,
    w.total_precip_mm
FROM weather w
LEFT JOIN traffic_incidents ti
    ON w.date = ti.start_dt::date
GROUP BY w.date, w.total_precip_mm
ORDER BY w.date;
"""

# Table for accident analysis with weather data
# Optimized for cloud storage - removed community_geom (dashboard only needs district_name text)
create_acc_fact_table = """
CREATE TABLE accident_facts (
    occurred_date TIMESTAMP NOT NULL,
    modified_dt TIMESTAMP,
    district_name TEXT,
    min_temp_c FLOAT,
    max_temp_c FLOAT,
    total_precip_mm FLOAT,
    accident_lon FLOAT NOT NULL,
    accident_lat FLOAT NOT NULL,
    accident_geom GEOMETRY(POINT, 4326),
    PRIMARY KEY (occurred_date, accident_lon, accident_lat)
);
    
-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_accident_facts_geom
    ON accident_facts USING GIST (accident_geom);
CREATE INDEX IF NOT EXISTS idx_accident_facts_date
    ON accident_facts (occurred_date);
CREATE INDEX IF NOT EXISTS idx_accident_facts_community
    ON accident_facts  (district_name);
CREATE INDEX IF NOT EXISTS idx_accident_facts_precip
    ON accident_facts (total_precip_mm);
"""

update_acc_fact_data = """
INSERT INTO accident_facts (
    occurred_date,
    modified_dt,
    district_name,
    min_temp_c,
    max_temp_c,
    total_precip_mm,
    accident_lon,
    accident_lat,
    accident_geom
)
SELECT DISTINCT ON (occurred_date, accident_lon, accident_lat) -- will keep the last modified_dt
    ti.start_dt::timestamp AS occurred_date,
    ti.modified_dt::timestamp AS modified_dt,
    cb.name AS district_name,
    w.min_temp_c,
    w.max_temp_c,
    w.total_precip_mm,
    ST_X(ti.geometry) AS accident_lon,
    ST_Y(ti.geometry) AS accident_lat,
    ti.geometry AS accident_geom
FROM traffic_incidents ti
LEFT JOIN community_boundaries cb
    ON ST_Contains(cb.geometry, ti.geometry)
LEFT JOIN weather w
    ON ti.start_dt::timestamp::date = w.date
WHERE ti.geometry IS NOT NULL
ORDER BY occurred_date, accident_lon, accident_lat, ti.modified_dt DESC
ON CONFLICT (occurred_date, accident_lon, accident_lat) DO UPDATE 
SET
    modified_dt = EXCLUDED.modified_dt,
    district_name = EXCLUDED.district_name,
    min_temp_c = EXCLUDED.min_temp_c,
    max_temp_c = EXCLUDED.max_temp_c,
    total_precip_mm = EXCLUDED.total_precip_mm,
    accident_geom = EXCLUDED.accident_geom
WHERE accident_facts.modified_dt < EXCLUDED.modified_dt;
"""


# Accidents with weather conditions at time and location of accident -
# FOR MULTIPLE WEATHER BASES, not implemented (yet)
# acc_weather_query = """
# SELECT
#     a.accident_id,
#     a.occurred_at,
#     a. geom AS accident_location,
#     w.weather_condition,
#     w.precipitation_mm,
#     w.temperature_c
# FROM traffic_incidents a
# -- do subquery for each accident to get closest weather record
# JOIN LATERAL (
#     SELECT
#         w.weather_condition,
#         w.precipitation_mm,
#         w.temperature_c
#     FROM weather w
#     WHERE ST_DWithin(
#         ST_SetSRID(ST_MakePoint(w.longitude, w.latitude), 4326),
#         a.geom,
#         10000 -- within 10 km?
#     )
#     AND w.recorded_at -- Need to get closest time before accident. idk column names
#     ORDER BY w.recorded_at DESC
#     LIMIT 1
# ) w ON TRUE;
# """

