from sqlalchemy import create_engine, text
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection
host = os.getenv("PGHOST", "localhost")
port = os.getenv("PGPORT", "5432")
dbname = os.getenv("PGDB", "a3_db")
user = os.getenv("PGUSER", "postgres")
password = os.getenv("PGPASSWORD","mcfruity")
engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

# query to count incidents percomunity to look at spatial join performance
test_query = """
SELECT cb.name, COUNT(ti.*) AS accident_count
FROM community_boundaries cb
LEFT JOIN traffic_incidents ti ON ST_Contains(cb.geometry, ti.geometry)
GROUP BY cb.name;
"""

# query with no joins to looks at the materialized view performance
view_query = """
SELECT district_name, COUNT(*) AS accident_count
FROM accident_geo_view
GROUP BY district_name;
"""

def benchmark(query, runs=3):
    # runs a query multiple times (3) and calculates the average speed 
    times = []
    with engine.connect() as conn:
        for _ in range(runs):
            start = time.time()
            conn.execute(text(query)).fetchall()
            times.append((time.time() - start) * 1000)
    return sum(times) / len(times)

def drop_indexes():
    #remove indexing to test performance without it
    with engine.connect() as conn:
        conn.execute(text("DROP INDEX IF EXISTS idx_traffic_geom;"))
        conn.execute(text("DROP INDEX IF EXISTS idx_boundaries_geom;"))
        conn.commit()

def create_indexes():
    #bring the indexes back after dropping them
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_traffic_geom
                ON traffic_incidents USING GIST (geometry);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_boundaries_geom
                ON community_boundaries USING GIST (geometry);
        """))
        conn.commit()


print("PERFORMANCE TEST\n")

# No indexes
drop_indexes()
time_no_idx = benchmark(test_query)
print(f"No indexes:{time_no_idx:>8.1f} ms (baseline)")

# With indexes
create_indexes()
time_with_idx = benchmark(test_query)
speedup_idx = time_no_idx / time_with_idx
print(f"With indexes: {time_with_idx:>8.1f} ms ({speedup_idx:.1f}x faster)")

# With Materialized view
time_mat_view = benchmark(view_query)
speedup_view = time_no_idx / time_mat_view
print(f"Materialized view: {time_mat_view:>8.1f} ms ({speedup_view:.1f}x faster)")
