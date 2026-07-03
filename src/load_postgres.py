"""Optional: load processed tables into PostgreSQL.

Usage:
    export DATABASE_URL=postgresql://user:pass@localhost:5432/olist
    python src/load_postgres.py

Then explore with the window-function queries in sql/analysis_queries.sql.
Requires: pip install sqlalchemy psycopg2-binary
"""

import os

import pandas as pd
from sqlalchemy import create_engine

if __name__ == "__main__":
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("Set DATABASE_URL first, e.g. "
                         "postgresql://user:pass@localhost:5432/olist")
    engine = create_engine(url)
    for table in ("order_items", "weekly_demand"):
        df = pd.read_csv(f"data/processed/{table}.csv")
        df.to_sql(table, engine, if_exists="replace", index=False)
        print(f"loaded {table}: {len(df):,} rows")
