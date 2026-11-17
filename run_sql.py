"""
Load the exported hourly tables into SQLite and run sql/queries.sql.
Usage: python run_sql.py     (run analysis.py first)
Output: outputs/dam.db and outputs/sql_results/Q*.csv
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

OUT = Path("outputs")
RES = OUT / "sql_results"
RES.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(OUT / "dam.db")
pd.read_csv(OUT / "market.csv").to_sql("market", conn, if_exists="replace", index=False)
pd.read_csv(OUT / "park_hourly.csv").to_sql("park_hourly", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS ix_ph ON park_hourly(ts_local, park)")
conn.execute("CREATE INDEX IF NOT EXISTS ix_m ON market(month, hour)")

sql_text = Path("queries.sql").read_text(encoding="utf-8")
# split into (title, statement) blocks: each block starts with "-- Qn: title"
blocks = re.split(r"^-- (Q\d+): (.*)$", sql_text, flags=re.M)[1:]
pd.set_option("display.width", 200, "display.max_columns", 20)
for i in range(0, len(blocks), 3):
    qid, title, body = blocks[i], blocks[i + 1], blocks[i + 2]
    stmt = body.strip().rstrip(";")
    df = pd.read_sql_query(stmt, conn)
    df.to_csv(RES / f"{qid}.csv", index=False)
    print(f"\n--- {qid}: {title} ---")
    print(df.to_string(index=False))

conn.close()
print(f"\nDatabase: {(OUT / 'dam.db').resolve()}")
