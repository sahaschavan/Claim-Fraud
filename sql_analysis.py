import sqlite3
from pathlib import Path
import pandas as pd
BASE = Path(__file__).resolve().parent
out_dir = BASE / "outputs" / "sql_results"
out_dir.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(BASE / "data" / "claims.db")
text = (BASE / "sql" / "queries.sql").read_text()
blocks = text.split("-- name:")[1:]
pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 20)
for block in blocks:
    name, query = block.split("\n", 1)
    name = name.strip()
    result = pd.read_sql_query(query, con)
    result.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"\n=== {name} ===")
    print(result.to_string(index=False))
con.close()
