import sqlite3
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "data" / "insurance_claims.csv"
DB_PATH = BASE / "data" / "claims.db"

df = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=[""])
print("raw shape:", df.shape)

df = df.dropna(axis=1, how="all")
df.columns = [c.strip().replace("-", "_") for c in df.columns]
df = df.replace("?", "Unknown")
for col in ["policy_bind_date", "incident_date"]:
    df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
df["fraud"] = (df["fraud_reported"] == "Y").astype(int)
print("duplicate rows:", df.duplicated().sum())
print("null values left:", int(df.isna().sum().sum()))
print("fraud rate: {:.1%}".format(df["fraud"].mean()))
df.insert(0, "claim_id", range(1, len(df) + 1))

con = sqlite3.connect(DB_PATH)
df.to_sql("claims", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_incident_type ON claims(incident_type)")
con.execute("CREATE INDEX IF NOT EXISTS idx_state ON claims(incident_state)")
con.commit()
print("loaded", con.execute("SELECT COUNT(*) FROM claims").fetchone()[0], "rows into", DB_PATH.name)
con.close()
