import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent
fig_dir = BASE / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(BASE / "data" / "claims.db")
df = pd.read_sql_query("SELECT * FROM claims", con)
con.close()

sns.set_theme(style="whitegrid")
overall = df["fraud"].mean() * 100

# chart 1: fraud rate by incident severity
rate = df.groupby("incident_severity")["fraud"].mean().mul(100).sort_values(ascending=False)
plt.figure(figsize=(7, 4))
ax = sns.barplot(x=rate.values, y=rate.index, color="#2a78d6")
ax.axvline(overall, color="#eb6834", linestyle="--", label=f"overall ({overall:.0f}%)")
ax.set_xlabel("Fraud rate (%)")
ax.set_ylabel("")
ax.set_title("Fraud rate by incident severity")
ax.legend()
plt.tight_layout()
plt.savefig(fig_dir / "01_fraud_by_severity.png", dpi=150)
plt.close()

# chart 2: claim amount for fraud vs non-fraud
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="fraud", y="total_claim_amount", color="#9cc3f0")
plt.xticks([0, 1], ["Not fraud", "Fraud"])
plt.xlabel("")
plt.ylabel("Total claim amount ($)")
plt.title("Claim amount: fraud vs not fraud")
plt.tight_layout()
plt.savefig(fig_dir / "02_claim_amount_by_fraud.png", dpi=150)
plt.close()

# chart 3: fraud rate by hobby (only hobbies with 20+ claims)
counts = df["insured_hobbies"].value_counts()
keep = counts[counts >= 20].index
rate_h = (df[df["insured_hobbies"].isin(keep)].groupby("insured_hobbies")["fraud"]
          .mean().mul(100).sort_values())
plt.figure(figsize=(7, 5))
ax = sns.barplot(x=rate_h.values, y=rate_h.index, color="#2a78d6")
ax.axvline(overall, color="#eb6834", linestyle="--")
ax.set_xlabel("Fraud rate (%)")
ax.set_ylabel("")
ax.set_title("Fraud rate by insured hobby (orange line = overall)")
plt.tight_layout()
plt.savefig(fig_dir / "03_fraud_by_hobby.png", dpi=150)
plt.close()

# chart 4 : correlation heatmap of the numeric columns
num_cols = ["age", "months_as_customer", "policy_annual_premium", "total_claim_amount",
            "injury_claim", "property_claim", "vehicle_claim", "witnesses",
            "bodily_injuries", "incident_hour_of_the_day", "fraud"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="Blues", cbar=False)
plt.title("Correlation of numeric columns")
plt.tight_layout()
plt.savefig(fig_dir / "04_correlation_heatmap.png", dpi=150)
plt.close()

print("saved 4 charts to", fig_dir)
