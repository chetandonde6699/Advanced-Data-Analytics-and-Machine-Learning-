import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === SET BASE PATH (WORKS ANYWHERE) ===
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "data")

# === LOAD DATA (NO .csv EXTENSIONS) ===
demand = pd.read_csv(os.path.join(BASE, "demand.csv"))
plants = pd.read_csv(os.path.join(BASE, "plants.csv"))
gen = pd.read_csv(os.path.join(BASE, "generation_costs.csv"))

# === REMOVE MISSING TARGETS ===
gen = gen.dropna(subset=["Cost_USD_per_MWh"])

# === MERGE THREE DATASETS ===
df = gen.merge(demand, on="Demand ID", how="left") \
        .merge(plants, on="Plant ID", how="left")

# === COLUMN IDENTIFICATION ===
num_cols = [c for c in df.columns if (c.startswith("DF") and c not in ["DF_region", "DF_daytype"]) or c.startswith("PF")]
cat_cols = ["Plant Type", "DF_region", "Region", "DF_daytype"]

# === IMPUTE NUMERIC ===
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# === IMPUTE CATEGORICAL ===
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown")

# === BASELINE RMSE PER PLANT ===
best_cost = df.groupby("Demand ID")["Cost_USD_per_MWh"].min().rename("BestCost")
df = df.merge(best_cost, on="Demand ID")
df["plant_error"] = df["Cost_USD_per_MWh"] - df["BestCost"]

plant_rmse = df.groupby("Plant ID")["plant_error"] \
               .apply(lambda x: np.sqrt(np.mean(x**2))) \
               .sort_values()

# === SAVE FOLDERS ===
OUT = os.path.join(HERE, "outputs")
PROC = os.path.join(HERE, "data_processed")
os.makedirs(OUT, exist_ok=True)
os.makedirs(PROC, exist_ok=True)

# === SAVE PLOT ===
plt.figure(figsize=(12,5))
plant_rmse.plot(kind="bar")
plt.ylabel("RMSE Error")
plt.title("Baseline RMSE per Plant")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "baseline_plant_rmse.png"))
plt.close()

# === PRUNE WORST 20 PLANTS ===
good_plants = plant_rmse.head(44).index
df = df[df["Plant ID"].isin(good_plants)]

# === SAVE CLEANED FILE ===
df.to_csv(os.path.join(PROC, "cleaned_merged_data.csv"), index=False)

print("Step 1 complete.")
