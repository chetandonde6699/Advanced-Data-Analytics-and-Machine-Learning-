import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "data_processed")
OUT = os.path.join(HERE, "outputs")

df = pd.read_csv(os.path.join(PROC, "cleaned_merged_data.csv"))

num_cols = [c for c in df.columns if (c.startswith("DF") and c not in ["DF_region", "DF_daytype"]) or c.startswith("PF")]
cat_cols = ["Plant Type", "DF_region", "Region", "DF_daytype"]
feature_cols = num_cols + cat_cols + ["Demand ID"]

X = df[feature_cols]
y = df["Cost_USD_per_MWh"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# 20-demand grouped split
unique_ids = df["Demand ID"].unique()
test_ids = np.random.choice(unique_ids, size=20, replace=False)

mask = df["Demand ID"].isin(test_ids)
X_train, X_test = X[~mask], X[mask]
y_train, y_test = y[~mask], y[mask]

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", rf)
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

test_df = df[mask].copy()
test_df["pred"] = pred

errors = []
for d_id, g in test_df.groupby("Demand ID"):
    idx = g["pred"].idxmin()
    chosen_actual = g.loc[idx, "Cost_USD_per_MWh"]
    best_actual = g["Cost_USD_per_MWh"].min()
    errors.append(chosen_actual - best_actual)

rmse = np.sqrt(np.mean(np.array(errors)**2))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "rf_train_test_results.txt"), "w") as f:
    f.write(f"Train/Test RMSE = {rmse}\n")

print("Step 3 complete. RMSE:", rmse)
