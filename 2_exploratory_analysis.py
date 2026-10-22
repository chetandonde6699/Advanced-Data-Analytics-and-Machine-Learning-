import pandas as pd
import matplotlib.pyplot as plt
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "data_processed")
OUT = os.path.join(HERE, "outputs")

df = pd.read_csv(os.path.join(PROC, "cleaned_merged_data.csv"))
os.makedirs(OUT, exist_ok=True)

# Cost Distribution
plt.figure(figsize=(8,4))
df["Cost_USD_per_MWh"].hist(bins=30)
plt.title("Cost Distribution")
plt.xlabel("Cost (USD/MWh)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "cost_distribution.png"))
plt.close()

# Summary statistics
df.describe().to_csv(os.path.join(OUT, "summary_statistics.csv"))

print("Step 2 complete.")
