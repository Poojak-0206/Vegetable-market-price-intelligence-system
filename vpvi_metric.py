import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("vegetable_market_prices_cleaned_2025.csv")

# ---------------------------------
# Calculate VPVI for each vegetable
# ---------------------------------
vpvi_df = df.groupby("Commodity")["Average_Price"].agg(["mean", "std", "count"]).reset_index()
vpvi_df.columns = ["Commodity", "Mean_Average_Price", "Std_Dev_Price", "Observation_Count"]

vpvi_df["VPVI"] = (vpvi_df["Std_Dev_Price"] / vpvi_df["Mean_Average_Price"]) * 100
vpvi_df["VPVI"] = vpvi_df["VPVI"].fillna(0).round(2)

# ---------------------------------
# Classify VPVI
# ---------------------------------
def classify_vpvi(v):
    if v <= 10:
        return "Stable"
    elif v <= 20:
        return "Moderately Volatile"
    else:
        return "Highly Volatile"

vpvi_df["Volatility_Category"] = vpvi_df["VPVI"].apply(classify_vpvi)

# Save separate summary file
vpvi_df = vpvi_df.sort_values(by="VPVI", ascending=False)
vpvi_df.to_csv("vegetable_vpvi_summary.csv", index=False)

# ---------------------------------
# Merge VPVI back into main dataset
# ---------------------------------
df = df.merge(vpvi_df[["Commodity", "VPVI", "Volatility_Category"]], on="Commodity", how="left")

# Save final dataset for dashboard
df.to_csv("vegetable_market_prices_final_2025.csv", index=False)

print("VPVI calculation complete!")
print("\nTop 10 most volatile vegetables:")
print(vpvi_df.head(10))

print("\nFinal dataset saved as vegetable_market_prices_final_2025.csv")
print("Final shape:", df.shape)