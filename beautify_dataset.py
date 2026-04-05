import pandas as pd
import re

# Load dataset
df = pd.read_csv("vegetable_market_prices_final_2025.csv")
vpvi_df = pd.read_csv("vegetable_vpvi_summary.csv")

# -----------------------------
# CLEAN MARKET NAMES
# -----------------------------
def clean_market_name(name):
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)  # remove extra spaces
    name = name.replace("(Uzhavar Sandhai )", " - Uzhavar Sandhai")
    name = name.replace("(Uzhavar Sandhai)", " - Uzhavar Sandhai")
    name = name.replace("Uzhavar Sandhai )", "Uzhavar Sandhai")
    name = name.replace("(", " - ")
    name = name.replace(")", "")
    return name.strip()

# -----------------------------
# CLEAN COMMODITY NAMES
# -----------------------------
def clean_commodity_name(name):
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)  # remove extra spaces
    name = name.replace("Brinjal", "Brinjal")
    return name.strip()

# Apply cleaning
df["Market"] = df["Market"].apply(clean_market_name)
df["Commodity"] = df["Commodity"].apply(clean_commodity_name)

vpvi_df["Commodity"] = vpvi_df["Commodity"].apply(clean_commodity_name)

# Save cleaned versions
df.to_csv("vegetable_market_prices_final_2025_clean_ui.csv", index=False)
vpvi_df.to_csv("vegetable_vpvi_summary_clean_ui.csv", index=False)

print("Dataset beautification complete!")
print("Files created:")
print("1. vegetable_market_prices_final_2025_clean_ui.csv")
print("2. vegetable_vpvi_summary_clean_ui.csv")