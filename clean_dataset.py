import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("vegetable_market_prices_filtered_2025.csv")

print("Columns in dataset:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# CLEAN COLUMN NAMES
# -------------------------------
df.columns = df.columns.str.strip()

# -------------------------------
# KEEP ONLY NEEDED COLUMNS
# -------------------------------
columns_to_keep = [
    "State", "District", "Market", "Commodity", "Variety", "Grade",
    "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"
]

df = df[columns_to_keep]

print("\nColumns after selection:")
print(df.columns.tolist())

# -------------------------------
# KEEP ONLY SELECTED VEGETABLES
# -------------------------------
vegetables = [
    "Amaranthus", "Beans", "Beetroot", "Bottle Gourd", "Brinjal",
    "Cabbage", "Carrot", "Cauliflower", "Chow Chow", "Cluster Beans",
    "Coriander", "Drumstick", "Garlic", "Ginger", "Green Chilli",
    "Ladies Finger", "Onion", "Potato", "Pumpkin", "Radish",
    "Snake Gourd", "Tomato"
]

df["Commodity"] = df["Commodity"].astype(str).str.strip().str.title()
df = df[df["Commodity"].isin(vegetables)]

# -------------------------------
# REMOVE MISSING VALUES
# -------------------------------
df.dropna(subset=["Commodity", "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"], inplace=True)

# -------------------------------
# REMOVE DUPLICATES
# -------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------
# FIX DATA TYPES
# -------------------------------
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce", dayfirst=True)
df["Min_Price"] = pd.to_numeric(df["Min_Price"], errors="coerce")
df["Max_Price"] = pd.to_numeric(df["Max_Price"], errors="coerce")
df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")

df.dropna(subset=["Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"], inplace=True)

# -------------------------------
# STANDARDIZE TEXT
# -------------------------------
text_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

# -------------------------------
# REMOVE ILLOGICAL PRICE ROWS
# -------------------------------
df = df[(df["Min_Price"] <= df["Modal_Price"]) & (df["Modal_Price"] <= df["Max_Price"])]

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["Price_Range"] = df["Max_Price"] - df["Min_Price"]
df["Average_Price"] = (df["Min_Price"] + df["Max_Price"] + df["Modal_Price"]) / 3
df["Price_Volatility_Percent"] = (df["Price_Range"] / df["Average_Price"]) * 100

df["Month"] = df["Arrival_Date"].dt.month_name()

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df["Season"] = df["Arrival_Date"].dt.month.apply(get_season)

# -------------------------------
# SAVE CLEANED FILE
# -------------------------------
df.to_csv("vegetable_market_prices_cleaned_2025.csv", index=False)

print("\nCleaning complete!")
print("Final shape:", df.shape)
print("\nFirst 5 cleaned rows:")
print(df.head())