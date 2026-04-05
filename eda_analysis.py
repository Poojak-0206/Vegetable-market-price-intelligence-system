import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder to save graphs
os.makedirs("graphs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("vegetable_market_prices_cleaned_2025.csv")

# Convert date column
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

# ---------------------------------
# 1. Basic Info
# ---------------------------------
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------
# 2. Top 10 Most Frequent Vegetables
# ---------------------------------
top_veg = df["Commodity"].value_counts().head(10)
print("\nTop 10 Most Frequent Vegetables:")
print(top_veg)

plt.figure(figsize=(10,6))
top_veg.plot(kind="bar")
plt.title("Top 10 Most Frequent Vegetables")
plt.xlabel("Commodity")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphs/top_10_vegetables.png")
plt.show()
plt.close()

# ---------------------------------
# 3. Average Price by Commodity
# ---------------------------------
avg_price_by_veg = df.groupby("Commodity")["Average_Price"].mean().sort_values(ascending=False).head(10)
print("\nTop 10 Vegetables by Average Price:")
print(avg_price_by_veg)

plt.figure(figsize=(10,6))
avg_price_by_veg.plot(kind="bar")
plt.title("Top 10 Vegetables by Average Price")
plt.xlabel("Commodity")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphs/average_price_by_commodity.png")
plt.show()
plt.close()

# ---------------------------------
# 4. Monthly Price Trend
# ---------------------------------
monthly_trend = df.groupby(df["Arrival_Date"].dt.month)["Average_Price"].mean()

print("\nMonthly Average Price Trend:")
print(monthly_trend)

plt.figure(figsize=(10,6))
monthly_trend.plot(marker="o")
plt.title("Monthly Average Price Trend")
plt.xlabel("Month Number")
plt.ylabel("Average Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/monthly_price_trend.png")
plt.show()
plt.close()

# ---------------------------------
# 5. Seasonal Price Trend
# ---------------------------------
seasonal_avg = df.groupby("Season")["Average_Price"].mean()

print("\nAverage Price by Season:")
print(seasonal_avg)

plt.figure(figsize=(8,5))
seasonal_avg.plot(kind="bar")
plt.title("Average Price by Season")
plt.xlabel("Season")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphs/seasonal_price_trend.png")
plt.show()
plt.close()

# ---------------------------------
# 6. Top 10 Most Volatile Vegetables
# ---------------------------------
volatility_by_veg = df.groupby("Commodity")["Price_Volatility_Percent"].mean().sort_values(ascending=False).head(10)

print("\nTop 10 Most Volatile Vegetables:")
print(volatility_by_veg)

plt.figure(figsize=(10,6))
volatility_by_veg.plot(kind="bar")
plt.title("Top 10 Most Volatile Vegetables")
plt.xlabel("Commodity")
plt.ylabel("Avg Price Volatility %")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphs/most_volatile_vegetables.png")
plt.show()
plt.close()

# ---------------------------------
# 7. Market-wise Average Price
# ---------------------------------
market_avg = df.groupby("Market")["Average_Price"].mean().sort_values(ascending=False).head(10)

print("\nTop 10 Markets by Average Price:")
print(market_avg)

plt.figure(figsize=(12,6))
market_avg.plot(kind="bar")
plt.title("Top 10 Markets by Average Vegetable Price")
plt.xlabel("Market")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphs/top_markets_by_average_price.png")
plt.show()
plt.close()

# ---------------------------------
# 8. Price Distribution Histogram
# ---------------------------------
plt.figure(figsize=(10,6))
plt.hist(df["Average_Price"], bins=30)
plt.title("Distribution of Average Vegetable Prices")
plt.xlabel("Average Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graphs/price_distribution_histogram.png")
plt.show()
plt.close()

# ---------------------------------
# 9. Boxplot for Outlier View
# ---------------------------------
plt.figure(figsize=(10,6))
sns.boxplot(y=df["Average_Price"])
plt.title("Boxplot of Average Price")
plt.tight_layout()
plt.savefig("graphs/average_price_boxplot.png")
plt.show()
plt.close()

# ---------------------------------
# 10. Correlation Heatmap
# ---------------------------------
numeric_cols = ["Min_Price", "Max_Price", "Modal_Price", "Price_Range", "Average_Price", "Price_Volatility_Percent"]
corr = df[numeric_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png")
plt.show()
plt.close()
# -----------------------------
# SCATTER PLOT: Price Range vs Price Volatility
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Price_Range",
    y="Price_Volatility_Percent",
    hue="Season"
)

plt.title("Price Range vs Price Volatility")
plt.xlabel("Price Range")
plt.ylabel("Price Volatility (%)")
plt.tight_layout()
plt.savefig("graphs/price_range_vs_volatility_scatter.png")
plt.show()
plt.close()

print("\nAll graphs saved successfully inside the 'graphs' folder.")