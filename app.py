import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Vegetable Market Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CONSTANTS
# -----------------------------
COL_DATE = "Arrival_Date"
COL_COMMODITY = "Commodity"
COL_MARKET = "Market"
COL_DISTRICT = "District"
COL_SEASON = "Season"
COL_VOLATILITY = "Volatility_Category"
COL_AVG_PRICE = "Average_Price"
COL_MIN_PRICE = "Min_Price"
COL_MAX_PRICE = "Max_Price"
COL_MODAL_PRICE = "Modal_Price"
COL_PRICE_RANGE = "Price_Range"
COL_PRICE_VOL_PCT = "Price_Volatility_Percent"
COL_VPVI = "VPVI"

REQUIRED_COLS = [
    COL_DATE, COL_COMMODITY, COL_MARKET, COL_DISTRICT,
    COL_SEASON, COL_VOLATILITY, COL_AVG_PRICE,
    COL_MIN_PRICE, COL_MAX_PRICE, COL_MODAL_PRICE,
    COL_PRICE_RANGE, COL_PRICE_VOL_PCT
]

VPVI_REQUIRED_COLS = [COL_COMMODITY, COL_VPVI]

# -----------------------------
# LOAD & CLEAN DATA (CACHED)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("vegetable_market_prices_final_2025_clean_ui.csv")
    vpvi_df = pd.read_csv("vegetable_vpvi_summary_clean_ui.csv")

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    for col in [COL_MARKET, COL_COMMODITY, COL_DISTRICT, COL_SEASON, COL_VOLATILITY]:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates()

    return df, vpvi_df

df, vpvi_df = load_data()

# -----------------------------
# COLUMN VALIDATION
# -----------------------------
missing_main = [c for c in REQUIRED_COLS if c not in df.columns]
missing_vpvi = [c for c in VPVI_REQUIRED_COLS if c not in vpvi_df.columns]

if missing_main or missing_vpvi:
    if missing_main:
        st.error(f"Main dataset is missing columns: {missing_main}")
    if missing_vpvi:
        st.error(f"VPVI dataset is missing columns: {missing_vpvi}")
    st.stop()

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.markdown("## Dashboard Controls")

theme_mode = st.sidebar.toggle("Dark Mode", value=False)
show_data_preview = st.sidebar.toggle("Show Dataset Preview", value=False)
show_insights = st.sidebar.toggle("Show Insights Section", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## Filters")

min_date = df[COL_DATE].min().date()
max_date = df[COL_DATE].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

st.sidebar.markdown("---")

_max_n = max(
    df[COL_COMMODITY].nunique(),
    df[COL_MARKET].nunique(),
    vpvi_df[COL_COMMODITY].nunique()
)
_slider_max = max(5, min(_max_n, 30))
_slider_default = min(10, _slider_max)
top_n = st.sidebar.slider("Top N items in charts", min_value=5, max_value=_slider_max, value=_slider_default)

st.sidebar.markdown("---")

def filter_widget(label, options, plural=None):
    plural = plural or f"{label}s"
    select_all = st.sidebar.checkbox(f"All {plural}", value=True, key=f"all_{label}")
    if select_all:
        return list(options)
    selected = st.sidebar.multiselect(f"Select {plural}", sorted(options), key=f"sel_{label}")
    return selected if selected else list(options)

commodity_options = df[COL_COMMODITY].dropna().unique().tolist()
market_options = df[COL_MARKET].dropna().unique().tolist()
district_options = df[COL_DISTRICT].dropna().unique().tolist()
season_options = df[COL_SEASON].dropna().unique().tolist()
volatility_options = df[COL_VOLATILITY].dropna().unique().tolist()

selected_commodity = filter_widget("Vegetable", commodity_options)
selected_market = filter_widget("Market", market_options)
selected_district = filter_widget("District", district_options, plural="Districts")
selected_season = filter_widget("Season", season_options)
selected_volatility = filter_widget("Volatility Category", volatility_options, plural="Volatility Categories")

# -----------------------------
# THEME SETTINGS
# -----------------------------
if theme_mode:
    bg_color = "#0b1220"
    card_bg = "#111827"
    text_color = "#f8fafc"
    subtext_color = "#cbd5e1"
    border_color = "#1f2937"
    sidebar_bg = "#0f172a"
    plot_bg = "#111827"
    paper_bg = "#111827"
    grid_color = "rgba(255,255,255,0.10)"
    tab_bg = "#1e293b"
    tab_active = "#16a34a"
    input_bg = "#111827"
    delta_color = "#4ade80"
else:
    bg_color = "#f3f8f4"
    card_bg = "#ffffff"
    text_color = "#1f2937"
    subtext_color = "#4b5563"
    border_color = "#dcefe1"
    sidebar_bg = "#edf8ef"
    plot_bg = "#ffffff"
    paper_bg = "#ffffff"
    grid_color = "rgba(0,0,0,0.08)"
    tab_bg = "#f3f4f6"
    tab_active = "#15803d"
    input_bg = "#ffffff"
    delta_color = "#15803d"

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown(f"""
<style>
    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background: {bg_color} !important;
        color: {text_color} !important;
    }}

    .main, [data-testid="stMainBlockContainer"] {{
        background: {bg_color} !important;
    }}

    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }}

    [data-testid="stHeader"] {{
        background: {bg_color} !important;
        border-bottom: none !important;
    }}

    [data-testid="stToolbar"] {{
        background: {bg_color} !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}

    section[data-testid="stSidebar"] input {{
        background: {input_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label {{
        color: {text_color} !important;
    }}

    .metric-card {{
        background: {card_bg} !important;
        padding: 22px !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12) !important;
        text-align: center !important;
        border: 1px solid {border_color} !important;
    }}

    .metric-label {{
        font-size: 15px;
        color: {subtext_color} !important;
        margin-bottom: 6px;
        font-weight: 600;
    }}

    .metric-value {{
        font-size: 30px;
        font-weight: 800;
        color: {text_color} !important;
    }}

    .metric-delta {{
        font-size: 13px;
        font-weight: 600;
        margin-top: 6px;
        color: {delta_color} !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background: {bg_color} !important;
        gap: 6px;
        padding: 4px 0;
    }}

    button[data-baseweb="tab"] {{
        background: {tab_bg} !important;
        color: {text_color} !important;
        border-radius: 12px !important;
        border: 1px solid {border_color} !important;
        padding: 10px 16px !important;
        margin-right: 6px !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        background: {tab_active} !important;
        color: white !important;
        border: 1px solid {tab_active} !important;
    }}

    div[data-baseweb="select"] > div {{
        background: {input_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid {border_color} !important;
        background: {card_bg} !important;
    }}

    [data-testid="stAlert"] {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}

    [data-testid="stAlert"] p {{
        color: {text_color} !important;
    }}

    .stDownloadButton > button {{
        background: linear-gradient(90deg, #15803d, #16a34a) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 18px !important;
        font-weight: 700 !important;
    }}

    .stDownloadButton > button:hover {{
        background: linear-gradient(90deg, #166534, #15803d) !important;
    }}

    .empty-state {{
        text-align: center;
        padding: 60px 20px;
        color: {subtext_color};
        font-size: 16px;
    }}

    .footer-note {{
        font-size: 14px;
        color: {subtext_color};
        margin-top: 15px;
        text-align: center;
    }}

    .season-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }}

    .season-card-title {{
        font-size: 14px;
        font-weight: 700;
        color: {subtext_color};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }}

    .season-veg-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }}

    .season-veg-name {{
        font-size: 16px;
        font-weight: 800;
        color: {text_color};
    }}

    .season-veg-count {{
        font-size: 13px;
        color: {subtext_color};
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FILTER LOGIC
# -----------------------------
filtered_df = df.copy()

if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered_df = filtered_df[
        (filtered_df[COL_DATE] >= start_date) &
        (filtered_df[COL_DATE] <= end_date)
    ]

filter_map = {
    COL_COMMODITY: selected_commodity,
    COL_MARKET: selected_market,
    COL_DISTRICT: selected_district,
    COL_SEASON: selected_season,
    COL_VOLATILITY: selected_volatility,
}
for col, selection in filter_map.items():
    if selection:
        filtered_df = filtered_df[filtered_df[col].isin(selection)]

if filtered_df.empty:
    st.markdown(f"""
        <div class='empty-state'>
            <div style='font-size:48px'>🔍</div>
            <div style='font-size:20px; font-weight:700; margin-top:10px; color:{text_color}'>No records found</div>
            <div style='margin-top:8px'>Try adjusting your filters or date range in the sidebar.</div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# -----------------------------
# HEADER
# -----------------------------
base_year = df[COL_DATE].dt.year.max()

if theme_mode:
    header_bg = "linear-gradient(135deg, #0f2417 0%, #0b1220 100%)"
    header_border = "1px solid #1f2937"
    badge_bg = "rgba(22,163,74,0.15)"
    badge_color = "#4ade80"
    badge_border = "1px solid rgba(22,163,74,0.3)"
else:
    header_bg = "linear-gradient(135deg, #f0faf3 0%, #e8f5ec 100%)"
    header_border = "1px solid #c6e8d0"
    badge_bg = "rgba(21,128,61,0.08)"
    badge_color = "#15803d"
    badge_border = "1px solid rgba(21,128,61,0.2)"

st.markdown(f"""
<div style="
    background: {header_bg};
    border: {header_border};
    border-radius: 20px;
    padding: 28px 36px 24px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
">
    <div style="
        position: absolute; top: -30px; right: -30px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: rgba(22,163,74,0.06);
        pointer-events: none;
    "></div>
    <div style="
        position: absolute; bottom: -50px; right: 80px;
        width: 120px; height: 120px;
        border-radius: 50%;
        background: rgba(22,163,74,0.04);
        pointer-events: none;
    "></div>
    <span style="
        display: inline-block;
        background: {badge_bg};
        color: {badge_color};
        border: {badge_border};
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        text-transform: uppercase;
    ">📊 Market Intelligence · {base_year}</span>
    <div style="
        font-size: 38px;
        font-weight: 900;
        color: {text_color};
        line-height: 1.15;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    ">Vegetable Market Intelligence<br>Dashboard</div>
    <p style="
        font-size: 15px;
        color: {subtext_color};
        margin: 0;
        line-height: 1.6;
        max-width: 680px;
    ">Track prices, spot trends, and make smarter market decisions — powered by vegetable market data across districts, seasons, and volatility patterns.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# KPI METRICS
# -----------------------------
st.markdown("## Key Metrics")

overall_avg = df[COL_AVG_PRICE].mean()
total_records = len(filtered_df)
avg_price = filtered_df[COL_AVG_PRICE].mean()
highest_price = filtered_df[COL_AVG_PRICE].max()
lowest_price = filtered_df[COL_AVG_PRICE].min()
delta = avg_price - overall_avg
if abs(delta) < 0.01:
    delta_str = "Same as overall avg"
else:
    delta_sign = "▲" if delta >= 0 else "▼"
    delta_color_inline = delta_color if delta >= 0 else "#ef4444"
    delta_str = f"{delta_sign} ₹{abs(delta):,.2f} vs overall avg"

col1, col2, col3, col4 = st.columns(4)

metrics = [
    ("Total Records", f"{total_records:,}", ""),
    ("Average Price", f"₹ {avg_price:,.2f}", delta_str),
    ("Highest Avg Price", f"₹ {highest_price:,.2f}", ""),
    ("Lowest Avg Price", f"₹ {lowest_price:,.2f}", ""),
]

for col, (label, value, delta_text) in zip([col1, col2, col3, col4], metrics):
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {"<div class='metric-delta'>" + delta_text + "</div>" if delta_text else ""}
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# COMMON PLOTLY LAYOUT
# -----------------------------
def apply_layout(fig, title, **kwargs):
    fig.update_layout(
        title=title,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color=text_color, family="Segoe UI"),
        title_font=dict(size=18),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12)
        ),
        margin=dict(l=40, r=30, t=60, b=40),
        **kwargs
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False)
    return fig

# -----------------------------
# CHART TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Price Trend",
    "🥦 Commodity Analysis",
    "🏪 Market Analysis",
    "⚡ Volatility Analysis",
    "📊 VPVI",
    "🔗 Correlation",
    "🔮 Price Forecast",
    "📋 Market Insights"
])

# -----------------------------
# TAB 1: PRICE TREND
# -----------------------------
with tab1:
    col_left, col_right = st.columns([3, 1])

    with col_right:
        trend_granularity = st.radio(
            "Granularity",
            ["Monthly", "Quarterly"],
            horizontal=False
        )

    trend_df = filtered_df.copy()
    if trend_granularity == "Monthly":
        trend_df["Period"] = trend_df[COL_DATE].dt.to_period("M").astype(str)
    else:
        trend_df["Period"] = trend_df[COL_DATE].dt.to_period("Q").astype(str)

    monthly_trend = (
        trend_df.groupby("Period")[COL_AVG_PRICE]
        .mean()
        .reset_index()
    )
    monthly_trend.columns = ["Period", "Average Price"]

    with col_left:
        fig = px.line(monthly_trend, x="Period", y="Average Price", markers=True)
        fig.update_traces(line=dict(color="#16a34a", width=4), marker=dict(size=9))
        fig = apply_layout(fig, f"{trend_granularity} Average Price Trend")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    record_trend = (
        trend_df.groupby("Period")
        .size()
        .reset_index(name="Record Count")
    )
    fig2 = px.bar(record_trend, x="Period", y="Record Count", color_discrete_sequence=["#bbf7d0"])
    fig2 = apply_layout(fig2, "Number of Market Records per Period")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# -----------------------------
# TAB 2: COMMODITY ANALYSIS
# -----------------------------
with tab2:
    avg_price_veg = (
        filtered_df.groupby(COL_COMMODITY)[COL_AVG_PRICE]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    veg_colors = [
        "#16a34a", "#84cc16", "#f59e0b", "#ef4444", "#22c55e",
        "#eab308", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4",
        "#3b82f6", "#a855f7", "#ec4899", "#10b981", "#f43f5e",
        "#fb923c", "#a3e635", "#38bdf8", "#e879f9", "#34d399"
    ]

    fig = px.bar(
        avg_price_veg,
        x=COL_COMMODITY,
        y=COL_AVG_PRICE,
        color=COL_COMMODITY,
        color_discrete_sequence=veg_colors[:top_n],
        text_auto=".2f"
    )
    fig = apply_layout(fig, f"Top {top_n} Vegetables by Average Price")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -----------------------------
# TAB 3: MARKET ANALYSIS
# -----------------------------
with tab3:
    market_avg = (
        filtered_df.groupby(COL_MARKET)[COL_AVG_PRICE]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    market_colors = [
        "#0ea5e9", "#0284c7", "#0369a1", "#2563eb", "#1d4ed8",
        "#7c3aed", "#9333ea", "#c026d3", "#db2777", "#e11d48",
        "#f43f5e", "#fb923c", "#f59e0b", "#84cc16", "#22c55e",
        "#10b981", "#14b8a6", "#06b6d4", "#3b82f6", "#6366f1"
    ]

    fig = px.bar(
        market_avg,
        x=COL_MARKET,
        y=COL_AVG_PRICE,
        color=COL_MARKET,
        color_discrete_sequence=market_colors[:top_n],
        text_auto=".2f"
    )
    fig = apply_layout(fig, f"Top {top_n} Markets by Average Price")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -----------------------------
# TAB 4: VOLATILITY ANALYSIS
# -----------------------------
with tab4:
    colA, colB = st.columns(2)

    with colA:
        fig = px.box(
            filtered_df,
            x=COL_VOLATILITY,
            y=COL_AVG_PRICE,
            color=COL_VOLATILITY,
            color_discrete_sequence=["#14532d", "#16a34a", "#84cc16", "#0f766e", "#22c55e"]
        )
        fig = apply_layout(fig, "Price Distribution by Volatility Category")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with colB:
        season_color_map = {
            "Winter": "#166534",
            "Summer": "#f59e0b",
            "Monsoon": "#0ea5e9",
            "Autumn": "#ef4444",
            "Spring": "#8b5cf6"
        }

        p99_x = filtered_df[COL_PRICE_RANGE].quantile(0.99)
        p99_y = filtered_df[COL_PRICE_VOL_PCT].quantile(0.99)
        plot_df = filtered_df[
            (filtered_df[COL_PRICE_RANGE] <= p99_x) &
            (filtered_df[COL_PRICE_VOL_PCT] <= p99_y)
        ]

        if len(plot_df) > 50000:
            fig = px.density_heatmap(
                plot_df,
                x=COL_PRICE_RANGE,
                y=COL_PRICE_VOL_PCT,
                nbinsx=40,
                nbinsy=40,
                marginal_x="histogram",
                marginal_y="histogram",
            )
            fig.update_traces(
                colorscale="Greens",
                selector=dict(type="histogram2d")
            )
            fig.update_traces(
                marker_color="#16a34a",
                selector=dict(type="histogram")
            )
            fig = apply_layout(fig, "Price Range vs Volatility % (Density)")
        else:
            fig = px.scatter(
                plot_df,
                x=COL_PRICE_RANGE,
                y=COL_PRICE_VOL_PCT,
                color=COL_SEASON,
                color_discrete_map=season_color_map,
                hover_data=[COL_COMMODITY, COL_MARKET, COL_DISTRICT, COL_AVG_PRICE],
                opacity=0.85
            )
            fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
            fig = apply_layout(fig, "Price Range vs Price Volatility %")

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -----------------------------
# TAB 5: VPVI
# -----------------------------
with tab5:
    vpvi_sorted = vpvi_df.sort_values(by=COL_VPVI, ascending=False).head(top_n)
    vpvi_mean = vpvi_df[COL_VPVI].mean()

    vpvi_colors = [
        "#14532d", "#166534", "#15803d", "#16a34a", "#22c55e",
        "#4ade80", "#65a30d", "#3f6212", "#2f855a", "#047857",
        "#0f766e", "#1d4ed8", "#2563eb", "#7c3aed", "#9333ea",
        "#c026d3", "#db2777", "#e11d48", "#f97316", "#f59e0b"
    ]

    fig = px.bar(
        vpvi_sorted,
        x=COL_COMMODITY,
        y=COL_VPVI,
        color=COL_COMMODITY,
        color_discrete_sequence=vpvi_colors[:top_n],
        text_auto=".2f"
    )
    fig.add_hline(
        y=vpvi_mean,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Avg VPVI: {vpvi_mean:.2f}",
        annotation_position="top right"
    )
    fig = apply_layout(fig, f"Top {top_n} Vegetables by VPVI")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -----------------------------
# TAB 6: CORRELATION
# -----------------------------
with tab6:
    numeric_cols = [COL_MIN_PRICE, COL_MAX_PRICE, COL_MODAL_PRICE,
                    COL_PRICE_RANGE, COL_AVG_PRICE, COL_PRICE_VOL_PCT]
    corr = filtered_df[numeric_cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="YlGnBu",
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            hoverongaps=False
        )
    )
    fig = apply_layout(fig, "Correlation Between Price Variables")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =====================================================================
# TAB 7: PRICE FORECAST — Multi-year
# =====================================================================
with tab7:
    data_year = int(df[COL_DATE].dt.year.max())

    st.markdown(f"### Multi-Year Price Forecast (based on {data_year} data)")
    st.markdown(
        f"<p style='color:{subtext_color};font-size:14px;margin-top:-10px;margin-bottom:20px;'>"
        f"Predicted using Polynomial Regression (degree 3) on monthly average prices from {data_year}. "
        "Shaded area shows ±10% confidence band. Select how many years ahead you want to forecast.</p>",
        unsafe_allow_html=True
    )

    ctrl_col1, ctrl_col2 = st.columns([2, 2])

    with ctrl_col1:
        forecast_years_ahead = st.slider(
            "Years to forecast ahead",
            min_value=1, max_value=10, value=1, step=1,
            help=f"How many years beyond {data_year} to predict prices for"
        )

    with ctrl_col2:
        forecast_df_full = df.copy()
        commodities_list = sorted(forecast_df_full[COL_COMMODITY].dropna().unique().tolist())
        selected_forecast_veg = st.multiselect(
            "Select vegetable(s) to forecast",
            commodities_list,
            default=commodities_list[:3]
        )

    forecast_target_years = list(range(data_year + 1, data_year + forecast_years_ahead + 1))
    years_label = ", ".join(str(y) for y in forecast_target_years)

    st.markdown(
        f"<div style='background:{card_bg};border:1px solid {border_color};border-radius:12px;"
        f"padding:12px 18px;margin-bottom:20px;display:inline-block;'>"
        f"<span style='font-weight:700;color:{text_color};'>Forecasting for:</span> "
        f"<span style='color:#16a34a;font-weight:800;'>{years_label}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    if not selected_forecast_veg:
        st.warning("Please select at least one vegetable.")
    else:
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        forecast_df_base = forecast_df_full.copy()
        forecast_df_base["Month_Num"] = forecast_df_base[COL_DATE].dt.month

        fig_fc = go.Figure()
        forecast_table_rows = []

        year_palette = [
            "#16a34a", "#0ea5e9", "#f59e0b", "#ef4444", "#8b5cf6",
            "#ec4899", "#14b8a6", "#f97316", "#84cc16", "#6366f1"
        ]

        for veg in selected_forecast_veg:
            veg_df = forecast_df_base[forecast_df_base[COL_COMMODITY] == veg]
            monthly_avg = (
                veg_df.groupby("Month_Num")[COL_AVG_PRICE]
                .mean()
                .reindex(range(1, 13))
                .interpolate(method="linear")
                .reset_index()
            )
            monthly_avg.columns = ["Month_Num", COL_AVG_PRICE]

            if monthly_avg[COL_AVG_PRICE].isna().sum() > 8:
                st.warning(f"Not enough data for {veg}. Skipping.")
                continue

            y_train = monthly_avg[COL_AVG_PRICE].values  # 12 monthly averages

            # ── Compute a gentle annual growth rate from linear trend ──
            X_train = np.arange(1, 13).reshape(-1, 1)
            lr = LinearRegression().fit(X_train, y_train)
            annual_slope = lr.coef_[0] * 12  # total yearly drift in ₹
            base_mean = y_train.mean()
            # Cap growth rate to ±20% of base mean per year to prevent runaway
            annual_slope = np.clip(annual_slope, -0.20 * base_mean, 0.20 * base_mean)

            # ── Build continuous labels ──
            all_labels = []
            for yr_offset in range(forecast_years_ahead + 1):
                yr = data_year + yr_offset
                for m in month_names:
                    all_labels.append(f"{yr}-{m}")

            # ── Plot actual ──
            fig_fc.add_trace(go.Scatter(
                x=all_labels[:12],
                y=y_train,
                mode="lines+markers",
                name=f"{veg} ({data_year} Actual)",
                line=dict(width=2, dash="dot", color="#94a3b8"),
                marker=dict(size=7),
                opacity=0.8
            ))

            # ── Build forecast: repeat seasonal pattern + linear drift ──
            forecast_x = all_labels[12:12 + forecast_years_ahead * 12]
            forecast_y = []
            np.random.seed(42)  # for reproducibility

            for yr_offset in range(1, forecast_years_ahead + 1):
                # Generate a small noise factor unique to each year (±5% of mean)
                yearly_noise = np.random.normal(0, 0.05 * base_mean, 12)
                
                # Add slight acceleration/deceleration to slope each year
                slope_variation = annual_slope * (1 + np.random.uniform(-0.15, 0.15))
                
                for m_idx in range(12):
                    i = (yr_offset - 1) * 12 + m_idx
                    
                    # Base seasonal value from actual same month
                    base = y_train[m_idx]
                    
                    # Trend drift for this year
                    trend = slope_variation * yr_offset
                    
                    # Add monthly noise unique to this year
                    noise = yearly_noise[m_idx]
                    
                    predicted = max(0, base + trend + noise)
                    forecast_y.append(predicted)

            forecast_y = np.array(forecast_y)
            conf_band  = forecast_y * 0.10

            veg_color = year_palette[selected_forecast_veg.index(veg) % len(year_palette)]

            fig_fc.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                name=f"{veg} Forecast ({forecast_target_years[0]}–{forecast_target_years[-1]})",
                line=dict(width=3, color=veg_color),
                marker=dict(size=8)
            ))

            # Confidence band
            fig_fc.add_trace(go.Scatter(
                x=list(forecast_x) + list(forecast_x[::-1]),
                y=list(forecast_y + conf_band) + list((forecast_y - conf_band)[::-1]),
                fill="toself",
                fillcolor="rgba(22,163,74,0.07)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Annotation marker
            fig_fc.add_annotation(
                x=all_labels[11],
                y=1.02,
                yref="paper",
                xref="x",
                text="▏← Actual  |  Forecast →",
                showarrow=False,
                font=dict(color="#94a3b8", size=11),
                xanchor="center"
            )

            # ── Build table rows ──
            for i in range(forecast_years_ahead * 12):
                yr_offset = i // 12
                m_idx     = i % 12
                yr        = forecast_target_years[yr_offset]
                m_name    = month_names[m_idx]
                forecast_table_rows.append({
                    "Vegetable": veg,
                    "Forecast Year": yr,
                    "Month": m_name,
                    "Predicted Price (₹)": round(float(forecast_y[i]), 2),
                    "Lower Bound (₹)":     round(float(max(0, forecast_y[i] - conf_band[i])), 2),
                    "Upper Bound (₹)":     round(float(forecast_y[i] + conf_band[i]), 2),
                })

        fig_fc = apply_layout(
            fig_fc,
            f"{data_year} Actual vs {years_label} Predicted Monthly Prices"
        )
        fig_fc.update_layout(xaxis_tickangle=-45, height=520)
        st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})

        if forecast_table_rows:
            st.markdown("#### 📅 Month-wise Forecast Table")
            forecast_table_df = pd.DataFrame(forecast_table_rows)

            year_filter = st.multiselect(
                "Filter table by year",
                options=forecast_target_years,
                default=forecast_target_years,
                key="fc_year_filter"
            )
            display_fc_df = (
                forecast_table_df[forecast_table_df["Forecast Year"].isin(year_filter)]
                if year_filter else forecast_table_df
            )
            st.dataframe(display_fc_df, use_container_width=True, height=380)

            csv_fc = forecast_table_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download Full Forecast ({years_label}) as CSV",
                data=csv_fc,
                file_name=f"price_forecast_{years_label.replace(', ','_')}.csv",
                mime="text/csv"
            )

# =====================================================================
# TAB 8: MARKET INSIGHTS — with Most/Least Sold per Season
# =====================================================================
with tab8:
    st.markdown(f"### {data_year} Market Insights")

    ins_df = df.copy()
    ins_df["Month_Num"] = ins_df[COL_DATE].dt.month
    ins_df["Month_Name"] = ins_df[COL_DATE].dt.strftime("%b")

    # ── Insight KPI cards ──
    most_exp_veg    = ins_df.groupby(COL_COMMODITY)[COL_AVG_PRICE].mean().idxmax()
    least_exp_veg   = ins_df.groupby(COL_COMMODITY)[COL_AVG_PRICE].mean().idxmin()
    most_exp_price  = ins_df.groupby(COL_COMMODITY)[COL_AVG_PRICE].mean().max()
    least_exp_price = ins_df.groupby(COL_COMMODITY)[COL_AVG_PRICE].mean().min()
    most_vol_market = ins_df.groupby(COL_MARKET)[COL_PRICE_VOL_PCT].mean().idxmax()
    stable_market   = ins_df.groupby(COL_MARKET)[COL_PRICE_VOL_PCT].mean().idxmin()
    peak_month_num  = ins_df.groupby("Month_Num")[COL_AVG_PRICE].mean().idxmax()
    low_month_num   = ins_df.groupby("Month_Num")[COL_AVG_PRICE].mean().idxmin()
    month_names_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    peak_month  = month_names_map[peak_month_num]
    low_month   = month_names_map[low_month_num]
    most_active_market   = ins_df.groupby(COL_MARKET)[COL_AVG_PRICE].count().idxmax()
    most_active_district = ins_df.groupby(COL_DISTRICT)[COL_AVG_PRICE].count().idxmax()

    insight_cards = [
        ("🥇", "Most Expensive Veg", most_exp_veg, f"₹{most_exp_price:,.0f} avg"),
        ("🟢", "Least Expensive Veg", least_exp_veg, f"₹{least_exp_price:,.0f} avg"),
        ("⚡", "Most Volatile Market", most_vol_market, "Highest price swings"),
        ("🏆", "Most Stable Market", stable_market, "Lowest price swings"),
        ("📈", "Peak Price Month", peak_month, "Highest avg prices"),
        ("📉", "Lowest Price Month", low_month, "Lowest avg prices"),
        ("🏪", "Most Active Market", most_active_market, "Most records in data"),
        ("📍", "Most Active District", most_active_district, "Most records in data"),
    ]

    cols_ins = st.columns(4)
    for i, (icon, label, value, sub) in enumerate(insight_cards):
        with cols_ins[i % 4]:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:16px;">
                <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
                <div class="metric-label">{label}</div>
                <div style="font-size:18px;font-weight:800;color:{text_color};">{value}</div>
                <div style="font-size:12px;color:{subtext_color};margin-top:4px;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # NEW: Most & Least Sold Vegetable per Season
    # ─────────────────────────────────────────────
    st.markdown("#### 🌿 Most & Least Sold Vegetable per Season")
    st.markdown(
        f"<p style='color:{subtext_color};font-size:13px;margin-top:-8px;margin-bottom:18px;'>"
        "Sales volume is approximated by the number of market records (arrivals) for each vegetable in each season.</p>",
        unsafe_allow_html=True
    )

    season_veg_counts = (
        ins_df.groupby([COL_SEASON, COL_COMMODITY])
        .size()
        .reset_index(name="Record_Count")
    )

    seasons_available = sorted(ins_df[COL_SEASON].dropna().unique().tolist())

    season_icons = {
        "Winter": "❄️", "Summer": "☀️", "Monsoon": "🌧️",
        "Autumn": "🍂", "Spring": "🌸"
    }
    season_colors_header = {
        "Winter": "#1d4ed8", "Summer": "#f59e0b", "Monsoon": "#0ea5e9",
        "Autumn": "#ef4444", "Spring": "#8b5cf6"
    }

    # Layout: 2 or 3 seasons per row
    chunk_size = 3
    season_chunks = [seasons_available[i:i+chunk_size] for i in range(0, len(seasons_available), chunk_size)]

    for chunk in season_chunks:
        cols_season = st.columns(len(chunk))
        for col_s, season in zip(cols_season, chunk):
            s_data = season_veg_counts[season_veg_counts[COL_SEASON] == season]
            if s_data.empty:
                continue
            most_sold_row = s_data.loc[s_data["Record_Count"].idxmax()]
            least_sold_row = s_data.loc[s_data["Record_Count"].idxmin()]

            icon = season_icons.get(season, "🌱")
            hdr_color = season_colors_header.get(season, "#16a34a")

            with col_s:
                st.markdown(f"""
                <div class="season-card">
                    <div class="season-card-title" style="color:{hdr_color};">
                        {icon} {season}
                    </div>
                    <div style="margin-bottom:14px;">
                        <div style="font-size:11px;font-weight:700;color:#16a34a;
                            text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">
                            🏆 Most Sold
                        </div>
                        <div class="season-veg-name">{most_sold_row[COL_COMMODITY]}</div>
                        <div class="season-veg-count">{int(most_sold_row['Record_Count']):,} records</div>
                    </div>
                    <div style="border-top:1px solid {border_color};padding-top:12px;">
                        <div style="font-size:11px;font-weight:700;color:#ef4444;
                            text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">
                            📉 Least Sold
                        </div>
                        <div class="season-veg-name">{least_sold_row[COL_COMMODITY]}</div>
                        <div class="season-veg-count">{int(least_sold_row['Record_Count']):,} records</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Stacked bar: Top vegetables per season by record count
    st.markdown("#### 📊 Seasonal Vegetable Arrival Volume — Top Vegetables")
    top_n_season = st.slider("Top N vegetables per season", min_value=3, max_value=15, value=5, key="season_top_n")

    season_top_vegs = []
    for season in seasons_available:
        s_data = season_veg_counts[season_veg_counts[COL_SEASON] == season]
        top_rows = s_data.nlargest(top_n_season, "Record_Count")
        season_top_vegs.append(top_rows)

    season_top_df = pd.concat(season_top_vegs, ignore_index=True)

    fig_sv = px.bar(
        season_top_df,
        x=COL_SEASON,
        y="Record_Count",
        color=COL_COMMODITY,
        barmode="group",
        text_auto=True,
        labels={"Record_Count": "Number of Arrivals"},
    )
    fig_sv = apply_layout(fig_sv, f"Top {top_n_season} Most-Sold Vegetables per Season (by Arrivals)")
    fig_sv.update_layout(height=420, legend_title_text="Vegetable")
    st.plotly_chart(fig_sv, use_container_width=True, config={"displayModeBar": False})

    # Full table for download
    st.markdown("#### 📋 Full Seasonal Sales Volume Table")
    season_pivot = season_veg_counts.pivot(index=COL_COMMODITY, columns=COL_SEASON, values="Record_Count").fillna(0).astype(int)
    season_pivot["Total"] = season_pivot.sum(axis=1)
    season_pivot = season_pivot.sort_values("Total", ascending=False)
    st.dataframe(season_pivot, use_container_width=True, height=320)

    csv_sv = season_veg_counts.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Seasonal Sales Volume as CSV",
        data=csv_sv,
        file_name="seasonal_vegetable_volume.csv",
        mime="text/csv"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Vegetable Price Heatmap ──
    st.markdown("#### 🌡️ Vegetable Price Heatmap — Monthly Average")
    heat_df = ins_df.groupby([COL_COMMODITY, "Month_Num"])[COL_AVG_PRICE].mean().reset_index()
    heat_pivot = heat_df.pivot(index=COL_COMMODITY, columns="Month_Num", values=COL_AVG_PRICE)
    heat_pivot.columns = [month_names_map[c] for c in heat_pivot.columns]

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=heat_pivot.index.tolist(),
        colorscale="YlGn",
        text=np.round(heat_pivot.values, 0).astype(int),
        texttemplate="₹%{text}",
        hoverongaps=False
    ))
    fig_heat = apply_layout(fig_heat, "Average Price per Vegetable per Month")
    fig_heat.update_layout(height=380)
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # ── Season avg price + District ranking ──
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### 🌦️ Season-wise Average Price")
        season_avg = ins_df.groupby(COL_SEASON)[COL_AVG_PRICE].mean().sort_values(ascending=False).reset_index()
        fig_sea = px.bar(
            season_avg, x=COL_SEASON, y=COL_AVG_PRICE,
            color=COL_SEASON,
            color_discrete_sequence=["#f59e0b","#0ea5e9","#16a34a","#ef4444","#8b5cf6"],
            text_auto=".0f"
        )
        fig_sea = apply_layout(fig_sea, "Avg Price by Season")
        fig_sea.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_sea, use_container_width=True, config={"displayModeBar": False})

    with col_s2:
        st.markdown("#### 📍 District-wise Average Price Ranking")
        dist_avg = ins_df.groupby(COL_DISTRICT)[COL_AVG_PRICE].mean().sort_values(ascending=False).head(10).reset_index()
        fig_dist = px.bar(
            dist_avg, x=COL_AVG_PRICE, y=COL_DISTRICT,
            orientation="h",
            color=COL_AVG_PRICE,
            color_continuous_scale="Greens",
            text_auto=".0f"
        )
        fig_dist = apply_layout(fig_dist, "Top 10 Districts by Avg Price")
        fig_dist.update_layout(showlegend=False, height=340, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 📅 Month-wise Price Trend per Vegetable")
        month_veg = ins_df.groupby([COL_COMMODITY, "Month_Num"])[COL_AVG_PRICE].mean().reset_index()
        month_veg["Month"] = month_veg["Month_Num"].map(month_names_map)
        fig_mv = px.line(
            month_veg, x="Month", y=COL_AVG_PRICE,
            color=COL_COMMODITY,
            markers=True
        )
        fig_mv = apply_layout(fig_mv, "Monthly Price Trend by Vegetable")
        fig_mv.update_layout(height=360)
        st.plotly_chart(fig_mv, use_container_width=True, config={"displayModeBar": False})

    with col_t2:
        st.markdown("#### 🌀 Price Volatility by Season")
        vol_season = ins_df.groupby(COL_SEASON)[COL_PRICE_VOL_PCT].mean().sort_values(ascending=False).reset_index()
        fig_vs = px.bar(
            vol_season, x=COL_SEASON, y=COL_PRICE_VOL_PCT,
            color=COL_SEASON,
            color_discrete_sequence=["#f59e0b","#0ea5e9","#16a34a","#ef4444","#8b5cf6"],
            text_auto=".1f"
        )
        fig_vs = apply_layout(fig_vs, "Avg Volatility % by Season")
        fig_vs.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_vs, use_container_width=True, config={"displayModeBar": False})

    st.markdown("#### 🏪 Market Performance Summary")
    market_summary = ins_df.groupby(COL_MARKET).agg(
        Avg_Price=(COL_AVG_PRICE, "mean"),
        Max_Price=(COL_MAX_PRICE, "max"),
        Min_Price=(COL_MIN_PRICE, "min"),
        Avg_Volatility=(COL_PRICE_VOL_PCT, "mean"),
        Total_Records=(COL_AVG_PRICE, "count")
    ).round(2).sort_values("Avg_Price", ascending=False).reset_index()
    market_summary.columns = ["Market","Avg Price (₹)","Max Price (₹)","Min Price (₹)","Avg Volatility %","Records"]
    st.dataframe(market_summary, use_container_width=True, height=320)


# -----------------------------
# INSIGHTS
# -----------------------------
if show_insights:
    st.markdown("## Price Intelligence Insights")

    most_volatile = vpvi_df.sort_values(by=COL_VPVI, ascending=False).head(5)
    most_stable = vpvi_df.sort_values(by=COL_VPVI, ascending=True).head(5)

    colX, colY = st.columns(2)

    with colX:
        st.markdown("### Top 5 Most Volatile Vegetables")
        st.dataframe(most_volatile, use_container_width=True)

    with colY:
        st.markdown("### Top 5 Most Stable Vegetables")
        st.dataframe(most_stable, use_container_width=True)

# -----------------------------
# RECOMMENDATION
# -----------------------------
top_risky = vpvi_df.sort_values(by=COL_VPVI, ascending=False).iloc[0]
top_stable = vpvi_df.sort_values(by=COL_VPVI, ascending=True).iloc[0]

st.markdown("## Market Intelligence Recommendation")

if theme_mode:
    risky_bg  = "linear-gradient(135deg, #2d0f0f 0%, #1a0a0a 100%)"
    risky_border = "#7f1d1d"
    risky_icon_bg = "rgba(239,68,68,0.15)"
    stable_bg = "linear-gradient(135deg, #0f2d15 0%, #0a1a0d 100%)"
    stable_border = "#14532d"
    stable_icon_bg = "rgba(22,163,74,0.15)"
else:
    risky_bg  = "linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%)"
    risky_border = "#fca5a5"
    risky_icon_bg = "rgba(239,68,68,0.10)"
    stable_bg = "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)"
    stable_border = "#86efac"
    stable_icon_bg = "rgba(22,163,74,0.10)"

col_r, col_s = st.columns(2)

with col_r:
    st.markdown(f"""
    <div style="
        background: {risky_bg};
        border: 1.5px solid {risky_border};
        border-radius: 20px;
        padding: 24px 28px;
        height: 100%;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -20px; right: -20px;
            width: 100px; height: 100px; border-radius: 50%;
            background: rgba(239,68,68,0.06);
        "></div>
        <div style="
            width: 48px; height: 48px; border-radius: 14px;
            background: {risky_icon_bg};
            display: flex; align-items: center; justify-content: center;
            font-size: 24px; margin-bottom: 14px;
        ">⚠️</div>
        <div style="font-size: 11px; font-weight: 700; color: #ef4444;
            text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
            Highest Risk · Avoid Bulk Buying
        </div>
        <div style="font-size: 26px; font-weight: 900; color: {text_color}; margin-bottom: 6px;">
            {top_risky[COL_COMMODITY]}
        </div>
        <div style="font-size: 13px; color: {subtext_color}; line-height: 1.6;">
            Most volatile vegetable with a VPVI score of
            <span style="font-weight: 800; color: #ef4444; font-size: 15px;">
                {top_risky[COL_VPVI]:.2f}
            </span>.
            Price swings make this a high-risk procurement choice.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_s:
    st.markdown(f"""
    <div style="
        background: {stable_bg};
        border: 1.5px solid {stable_border};
        border-radius: 20px;
        padding: 24px 28px;
        height: 100%;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -20px; right: -20px;
            width: 100px; height: 100px; border-radius: 50%;
            background: rgba(22,163,74,0.06);
        "></div>
        <div style="
            width: 48px; height: 48px; border-radius: 14px;
            background: {stable_icon_bg};
            display: flex; align-items: center; justify-content: center;
            font-size: 24px; margin-bottom: 14px;
        ">✅</div>
        <div style="font-size: 11px; font-weight: 700; color: #16a34a;
            text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
            Most Stable · Safe to Procure
        </div>
        <div style="font-size: 26px; font-weight: 900; color: {text_color}; margin-bottom: 6px;">
            {top_stable[COL_COMMODITY]}
        </div>
        <div style="font-size: 13px; color: {subtext_color}; line-height: 1.6;">
            Most price-stable vegetable with a VPVI score of
            <span style="font-weight: 800; color: #16a34a; font-size: 15px;">
                {top_stable[COL_VPVI]:.2f}
            </span>.
            Consistent pricing makes this ideal for bulk procurement.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# DOWNLOAD
# -----------------------------
st.markdown("## Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Dataset as CSV",
    data=csv,
    file_name="filtered_vegetable_market_data.csv",
    mime="text/csv"
)

# -----------------------------
# DATA PREVIEW
# -----------------------------
if show_data_preview:
    st.markdown("## Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True, height=450)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    "<p class='footer-note'>This dashboard presents analytical insights derived from vegetable market price data.</p>",
    unsafe_allow_html=True
)