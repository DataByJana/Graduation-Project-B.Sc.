import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from utils.apify_loader import fetch_tweets
from utils.fda_loader import fetch_fda_recalls
from utils.model_loader import load_models
from utils.inference import run_pipeline


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Foodborne Illness Early Warning Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# THEME STATE
# =========================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

if "theme_toggle" not in st.session_state:
    st.session_state.theme_toggle = True


def sync_theme():
    st.session_state.theme_mode = "Dark" if st.session_state.theme_toggle else "Light"


is_dark = st.session_state.theme_mode == "Dark"


# =========================================================
# COLORS
# =========================================================
if is_dark:
    BG = "linear-gradient(180deg, #020817 0%, #071226 100%)"
    CARD_BG = "rgba(10, 20, 40, 0.95)"
    BAR_BG = "rgba(10, 20, 40, 0.92)"
    BORDER = "#1d3a66"
    TEXT = "#eaf2ff"
    MUTED = "#9db2ce"
    GRID = "rgba(120, 160, 220, 0.12)"
    MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"
else:
    BG = "linear-gradient(180deg, #f6f9ff 0%, #eef4ff 100%)"
    CARD_BG = "rgba(255,255,255,0.95)"
    BAR_BG = "rgba(255,255,255,0.92)"
    BORDER = "#dbe7ff"
    TEXT = "#243247"
    MUTED = "#6c7a93"
    GRID = "rgba(120, 140, 170, 0.18)"
    MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json"


# =========================================================
# CSS
# =========================================================
st.markdown(f"""
<style>
    .stApp {{
        background: {BG};
    }}

    .block-container {{
        max-width: 1750px;
        padding-top: 0.7rem;
        padding-bottom: 1.5rem;
    }}

    .top-bar {{
        background: {BAR_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }}

    .card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(40, 70, 140, 0.08);
        margin-bottom: 14px;
    }}

    .kpi-card {{
        min-height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .kpi-blue {{
        background: {"linear-gradient(135deg, rgba(7,32,68,0.95), rgba(11,30,58,0.95))" if is_dark else "linear-gradient(135deg, rgba(229,242,255,0.98), rgba(242,247,255,0.98))"};
        border: 1px solid {"#3d7fd4" if is_dark else "#8ac5ff"};
    }}

    .kpi-pink {{
        background: {"linear-gradient(135deg, rgba(65,18,38,0.95), rgba(53,18,42,0.95))" if is_dark else "linear-gradient(135deg, rgba(255,234,241,0.98), rgba(255,245,248,0.98))"};
        border: 1px solid {"#d86a98" if is_dark else "#ff90b1"};
    }}

    .kpi-green {{
        background: {"linear-gradient(135deg, rgba(14,52,33,0.95), rgba(15,42,34,0.95))" if is_dark else "linear-gradient(135deg, rgba(231,255,239,0.98), rgba(245,255,249,0.98))"};
        border: 1px solid {"#5fb87d" if is_dark else "#7ed79d"};
    }}

    .kpi-title {{
        font-size: 15px;
        color: {TEXT};
        font-weight: 600;
        margin-bottom: 10px;
    }}

    .kpi-value {{
        font-size: 38px;
        font-weight: 800;
        line-height: 1.1;
    }}

    .kpi-blue .kpi-value {{ color: #47bfff; }}
    .kpi-pink .kpi-value {{ color: #ff6489; }}
    .kpi-green .kpi-value {{ color: #63d98d; }}

    .section-title {{
        font-size: 18px;
        font-weight: 700;
        color: {TEXT};
        margin-bottom: 10px;
    }}

    .footer-note {{
        color: {MUTED};
        font-size: 13px;
        text-align: center;
        margin-top: 6px;
    }}

    .stMarkdown, .stCaption, p, div {{
        color: {TEXT};
    }}

    .stTextInput label, .stSelectbox label {{
        color: {TEXT} !important;
        font-weight: 600;
    }}

    div[data-testid="stToggle"] label {{
        color: {TEXT} !important;
        font-weight: 600 !important;
    }}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOCATION HELPERS
# =========================================================
KNOWN_LOCATIONS = {
    "riyadh": (24.7136, 46.6753, "Riyadh", "Saudi Arabia"),
    "jeddah": (21.5433, 39.1728, "Jeddah", "Saudi Arabia"),
    "makkah": (21.3891, 39.8579, "Makkah", "Saudi Arabia"),
    "mecca": (21.3891, 39.8579, "Makkah", "Saudi Arabia"),
    "madinah": (24.5247, 39.5692, "Madinah", "Saudi Arabia"),
    "dammam": (26.4207, 50.0888, "Dammam", "Saudi Arabia"),
    "khobar": (26.2172, 50.1971, "Khobar", "Saudi Arabia"),
    "dubai": (25.2048, 55.2708, "Dubai", "UAE"),
    "abu dhabi": (24.4539, 54.3773, "Abu Dhabi", "UAE"),
    "kuwait": (29.3759, 47.9774, "Kuwait City", "Kuwait"),
    "doha": (25.2854, 51.5310, "Doha", "Qatar"),
    "london": (51.5072, -0.1276, "London", "United Kingdom"),
    "new york": (40.7128, -74.0060, "New York", "USA"),
    "los angeles": (34.0522, -118.2437, "Los Angeles", "USA"),
    "washington": (38.9072, -77.0369, "Washington", "USA"),
    "california": (36.7783, -119.4179, "California", "USA"),
    "texas": (31.9686, -99.9018, "Texas", "USA"),
    "florida": (27.6648, -81.5158, "Florida", "USA"),
}


def geocode_location(location):
    text = str(location).lower()

    for key, (lat, lon, city, country) in KNOWN_LOCATIONS.items():
        if key in text:
            return city, country, lat, lon

    return "Unknown", "Unknown", 24.7136, 46.6753


def normalize_symptom_from_text(text):
    text = str(text).lower()

    if any(x in text for x in ["vomit", "vomiting", "throwing up", "throw up"]):
        return "Vomiting"
    if any(x in text for x in ["fever", "temperature", "feverish"]):
        return "Fever"
    if any(x in text for x in ["diarrhea", "diarrhoea", "loose stool"]):
        return "Diarrhea"
    if any(x in text for x in ["nausea", "nauseous", "queasy"]):
        return "Nausea"

    return "Unknown"


# =========================================================
# LOAD DATA + MODELS
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_and_process(max_tweets, max_fda):
    apify_token = st.secrets.get("APIFY_TOKEN", "")

    tweets = fetch_tweets(apify_token=apify_token, max_items=max_tweets)
    fda = fetch_fda_recalls(limit=max_fda)

    combined = pd.concat([tweets, fda], ignore_index=True, sort=False)

    alert_pipe, ner_pipe, category_pipe = load_models()

    return run_pipeline(combined, alert_pipe, ner_pipe, category_pipe)


with st.sidebar:
    st.header("Data Settings")

    max_tweets = st.slider("Max tweets", 50, 1000, 50, 50)
    max_fda = st.slider("Max FDA records", 50, 1000, 200, 50)

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("Clear all cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


with st.spinner("Fetching data and running AI models..."):
    final_df = load_and_process(max_tweets, max_fda)


if final_df.empty:
    st.error("No alert records found. Check API token, model labels, or data availability.")
    st.stop()


# =========================================================
# NORMALIZE DASHBOARD DATA
# =========================================================
df = final_df.copy()

df["date"] = pd.to_datetime(df.get("created_at"), errors="coerce").dt.tz_localize(None)

df["title"] = df.get("text", "").astype(str)

if "final_location" not in df.columns:
    df["final_location"] = "Unknown"

df["city"] = "Unknown"
df["country"] = "Unknown"
df["lat"] = 24.7136
df["lon"] = 46.6753

for idx, row in df.iterrows():
    city, country, lat, lon = geocode_location(row.get("final_location", "Unknown"))
    df.at[idx, "city"] = city
    df.at[idx, "country"] = country
    df.at[idx, "lat"] = lat
    df.at[idx, "lon"] = lon

if "product_category" not in df.columns:
    df["product_category"] = "Unknown"

df["product_category"] = (
    df["product_category"]
    .fillna("Unknown")
    .astype(str)
    .replace("", "Unknown")
)

if "hazard" not in df.columns:
    df["hazard"] = "Unknown"

df["hazard"] = df["hazard"].fillna("Unknown").astype(str).replace("", "Unknown")

df["symptom"] = df["title"].apply(normalize_symptom_from_text)

df.loc[df["symptom"] == "Unknown", "symptom"] = df.loc[
    df["symptom"] == "Unknown", "hazard"
].replace("", "Unknown")

df["cases"] = 1
df["alert_level"] = "Medium"

if "source" not in df.columns:
    df["source"] = "Unknown"


# =========================================================
# TOP BAR FILTERS
# =========================================================
st.markdown('<div class="top-bar">', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns([1.35, 1.45, 1.1, 1.1, 1.0, 0.7], gap="medium")

with c1:
    st.markdown(
        f"""
        <div style="height:72px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:22px; font-weight:800; color:{TEXT};">
                Filters & Search
            </div>
            <div style="font-size:12px; color:{MUTED};">
                Explore alerts by city, symptom, date range, and source
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    search_text = st.text_input(
        "Search",
        placeholder="City / title / symptom",
        key="search_text"
    )

with c3:
    symptoms = ["All"] + sorted(df["symptom"].dropna().astype(str).unique().tolist())
    selected_symptom = st.selectbox("Symptoms", symptoms)

with c4:
    selected_range = st.selectbox(
        "Date Range",
        ["Last 7 Days", "Last 14 Days", "Last 30 Days", "All"],
        index=3
    )

with c5:
    sources = ["All"] + sorted(df["source"].dropna().astype(str).unique().tolist())
    selected_source = st.selectbox("Source", sources)

with c6:
    st.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
    st.toggle("Dark", key="theme_toggle", on_change=sync_theme)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# APPLY FILTERS
# =========================================================
filtered_df = df.copy()

if selected_symptom != "All":
    filtered_df = filtered_df[filtered_df["symptom"] == selected_symptom]

today = pd.to_datetime(pd.Timestamp.today().date())

if selected_range == "Last 7 Days":
    filtered_df = filtered_df[filtered_df["date"] >= today - pd.Timedelta(days=6)]
elif selected_range == "Last 14 Days":
    filtered_df = filtered_df[filtered_df["date"] >= today - pd.Timedelta(days=13)]
elif selected_range == "Last 30 Days":
    filtered_df = filtered_df[filtered_df["date"] >= today - pd.Timedelta(days=29)]

if selected_source != "All":
    filtered_df = filtered_df[filtered_df["source"] == selected_source]

if search_text:
    q = search_text.strip().lower()
    filtered_df = filtered_df[
        filtered_df["city"].astype(str).str.lower().str.contains(q, na=False) |
        filtered_df["country"].astype(str).str.lower().str.contains(q, na=False) |
        filtered_df["symptom"].astype(str).str.lower().str.contains(q, na=False) |
        filtered_df["product_category"].astype(str).str.lower().str.contains(q, na=False) |
        filtered_df["title"].astype(str).str.lower().str.contains(q, na=False)
    ]


# =========================================================
# KPI CARDS
# =========================================================
total_alerts = len(filtered_df)
top_category = filtered_df["product_category"].mode()[0] if not filtered_df.empty else "-"
top_symptom = filtered_df["symptom"].mode()[0] if not filtered_df.empty else "-"
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="card kpi-card kpi-blue">
        <div class="kpi-title">Total Alerts Detected</div>
        <div class="kpi-value">{total_alerts}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card kpi-card kpi-pink">
        <div class="kpi-title">Most Common Product Category</div>
        <div class="kpi-value" style="font-size:30px;">{top_category}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card kpi-card kpi-green">
        <div class="kpi-title">Most Mentioned Symptom</div>
        <div class="kpi-value">{top_symptom}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MAP
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Map: Alerts per Region</div>', unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    map_df = filtered_df.copy()

    map_df["cases"] = pd.to_numeric(map_df["cases"], errors="coerce").fillna(1)
    map_df["radius_outer"] = 18000 + (map_df["cases"] * 2200)
    map_df["radius_mid"] = 10000 + (map_df["cases"] * 1400)
    map_df["radius_core"] = 4200 + (map_df["cases"] * 650)

    map_df["outer_color"] = [[255, 68, 68, 26]] * len(map_df)
    map_df["mid_color"] = [[255, 95, 95, 58]] * len(map_df)
    map_df["core_color"] = [[255, 74, 74, 205]] * len(map_df)
    map_df["ring_color"] = [[255, 165, 165, 190]] * len(map_df)

    outer_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius_outer",
        get_fill_color="outer_color",
        pickable=True,
        opacity=0.18
    )

    mid_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius_mid",
        get_fill_color="mid_color",
        pickable=True,
        opacity=0.28
    )

    ring_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius_mid",
        get_fill_color=[0, 0, 0, 0],
        get_line_color="ring_color",
        stroked=True,
        filled=False,
        line_width_min_pixels=2,
        pickable=False,
        opacity=0.55
    )

    core_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius_core",
        get_fill_color="core_color",
        pickable=True,
        opacity=0.95
    )

    tooltip = {
        "html": """
        <b>City:</b> {city}<br/>
        <b>Country:</b> {country}<br/>
        <b>Category:</b> {product_category}<br/>
        <b>Symptom:</b> {symptom}<br/>
        <b>Source:</b> {source}<br/>
        <b>Cases:</b> {cases}
        """,
        "style": {
            "backgroundColor": "white",
            "color": "#1f2937",
            "borderRadius": "10px"
        }
    }

    deck = pdk.Deck(
        layers=[outer_layer, mid_layer, ring_layer, core_layer],
        initial_view_state=pdk.ViewState(latitude=23, longitude=24, zoom=1.15, pitch=0),
        map_style=MAP_STYLE,
        tooltip=tooltip
    )

    st.pydeck_chart(deck, use_container_width=True, height=500)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# CHARTS
# =========================================================
left_col, right_col = st.columns([1.35, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Product Category Chart</div>', unsafe_allow_html=True)

    cat_df = (
        filtered_df["product_category"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
        .reset_index()
    )
    cat_df.columns = ["Product Category", "Count"]

    if cat_df.empty:
        st.info("No category data.")
    else:
        fig_bar = px.bar(
            cat_df,
            x="Product Category",
            y="Count",
            color="Product Category"
        )

        fig_bar.update_traces(
            marker_line_width=1.5,
            marker_line_color="rgba(255,255,255,0.55)"
        )

        fig_bar.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="",
            font=dict(color=TEXT)
        )

        fig_bar.update_xaxes(showgrid=False, categoryorder="total descending")
        fig_bar.update_yaxes(gridcolor=GRID)

        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Symptom Frequency</div>', unsafe_allow_html=True)

    symptom_df = (
        filtered_df["symptom"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
        .reset_index()
    )
    symptom_df.columns = ["Symptom", "Count"]

    if symptom_df.empty:
        st.info("No symptom data.")
    else:
        symptom_colors = {
            "Vomiting": "#63d89b",
            "Fever": "#ff7c5c",
            "Diarrhea": "#7da8ff",
            "Nausea": "#aa6cf7",
            "Unknown": "#cccccc"
        }

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=symptom_df["Symptom"],
                    values=symptom_df["Count"],
                    hole=0.58,
                    marker=dict(
                        colors=[
                            symptom_colors.get(s, "#cccccc")
                            for s in symptom_df["Symptom"]
                        ]
                    ),
                    textinfo="label+percent"
                )
            ]
        )

        fig_donut.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="v",
                x=1.0,
                y=0.5,
                font=dict(color=TEXT)
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT)
        )

        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TABLE + DOWNLOAD
# =========================================================
with st.expander("Show Detected Alerts Table"):
    show_cols = [
        "source",
        "date",
        "title",
        "hazard",
        "product",
        "product_category",
        "final_location",
        "city",
        "country",
        "alert_score",
        "category_score",
        "url"
    ]

    show_cols = [c for c in show_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[show_cols].sort_values("date", ascending=False),
        use_container_width=True
    )

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download results as CSV",
    csv,
    "foodborne_alerts.csv",
    "text/csv",
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================
twitter_count = (df["source"] == "Twitter/X").sum() if not df.empty else 0
fda_count = (df["source"] == "FDA").sum() if not df.empty else 0

st.markdown(
    f"""
    <div class="footer-note">
        Live data rows: {len(df)} |
        FDA rows: {fda_count} |
        Twitter/X rows: {twitter_count} |
        Filtered rows: {len(filtered_df)} |
        Mode: {st.session_state.theme_mode}
    </div>
    """,
    unsafe_allow_html=True
)