import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="foodborne_dashboard_project")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

INVALID_LOCATIONS = {
    "unknown",
    "none",
    "nan",
    "",
    "north america",
    "worldwide",
    "global",
    "death city",
}

def first_non_empty(*values) -> str:
    for value in values:
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return "Unknown"

def resolve_final_location(row: pd.Series) -> str:
    """Apply required location logic for tweets and FDA records."""
    ner_location = row.get("extracted_location", "")
    if row.get("source") == "Twitter/X":
        return first_non_empty(
            ner_location,
            row.get("metadata_location"),
            "Unknown"
        )
    return first_non_empty(
        ner_location,
        row.get("metadata_location"),
        row.get("raw_location_fields"),
        "Unknown"
    )

def is_valid_location(location: str) -> bool:
    if location is None or pd.isna(location):
        return False
    location = str(location).strip().lower()
    if location in INVALID_LOCATIONS:
        return False
    return True

def geocode_location(location: str) -> pd.Series:
    if not is_valid_location(location):
        return pd.Series([None, None])
    try:
        result = geocode(str(location))
        if result:
            return pd.Series([result.latitude, result.longitude])
    except Exception:
        pass
    return pd.Series([None, None])

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_location_cached(location: str):
    result = geocode_location(location)
    return result.iloc[0], result.iloc[1]
