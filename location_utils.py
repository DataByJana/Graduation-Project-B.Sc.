import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import streamlit as st


def first_non_empty(*values) -> str:
    for value in values:
        if value is not None and not pd.isna(value) and str(value).strip():
            value = str(value).strip()
            if value.lower() not in ["unknown", "none", "nan", ""]:
                return value
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


geolocator = Nominatim(user_agent="foodborne_dashboard_project")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


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

import streamlit as st

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_location_cached(location: str):
    result = geocode_location(location)
    return result.iloc[0], result.iloc[1]