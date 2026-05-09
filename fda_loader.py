import requests
import pandas as pd
import streamlit as st
from .preprocessing import normalize_date, safe_get

OPENFDA_URL = "https://api.fda.gov/food/enforcement.json"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fda_recalls(limit: int = 100) -> pd.DataFrame:
    """Fetch openFDA food enforcement records and normalize fields."""
    params = {
        "limit": min(int(limit), 1000),
        "sort": "recall_initiation_date:desc",
    }
    try:
        response = requests.get(OPENFDA_URL, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as exc:
        st.warning(f"FDA fetch failed: {exc}")
        return pd.DataFrame()

    rows = []
    for item in results:
        product_description = safe_get(item, ["product_description"])
        reason = safe_get(item, ["reason_for_recall"])
        state = safe_get(item, ["state"])
        country = safe_get(item, ["country"])
        distribution = safe_get(item, ["distribution_pattern"])
        rows.append({
            "source": "FDA",
            "record_id": safe_get(item, ["recall_number", "event_id"]),
            "text": f"{product_description}. {reason}".strip(),
            "created_at": normalize_date(safe_get(item, ["recall_initiation_date", "report_date"])),
            "metadata_location": ", ".join([x for x in [state, country] if x]),
            "author": safe_get(item, ["recalling_firm"]),
            "url": "",
            "fda_classification": safe_get(item, ["classification"]),
            "fda_status": safe_get(item, ["status"]),
            "fda_firm": safe_get(item, ["recalling_firm"]),
            "raw_location_fields": " | ".join([x for x in [state, country, distribution] if x]),
            "product_description": product_description,
            "reason_for_recall": reason,
            "distribution_pattern": distribution,
        })
    return pd.DataFrame(rows)
