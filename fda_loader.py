import requests
import pandas as pd
import streamlit as st
from .preprocessing import normalize_date, safe_get, clean_text

OPENFDA_URL = "https://api.fda.gov/food/enforcement.json"


def build_fda_title(product_description, reason_for_recall, recalling_firm=""):
    product_description = clean_text(product_description)
    reason_for_recall = clean_text(reason_for_recall)
    recalling_firm = clean_text(recalling_firm)

    if recalling_firm and product_description:
        return f"{recalling_firm} recalls {product_description[:80]}"
    if product_description:
        return f"Recall of {product_description[:100]}"
    if reason_for_recall:
        return f"Food recall due to {reason_for_recall[:100]}"

    return "FDA Food Recall"


def build_fda_text(item):
    product_description = clean_text(safe_get(item, ["product_description"]))
    reason = clean_text(safe_get(item, ["reason_for_recall"]))
    classification = clean_text(safe_get(item, ["classification"]))
    recalling_firm = clean_text(safe_get(item, ["recalling_firm"]))
    city = clean_text(safe_get(item, ["city"]))
    state = clean_text(safe_get(item, ["state"]))
    country = clean_text(safe_get(item, ["country"]))
    distribution = clean_text(safe_get(item, ["distribution_pattern"]))

    location = ", ".join([x for x in [city, state, country] if x])

    sentences = []

    if product_description:
        sentences.append(f"The recalled product is {product_description}.")

    if reason:
        sentences.append(f"The reason for the recall is {reason}.")

    if recalling_firm:
        sentences.append(f"The recalling firm is {recalling_firm}.")

    extra = []
    if classification:
        extra.append(f"classification: {classification}")
    if location:
        extra.append(f"location: {location}")
    if distribution:
        extra.append(f"distribution: {distribution}")

    if extra:
        sentences.append("Additional recall details include " + "; ".join(extra) + ".")

    return clean_text(" ".join(sentences))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fda_recalls(limit: int = 100) -> pd.DataFrame:
    """Fetch openFDA food enforcement records and match training-style preprocessing."""
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
        recalling_firm = safe_get(item, ["recalling_firm"])

        if not product_description or not reason:
            continue

        parsed_date = normalize_date(
            safe_get(item, ["report_date", "recall_initiation_date"])
        )

        if pd.isna(parsed_date):
            continue

        state = clean_text(safe_get(item, ["state"]))
        country = clean_text(safe_get(item, ["country"])) or "US"

        if country == "united states":
            country = "US"

        distribution = safe_get(item, ["distribution_pattern"])

        raw_text = build_fda_text(item)
        cleaned_text = clean_text(raw_text)

        title = build_fda_title(
            product_description=product_description,
            reason_for_recall=reason,
            recalling_firm=recalling_firm,
        )

        rows.append({
            # App columns
            "source": "FDA",
            "record_id": safe_get(item, ["recall_number", "event_id"]),
            "title": title,
            "text": cleaned_text,
            "raw_text": raw_text,
            "created_at": parsed_date,
            "metadata_location": ", ".join([x for x in [state, country] if x]),
            "author": recalling_firm,
            "url": "",

            # FDA metadata
            "fda_classification": safe_get(item, ["classification"]),
            "fda_status": safe_get(item, ["status"]),
            "fda_firm": recalling_firm,
            "raw_location_fields": " | ".join(
                [x for x in [state, country, distribution] if x]
            ),
            "product_description": product_description,
            "reason_for_recall": reason,
            "distribution_pattern": distribution,

            # Training-style columns
            "year": int(parsed_date.year),
            "month": int(parsed_date.month),
            "day": int(parsed_date.day),
            "country": country,
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(
            subset=["year", "month", "day", "country", "title", "text"],
            keep="first"
        )
        df = df[df["text"].astype(str).str.strip() != ""]
        df = df.reset_index(drop=True)

    return df

