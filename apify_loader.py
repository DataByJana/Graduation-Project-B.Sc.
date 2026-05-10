from typing import List, Dict, Any
import pandas as pd
import streamlit as st
from apify_client import ApifyClient
from .preprocessing import safe_get, normalize_date, clean_text

DEFAULT_QUERIES = [
    '(food poisoning OR foodborne illness OR salmonella OR listeria OR "e coli" OR "food recall" OR "contaminated food") lang:en -filter:retweets',
    '(food OR recall OR salmonella OR listeria OR "e coli" OR contaminated OR vomiting OR diarrhea) lang:en -filter:retweets',
    '"food poisoning" lang:en -filter:retweets',
    '"foodborne illness" lang:en -filter:retweets',
    '"food recall" lang:en -filter:retweets',
    '"contaminated food" lang:en -filter:retweets',
    'salmonella lang:en -filter:retweets',
    'listeria lang:en -filter:retweets',
    '"e coli" lang:en -filter:retweets',
    'norovirus lang:en -filter:retweets',
    '"recall alert" food lang:en -filter:retweets',
    '"sick after eating" lang:en -filter:retweets',
    '"vomiting after eating" lang:en -filter:retweets',
    '"diarrhea after eating" lang:en -filter:retweets',
    '"stomach pain" food lang:en -filter:retweets',
    '"food safety" alert lang:en -filter:retweets',
    '"product recall" food lang:en -filter:retweets',
    '"FDA recall" food lang:en -filter:retweets',
    '"chicken recall" lang:en -filter:retweets',
    '"milk recall" lang:en -filter:retweets',
    '"cheese recall" lang:en -filter:retweets',
    '"eggs recall" lang:en -filter:retweets',
    '"lettuce recall" lang:en -filter:retweets',
]


def _profile_location(item: Dict[str, Any]) -> str:
    author = item.get("author") or item.get("user") or {}

    if isinstance(author, dict):
        return safe_get(author, ["location", "profileLocation"])

    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_tweets(
    apify_token: str,
    max_items: int = 300,
    queries: List[str] = None,
) -> pd.DataFrame:
    """Fetch tweets from Apify and apply preprocessing before model inference."""
    if not apify_token:
        return pd.DataFrame()

    queries = queries or DEFAULT_QUERIES
    client = ApifyClient(apify_token)

    run_input = {
        "searchTerms": queries,
        "maxItems": int(max_items),
        "sort": "Latest",
        "tweetLanguage": "en",
    }

    try:
        run = client.actor("apidojo/tweet-scraper").call(run_input=run_input)
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    except Exception as exc:
        st.warning(f"Apify fetch failed: {exc}")
        return pd.DataFrame()

    rows = []

    for item in items:
        raw_text = safe_get(item, ["text", "full_text", "content", "tweetText"])
        cleaned_text = clean_text(raw_text)

        rows.append({
            "source": "Twitter/X",
            "record_id": safe_get(item, ["id", "tweet_id", "url"]),
            "text": cleaned_text,
            "raw_text": raw_text,
            "created_at": normalize_date(
                safe_get(item, ["createdAt", "created_at", "date"])
            ),
            "metadata_location": (
                safe_get(item, ["place", "location"])
                or _profile_location(item)
            ),
            "author": safe_get(
                item.get("author", {})
                if isinstance(item.get("author"), dict)
                else {},
                ["userName", "username", "name"],
            ),
            "url": safe_get(item, ["url", "twitterUrl"]),
            "fda_classification": "",
            "fda_status": "",
            "fda_firm": "",
            "raw_location_fields": "",
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["text"], keep="first")
        df = df[df["text"].astype(str).str.strip() != ""]
        df = df.reset_index(drop=True)

    return df
