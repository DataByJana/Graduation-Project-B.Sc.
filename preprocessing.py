import re
import html
import pandas as pd


def clean_text(text: str) -> str:
    """Clean and normalize tweet/FDA text while keeping meaningful food-safety terms."""
    if text is None or pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_date(value):
    """Return pandas datetime from several API date formats."""
    if value is None or pd.isna(value) or value == "":
        return pd.NaT
    # FDA uses YYYYMMDD; Apify may return ISO strings.
    value = str(value)
    for fmt in ("%Y%m%d", None):
        try:
            return pd.to_datetime(value, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(value, errors="coerce")


def safe_get(obj: dict, keys, default=""):
    """Safely get the first available field from a dictionary."""
    if not isinstance(obj, dict):
        return default
    for key in keys:
        val = obj.get(key)
        if val not in (None, "", [], {}):
            return val
    return default
