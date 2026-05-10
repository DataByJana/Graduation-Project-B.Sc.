import re
import html
import pandas as pd

try:
    import emoji
except ImportError:
    emoji = None

try:
    import ftfy
except ImportError:
    ftfy = None


def clean_text(text: str) -> str:
    """
    Clean and normalize tweet/FDA text.
    Used before sending text to the model.
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)

    if ftfy is not None:
        text = ftfy.fix_text(text)

    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"#", "", text)

    if emoji is not None:
        text = emoji.demojize(text)

    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


def normalize_date(value):
    """Return pandas datetime from several API date formats."""
    if value is None or pd.isna(value) or value == "":
        return pd.NaT

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