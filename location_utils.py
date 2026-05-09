import pandas as pd


def first_non_empty(*values) -> str:
    for value in values:
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return "Unknown"


def resolve_final_location(row: pd.Series) -> str:
    """Apply required location logic for tweets and FDA records."""
    ner_location = row.get("extracted_location", "")
    if row.get("source") == "Twitter/X":
        return first_non_empty(ner_location, row.get("metadata_location"), "Unknown")
    return first_non_empty(ner_location, row.get("metadata_location"), row.get("raw_location_fields"), "Unknown")
