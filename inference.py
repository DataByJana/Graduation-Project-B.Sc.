from typing import List, Dict
import pandas as pd
import torch

from .preprocessing import clean_text
from .location_utils import resolve_final_location


ALERT_LABELS = {"alert", "foodborne_alert", "positive", "1", "LABEL_1"}


HAZARD_KEYWORDS = {
    "Salmonella": ["salmonella"],
    "E. coli": ["e. coli", "ecoli", "escherichia coli"],
    "Listeria": ["listeria"],
    "Norovirus": ["norovirus"],
    "Allergen": ["allergen", "undeclared allergen", "allergy"],
    "Foreign Material": ["foreign material", "plastic", "metal", "glass"],
    "Mold": ["mold", "mould"],
    "Food Poisoning": ["food poisoning", "foodborne illness", "contaminated", "contamination"],
}


LOCATION_KEYWORDS = {
    "Riyadh": ["riyadh"],
    "Jeddah": ["jeddah"],
    "Dammam": ["dammam"],
    "Makkah": ["makkah", "mecca"],
    "Madinah": ["madinah", "medina"],
    "Dubai": ["dubai"],
    "London": ["london"],
    "New York": ["new york"],
    "California": ["california"],
    "Texas": ["texas"],
    "Florida": ["florida"],
}


def detect_hazard_fallback(text):
    text = str(text).lower()
    for hazard, words in HAZARD_KEYWORDS.items():
        if any(w in text for w in words):
            return hazard
    return "Food Safety Concern"


def detect_location_fallback(text):
    text = str(text).lower()
    for loc, words in LOCATION_KEYWORDS.items():
        if any(w in text for w in words):
            return loc
    return ""


def _label(result) -> str:
    if isinstance(result, list):
        result = result[0] if result else {}
    return str(result.get("label", ""))


def _score(result) -> float:
    if isinstance(result, list):
        result = result[0] if result else {}
    try:
        return float(result.get("score", 0.0))
    except Exception:
        return 0.0


def is_alert_label(label: str) -> bool:
    return label.strip().lower() in {x.lower() for x in ALERT_LABELS}


def _extract_entities(entities: List[Dict]) -> Dict[str, str]:
    buckets = {"hazard": [], "product": [], "location": []}

    for ent in entities or []:
        group = str(ent.get("entity_group") or ent.get("entity") or "").lower()
        word = str(ent.get("word") or "").strip()

        if not word:
            continue

        if any(x in group for x in ["hazard", "haz", "pathogen", "contaminant", "microorganism"]):
            buckets["hazard"].append(word)

        elif any(x in group for x in ["product", "prod", "food", "item"]):
            buckets["product"].append(word)

        elif any(x in group for x in ["location", "loc", "gpe", "city", "state", "country", "place"]):
            buckets["location"].append(word)

    return {k: "; ".join(dict.fromkeys(v)) for k, v in buckets.items()}


def _predict_alert_one(text: str, alert_obj: dict) -> Dict:
    if alert_obj is None or not isinstance(alert_obj, dict):
        return {"label": "LABEL_1", "score": 1.0}

    try:
        model = alert_obj.get("model")
        tokenizer = alert_obj.get("tokenizer")
        id2label = alert_obj.get("id2label")
        label_names = alert_obj.get("label_names")
        max_length = alert_obj.get("max_length", 128)

        if model is None or tokenizer is None:
            return {"label": "LABEL_1", "score": 1.0}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = int(torch.argmax(probs, dim=1).item())
            score = float(probs[0][pred_id].item())

        if id2label is not None:
            label = str(id2label.get(pred_id, pred_id))
        elif label_names is not None:
            label = str(label_names[pred_id])
        else:
            label = str(pred_id)

        return {"label": label, "score": score}

    except Exception:
        return {"label": "LABEL_1", "score": 1.0}


def _predict_category_one(text: str, category_obj: dict) -> Dict:
    if category_obj is None or not isinstance(category_obj, dict):
        return {"label": "Unknown", "score": 0.0}

    try:
        model = category_obj.get("model")
        tokenizer = category_obj.get("tokenizer")
        label_encoder = category_obj.get("label_encoder")
        id2label = category_obj.get("id2label")
        max_length = category_obj.get("max_length", 512)

        if model is None or tokenizer is None:
            return {"label": "Unknown", "score": 0.0}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = int(torch.argmax(probs, dim=1).item())
            score = float(probs[0][pred_id].item())

        if label_encoder is not None:
            label = str(label_encoder.inverse_transform([pred_id])[0])
        elif id2label is not None:
            label = str(id2label.get(pred_id, pred_id))
        elif hasattr(model.config, "id2label"):
            label = str(model.config.id2label.get(pred_id, pred_id))
        else:
            label = "Unknown"

        return {"label": label, "score": score}

    except Exception:
        return {"label": "Unknown", "score": 0.0}


def _predict_ner_one(text: str, ner_obj: dict) -> List[Dict]:
    if ner_obj is None or not isinstance(ner_obj, dict):
        return []

    try:
        model = ner_obj.get("model")
        tokenizer = ner_obj.get("tokenizer")
        id2label = ner_obj.get("id2label")
        max_length = ner_obj.get("max_length", 128)

        if model is None or tokenizer is None or id2label is None:
            return []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        encoded = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        offset_mapping = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].cpu().tolist())

        entities = []
        current_group = None
        current_words = []

        for token, pred_id, offset in zip(tokens, pred_ids, offset_mapping):
            label = id2label.get(pred_id, "O")

            if token in tokenizer.all_special_tokens or offset == [0, 0]:
                continue

            clean_token = (
                token.replace("##", "")
                .replace("▁", " ")
                .replace("Ġ", " ")
                .strip()
            )

            if label == "O":
                if current_group and current_words:
                    entities.append({
                        "entity_group": current_group,
                        "word": " ".join(current_words),
                    })
                current_group = None
                current_words = []
                continue

            if "-" not in label:
                continue

            prefix, ent_type = label.split("-", 1)

            if prefix == "B":
                if current_group and current_words:
                    entities.append({
                        "entity_group": current_group,
                        "word": " ".join(current_words),
                    })
                current_group = ent_type
                current_words = [clean_token]

            elif prefix == "I" and current_group == ent_type:
                current_words.append(clean_token)

        if current_group and current_words:
            entities.append({
                "entity_group": current_group,
                "word": " ".join(current_words),
            })

        return entities

    except Exception:
        return []


def run_pipeline(df: pd.DataFrame, alert_pipe, ner_pipe, category_pipe) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()

    if df.empty:
        return df

    # alert_pipe is now a pickle dict, not a Hugging Face pipeline
    alert_outputs = [
        _predict_alert_one(text, alert_pipe)
        for text in df["clean_text"].tolist()
    ]

    df["alert_label"] = [_label(x) for x in alert_outputs]
    df["alert_score"] = [_score(x) for x in alert_outputs]

    df = df[df["alert_label"].apply(is_alert_label)].copy()

    if df.empty:
        return df

    # ner_pipe is now a pickle dict, not a Hugging Face pipeline
    ner_outputs = [
        _predict_ner_one(text, ner_pipe)
        for text in df["clean_text"].tolist()
    ]

    extracted = [_extract_entities(x) for x in ner_outputs]

    df["hazard"] = [
        x.get("hazard", "") or detect_hazard_fallback(text)
        for x, text in zip(extracted, df["clean_text"])
    ]

    df["product"] = [
        x.get("product", "") or "Unknown"
        for x in extracted
    ]

    df["extracted_location"] = [
        x.get("location", "") or detect_location_fallback(text)
        for x, text in zip(extracted, df["clean_text"])
    ]

    # category_pipe is now a pickle dict, not a Hugging Face pipeline
    cat_outputs = [
        _predict_category_one(text, category_pipe)
        for text in df["clean_text"].tolist()
    ]

    df["product_category"] = [
        _label(x).replace("_", " ").title()
        for x in cat_outputs
    ]

    df["category_score"] = [_score(x) for x in cat_outputs]

    df["final_location"] = df.apply(resolve_final_location, axis=1)

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

    return df.reset_index(drop=True)