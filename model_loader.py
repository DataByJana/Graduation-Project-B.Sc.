import os
import pickle
import streamlit as st


ALERT_MODEL_PATH = "models/alert_classifier/bertweet_alert_detection_model.pkl"
NER_MODEL_PATH = "models/ner_model/biobert_best_ner_model.pkl"
CATEGORY_MODEL_PATH = "models/category_classifier/modernbert_product_best_model.pkl"


def load_pickle_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def patch_modernbert_sliding_window(model):
    """
    Fix ModernBERT compatibility issue:
    'ModernBertAttention' object has no attribute 'sliding_window'
    """
    if model is None:
        return model

    try:
        config = getattr(model, "config", None)
        sliding_window = getattr(config, "sliding_window", None)

        for module in model.modules():
            if module.__class__.__name__ == "ModernBertAttention":
                if not hasattr(module, "sliding_window"):
                    module.sliding_window = sliding_window

        return model

    except Exception:
        return model


@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    """
    Load local pickle models once per Streamlit session.
    This does NOT use Hugging Face pipeline folders.
    """

    alert_obj = load_pickle_model(ALERT_MODEL_PATH)
    ner_obj = load_pickle_model(NER_MODEL_PATH)
    category_obj = load_pickle_model(CATEGORY_MODEL_PATH)

    if isinstance(category_obj, dict) and "model" in category_obj:
        category_obj["model"] = patch_modernbert_sliding_window(category_obj["model"])

    return alert_obj, ner_obj, category_obj
