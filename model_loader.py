import os
import pickle
import streamlit as st
from huggingface_hub import hf_hub_download

REPO_ID = "DataByJana/GP-models"

ALERT_MODEL_FILE = "bertweet_alert_detection_model.pkl"
NER_MODEL_FILE = "biobert_best_ner_model.pkl"
CATEGORY_MODEL_FILE = "modernbert_product_best_model.pkl"

def load_pickle_model(filename):
    path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    with open(path, "rb") as f:
        return pickle.load(f)

def patch_modernbert_sliding_window(model):
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
    alert_obj = load_pickle_model(ALERT_MODEL_FILE)
    ner_obj = load_pickle_model(NER_MODEL_FILE)
    category_obj = load_pickle_model(CATEGORY_MODEL_FILE)

    if isinstance(category_obj, dict) and "model" in category_obj:
        category_obj["model"] = patch_modernbert_sliding_window(category_obj["model"])

    return alert_obj, ner_obj, category_obj