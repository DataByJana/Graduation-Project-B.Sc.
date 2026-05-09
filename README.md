# Foodborne Illness Early Warning Dashboard

Streamlit dashboard that combines Twitter/X signals from Apify with openFDA food enforcement records, runs three trained NLP models, and displays alert analytics.

## System pipeline

```text
Apify Twitter/X Scraper                  openFDA Food Enforcement API
        |                                             |
        v                                             v
 Normalize tweet fields                  Normalize FDA recall fields
        |                                             |
        +-------------------+-------------------------+
                            v
                    Clean and normalize text
                            v
                    Alert classifier
                            v
              Keep alert-related records only
                            v
          +-----------------+------------------+
          |                                    |
          v                                    v
 NER model: hazard/product/location     Product/category classifier
          |                                    |
          +-----------------+------------------+
                            v
             Resolve final location by source
                            |
        Twitter/X: NER location first, else metadata/profile location
        FDA: NER location first, else state/country/distribution_pattern
                            v
                    Final dataframe
                            v
                Streamlit dashboard KPIs,
             filters, charts, tables, CSV download
```

## Folder structure

```text
foodborne_dashboard_project/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml
├── models/
│   ├── alert_classifier/
│   ├── ner_model/
│   └── category_classifier/
└── utils/
    ├── apify_loader.py
    ├── fda_loader.py
    ├── preprocessing.py
    ├── model_loader.py
    ├── inference.py
    └── location_utils.py
```

## Model folders

Export or copy your Hugging Face compatible models into:

```text
models/alert_classifier/
models/ner_model/
models/category_classifier/
```

Each folder should include files such as:

```text
config.json
model.safetensors or pytorch_model.bin
tokenizer.json / vocab files
special_tokens_map.json
tokenizer_config.json
```

The code uses `transformers.pipeline`, so the models must be loadable with `pipeline(..., model="models/...", tokenizer="models/...")`.

## Alert label note

In `utils/inference.py`, update `ALERT_LABELS` if your alert classifier uses different positive labels.

Current accepted alert labels:

```python
{"alert", "foodborne_alert", "positive", "1", "LABEL_1"}
```

## Run locally

```bash
cd foodborne_dashboard_project
pip install -r requirements.txt
streamlit run app.py
```

Add your Apify token in `.streamlit/secrets.toml`:

```toml
APIFY_TOKEN = "YOUR_APIFY_TOKEN_HERE"
```

## Deploy on Streamlit Community Cloud

1. Push this project folder to a GitHub repository.
2. Make sure `app.py`, `requirements.txt`, `utils/`, and `models/` are included.
3. Go to Streamlit Community Cloud.
4. Create a new app from your GitHub repository.
5. Set the main file path to:

```text
app.py
```

6. Add your secret in the Streamlit app settings:

```toml
APIFY_TOKEN = "YOUR_APIFY_TOKEN_HERE"
```

7. Deploy.

## Dashboard features

KPIs:
- Total alerts detected
- Number of Twitter/X alerts
- Number of FDA alerts
- Locations found
- Products found
- Most common hazard
- Most common product category

Charts:
- Alerts by source
- Product category frequency
- Hazard frequency
- Alerts over time
- FDA classification distribution

Tables:
- Full detected alerts table
- Twitter/X alerts table
- FDA alerts table

Controls:
- Source filter
- Category filter
- Hazard filter
- Location filter
- Date range filter
- Refresh data button
- Download CSV button
