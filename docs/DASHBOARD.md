# Dashboard (Streamlit Explorer)

This project includes a lightweight Streamlit app for browsing analysis outputs in `data/glo_reports.db`.

## Run

```bash
# Install deps
source venv/bin/activate
pip install -r requirements.txt

# Launch
streamlit run dashboard/app.py
```

The app defaults to `data/glo_reports.db`. You can point it to a different SQLite path in the sidebar.

## What it shows

- **Overview**: quick table counts + latest quarter
- **Topics**: pick a `topic_models.id`, browse topics, view representative snippets, and quarter trends
- **Relations**: browse top co-occurrence edges + evidence snippets
- **Money**: filter money mentions by context/amount/entity and inspect linked entities
- **Sections**: heading-family distribution + top headings by count

