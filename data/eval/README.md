# NER Evaluation Data

Place gold-standard labeled entities here to evaluate extraction quality.

## Expected CSV format

Minimum columns:
- `filename` (or `document_id`)
- `page_number`
- `entity_type`
- `start_char`
- `end_char`

Optional:
- `entity_text`

Example:
```
filename,page_number,entity_type,start_char,end_char,entity_text
drgr-h5b-2025-q4.pdf,12,DISASTER,105,121,Hurricane Harvey
```

## Run evaluation

```
python src/ner_evaluate.py --gold data/eval/gold_entities.csv
```
