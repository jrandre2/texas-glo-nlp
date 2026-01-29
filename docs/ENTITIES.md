# Entity Reference

Documentation of all named entity types extracted by the NLP pipeline.

## Table of Contents

- [Entity Overview](#entity-overview)
- [Standard spaCy Entities](#standard-spacy-entities)
- [Custom Domain Entities](#custom-domain-entities)
- [Regex Pattern Entities](#regex-pattern-entities)
- [Adding New Patterns](#adding-new-patterns)

---

## Entity Overview

The NLP pipeline extracts 27 entity types totaling 4.2M+ entities:

| Category | Entity Types | Count |
|----------|-------------|-------|
| Standard spaCy | MONEY, ORG, CARDINAL, DATE, GPE, etc. | 3.8M |
| Custom Domain | DISASTER, FEMA_DECLARATION, TX_COUNTY, PROGRAM | 180K |
| Regex Patterns | MONEY, DAMAGE_METRIC, RAINFALL, WIND_SPEED, QUARTER | 25K |

---

## Standard spaCy Entities

Entities recognized by the spaCy `en_core_web_sm` or `en_core_web_trf` model.

### PERSON

People's names, including titles.

**Examples**: `John Smith`, `Governor Abbott`, `Secretary Carson`

### ORG (Organization)

Companies, agencies, institutions.

**Examples**: `Texas GLO`, `HUD`, `FEMA`, `Harris County`, `City of Houston`

**Count**: 1,152,944

### GPE (Geopolitical Entity)

Countries, cities, states.

**Examples**: `Texas`, `Houston`, `United States`, `Harris County`

**Count**: 198,125

### DATE

Dates and time periods.

**Examples**: `August 25, 2017`, `Q4 2024`, `2019`, `September`

**Count**: 351,447

### MONEY

Monetary values (also extracted via regex).

**Examples**: `$5,685,492,029.81`, `$1.3 billion`, `$50 million`

**Count**: 1,284,980

### CARDINAL

Numerals that don't fit other categories.

**Examples**: `442`, `153,540`, `82`

**Count**: 488,396

### ORDINAL

Position indicators.

**Examples**: `first`, `second`, `Q1`, `Phase 2`

### PERCENT

Percentage values.

**Examples**: `95%`, `73 percent`, `0.5%`

**Count**: 28,097

### QUANTITY

Measurements with units.

**Examples**: `60 inches`, `150 mph`, `1,200 homes`

**Count**: 68,573

### FAC (Facility)

Buildings, infrastructure.

**Examples**: `City Hall`, `Highway 290`, `Port of Houston`

**Count**: 180,231

### LOC (Location)

Non-GPE locations.

**Examples**: `Lower Rio Grande Valley`, `Gulf Coast`, `Galveston Bay`

**Count**: 30,065

### PRODUCT

Objects, vehicles, foods.

**Examples**: `CDBG-DR`, `DRGR`, `HMGP`

### LAW

Named laws and regulations.

**Examples**: `Stafford Act`, `CDBG regulations`, `42 U.S.C. 5306`

**Count**: 9,851

### EVENT

Named events (hurricanes, etc.).

**Examples**: `Hurricane Harvey`, `Tropical Storm Imelda`

**Count**: 6,916

### WORK_OF_ART

Titles of documents, reports.

**Examples**: `Action Plan`, `Quarterly Performance Report`

### NORP

Nationalities, religious groups.

**Examples**: `American`, `Texan`, `Hispanic`

### TIME

Times of day.

**Examples**: `5:00 PM`, `noon`, `midnight`

---

## Custom Domain Entities

Entities defined using spaCy EntityRuler with custom patterns.

### DISASTER

Named disaster events.

**Patterns**:
- `Hurricane {Name}` (Hurricane Harvey, Hurricane Ike)
- `Tropical Storm {Name}` (Tropical Storm Imelda)
- `{Year} {Type}` (2019 Floods)

**Examples**:
| Entity | Mentions |
|--------|----------|
| Hurricane Ike | 23,933 |
| Hurricane Harvey | 13,213 |
| Hurricane Dolly | 3,431 |
| Hurricane Rita | 724 |
| Tropical Storm Imelda | 650 |

**Count**: 50,676 (24 unique)

### FEMA_DECLARATION

FEMA disaster declaration numbers.

**Patterns**:
- `DR-{4 digits}` (DR-4332, DR-1791)
- `FEMA-{4 digits}` (FEMA-4332, FEMA-1791)
- `FEMA-{4 digits}-TX` (FEMA-4332-TX)

**Examples**:
| Entity | Mentions |
|--------|----------|
| DR-4029 | 82 |
| FEMA-4272 | 76 |
| FEMA-4332 | 72 |
| DR-4245 | 66 |
| FEMA-1791 | 53 |

**Count**: 893 (23 unique)

### TX_COUNTY

Texas county names.

**Recognized Counties** (54 total):
```
Harris, Jefferson, Galveston, Chambers, Orange, Brazoria,
Fort Bend, Montgomery, Liberty, Waller, Austin, Colorado,
Wharton, Matagorda, Jackson, Victoria, Calhoun, Refugio,
Aransas, San Patricio, Nueces, Kleberg, Kenedy, Willacy,
Cameron, Hidalgo, Starr, Webb, Zapata, Jim Hogg, Brooks,
Duval, Jim Wells, Live Oak, Bee, Goliad, DeWitt, Lavaca,
Fayette, Bastrop, Lee, Washington, Grimes, Walker, San Jacinto,
Polk, Tyler, Hardin, Jasper, Newton, Sabine, Angelina, Nacogdoches
```

**Top Counties**:
| County | Mentions |
|--------|----------|
| Harris | 15,234 |
| Jefferson | 8,921 |
| Galveston | 7,456 |
| Cameron | 5,123 |
| Hidalgo | 4,892 |

**Count**: 103,497 (104 unique)

### PROGRAM

Recovery program names.

**Patterns**:
- `Homeowner Assistance Program`
- `Homeowner Reimbursement Program`
- `Local Buyout Program`
- `Affordable Rental Program`
- `Housing Assistance Program`
- `Infrastructure Program`
- `Economic Revitalization`
- `Mitigation Program`

**Examples**:
| Program | Mentions |
|---------|----------|
| Homeowner Assistance Program | 8,234 |
| Housing Program | 5,621 |
| Infrastructure Program | 4,123 |
| Buyout Program | 3,456 |

**Count**: 24,638 (24 unique)

### GRANT_NUMBER

HUD grant numbers.

**Pattern**: `B-{2 digits}-{2 letters}-{2 digits}-{4 digits}`

**Examples**: `B-17-DM-48-0001`, `B-08-DI-48-0001`

---

## Regex Pattern Entities

Entities extracted using regular expressions.

### MONEY (Regex)

Dollar amounts in various formats.

**Patterns**:
```python
r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?'
```

**Formats Recognized**:
- `$5,685,492,029.81` (exact amounts)
- `$1.3 billion` (billions)
- `$50 million` (millions)
- `$1.2M` (abbreviated millions)
- `$500B` (abbreviated billions)

**Examples**:
| Amount | Mentions |
|--------|----------|
| $0 | 45,234 |
| $1,000,000 | 12,345 |
| $5.6 billion | 234 |

### DAMAGE_METRIC

Quantified damage statistics.

**Patterns**:
```python
r'\d+(?:,\d{3})*\s+(?:homes?|units?|structures?|buildings?|families|households|residents|fatalities|deaths|casualties|injuries)'
```

**Examples**:
| Metric | Sample |
|--------|--------|
| Homes | `1,200 homes destroyed` |
| Fatalities | `82 fatalities` |
| Families | `50,000 families affected` |
| Units | `3,500 housing units` |

**Count**: 2,867 (408 unique)

### RAINFALL

Precipitation measurements.

**Patterns**:
```python
r'\d+(?:\.\d+)?\s*(?:inches?|in\.)\s*(?:of\s+)?(?:rain(?:fall)?)?'
```

**Examples**:
| Amount | Sample |
|--------|--------|
| 60 inches | `60 inches of rainfall` |
| 25.5 in | `25.5 in. of rain` |
| 40 inches | `40 inches of precipitation` |

**Count**: 22,163 (433 unique)

### WIND_SPEED

Wind speed measurements.

**Patterns**:
```python
r'\d+(?:\.\d+)?\s*(?:mph|miles?\s*per\s*hour)'
```

**Examples**:
- `150 mph winds`
- `85 miles per hour`
- `110 mph sustained`

**Count**: 211 (14 unique)

### QUARTER

Fiscal quarters.

**Patterns**:
```python
r'Q[1-4]\s+\d{4}'
r'\d{4}\s+Q[1-4]'
r'(?:first|second|third|fourth)\s+quarter\s+\d{4}'
```

**Examples**:
- `Q4 2024`
- `2023 Q3`
- `first quarter 2025`

**Count**: 4

---

## Adding New Patterns

### Adding Custom Entity Patterns

Edit `src/nlp_processor.py`:

```python
# In DISASTER_PATTERNS list
DISASTER_PATTERNS = [
    # Add new disaster pattern
    {"label": "DISASTER", "pattern": [{"LOWER": "winter"}, {"LOWER": "storm"}, {"IS_TITLE": True}]},
    # Pattern: "Winter Storm Uri"
]

# In COUNTY_PATTERNS - add county names
{"label": "TX_COUNTY", "pattern": [{"LOWER": "travis"}, {"LOWER": "county"}]},
```

### Adding Regex Patterns

```python
# In REGEX_PATTERNS dictionary
REGEX_PATTERNS = {
    # Add new pattern
    'POPULATION': r'\d+(?:,\d{3})*\s+(?:residents|population|people)',
}
```

### Testing New Patterns

```python
from src.nlp_processor import NLPProcessor

processor = NLPProcessor()

# Test extraction
test_text = "Winter Storm Uri caused damage to 5,000 homes in Travis County."
entities = processor.extract_all_entities(test_text)

for ent in entities:
    print(f"{ent['type']}: {ent['text']}")
```

### Reprocessing After Changes

```bash
# Reprocess all documents with new patterns
python src/nlp_processor.py --reprocess

# Export updated entities
python src/nlp_processor.py --export
```
