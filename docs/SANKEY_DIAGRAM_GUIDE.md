# Sankey Diagram Best Practices Guide

A guide for creating effective funding flow Sankey diagrams from Texas GLO disaster recovery data.

---

## Overview

Sankey diagrams visualize flow quantities between nodes. For CDBG-DR funding, they show how federal dollars flow from HUD through state agencies to program categories.

---

## Data Format

### Input JSON Structure

The Sankey data files (in `outputs/exports/harvey/`: `harvey_sankey_infrastructure.json`, `harvey_sankey_housing.json`, `harvey_sankey_recipients.json`) use this structure:

```json
{
  "quarter": "Q4 2025",
  "program_type": "Infrastructure",
  "nodes": [
    {"id": "HUD", "name": "HUD", "level": 0},
    {"id": "Texas GLO", "name": "Texas GLO", "level": 1},
    {"id": "Program Category", "name": "Program Category", "level": 2}
  ],
  "links": [
    {"source": "HUD", "target": "Texas GLO", "value": 4416746991.02},
    {"source": "Texas GLO", "target": "Program Category", "value": 1927075533.54}
  ],
  "summary": {
    "total_budget": 4416746991.02,
    "programs": 1,
    "organizations": 3,
    "counties": 62
  }
}
```

### Node Levels

| Level | Description | Example |
|-------|-------------|---------|
| 0 | Federal source | HUD |
| 1 | State administrator | Texas GLO |
| 2 | Program categories | Homeowner Assistance, Infrastructure Projects |
| 3 | (Optional) Subrecipients | Harris County, City of Houston |

---

## Recommended Approach: Matplotlib Custom Drawing

After testing Plotly, D3.js, and matplotlib, **matplotlib with custom drawing** provides the best control over node positioning and label placement.

### Why Matplotlib?

| Approach | Pros | Cons |
|----------|------|------|
| Plotly | Interactive, easy setup | Labels overlap, limited position control |
| D3.js | Flexible, web-native | Requires HTML-to-PDF conversion |
| **Matplotlib** | **Full control, direct PDF export** | Requires manual path drawing |

### Key Implementation Decisions

1. **Remove 100% pass-through nodes**: If HUD → Texas GLO is always 100%, omit HUD to reduce visual clutter
2. **Use uniform colors for leaf nodes**: Colorful category nodes distract when not expanding further
3. **Maximize vertical space**: Stretch diagram to use 90%+ of available height
4. **Stack flows proportionally**: Source node flows should be stacked, not centered

---

## Implementation Guide

### File Location

```
scripts/generate_sankey_matplotlib.py
```

### Core Parameters (Tuned for Best Results)

```python
# Figure size for landscape PDF
figsize = (16, 10)

# Node positioning
node_width = 3
glo_x = 8           # Source node X position (left side)
cat_x = 45          # Target nodes X position (gives room for labels)

# Vertical spacing
top_margin = 90     # Top of diagram area
bottom_margin = 6   # Bottom of diagram area
gap = 1.5           # Gap between category nodes
min_height = 4      # Minimum node height for visibility

# Colors
source_color = '#2c5282'    # Texas GLO blue
category_color = '#718096'  # Uniform gray for all categories
flow_alpha = 0.4            # Flow transparency
```

### Drawing Flow Paths with Bezier Curves

```python
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

vertices = [
    (glo_right, glo_source_top),          # Start at source
    (glo_right + 12, glo_source_top),     # Control point 1
    (cat_x - 12, cat_target_top),         # Control point 2
    (cat_x, cat_target_top),              # End at target top
    (cat_x, cat_target_bottom),           # Target bottom
    (cat_x - 12, cat_target_bottom),      # Control point 3
    (glo_right + 12, glo_source_bottom),  # Control point 4
    (glo_right, glo_source_bottom),       # Back to source
    (glo_right, glo_source_top),          # Close path
]
codes = [
    MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
    MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
    MPath.CLOSEPOLY
]
```

### Proportional Node Heights

```python
# Calculate heights proportional to values
total_value = sum(link['value'] for link in category_links)
space_for_nodes = available_height - (gap * (num_categories - 1))

for link in category_links:
    proportion = link['value'] / total_value
    height = max(min_height, proportion * space_for_nodes)
```

### Currency Formatting

```python
def format_currency(value):
    if value >= 1e9:
        return f'${value/1e9:.2f}B'
    if value >= 1e6:
        return f'${value/1e6:.1f}M'
    if value >= 1e3:
        return f'${value/1e3:.0f}K'
    return f'${value:.0f}'
```

---

## Common Pitfalls to Avoid

### 1. Label Overlapping

**Problem**: With many categories, labels overlap vertically.

**Solution**:
- Use minimum node heights (`min_height = 4`)
- Reduce gap between nodes (`gap = 1.5`)
- Scale nodes down proportionally if they exceed available space

### 2. Flows Not Aligned with Nodes

**Problem**: Flow endpoints don't match node edges.

**Solution**: Track cumulative Y position on source node:
```python
glo_current_y = glo_y_center + glo_height / 2
for pos in category_positions:
    flow_thickness = (pos['value'] / total_budget) * glo_height
    glo_source_top = glo_current_y
    glo_source_bottom = glo_current_y - flow_thickness
    glo_current_y = glo_source_bottom  # Stack flows
```

### 3. Wasted Space

**Problem**: Diagram doesn't use full page.

**Solution**:
- Increase figsize: `figsize=(16, 10)`
- Reduce margins: `top_margin=90, bottom_margin=6`
- Use 95% of available height for source node

### 4. Distracting Colors

**Problem**: Random colors for category nodes add visual noise.

**Solution**: Use uniform gray (`#718096`) for all target nodes when not expanding them further.

---

## Output Formats

### PDF (Print Quality)

```python
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
```

### PNG (GitHub/Web Display)

Use PyMuPDF for high-quality conversion:

```python
import fitz
doc = fitz.open("diagram.pdf")
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
pix.save("diagram.png")
```

---

## Data Terminology

### Budget vs Expenditure

| Term | Meaning |
|------|---------|
| **Budget/Allocated** | Planned funding amount (shown in Sankey) |
| **Obligated** | Committed via contract |
| **Disbursed** | Sent to subrecipient |
| **Expended** | Actually spent on activities |

The Sankey diagrams show **budget allocations**, not actual expenditure. For spending data, query the `harvey_activities` table.

---

## Quick Reference

### Generate New Sankey PDFs

```bash
cd "/Volumes/T9/Texas GLO Action Plan Project"
source venv/bin/activate
python scripts/generate_sankey_matplotlib.py
```

### Convert to PNG

```python
import fitz
doc = fitz.open("outputs/visualizations/harvey_sankey_5b.pdf")
pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
pix.save("outputs/visualizations/harvey_sankey_5b.png")
```

### Update Data Source

Edit `src/funding_tracker.py` to modify the Sankey data generation, then regenerate JSON exports.

---

## Files Reference

| File | Purpose |
|------|---------|
| `outputs/exports/harvey/harvey_sankey_infrastructure.json` | 5B grant flow data |
| `outputs/exports/harvey/harvey_sankey_housing.json` | 57M grant flow data |
| `scripts/generate_sankey_matplotlib.py` | PDF generator |
| `outputs/visualizations/harvey_sankey_5b.pdf` | Infrastructure Sankey PDF |
| `outputs/visualizations/harvey_sankey_5b.png` | Infrastructure Sankey PNG |
| `outputs/visualizations/harvey_sankey_57m.pdf` | Housing Sankey PDF |
| `outputs/visualizations/harvey_sankey_57m.png` | Housing Sankey PNG |
| `outputs/exports/harvey/harvey_sankey_recipients.json` | Recipient organization flow data |
| `scripts/generate_sankey_recipients.py` | Recipient PDF generator |
| `outputs/visualizations/harvey_sankey_recipients.pdf` | Recipient Sankey PDF |
| `outputs/visualizations/harvey_sankey_recipients.png` | Recipient Sankey PNG |

---

*Guide created January 2025 based on iterative refinement of Harvey CDBG-DR funding visualizations.*
