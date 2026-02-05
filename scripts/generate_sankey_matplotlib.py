#!/usr/bin/env python3
"""
Generate Sankey diagram PDFs using matplotlib.
Provides precise control over node positioning and label placement.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = ROOT / 'outputs' / 'exports' / 'harvey'
OUTPUT_DIR = ROOT / 'outputs' / 'visualizations'

# Color scheme
COLORS = {
    'HUD': '#1a365d',
    'Texas GLO': '#2c5282',
    'Homeowner Assistance Program': '#c53030',
    'Affordable Rental': '#dd6b20',
    'Local Buyout/Acquisition': '#d69e2e',
    'Infrastructure Projects': '#38a169',
    'Administration': '#805ad5',
    'Homeowner Reimbursement': '#d53f8c',
    'Economic Revitalization': '#00b5d8',
    'Planning': '#718096',
    'Other': '#a0aec0',
    'Single Family Housing': '#ed8936',
    'Homebuyer Assistance': '#48bb78',
    'PREPS Program': '#9f7aea',
    'Public Services': '#667eea',
}

def format_currency(value):
    """Format value as currency."""
    if value >= 1e9:
        return f'${value/1e9:.2f}B'
    if value >= 1e6:
        return f'${value/1e6:.1f}M'
    if value >= 1e3:
        return f'${value/1e3:.0f}K'
    return f'${value:.0f}'


def create_sankey_pdf(sankey_data, title, output_path, figsize=(14, 8)):
    """Create a Sankey diagram using custom drawing for better control."""

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 97, title, ha='center', va='top', fontsize=16, fontweight='bold', color='#1a365d')
    ax.text(50, 93, f"Q4 2025 | Total: {format_currency(sankey_data['summary']['total_budget'])}",
            ha='center', va='top', fontsize=11, color='#666')

    total_budget = sankey_data['summary']['total_budget']

    # Get links from Texas GLO to categories
    links = sankey_data['links']
    category_links = [l for l in links if l['source'] == 'Texas GLO']
    category_links.sort(key=lambda x: -x['value'])

    # Node dimensions - expanded to use more space
    node_width = 3
    glo_x = 8  # Texas GLO closer to left edge
    cat_x = 45  # Categories moved left to give more label room

    # Available vertical space - use more of the page
    top_margin = 90
    bottom_margin = 6
    available_height = top_margin - bottom_margin

    # Uniform color for all category nodes
    category_color = '#718096'  # Gray

    # Calculate heights proportional to values (with minimum height for visibility)
    min_height = 4
    total_value = sum(l['value'] for l in category_links)
    num_categories = len(category_links)
    gap = 1.5  # Gap between nodes

    # Calculate total space needed for gaps
    total_gap_space = gap * (num_categories - 1)
    space_for_nodes = available_height - total_gap_space

    category_positions = []
    current_y = top_margin

    for link in category_links:
        # Height proportional to value
        proportion = link['value'] / total_value
        height = max(min_height, proportion * space_for_nodes)

        y_center = current_y - height / 2
        category_positions.append({
            'name': link['target'],
            'value': link['value'],
            'y_top': current_y,
            'y_center': y_center,
            'y_bottom': current_y - height,
            'height': height,
        })
        current_y = current_y - height - gap

    # Scale if needed to fit
    actual_bottom = category_positions[-1]['y_bottom']
    if actual_bottom < bottom_margin:
        # Need to scale down
        scale_factor = (available_height - total_gap_space) / sum(p['height'] for p in category_positions)
        current_y = top_margin
        for pos in category_positions:
            pos['height'] = max(min_height * 0.7, pos['height'] * scale_factor)
            pos['y_top'] = current_y
            pos['y_center'] = current_y - pos['height'] / 2
            pos['y_bottom'] = current_y - pos['height']
            current_y = pos['y_bottom'] - gap

    # Draw Texas GLO node (source - no HUD since it's 100% pass-through)
    glo_height = available_height * 0.95  # Use nearly full height
    glo_y_center = (top_margin + bottom_margin) / 2
    glo_rect = mpatches.FancyBboxPatch(
        (glo_x, glo_y_center - glo_height/2), node_width, glo_height,
        boxstyle="round,pad=0.02", facecolor=COLORS['Texas GLO'], edgecolor='white', linewidth=2
    )
    ax.add_patch(glo_rect)
    ax.text(glo_x - 1, glo_y_center, 'Texas GLO', ha='right', va='center', fontsize=12, fontweight='bold')
    ax.text(glo_x - 1, glo_y_center - 4, format_currency(total_budget), ha='right', va='center', fontsize=10, color='#666')

    # Import for drawing paths
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    glo_right = glo_x + node_width

    # Track cumulative position on GLO node for stacking flows
    glo_current_y = glo_y_center + glo_height / 2

    for pos in category_positions:
        name = pos['name']

        # Draw category node (uniform gray color)
        rect = mpatches.FancyBboxPatch(
            (cat_x, pos['y_bottom']), node_width, pos['height'],
            boxstyle="round,pad=0.02", facecolor=category_color, edgecolor='white', linewidth=1.5
        )
        ax.add_patch(rect)

        # Draw label to the right
        ax.text(cat_x + node_width + 1.5, pos['y_center'], name,
                ha='left', va='center', fontsize=10, fontweight='500')
        ax.text(cat_x + node_width + 1.5, pos['y_center'] - 2.2, format_currency(pos['value']),
                ha='left', va='center', fontsize=9, color='#666')

        # Flow thickness proportional to value
        flow_thickness = (pos['value'] / total_budget) * glo_height

        # Source position on GLO (stacked from top)
        glo_source_top = glo_current_y
        glo_source_bottom = glo_current_y - flow_thickness
        glo_current_y = glo_source_bottom

        # Target position on category
        cat_target_top = pos['y_top']
        cat_target_bottom = pos['y_bottom']

        # Draw curved flow
        vertices = [
            (glo_right, glo_source_top),
            (glo_right + 12, glo_source_top),
            (cat_x - 12, cat_target_top),
            (cat_x, cat_target_top),
            (cat_x, cat_target_bottom),
            (cat_x - 12, cat_target_bottom),
            (glo_right + 12, glo_source_bottom),
            (glo_right, glo_source_bottom),
            (glo_right, glo_source_top),
        ]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        path = MPath(vertices, codes)
        patch = PathPatch(path, facecolor=COLORS['Texas GLO'], edgecolor='none', alpha=0.4)
        ax.add_patch(patch)

    # Footer
    ax.text(50, 2, "Data Source: Texas GLO DRGR Reports | Generated from Harvey CDBG-DR Funding Analysis",
            ha='center', va='bottom', fontsize=9, color='#666')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Generated: {output_path}")


def main():
    print("Generating Sankey PDF files with matplotlib...")

    # Load data
    with open(EXPORTS_DIR / 'harvey_sankey_infrastructure.json') as f:
        infra_data = json.load(f)

    with open(EXPORTS_DIR / 'harvey_sankey_housing.json') as f:
        housing_data = json.load(f)

    # Create Infrastructure (5B) Sankey - landscape tabloid size
    print("\nCreating Infrastructure (5B Grant) Sankey...")
    create_sankey_pdf(
        infra_data,
        "Harvey CDBG-DR 5B Grant Funding Flow",
        OUTPUT_DIR / 'harvey_sankey_5b.pdf',
        figsize=(16, 10)  # Wider landscape format
    )

    # Create Housing (57M) Sankey
    print("\nCreating Housing (57M Grant) Sankey...")
    create_sankey_pdf(
        housing_data,
        "Harvey CDBG-DR 57M Grant Funding Flow",
        OUTPUT_DIR / 'harvey_sankey_57m.pdf',
        figsize=(14, 8)
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
