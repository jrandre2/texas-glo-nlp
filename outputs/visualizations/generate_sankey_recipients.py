#!/usr/bin/env python3
"""
Generate Sankey diagram PDF showing funding flow to recipient entities.
Shows Texas GLO → Recipients with geographic breakdown.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

# Paths
EXPORTS_DIR = Path(__file__).parent.parent / 'exports'
OUTPUT_DIR = Path(__file__).parent

# Color scheme
COLORS = {
    'Texas GLO': '#2c5282',
    'Houston Metro': '#c53030',      # Red for Houston area
    'City of Houston': '#c53030',
    'Harris County': '#d53f8c',      # Pink
    'GLO Direct': '#38a169',         # Green
    'Aransas County': '#48bb78',
    'Refugio County': '#68d391',
    'Liberty': '#9ae6b4',
    'Other Counties': '#c6f6d5',
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


def create_recipient_sankey_pdf(output_path, figsize=(18, 11)):
    """Create a Sankey diagram showing funding flow to recipients with county breakdown."""

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 98, "Harvey CDBG-DR Funding by Recipient",
            ha='center', va='top', fontsize=20, fontweight='bold', color='#1a365d')
    ax.text(50, 94, "Q4 2025 | Total: $4.47B | 616 Activities | 62 Counties",
            ha='center', va='top', fontsize=13, color='#666')

    total_budget = 4474546991.02

    # Define the data
    # Level 1: Texas GLO → Main Recipients
    level1_recipients = [
        {'name': 'GLO Direct\n(62 Counties)', 'value': 2727092535.89, 'activities': 538, 'color': '#38a169'},
        {'name': 'City of Houston', 'value': 1035382736.84, 'activities': 17, 'color': '#c53030'},
        {'name': 'Harris County', 'value': 709103718.29, 'activities': 51, 'color': '#d53f8c'},
    ]

    # Level 2: GLO Direct → Top Counties
    level2_counties = [
        {'name': 'Aransas', 'value': 28797241.85, 'activities': 3},
        {'name': 'Refugio', 'value': 11727057.24, 'activities': 6},
        {'name': 'Liberty', 'value': 11155242.11, 'activities': 5},
        {'name': 'Brazoria', 'value': 10917829.0, 'activities': 1},
        {'name': 'San Jacinto', 'value': 9195613.82, 'activities': 4},
        {'name': 'Hardin', 'value': 7826814.66, 'activities': 4},
        {'name': 'Montgomery', 'value': 7175792.78, 'activities': 3},
        {'name': 'Newton', 'value': 6171945.58, 'activities': 3},
    ]

    # Node dimensions
    node_width = 3.5
    glo_x = 6
    recipient_x = 35
    county_x = 65

    # Vertical spacing
    top_margin = 88
    bottom_margin = 10
    available_height = top_margin - bottom_margin
    gap = 3

    # Calculate Level 1 positions
    level1_total = sum(r['value'] for r in level1_recipients)
    space_for_l1 = available_height - (gap * (len(level1_recipients) - 1))

    l1_positions = []
    current_y = top_margin
    for r in level1_recipients:
        proportion = r['value'] / level1_total
        height = max(12, proportion * space_for_l1)
        l1_positions.append({
            **r,
            'y_top': current_y,
            'y_center': current_y - height / 2,
            'y_bottom': current_y - height,
            'height': height,
        })
        current_y = current_y - height - gap

    # Draw Texas GLO node
    glo_height = available_height * 0.95
    glo_y_center = (top_margin + bottom_margin) / 2
    glo_rect = mpatches.FancyBboxPatch(
        (glo_x, glo_y_center - glo_height/2), node_width, glo_height,
        boxstyle="round,pad=0.02", facecolor=COLORS['Texas GLO'], edgecolor='white', linewidth=2
    )
    ax.add_patch(glo_rect)
    ax.text(glo_x - 0.5, glo_y_center + 3, 'Texas', ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(glo_x - 0.5, glo_y_center, 'GLO', ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(glo_x - 0.5, glo_y_center - 4, '$4.47B', ha='right', va='center', fontsize=10, color='#666')

    glo_right = glo_x + node_width
    glo_current_y = glo_y_center + glo_height / 2

    # Draw Level 1 recipients and flows
    for pos in l1_positions:
        color = pos['color']

        # Draw recipient node
        rect = mpatches.FancyBboxPatch(
            (recipient_x, pos['y_bottom']), node_width, pos['height'],
            boxstyle="round,pad=0.02", facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(rect)

        # Label
        pct = (pos['value'] / total_budget) * 100
        ax.text(recipient_x + node_width + 1, pos['y_center'] + 2,
                pos['name'], ha='left', va='center', fontsize=11, fontweight='bold')
        ax.text(recipient_x + node_width + 1, pos['y_center'] - 1.5,
                f"{format_currency(pos['value'])} ({pct:.0f}%)",
                ha='left', va='center', fontsize=10, color='#444')
        ax.text(recipient_x + node_width + 1, pos['y_center'] - 4.5,
                f"{pos['activities']} activities",
                ha='left', va='center', fontsize=9, color='#666')

        # Flow from GLO
        flow_thickness = (pos['value'] / total_budget) * glo_height
        glo_source_top = glo_current_y
        glo_source_bottom = glo_current_y - flow_thickness
        glo_current_y = glo_source_bottom

        vertices = [
            (glo_right, glo_source_top),
            (glo_right + 10, glo_source_top),
            (recipient_x - 10, pos['y_top']),
            (recipient_x, pos['y_top']),
            (recipient_x, pos['y_bottom']),
            (recipient_x - 10, pos['y_bottom']),
            (glo_right + 10, glo_source_bottom),
            (glo_right, glo_source_bottom),
            (glo_right, glo_source_top),
        ]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        path = MPath(vertices, codes)
        patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=0.35)
        ax.add_patch(patch)

    # Calculate Level 2 (county) positions within GLO Direct space
    glo_direct_pos = l1_positions[0]  # First one is GLO Direct
    county_top = glo_direct_pos['y_top']
    county_bottom = glo_direct_pos['y_bottom']
    county_height = county_top - county_bottom

    # Add "Other Counties" to sum up remaining
    top_county_total = sum(c['value'] for c in level2_counties)
    other_value = glo_direct_pos['value'] - top_county_total
    all_counties = level2_counties + [{'name': 'Other\n(54 counties)', 'value': other_value, 'activities': 509}]

    county_gap = 1.5
    space_for_counties = county_height - (county_gap * (len(all_counties) - 1))
    county_total = sum(c['value'] for c in all_counties)

    l2_positions = []
    current_y = county_top
    for c in all_counties:
        proportion = c['value'] / county_total
        height = max(3, proportion * space_for_counties)
        l2_positions.append({
            **c,
            'y_top': current_y,
            'y_center': current_y - height / 2,
            'y_bottom': current_y - height,
            'height': height,
        })
        current_y = current_y - height - county_gap

    # Draw Level 2 counties and flows
    glo_direct_right = recipient_x + node_width
    glo_direct_current_y = glo_direct_pos['y_top']

    # Generate gradient green colors for counties
    greens = ['#276749', '#2f855a', '#38a169', '#48bb78', '#68d391', '#9ae6b4', '#c6f6d5', '#f0fff4', '#718096']

    for i, pos in enumerate(l2_positions):
        color = greens[min(i, len(greens)-1)]

        # Draw county node
        rect = mpatches.FancyBboxPatch(
            (county_x, pos['y_bottom']), node_width * 0.8, pos['height'],
            boxstyle="round,pad=0.01", facecolor=color, edgecolor='white', linewidth=1
        )
        ax.add_patch(rect)

        # Label
        ax.text(county_x + node_width * 0.8 + 0.8, pos['y_center'],
                f"{pos['name']} - {format_currency(pos['value'])}",
                ha='left', va='center', fontsize=9)

        # Flow from GLO Direct
        flow_thickness = (pos['value'] / glo_direct_pos['value']) * glo_direct_pos['height']
        src_top = glo_direct_current_y
        src_bottom = glo_direct_current_y - flow_thickness
        glo_direct_current_y = src_bottom

        vertices = [
            (glo_direct_right, src_top),
            (glo_direct_right + 10, src_top),
            (county_x - 10, pos['y_top']),
            (county_x, pos['y_top']),
            (county_x, pos['y_bottom']),
            (county_x - 10, pos['y_bottom']),
            (glo_direct_right + 10, src_bottom),
            (glo_direct_right, src_bottom),
            (glo_direct_right, src_top),
        ]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        path = MPath(vertices, codes)
        patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=0.3)
        ax.add_patch(patch)

    # Footer
    ax.text(50, 4, "City of Houston and Harris County manage their own projects; GLO Direct administers projects in 62 other counties",
            ha='center', va='bottom', fontsize=9, style='italic', color='#666')
    ax.text(50, 1, "Data Source: Texas GLO DRGR Reports | Harvey CDBG-DR Funding Analysis",
            ha='center', va='bottom', fontsize=9, color='#888')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Generated: {output_path}")


def main():
    print("Generating Recipient Sankey PDF...")
    create_recipient_sankey_pdf(OUTPUT_DIR / 'harvey_sankey_recipients.pdf')
    print("\nDone!")


if __name__ == '__main__':
    main()
