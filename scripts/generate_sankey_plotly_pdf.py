#!/usr/bin/env python3
"""
Generate Sankey diagram PDFs using Plotly.
Creates proper landscape PDF with controlled spacing.
"""

import json
from pathlib import Path
import plotly.graph_objects as go

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


def create_sankey_figure(sankey_data, title):
    """Create a Plotly Sankey figure from the data."""

    # Build node index
    node_ids = [n['id'] for n in sankey_data['nodes']]

    # Calculate node values for labels
    node_values = {}
    for link in sankey_data['links']:
        target = link['target']
        if target not in node_values:
            node_values[target] = 0
        node_values[target] += link['value']

    # For HUD and GLO, use the total budget
    node_values['HUD'] = sankey_data['summary']['total_budget']
    node_values['Texas GLO'] = sankey_data['summary']['total_budget']

    # Create labels with values
    node_labels_with_values = []
    for n in sankey_data['nodes']:
        val = node_values.get(n['id'], 0)
        node_labels_with_values.append(f"{n['name']}<br>{format_currency(val)}")

    # Get node colors
    node_colors = [COLORS.get(n['id'], '#c53030') for n in sankey_data['nodes']]

    # Build link indices
    source_indices = [node_ids.index(link['source']) for link in sankey_data['links']]
    target_indices = [node_ids.index(link['target']) for link in sankey_data['links']]
    values = [link['value'] for link in sankey_data['links']]

    # Link colors (based on source, semi-transparent)
    link_colors = []
    for link in sankey_data['links']:
        base_color = COLORS.get(link['source'], '#1a365d')
        # Convert hex to rgba
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        link_colors.append(f'rgba({r},{g},{b},0.4)')

    # Calculate evenly spaced Y positions for target nodes (level 2)
    # First two nodes are HUD and Texas GLO
    num_targets = len(node_ids) - 2
    # Spread targets evenly from 0.02 to 0.98 with extra padding
    target_y_positions = []
    for i in range(num_targets):
        # Even distribution with margins
        y_pos = 0.03 + (i * 0.94 / max(1, num_targets - 1))
        target_y_positions.append(y_pos)

    # Create figure with manual node positioning
    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',  # Use fixed positioning
        node=dict(
            pad=60,  # Increased vertical padding between nodes
            thickness=20,  # Node width
            line=dict(color='white', width=2),
            label=node_labels_with_values,
            color=node_colors,
            # Manual x positions: HUD at left, GLO at 1/4, targets at right
            x=[0.001, 0.20] + [0.85] * num_targets,
            # Manual y positions: HUD and GLO centered, targets spread evenly
            y=[0.5, 0.5] + target_y_positions,
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors,
        ),
        textfont=dict(size=11, family='Arial'),
    )])

    # Update layout for landscape PDF
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:14px'>Q4 2025 | Total: {format_currency(sankey_data['summary']['total_budget'])}</span>",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#1a365d'),
        ),
        font=dict(size=11, family='Arial'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=10, r=180, t=80, b=40),  # Extra right margin for labels
        annotations=[
            dict(
                text="Data Source: Texas GLO DRGR Reports | Generated from Harvey CDBG-DR Funding Analysis",
                xref="paper", yref="paper",
                x=0.5, y=-0.02,
                showarrow=False,
                font=dict(size=10, color='#666'),
            )
        ],
    )

    return fig


def main():
    print("Generating Sankey PDF files with Plotly...")

    # Load data
    with open(EXPORTS_DIR / 'harvey_sankey_infrastructure.json') as f:
        infra_data = json.load(f)

    with open(EXPORTS_DIR / 'harvey_sankey_housing.json') as f:
        housing_data = json.load(f)

    # Create Infrastructure (5B) Sankey
    print("\nCreating Infrastructure (5B Grant) Sankey...")
    fig_infra = create_sankey_figure(
        infra_data,
        "Harvey CDBG-DR 5B Grant Funding Flow"
    )

    # Export to PDF - extra tall landscape for label spacing
    pdf_path_infra = OUTPUT_DIR / 'harvey_sankey_5b.pdf'
    fig_infra.write_image(
        str(pdf_path_infra),
        format='pdf',
        width=1600,
        height=1400,  # Taller to spread out labels
        scale=2,
    )
    print(f"  Generated: {pdf_path_infra}")

    # Create Housing (57M) Sankey
    print("\nCreating Housing (57M Grant) Sankey...")
    fig_housing = create_sankey_figure(
        housing_data,
        "Harvey CDBG-DR 57M Grant Funding Flow"
    )

    pdf_path_housing = OUTPUT_DIR / 'harvey_sankey_57m.pdf'
    fig_housing.write_image(
        str(pdf_path_housing),
        format='pdf',
        width=1400,
        height=900,
        scale=2,
    )
    print(f"  Generated: {pdf_path_housing}")

    print("\nDone! PDF files created:")
    print(f"  - {pdf_path_infra}")
    print(f"  - {pdf_path_housing}")


if __name__ == '__main__':
    main()
