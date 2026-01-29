#!/usr/bin/env python3
"""
Generate a landscape PDF of the Harvey Sankey diagram.
Uses weasyprint to convert HTML to PDF with proper landscape orientation.
"""

import json
from pathlib import Path

# Paths
EXPORTS_DIR = Path(__file__).parent.parent / 'exports'
OUTPUT_DIR = Path(__file__).parent

def generate_sankey_html(sankey_data, title, width=1400, height=700):
    """Generate optimized HTML for PDF export."""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @page {{
            size: 17in 11in landscape;
            margin: 0.5in;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: white;
            padding: 20px;
        }}
        h1 {{
            font-size: 24px;
            color: #1a365d;
            margin-bottom: 10px;
            text-align: center;
        }}
        .subtitle {{
            font-size: 14px;
            color: #666;
            text-align: center;
            margin-bottom: 20px;
        }}
        #sankey-chart {{
            width: 100%;
            display: flex;
            justify-content: center;
        }}
        svg {{
            display: block;
        }}
        .link {{
            fill: none;
            stroke-opacity: 0.35;
        }}
        .node rect {{
            stroke: #fff;
            stroke-width: 2px;
        }}
        .node-label {{
            font-size: 12px;
            font-weight: 500;
        }}
        .value-label {{
            font-size: 10px;
            fill: #666;
        }}
        .legend {{
            margin-top: 20px;
            text-align: center;
            font-size: 11px;
            color: #666;
        }}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/d3-sankey@0.12.3/dist/d3-sankey.min.js"></script>
</head>
<body>
    <h1>{title}</h1>
    <div class="subtitle">Q4 2025 | Total: ${sankey_data['summary']['total_budget']/1e9:.2f}B</div>
    <div id="sankey-chart"></div>
    <div class="legend">Data Source: Texas GLO DRGR Reports | Generated from Harvey CDBG-DR Funding Analysis</div>

    <script>
        const data = {json.dumps(sankey_data)};

        const colors = {{
            hud: '#1a365d',
            glo: '#2c5282',
            category: '#c53030'
        }};

        function formatCurrency(value) {{
            if (value >= 1e9) return '$' + (value / 1e9).toFixed(2) + 'B';
            if (value >= 1e6) return '$' + (value / 1e6).toFixed(1) + 'M';
            if (value >= 1e3) return '$' + (value / 1e3).toFixed(0) + 'K';
            return '$' + value.toFixed(0);
        }}

        const container = d3.select('#sankey-chart');
        const width = {width};
        const height = {height};
        const margin = {{ top: 20, right: 300, bottom: 20, left: 20 }};

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${{width}} ${{height}}`);

        const sankey = d3.sankey()
            .nodeId(d => d.id)
            .nodeWidth(20)
            .nodePadding(35)
            .extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]]);

        const graph = sankey({{
            nodes: data.nodes.map(d => ({{...d}})),
            links: data.links.map(d => ({{
                source: d.source,
                target: d.target,
                value: d.value
            }}))
        }});

        function getColor(d) {{
            if (d.id === 'HUD') return colors.hud;
            if (d.id === 'Texas GLO') return colors.glo;
            return colors.category;
        }}

        // Draw links
        svg.append('g')
            .attr('fill', 'none')
            .selectAll('path')
            .data(graph.links)
            .join('path')
            .attr('class', 'link')
            .attr('d', d3.sankeyLinkHorizontal())
            .attr('stroke', d => getColor(d.source))
            .attr('stroke-width', d => Math.max(1, d.width));

        // Draw nodes
        const node = svg.append('g')
            .selectAll('g')
            .data(graph.nodes)
            .join('g')
            .attr('class', 'node');

        node.append('rect')
            .attr('x', d => d.x0)
            .attr('y', d => d.y0)
            .attr('height', d => Math.max(1, d.y1 - d.y0))
            .attr('width', d => d.x1 - d.x0)
            .attr('fill', d => getColor(d));

        // Get right-side nodes and sort by y position for label collision detection
        const rightNodes = graph.nodes.filter(d => d.x0 > width / 2);
        rightNodes.sort((a, b) => ((a.y0 + a.y1) / 2) - ((b.y0 + b.y1) / 2));

        // Calculate label positions with collision avoidance
        const labelPositions = new Map();
        const minLabelSpacing = 28; // minimum pixels between labels
        let lastLabelY = -100;

        rightNodes.forEach(d => {{
            let labelY = (d.y0 + d.y1) / 2;
            // If too close to previous label, push down
            if (labelY - lastLabelY < minLabelSpacing) {{
                labelY = lastLabelY + minLabelSpacing;
            }}
            labelPositions.set(d.id, labelY);
            lastLabelY = labelY;
        }});

        // Add labels with collision-avoided positions
        node.append('text')
            .attr('class', 'node-label')
            .attr('x', d => d.x0 < width / 2 ? d.x1 + 8 : d.x1 + 8)
            .attr('y', d => {{
                if (d.x0 < width / 2) return (d.y1 + d.y0) / 2;
                return labelPositions.get(d.id) || (d.y1 + d.y0) / 2;
            }})
            .attr('dy', '-0.2em')
            .attr('text-anchor', 'start')
            .text(d => d.name);

        // Add value labels
        node.append('text')
            .attr('class', 'value-label')
            .attr('x', d => d.x0 < width / 2 ? d.x1 + 8 : d.x1 + 8)
            .attr('y', d => {{
                if (d.x0 < width / 2) return (d.y1 + d.y0) / 2 + 14;
                return (labelPositions.get(d.id) || (d.y1 + d.y0) / 2) + 12;
            }})
            .attr('dy', '0.35em')
            .attr('text-anchor', 'start')
            .text(d => formatCurrency(d.value));

        // Draw connector lines from nodes to their labels (for right side nodes that were repositioned)
        svg.append('g')
            .attr('class', 'label-connectors')
            .selectAll('path')
            .data(rightNodes.filter(d => {{
                const origY = (d.y0 + d.y1) / 2;
                const newY = labelPositions.get(d.id);
                return Math.abs(newY - origY) > 5;
            }}))
            .join('path')
            .attr('d', d => {{
                const nodeY = (d.y0 + d.y1) / 2;
                const labelY = labelPositions.get(d.id);
                return `M${{d.x1}},${{nodeY}} L${{d.x1 + 5}},${{labelY}}`;
            }})
            .attr('stroke', '#999')
            .attr('stroke-width', 1)
            .attr('fill', 'none');
    </script>
</body>
</html>'''
    return html


def main():
    print("Generating Sankey PDF files...")

    # Load data
    with open(EXPORTS_DIR / 'harvey_sankey_infrastructure.json') as f:
        infra_data = json.load(f)

    with open(EXPORTS_DIR / 'harvey_sankey_housing.json') as f:
        housing_data = json.load(f)

    # Generate Infrastructure HTML
    infra_html = generate_sankey_html(
        infra_data,
        "Harvey CDBG-DR 5B Grant Funding Flow ($4.42B)",
        width=1500, height=750
    )
    infra_html_path = OUTPUT_DIR / 'harvey_sankey_5b_landscape.html'
    with open(infra_html_path, 'w') as f:
        f.write(infra_html)
    print(f"  Generated: {infra_html_path}")

    # Generate Housing HTML
    housing_html = generate_sankey_html(
        housing_data,
        "Harvey CDBG-DR 57M Grant Funding Flow ($57.8M)",
        width=1200, height=500
    )
    housing_html_path = OUTPUT_DIR / 'harvey_sankey_57m_landscape.html'
    with open(housing_html_path, 'w') as f:
        f.write(housing_html)
    print(f"  Generated: {housing_html_path}")

    print("\nTo create PDFs:")
    print("  1. Open the HTML files in a browser")
    print("  2. Print (Cmd+P / Ctrl+P)")
    print("  3. Select 'Save as PDF'")
    print("  4. Choose Landscape orientation")
    print("  5. Set paper size to Tabloid (11x17) for best results")

    print("\n  Open the HTML files in a browser and print to PDF for best results.")


if __name__ == '__main__':
    main()
