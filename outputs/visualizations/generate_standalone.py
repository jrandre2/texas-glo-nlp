#!/usr/bin/env python3
"""
Generate standalone Harvey Dashboard HTML with all data embedded.
"""

import json
import csv
from pathlib import Path

# Paths
EXPORTS_DIR = Path(__file__).parent.parent / 'exports'
OUTPUT_FILE = Path(__file__).parent / 'harvey_dashboard_standalone.html'

def load_data():
    """Load all data files."""
    # Load separate Sankey data files
    with open(EXPORTS_DIR / 'harvey_sankey_infrastructure.json') as f:
        sankey_infrastructure = json.load(f)

    with open(EXPORTS_DIR / 'harvey_sankey_housing.json') as f:
        sankey_housing = json.load(f)

    # Load quarterly trends
    with open(EXPORTS_DIR / 'harvey_quarterly_trends.json') as f:
        trends_data = json.load(f)

    # Load county allocations
    county_data = []
    with open(EXPORTS_DIR / 'harvey_county_allocations.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            county_data.append(row)

    # Load org allocations
    org_data = []
    with open(EXPORTS_DIR / 'harvey_org_allocations.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            org_data.append(row)

    return sankey_infrastructure, sankey_housing, trends_data, county_data, org_data

def generate_html(sankey_infrastructure, sankey_housing, trends_data, county_data, org_data):
    """Generate the complete HTML file."""

    # Get latest quarter
    latest_quarter = sankey_infrastructure['quarter']
    infra_budget = sankey_infrastructure['summary']['total_budget']
    housing_budget = sankey_housing['summary']['total_budget']
    total_budget = infra_budget + housing_budget

    # Format budgets for display
    total_budget_display = f"${total_budget / 1e9:.2f}B"
    infra_budget_display = f"${infra_budget / 1e9:.2f}B"
    housing_budget_display = f"${housing_budget / 1e6:.1f}M"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Harvey CDBG-DR Funding Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%); color: white; padding: 30px 20px; text-align: center; }}
        .header h1 {{ font-size: 2rem; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; font-size: 1.1rem; }}
        .stats-row {{ display: flex; justify-content: center; gap: 40px; margin-top: 20px; flex-wrap: wrap; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 1.8rem; font-weight: bold; }}
        .stat-label {{ font-size: 0.9rem; opacity: 0.8; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .section {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; overflow: hidden; }}
        .section-header {{ background: #f8f9fa; padding: 15px 20px; border-bottom: 1px solid #e9ecef; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
        .section-header h2 {{ font-size: 1.2rem; color: #2d3748; }}
        .section-content {{ padding: 20px; min-height: 400px; }}
        #sankey-chart {{ min-height: 820px; overflow-x: auto; }}
        .tabs {{ display: flex; gap: 10px; }}
        .tab {{ padding: 6px 12px; border: 1px solid #e9ecef; border-radius: 4px; cursor: pointer; background: white; font-size: 0.9rem; }}
        .tab:hover {{ background: #f8f9fa; }}
        .tab.active {{ background: #2c5282; color: white; border-color: #2c5282; }}
        .data-table {{ width: 100%; border-collapse: collapse; }}
        .data-table th, .data-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e9ecef; }}
        .data-table th {{ background: #f8f9fa; font-weight: 600; }}
        .data-table .amount {{ text-align: right; font-family: monospace; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9rem; }}
        .footer a {{ color: #2c5282; }}
        .error {{ color: #c53030; padding: 20px; text-align: center; }}
        .loading {{ color: #666; padding: 20px; text-align: center; }}
        svg {{ display: block; }}
        .grid line {{ stroke: #e9ecef; }}
        .grid path {{ stroke: none; }}
        .bar {{ fill: #dd6b20; cursor: pointer; }}
        .bar:hover {{ fill: #c05621; }}
        .link {{ stroke-opacity: 0.5; }}
        .link:hover {{ stroke-opacity: 0.8; }}
        .node rect {{ stroke: #fff; stroke-width: 2px; }}
        .node text {{ font-size: 12px; }}
        .axis text {{ font-size: 11px; }}
        .line {{ fill: none; stroke-width: 2; }}
        .dot {{ stroke: white; stroke-width: 2; }}
        .tooltip {{ position: fixed; padding: 10px; background: rgba(0,0,0,0.85); color: white; border-radius: 4px; font-size: 13px; pointer-events: none; z-index: 1000; max-width: 250px; }}
        .tooltip .title {{ font-weight: bold; margin-bottom: 5px; }}
        .tooltip .value {{ font-size: 1.1em; color: #68d391; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Hurricane Harvey CDBG-DR Funding Flow</h1>
        <p>Tracking $4.47B in disaster recovery funds across 616 activities in 62 Texas counties</p>
        <div class="stats-row">
            <div class="stat"><div class="stat-value">{total_budget_display}</div><div class="stat-label">Total Budget</div></div>
            <div class="stat"><div class="stat-value">616</div><div class="stat-label">Activities</div></div>
            <div class="stat"><div class="stat-value">62</div><div class="stat-label">Counties</div></div>
            <div class="stat"><div class="stat-value">12.5%</div><div class="stat-label">Completion Rate</div></div>
            <div class="stat"><div class="stat-value">{latest_quarter}</div><div class="stat-label">Latest Quarter</div></div>
        </div>
    </div>

    <div class="container">
        <div class="section">
            <div class="section-header">
                <h2>Funding Flow: HUD &rarr; Texas GLO &rarr; Programs</h2>
                <div class="tabs" id="sankey-tabs">
                    <div class="tab active" data-sankey="infrastructure">5B Grant ({infra_budget_display})</div>
                    <div class="tab" data-sankey="housing">57M Grant ({housing_budget_display})</div>
                </div>
            </div>
            <div class="section-content" id="sankey-chart">
                <p class="loading">Loading Sankey diagram...</p>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Quarterly Budget Trends (2019-2025)</h2>
                <div class="tabs">
                    <div class="tab active" data-view="budget">Budget</div>
                    <div class="tab" data-view="activities">Activities</div>
                </div>
            </div>
            <div class="section-content" id="timeline-chart">
                <p class="loading">Loading timeline...</p>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Top 15 Counties by Funding ({latest_quarter})</h2>
            </div>
            <div class="section-content" id="county-chart">
                <p class="loading">Loading county chart...</p>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Organization Allocations ({latest_quarter})</h2>
            </div>
            <div class="section-content">
                <table class="data-table" id="org-table">
                    <thead><tr><th>Organization</th><th>Program</th><th class="amount">Allocated</th><th class="amount">Activities</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Data Source: Texas GLO DRGR Reports (2018-2025) | Generated by <a href="https://github.com/jrandre2/texas-glo-nlp">Texas GLO NLP Project</a></p>
    </div>

    <div class="tooltip" id="tooltip" style="display: none;"></div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/d3-sankey@0.12.3/dist/d3-sankey.min.js"></script>

    <script>
        // ============ EMBEDDED DATA ============
        const sankeyInfrastructure = {json.dumps(sankey_infrastructure)};
        const sankeyHousing = {json.dumps(sankey_housing)};
        const trendsData = {json.dumps(trends_data)};
        const countyData = {json.dumps(county_data)};
        const orgData = {json.dumps(org_data)};
        const latestQuarter = "{latest_quarter}";
        let currentSankeyView = 'infrastructure';

        // ============ COLOR SCHEME ============
        const colors = {{
            hud: '#1a365d',
            glo: '#2c5282',
            infrastructure: '#38a169',
            housing: '#d69e2e',
            category: '#e53e3e',      // Red for spending categories
            organization: '#805ad5',
            county: '#dd6b20'
        }};

        // ============ UTILITY FUNCTIONS ============
        function formatCurrency(value, precise = false) {{
            if (precise) {{
                // Full precision with commas
                return '$' + value.toLocaleString('en-US', {{minimumFractionDigits: 0, maximumFractionDigits: 0}});
            }}
            // Abbreviated format
            if (value >= 1e9) return '$' + (value / 1e9).toFixed(3) + 'B';
            if (value >= 1e6) return '$' + (value / 1e6).toFixed(2) + 'M';
            if (value >= 1e3) return '$' + (value / 1e3).toFixed(1) + 'K';
            return '$' + value.toFixed(0);
        }}

        function formatNumber(value) {{
            return value.toLocaleString();
        }}

        function showTooltip(event, html) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = html;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.clientX + 10) + 'px';
            tooltip.style.top = (event.clientY + 10) + 'px';
        }}

        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        // ============ SANKEY DIAGRAM ============
        function drawSankey(data) {{
            console.log('Drawing Sankey diagram...');
            const container = d3.select('#sankey-chart');
            container.selectAll('*').remove();

            const containerWidth = container.node().getBoundingClientRect().width;
            const width = Math.max(1200, containerWidth);
            const height = 800;
            const margin = {{ top: 20, right: 250, bottom: 20, left: 20 }};

            const svg = container.append('svg')
                .attr('width', width)
                .attr('height', height);

            try {{
                // Create sankey generator
                const sankey = d3.sankey()
                    .nodeId(d => d.id)
                    .nodeWidth(18)
                    .nodePadding(20)
                    .extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]]);

                // Deep copy data
                const graph = sankey({{
                    nodes: data.nodes.map(d => ({{...d}})),
                    links: data.links.map(d => ({{
                        source: d.source,
                        target: d.target,
                        value: d.value
                    }}))
                }});

                // Color function (adapts to single-program view where categories are level 2)
                function getColor(d) {{
                    if (d.id === 'HUD') return colors.hud;
                    if (d.id === 'Texas GLO') return colors.glo;
                    if (d.id === 'Infrastructure') return colors.infrastructure;
                    if (d.id === 'Housing') return colors.housing;
                    // Organizations (Harris County, City of Houston)
                    if (d.id === 'Harris County' || d.id === 'City of Houston') return colors.organization;
                    // Spending categories (level 2 in single-program view, level 3 in combined)
                    if (d.level === 2 || d.level === 3) return colors.category;
                    return colors.county;
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
                    .attr('stroke-width', d => Math.max(1, d.width))
                    .on('mouseover', function(event, d) {{
                        d3.select(this).attr('stroke-opacity', 0.8);
                        showTooltip(event, `
                            <div class="title">${{d.source.name}} &rarr; ${{d.target.name}}</div>
                            <div class="value">${{formatCurrency(d.value, true)}}</div>
                        `);
                    }})
                    .on('mouseout', function() {{
                        d3.select(this).attr('stroke-opacity', 0.5);
                        hideTooltip();
                    }});

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
                    .attr('fill', d => getColor(d))
                    .on('mouseover', function(event, d) {{
                        showTooltip(event, `
                            <div class="title">${{d.name}}</div>
                            <div class="value">${{formatCurrency(d.value, true)}}</div>
                        `);
                    }})
                    .on('mouseout', hideTooltip);

                // Add labels - smaller font for categories (level 3) and orgs (level 4)
                node.append('text')
                    .attr('x', d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
                    .attr('y', d => (d.y1 + d.y0) / 2)
                    .attr('dy', '0.35em')
                    .attr('text-anchor', d => d.x0 < width / 2 ? 'start' : 'end')
                    .text(d => d.level >= 3 ? d.name + ' ' + formatCurrency(d.value) : d.name)
                    .style('font-size', d => d.level >= 3 ? '10px' : '12px')
                    .style('font-weight', d => d.level >= 3 ? '400' : '500');

                // Add value labels (only for top-level nodes: HUD, GLO, Programs)
                node.filter(d => d.level < 3)
                    .append('text')
                    .attr('x', d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
                    .attr('y', d => (d.y1 + d.y0) / 2 + 14)
                    .attr('dy', '0.35em')
                    .attr('text-anchor', d => d.x0 < width / 2 ? 'start' : 'end')
                    .text(d => formatCurrency(d.value))
                    .style('font-size', '10px')
                    .style('fill', '#666');

                console.log('Sankey diagram complete');
            }} catch (error) {{
                console.error('Sankey error:', error);
                container.html('<p class="error">Error drawing Sankey: ' + error.message + '</p>');
            }}
        }}

        // ============ TIMELINE CHART ============
        function drawTimeline(data, viewType) {{
            console.log('Drawing timeline chart:', viewType);
            const container = d3.select('#timeline-chart');
            container.selectAll('*').remove();

            const containerWidth = container.node().getBoundingClientRect().width;
            const width = Math.max(800, containerWidth);
            const height = 350;
            const margin = {{ top: 30, right: 120, bottom: 70, left: 80 }};

            const svg = container.append('svg')
                .attr('width', width)
                .attr('height', height);

            // Sort quarters chronologically
            const parseQuarter = q => {{
                const match = q.match(/Q(\\d) (\\d{{4}})/);
                return match ? parseInt(match[2]) * 10 + parseInt(match[1]) : 0;
            }};

            const quarters = [...new Set(data.map(d => d.quarter))].sort((a, b) => parseQuarter(a) - parseQuarter(b));

            // Create scales
            const x = d3.scalePoint()
                .domain(quarters)
                .range([margin.left, width - margin.right])
                .padding(0.5);

            const yField = viewType === 'budget' ? 'total_budget' : 'activity_count';
            const yMax = d3.max(data, d => d[yField]);
            const y = d3.scaleLinear()
                .domain([0, yMax * 1.1])
                .range([height - margin.bottom, margin.top]);

            // Draw grid
            svg.append('g')
                .attr('class', 'grid')
                .attr('transform', `translate(0,${{height - margin.bottom}})`)
                .call(d3.axisBottom(x).tickSize(-(height - margin.top - margin.bottom)).tickFormat(''));

            // Draw axes
            svg.append('g')
                .attr('transform', `translate(0,${{height - margin.bottom}})`)
                .call(d3.axisBottom(x))
                .selectAll('text')
                .attr('transform', 'rotate(-45)')
                .style('text-anchor', 'end');

            svg.append('g')
                .attr('transform', `translate(${{margin.left}},0)`)
                .call(d3.axisLeft(y).tickFormat(viewType === 'budget' ? d => formatCurrency(d) : d => formatNumber(d)));

            // Draw lines for each program
            const programs = ['Infrastructure', 'Housing'];
            const programColors = {{ Infrastructure: colors.infrastructure, Housing: colors.housing }};

            programs.forEach(program => {{
                const programData = data
                    .filter(d => d.program_type === program)
                    .sort((a, b) => parseQuarter(a.quarter) - parseQuarter(b.quarter));

                if (programData.length === 0) return;

                const line = d3.line()
                    .x(d => x(d.quarter))
                    .y(d => y(d[yField]))
                    .defined(d => d[yField] !== undefined && d[yField] !== null);

                svg.append('path')
                    .datum(programData)
                    .attr('class', 'line')
                    .attr('d', line)
                    .attr('stroke', programColors[program]);

                svg.selectAll(`.dot-${{program}}`)
                    .data(programData)
                    .join('circle')
                    .attr('class', `dot dot-${{program}}`)
                    .attr('cx', d => x(d.quarter))
                    .attr('cy', d => y(d[yField]))
                    .attr('r', 4)
                    .attr('fill', programColors[program])
                    .on('mouseover', function(event, d) {{
                        showTooltip(event, `
                            <div class="title">${{d.quarter}} - ${{program}}</div>
                            <div class="value">${{viewType === 'budget' ? formatCurrency(d[yField]) : formatNumber(d[yField]) + ' activities'}}</div>
                        `);
                    }})
                    .on('mouseout', hideTooltip);
            }});

            // Legend
            const legend = svg.append('g')
                .attr('transform', `translate(${{width - 100}}, 40)`);

            programs.forEach((program, i) => {{
                const g = legend.append('g').attr('transform', `translate(0, ${{i * 25}})`);
                g.append('rect').attr('width', 15).attr('height', 15).attr('fill', programColors[program]);
                g.append('text').attr('x', 20).attr('y', 12).text(program).style('font-size', '12px');
            }});

            console.log('Timeline chart complete');
        }}

        // ============ COUNTY CHART ============
        function drawCountyChart(data, quarter) {{
            console.log('Drawing county chart for', quarter);
            const container = d3.select('#county-chart');
            container.selectAll('*').remove();

            // Filter for the specified quarter and aggregate by county
            const filtered = data.filter(d => d.Quarter === quarter && d.County !== 'Statewide');

            const countyTotals = {{}};
            filtered.forEach(d => {{
                const county = d.County;
                if (!countyTotals[county]) {{
                    countyTotals[county] = {{ county: county, allocated: 0, activities: 0 }};
                }}
                countyTotals[county].allocated += parseFloat(d.Allocated) || 0;
                countyTotals[county].activities += parseInt(d['Activity Count']) || 0;
            }});

            const sortedData = Object.values(countyTotals)
                .sort((a, b) => b.allocated - a.allocated)
                .slice(0, 15);

            if (sortedData.length === 0) {{
                container.html('<p class="error">No county data available for ' + quarter + '</p>');
                return;
            }}

            const containerWidth = container.node().getBoundingClientRect().width;
            const width = Math.max(600, containerWidth);
            const height = 450;
            const margin = {{ top: 20, right: 120, bottom: 40, left: 160 }};

            const svg = container.append('svg')
                .attr('width', width)
                .attr('height', height);

            // Scales
            const y = d3.scaleBand()
                .domain(sortedData.map(d => d.county))
                .range([margin.top, height - margin.bottom])
                .padding(0.2);

            const x = d3.scaleLinear()
                .domain([0, d3.max(sortedData, d => d.allocated) * 1.1])
                .range([margin.left, width - margin.right]);

            // Grid
            svg.append('g')
                .attr('class', 'grid')
                .attr('transform', `translate(${{margin.left}},0)`)
                .call(d3.axisLeft(y).tickSize(-(width - margin.left - margin.right)).tickFormat(''));

            // Bars
            svg.selectAll('.bar')
                .data(sortedData)
                .join('rect')
                .attr('class', 'bar')
                .attr('y', d => y(d.county))
                .attr('x', margin.left)
                .attr('height', y.bandwidth())
                .attr('width', d => Math.max(0, x(d.allocated) - margin.left))
                .on('mouseover', function(event, d) {{
                    d3.select(this).attr('opacity', 0.8);
                    showTooltip(event, `
                        <div class="title">${{d.county}}</div>
                        <div class="value">${{formatCurrency(d.allocated, true)}}</div>
                        <div>${{d.activities}} activities</div>
                    `);
                }})
                .on('mouseout', function() {{
                    d3.select(this).attr('opacity', 1);
                    hideTooltip();
                }});

            // Value labels
            svg.selectAll('.value-label')
                .data(sortedData)
                .join('text')
                .attr('y', d => y(d.county) + y.bandwidth() / 2)
                .attr('x', d => x(d.allocated) + 5)
                .attr('dy', '0.35em')
                .text(d => formatCurrency(d.allocated))
                .style('font-size', '11px')
                .style('fill', '#666');

            // Axes
            svg.append('g')
                .attr('class', 'axis')
                .attr('transform', `translate(${{margin.left}},0)`)
                .call(d3.axisLeft(y));

            svg.append('g')
                .attr('class', 'axis')
                .attr('transform', `translate(0,${{height - margin.bottom}})`)
                .call(d3.axisBottom(x).tickFormat(formatCurrency));

            console.log('County chart complete');
        }}

        // ============ ORGANIZATION TABLE ============
        function populateOrgTable(data, quarter) {{
            console.log('Populating org table for', quarter);
            const tbody = document.querySelector('#org-table tbody');
            tbody.innerHTML = '';

            // Filter for quarter and aggregate
            const filtered = data.filter(d => d.Quarter === quarter);

            const orgTotals = {{}};
            filtered.forEach(d => {{
                const key = d.Organization + '|' + d['Program Type'];
                if (!orgTotals[key]) {{
                    orgTotals[key] = {{
                        organization: d.Organization === 'Unknown' ? 'Texas GLO Direct' : d.Organization,
                        program: d['Program Type'],
                        allocated: 0,
                        activities: 0
                    }};
                }}
                orgTotals[key].allocated += parseFloat(d.Allocated) || 0;
                orgTotals[key].activities += parseInt(d['Activity Count']) || 0;
            }});

            const sortedData = Object.values(orgTotals).sort((a, b) => b.allocated - a.allocated);

            sortedData.forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${{row.organization}}</td>
                    <td>${{row.program}}</td>
                    <td class="amount">${{formatCurrency(row.allocated)}}</td>
                    <td class="amount">${{formatNumber(row.activities)}}</td>
                `;
                tbody.appendChild(tr);
            }});

            console.log('Org table complete');
        }}

        // ============ TAB HANDLING ============
        // Timeline tabs
        document.querySelectorAll('.tabs:not(#sankey-tabs) .tab').forEach(tab => {{
            tab.addEventListener('click', function() {{
                this.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                drawTimeline(trendsData, this.dataset.view);
            }});
        }});

        // Sankey tabs
        document.querySelectorAll('#sankey-tabs .tab').forEach(tab => {{
            tab.addEventListener('click', function() {{
                document.querySelectorAll('#sankey-tabs .tab').forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                currentSankeyView = this.dataset.sankey;
                const data = currentSankeyView === 'infrastructure' ? sankeyInfrastructure : sankeyHousing;
                drawSankey(data);
            }});
        }});

        // ============ INITIALIZATION ============
        function init() {{
            console.log('Initializing Harvey Dashboard...');
            console.log('D3 version:', d3.version);
            console.log('d3.sankey available:', typeof d3.sankey === 'function');

            try {{
                // Default to Infrastructure (5B) Sankey
                drawSankey(sankeyInfrastructure);
                drawTimeline(trendsData, 'budget');
                drawCountyChart(countyData, latestQuarter);
                populateOrgTable(orgData, latestQuarter);
                console.log('Dashboard initialization complete!');
            }} catch (error) {{
                console.error('Dashboard initialization error:', error);
            }}
        }}

        // Wait for everything to load
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', () => setTimeout(init, 100));
        }} else {{
            setTimeout(init, 100);
        }}
    </script>
</body>
</html>'''

    return html


def main():
    print("Loading data files...")
    sankey_infra, sankey_housing, trends_data, county_data, org_data = load_data()

    print(f"  Infrastructure Sankey: {len(sankey_infra['nodes'])} nodes, {len(sankey_infra['links'])} links")
    print(f"  Housing Sankey: {len(sankey_housing['nodes'])} nodes, {len(sankey_housing['links'])} links")
    print(f"  Trends: {len(trends_data)} records")
    print(f"  Counties: {len(county_data)} records")
    print(f"  Organizations: {len(org_data)} records")

    print("\nGenerating HTML...")
    html = generate_html(sankey_infra, sankey_housing, trends_data, county_data, org_data)

    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)

    print(f"\nGenerated: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
