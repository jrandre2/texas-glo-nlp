#!/usr/bin/env python3
"""
Build a single-file, click-to-view HTML portal for non-technical users.

Writes: TEAM_PORTAL.html (project root)
"""

from __future__ import annotations

import html
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "glo_reports.db"
OUTPUT_HTML = ROOT / "TEAM_PORTAL.html"


def _human_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "—"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TB"


def _fmt_dt(ts: Optional[float]) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True)
class Artifact:
    label: str
    relpath: str
    description: str
    kind: str

    def path(self) -> Path:
        return ROOT / self.relpath

    def exists(self) -> bool:
        return self.path().exists()

    def size_bytes(self) -> Optional[int]:
        try:
            return self.path().stat().st_size
        except OSError:
            return None

    def mtime(self) -> Optional[float]:
        try:
            return self.path().stat().st_mtime
        except OSError:
            return None


def _artifacts() -> Dict[str, List[Artifact]]:
    return {
        "Dashboards": [
            Artifact(
                label="Harvey dashboard (standalone)",
                relpath="outputs/visualizations/harvey_dashboard_standalone.html",
                description="Interactive overview of Harvey program flows, rollups, and trends.",
                kind="html",
            ),
            Artifact(
                label="Harvey extended dashboard",
                relpath="outputs/visualizations/harvey_extended_dashboard.html",
                description="Additional Harvey views and drill-down tables (lighter-weight HTML).",
                kind="html",
            ),
            Artifact(
                label="Harvey dashboard v2",
                relpath="outputs/visualizations/harvey_dashboard_v2.html",
                description="Alternative Harvey dashboard layout.",
                kind="html",
            ),
        ],
        "Reports (HTML)": [
            Artifact(
                label="Harvey: fund switching / reallocation statements",
                relpath="outputs/reports/harvey_action_plan_fund_switch_report.html",
                description="Extracted statements related to reallocating/switching funds (with snippets and counts).",
                kind="html",
            ),
            Artifact(
                label="Harvey: housing progress by ZIP × quarter",
                relpath="outputs/reports/harvey_housing_zip_progress_report.html",
                description="Quarter-by-quarter Housing activity trends with ZIP-level panel export (allocated).",
                kind="html",
            ),
        ],
        "Sankey (Images/PDF)": [
            Artifact(
                label="Harvey 5B Sankey (PNG)",
                relpath="outputs/visualizations/harvey_sankey_5b.png",
                description="Quick slide-friendly Sankey image for the 5B grant (budget allocations).",
                kind="png",
            ),
            Artifact(
                label="Harvey 5B Sankey (PDF)",
                relpath="outputs/visualizations/harvey_sankey_5b.pdf",
                description="Print-friendly Sankey PDF for the 5B grant.",
                kind="pdf",
            ),
            Artifact(
                label="Harvey 57M Sankey (PNG)",
                relpath="outputs/visualizations/harvey_sankey_57m.png",
                description="Quick Sankey image for the 57M grant (budget allocations).",
                kind="png",
            ),
            Artifact(
                label="Harvey 57M Sankey (PDF)",
                relpath="outputs/visualizations/harvey_sankey_57m.pdf",
                description="Print-friendly Sankey PDF for the 57M grant.",
                kind="pdf",
            ),
            Artifact(
                label="Recipients Sankey (PNG)",
                relpath="outputs/visualizations/harvey_sankey_recipients.png",
                description="Sankey view focused on recipient organizations.",
                kind="png",
            ),
            Artifact(
                label="Recipients Sankey (PDF)",
                relpath="outputs/visualizations/harvey_sankey_recipients.pdf",
                description="Print-friendly recipients Sankey PDF.",
                kind="pdf",
            ),
        ],
        "Spatial Maps (HTML)": [
            Artifact(
                label="ZIP choropleth (latest quarter)",
                relpath="outputs/exports/spatial_zip_latest_quarter.html",
                description="Very large interactive map; may take time to load.",
                kind="html",
            ),
            Artifact(
                label="Tract choropleth (latest quarter)",
                relpath="outputs/exports/spatial_tract_latest_quarter.html",
                description="Interactive tract map for the latest quarter.",
                kind="html",
            ),
            Artifact(
                label="Tract choropleth (Harris County)",
                relpath="outputs/exports/spatial_tract_harris.html",
                description="Tract map filtered to Harris County.",
                kind="html",
            ),
            Artifact(
                label="Tract choropleth (all mentions)",
                relpath="outputs/exports/spatial_tract_all.html",
                description="All tract mentions (no time filter).",
                kind="html",
            ),
            Artifact(
                label="Statewide choropleth (general)",
                relpath="outputs/exports/spatial_choropleth.html",
                description="Statewide choropleth (very large).",
                kind="html",
            ),
        ],
        "Key Tables (CSV/JSON)": [
            Artifact(
                label="Texas disaster financial summary (CSV)",
                relpath="outputs/exports/texas_disaster_financial_summary.csv",
                description="High-level totals by disaster/program (obligated/disbursed/expended).",
                kind="csv",
            ),
            Artifact(
                label="Linked entities summary (CSV)",
                relpath="outputs/exports/linked_entities_summary.csv",
                description="Entity → national grant links and rollups.",
                kind="csv",
            ),
            Artifact(
                label="Harvey county allocations (CSV)",
                relpath="outputs/exports/harvey_county_allocations.csv",
                description="Harvey rollups by county (quarter-aware).",
                kind="csv",
            ),
            Artifact(
                label="Harvey org allocations (CSV)",
                relpath="outputs/exports/harvey_org_allocations.csv",
                description="Harvey rollups by recipient organization.",
                kind="csv",
            ),
            Artifact(
                label="Harvey fund switch statements (CSV)",
                relpath="outputs/exports/harvey_action_plan_fund_switch_statements.csv",
                description="Row-level statement snippets about moving/reallocating funds (heuristic extraction).",
                kind="csv",
            ),
            Artifact(
                label="Harvey fund switch doc summary (CSV)",
                relpath="outputs/exports/harvey_action_plan_fund_switch_doc_summary.csv",
                description="Document-level counts of detected fund-switch statements.",
                kind="csv",
            ),
            Artifact(
                label="Harvey fund switch semantic candidates (CSV)",
                relpath="outputs/exports/harvey_action_plan_fund_switch_semantic_paragraph_candidates.csv",
                description="Paragraph-level narrative candidates ranked by transformer embedding similarity + clustered into themes.",
                kind="csv",
            ),
            Artifact(
                label="Harvey fund switch semantic dedup groups (CSV)",
                relpath="outputs/exports/harvey_action_plan_fund_switch_semantic_dedup_groups.csv",
                description="Near-duplicate groups of semantic candidates (collapsed across quarters/PDFs).",
                kind="csv",
            ),
            Artifact(
                label="Harvey fund switch BERTopic topics (CSV)",
                relpath="outputs/exports/harvey_action_plan_fund_switch_bertopic_topics.csv",
                description="Exploratory topics discovered from narrative paragraphs (ranked by movement-like phrasing).",
                kind="csv",
            ),
            Artifact(
                label="Harvey fund switch BERTopic paragraph assignments (CSV)",
                relpath="outputs/exports/harvey_action_plan_fund_switch_bertopic_paragraphs.csv",
                description="Paragraph-level topic assignments + confidence for BERTopic (use to review clusters and deduplicate).",
                kind="csv",
            ),
            Artifact(
                label="Harvey fund switch relocation justification timeline (CSV)",
                relpath="outputs/exports/harvey_action_plan_fund_switch_relocation_justification_timeline.csv",
                description="Quarter-by-quarter counts of relocation/buyout-related justifications (from BERTopic paragraph assignments).",
                kind="csv",
            ),
            Artifact(
                label="Harvey housing quarter summary (CSV)",
                relpath="outputs/exports/harvey_housing_quarter_summary.csv",
                description="Quarter-by-quarter summary for all Harvey Housing activities.",
                kind="csv",
            ),
            Artifact(
                label="Harvey housing ZIP × quarter panel (CSV)",
                relpath="outputs/exports/harvey_housing_zip_quarter_panel.csv",
                description="ZIP-by-quarter panel with allocated budgets/outcomes to avoid double-counting multi-ZIP activities.",
                kind="csv",
            ),
            Artifact(
                label="Topic examples (CSV)",
                relpath="outputs/exports/topic_examples.csv",
                description="Top terms + representative snippets for each discovered topic.",
                kind="csv",
            ),
            Artifact(
                label="Topic trends by quarter (CSV)",
                relpath="outputs/exports/topic_trends_by_quarter.csv",
                description="Topic frequency over time by category/year/quarter.",
                kind="csv",
            ),
            Artifact(
                label="Entity relations (top edges) (CSV)",
                relpath="outputs/exports/entity_relations_top_edges.csv",
                description="High-frequency entity co-mentions with weights (sentence co-occurrence).",
                kind="csv",
            ),
            Artifact(
                label="Money mentions by quarter (CSV)",
                relpath="outputs/exports/money_mentions_by_quarter.csv",
                description="NLP-derived money mentions over time, grouped by context label.",
                kind="csv",
            ),
            Artifact(
                label="Money mentions: top linked entities (CSV)",
                relpath="outputs/exports/money_mentions_top_entities.csv",
                description="Entities most often co-mentioned with money mentions, by context label.",
                kind="csv",
            ),
            Artifact(
                label="Money mentions (row-level, capped) (CSV)",
                relpath="outputs/exports/money_mentions.csv",
                description="Row-level money mentions with sentence snippets (large).",
                kind="csv",
            ),
        ],
        "Model-Ready Datasets (CSV)": [
            Artifact(
                label="Panel: document (CSV)",
                relpath="outputs/model_ready/panels/panel_document.csv",
                description="One row per PDF report (document metadata + entity/money rollups + activity counts).",
                kind="csv",
            ),
            Artifact(
                label="Panel: disaster × quarter (CSV)",
                relpath="outputs/model_ready/panels/panel_disaster_quarter.csv",
                description="One row per (category, disaster_code, year, quarter) with aggregated report-level features.",
                kind="csv",
            ),
            Artifact(
                label="Panel: county × quarter (CSV)",
                relpath="outputs/model_ready/panels/panel_county_quarter.csv",
                description="County-by-quarter panel from unique activities (county inferred from location mentions within activity pages).",
                kind="csv",
            ),
            Artifact(
                label="Panel: city × quarter (CSV)",
                relpath="outputs/model_ready/panels/panel_city_quarter.csv",
                description="City-by-quarter panel from unique activities (city inferred from location mentions; county included when available).",
                kind="csv",
            ),
            Artifact(
                label="Panel: state × quarter (CSV)",
                relpath="outputs/model_ready/panels/panel_state_quarter.csv",
                description="Statewide rollup by (year, quarter) across all disasters (unique activities).",
                kind="csv",
            ),
            Artifact(
                label="Activities (long) (CSV)",
                relpath="outputs/model_ready/long/activities.csv",
                description="One row per detected activity-group (header page + continuation pages) with status/type, geo hints, money aggregates, and beneficiary summaries (when present).",
                kind="csv",
            ),
            Artifact(
                label="Activities (unique, long) (CSV)",
                relpath="outputs/model_ready/long/activities_unique.csv",
                description="Deduplicated activities by (category, disaster_code, year, quarter, activity_key) for modeling/panels.",
                kind="csv",
            ),
            Artifact(
                label="Money mentions by quarter (CSV)",
                relpath="outputs/model_ready/long/money_mentions_by_quarter.csv",
                description="Money mention counts/sums by (category, year, quarter, context_label).",
                kind="csv",
            ),
            Artifact(
                label="Topic trends by quarter (CSV)",
                relpath="outputs/model_ready/long/topic_trends_by_quarter.csv",
                description="Topic frequency over time (long format).",
                kind="csv",
            ),
            Artifact(
                label="Entity counts by quarter (CSV)",
                relpath="outputs/model_ready/long/entity_counts_by_quarter.csv",
                description="Entity type counts by quarter (long format).",
                kind="csv",
            ),
            Artifact(
                label="Keyword pages by quarter (CSV)",
                relpath="outputs/model_ready/long/keyword_pages_by_quarter.csv",
                description="Counts of pages/documents containing selected keywords (e.g., payroll, deaths).",
                kind="csv",
            ),
            Artifact(
                label="Beneficiary measures (long) (CSV)",
                relpath="outputs/model_ready/long/beneficiary_measures.csv",
                description="Row-level parsed accomplishment/beneficiary counts (coverage varies by report format).",
                kind="csv",
            ),
            Artifact(
                label="Severity proxies by quarter (CSV)",
                relpath="outputs/model_ready/long/severity_proxies_by_quarter.csv",
                description="Quarterly rainfall/wind-speed proxies parsed from entity text (use as weak signals; prefer external joins).",
                kind="csv",
            ),
            Artifact(
                label="FEMA declarations by quarter (CSV)",
                relpath="outputs/model_ready/long/fema_declarations_by_quarter.csv",
                description="FEMA declaration numbers by quarter to support external joins.",
                kind="csv",
            ),
            Artifact(
                label="Build manifest (JSON)",
                relpath="outputs/model_ready/meta/manifest.json",
                description="Build timestamp + row counts for each model-ready dataset.",
                kind="json",
            ),
        ],
        "Docs": [
            Artifact(
                label="Start Here (non-technical)",
                relpath="docs/START_HERE.md",
                description="Plain-language guide and directory map.",
                kind="md",
            ),
            Artifact(
                label="Glossary",
                relpath="docs/GLOSSARY.md",
                description="Key terms used in the project.",
                kind="md",
            ),
            Artifact(
                label="Dashboard (Streamlit) guide",
                relpath="docs/DASHBOARD.md",
                description="How to run the interactive explorer.",
                kind="md",
            ),
            Artifact(
                label="Workflows",
                relpath="docs/WORKFLOWS.md",
                description="Step-by-step operational workflows (technical).",
                kind="md",
            ),
            Artifact(
                label="NLP analyses index",
                relpath="docs/ANALYSES.md",
                description="Sections/topics/relations/money-context documentation (technical).",
                kind="md",
            ),
            Artifact(
                label="Model-ready datasets",
                relpath="docs/MODEL_READY.md",
                description="What model-ready datasets exist + how to build them.",
                kind="md",
            ),
            Artifact(
                label="Modeling variables map",
                relpath="docs/MODELING_VARIABLES.md",
                description="Mapping from requested SEM/EDA variables to available outputs/proxies.",
                kind="md",
            ),
            Artifact(
                label="Harvey: fund switch extraction",
                relpath="docs/HARVEY_ACTION_PLAN_FUND_SWITCH.md",
                description="How fund reallocation statements are detected + outputs + limitations.",
                kind="md",
            ),
            Artifact(
                label="Harvey: housing ZIP progress",
                relpath="docs/HARVEY_HOUSING_ZIP_PROGRESS.md",
                description="How the ZIP × quarter housing panel is built + outputs + coverage notes.",
                kind="md",
            ),
            Artifact(
                label="GitHub sharing guide",
                relpath="docs/GITHUB_SHARING.md",
                description="What to commit vs share externally (large files).",
                kind="md",
            ),
        ],
    }


def _db_stats(db_path: Path) -> Dict[str, object]:
    if not db_path.exists():
        return {"error": f"DB not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    def count(table: str) -> Optional[int]:
        row = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        if not row:
            return None
        return int(cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    latest = cur.execute(
        """
        SELECT year, quarter
        FROM documents
        WHERE year IS NOT NULL AND quarter IS NOT NULL
        ORDER BY year DESC, quarter DESC
        LIMIT 1
        """
    ).fetchone()
    latest_q = None
    if latest and latest[0] and latest[1]:
        latest_q = {"year": int(latest[0]), "quarter": int(latest[1])}

    money_context_rows = []
    if cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='money_mentions'").fetchone():
        money_context_rows = cur.execute(
            """
            SELECT COALESCE(context_label,'unknown') as label, COUNT(*) as n
            FROM money_mentions
            GROUP BY COALESCE(context_label,'unknown')
            ORDER BY n DESC
            """
        ).fetchall()

    counts = {
        "documents": count("documents"),
        "entities": count("entities"),
        "location_mentions": count("location_mentions"),
        "topic_assignments": count("topic_assignments"),
        "entity_relations": count("entity_relations"),
        "money_mentions": count("money_mentions"),
    }

    conn.close()

    return {
        "db_path": str(db_path),
        "latest_quarter": latest_q,
        "counts": counts,
        "money_context": [{"label": str(r[0]), "n": int(r[1])} for r in money_context_rows],
    }


def _render_artifact_list(artifacts: Sequence[Artifact]) -> str:
    rows = []
    for a in artifacts:
        if not a.exists():
            continue
        size = _human_bytes(a.size_bytes())
        mtime = _fmt_dt(a.mtime())
        label = html.escape(a.label)
        desc = html.escape(a.description)
        href = html.escape(a.relpath)
        rows.append(
            f"""
            <div class="card">
              <div class="card-title"><a href="{href}">{label}</a></div>
              <div class="card-meta">{size} • updated {mtime}</div>
              <div class="card-desc">{desc}</div>
            </div>
            """.strip()
        )
    if not rows:
        return '<div class="muted">No artifacts found for this section.</div>'
    return "\n".join(rows)


def build_portal(db_path: Path) -> str:
    stats = _db_stats(db_path)
    latest = stats.get("latest_quarter") or {}
    latest_str = "Unknown"
    if isinstance(latest, dict) and latest.get("year") and latest.get("quarter"):
        latest_str = f"Q{latest['quarter']} {latest['year']}"

    counts = (stats.get("counts") or {}) if isinstance(stats, dict) else {}
    c_docs = counts.get("documents")
    c_entities = counts.get("entities")
    c_money = counts.get("money_mentions")
    c_rel = counts.get("entity_relations")

    money_context = stats.get("money_context") if isinstance(stats, dict) else None
    context_html = ""
    if isinstance(money_context, list) and money_context:
        rows = "".join(
            f"<tr><td>{html.escape(str(r.get('label')))}</td><td style=\"text-align:right\">{int(r.get('n') or 0):,}</td></tr>"
            for r in money_context[:10]
        )
        context_html = f"""
        <details>
          <summary>Money mention context breakdown (top labels)</summary>
          <table class="mini">
            <thead><tr><th>Context</th><th style="text-align:right">Mentions</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </details>
        """.strip()

    built_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    artifacts_by_group = _artifacts()

    sections_html = []
    for group, group_artifacts in artifacts_by_group.items():
        sections_html.append(
            f"""
            <section>
              <h2>{html.escape(group)}</h2>
              {_render_artifact_list(group_artifacts)}
            </section>
            """.strip()
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Texas GLO DRGR – Team Portal</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #101a2e;
      --card: #122042;
      --text: #e9eefc;
      --muted: #b7c3e0;
      --link: #8ab4ff;
      --border: rgba(255,255,255,0.10);
      --shadow: 0 6px 22px rgba(0,0,0,0.35);
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, var(--bg), #070b14 60%);
      color: var(--text);
    }}
    header {{
      padding: 28px 22px 14px;
      border-bottom: 1px solid var(--border);
      background: radial-gradient(1200px 500px at 20% 0%, rgba(138,180,255,0.20), transparent 60%);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      letter-spacing: 0.2px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      max-width: 1100px;
    }}
    .stats {{
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .stat {{
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 13px;
    }}
    .stat b {{
      display: block;
      font-size: 16px;
      margin-bottom: 2px;
    }}
    main {{
      padding: 18px 22px 60px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    section {{
      margin-top: 26px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0.2px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
      margin: 10px 0;
    }}
    .card-title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .card-meta {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .card-desc {{
      color: var(--text);
      font-size: 13px;
      opacity: 0.95;
      line-height: 1.35;
    }}
    a {{
      color: var(--link);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    details {{
      margin-top: 12px;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    summary {{
      cursor: pointer;
      color: var(--text);
      font-weight: 600;
      font-size: 13px;
    }}
    table.mini {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 12px;
      color: var(--text);
    }}
    table.mini th, table.mini td {{
      border-bottom: 1px solid var(--border);
      padding: 6px 4px;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 30px;
      padding-top: 14px;
      border-top: 1px solid var(--border);
    }}
    code {{
      background: rgba(255,255,255,0.08);
      padding: 2px 6px;
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Texas GLO DRGR – Team Portal</h1>
    <div class="sub">
      Click any link below to open dashboards, maps, or tables. This portal is generated from the local project folder.
      <br/>Built at <b>{html.escape(built_at)}</b>. Latest documents in DB: <b>{html.escape(latest_str)}</b>.
      <br/><span class="muted">Note: some HTML maps are very large and may take time to load.</span>
    </div>
    <div class="stats">
      <div class="stat"><b>{(int(c_docs) if c_docs is not None else 0):,}</b>Documents</div>
      <div class="stat"><b>{(int(c_entities) if c_entities is not None else 0):,}</b>Entities (NER)</div>
      <div class="stat"><b>{(int(c_rel) if c_rel is not None else 0):,}</b>Relation edges</div>
      <div class="stat"><b>{(int(c_money) if c_money is not None else 0):,}</b>Money mentions (NLP)</div>
    </div>
    {context_html}
  </header>
  <main>
    <section>
      <h2>How to use</h2>
      <div class="card">
        <div class="card-desc">
          <ul>
            <li><b>Dashboards / maps:</b> open the <code>.html</code> links in your browser.</li>
            <li><b>Tables:</b> open <code>.csv</code> in Excel/Sheets. Use filters to explore.</li>
            <li><b>Master database:</b> <code>data/glo_reports.db</code> can be opened with “DB Browser for SQLite”.</li>
            <li><b>Interpretation:</b> “money mentions” are extracted from narrative text and are <i>not</i> official accounting totals.</li>
          </ul>
        </div>
      </div>
    </section>

    {"".join(sections_html)}

    <footer>
      Tip: for interactive filtering/search, run the Streamlit explorer:
      <code>streamlit run dashboard/app.py</code>
    </footer>
  </main>
</body>
</html>
"""


def main() -> None:
    db_path = DEFAULT_DB_PATH
    html_text = build_portal(db_path)
    OUTPUT_HTML.write_text(html_text, encoding="utf-8")
    # Remove macOS AppleDouble artifacts that confuse non-technical users
    try:
        apple_double = OUTPUT_HTML.parent / f"._{OUTPUT_HTML.name}"
        if apple_double.exists():
            apple_double.unlink()
    except OSError:
        pass
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
