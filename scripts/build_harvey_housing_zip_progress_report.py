#!/usr/bin/env python3
"""
Harvey Housing Progress by ZIP (Heuristic panel + report)

Purpose:
  Build a ZIP-code by quarter panel for the Harvey Housing program using the
  Harvey QPR-derived tables in the SQLite DB.

Important:
  ZIP codes are extracted from QPR text (best-effort). Many activities do not
  include ZIPs in narrative/text in a consistent way. Treat this as screening /
  exploratory analysis, not an authoritative geographic allocation.

Because many activities list multiple ZIP codes, this script uses a simple
allocation approach:
  - For activity-quarter metrics (budget/households/etc.), divide by the number
    of ZIPs attached to that activity in that quarter, then sum by ZIP.
This avoids double-counting when aggregating across ZIPs, but is still an
assumption.

Writes:
  outputs/exports/harvey_housing_zip_quarter_panel.csv
  outputs/exports/harvey_housing_quarter_summary.csv
  outputs/reports/harvey_housing_zip_progress_report.html
  outputs/reports/assets/harvey_housing_zip_*.png
"""

from __future__ import annotations

import argparse
import html
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "glo_reports.db"
EXPORTS_DIR = ROOT / "outputs" / "exports"
REPORTS_DIR = ROOT / "outputs" / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"


ZIP_RE = re.compile(r"^\d{5}$")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_sql(con: sqlite3.Connection, query: str, params: Tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, con, params=params)


def _quarter_label(year: Any, quarter_num: Any) -> Optional[str]:
    try:
        y = int(year)
        q = int(quarter_num)
        return f"Q{q} {y}"
    except Exception:
        return None


def build_outputs(db_path: Path) -> Dict[str, Path]:
    _safe_mkdir(EXPORTS_DIR)
    _safe_mkdir(REPORTS_DIR)
    _safe_mkdir(ASSETS_DIR)

    con = sqlite3.connect(str(db_path))

    # Base: housing activities by quarter
    acts = _read_sql(
        con,
        """
        SELECT
          document_id,
          quarter,
          year,
          quarter_num,
          activity_code,
          responsible_org,
          county,
          total_budget,
          status
        FROM harvey_activities
        WHERE program_type = 'Housing'
          AND activity_code IS NOT NULL
          AND activity_code != 'Projected'
          AND year IS NOT NULL
          AND quarter_num IS NOT NULL
        """,
    )

    if acts.empty:
        con.close()
        raise RuntimeError("No Harvey Housing activities found in harvey_activities.")

    acts["quarter_label"] = acts.apply(lambda r: _quarter_label(r["year"], r["quarter_num"]), axis=1)
    acts["status"] = acts["status"].fillna("Unknown")

    # Quarter summary (overall; not ZIP-based)
    quarter_summary = (
        acts.groupby(["year", "quarter_num", "quarter", "quarter_label"], dropna=False)
        .agg(
            n_activity_rows=("activity_code", "count"),
            n_unique_activities=("activity_code", pd.Series.nunique),
            n_documents=("document_id", pd.Series.nunique),
            sum_total_budget_usd=("total_budget", "sum"),
        )
        .reset_index()
        .sort_values(["year", "quarter_num"])
    )

    status_pivot = (
        acts.pivot_table(
            index=["year", "quarter_num", "quarter", "quarter_label"],
            columns="status",
            values="activity_code",
            aggfunc=pd.Series.nunique,
            fill_value=0,
        )
        .reset_index()
    )
    status_cols = [c for c in status_pivot.columns if c not in {"year", "quarter_num", "quarter", "quarter_label"}]
    status_pivot = status_pivot.rename(columns={c: f"n_status_{str(c).strip().lower().replace(' ', '_')}" for c in status_cols})
    quarter_summary = quarter_summary.merge(status_pivot, on=["year", "quarter_num", "quarter", "quarter_label"], how="left")

    # Locations (ZIPs) – dedupe to activity_code×quarter×zip
    loc = _read_sql(
        con,
        """
        SELECT activity_code, quarter, location_value AS zip_code, city AS city_hint, county AS county_hint
        FROM harvey_activity_locations
        WHERE location_type = 'zip_code'
          AND activity_code IS NOT NULL
          AND activity_code != 'Projected'
        """,
    )
    if loc.empty:
        con.close()
        raise RuntimeError("No ZIP locations found in harvey_activity_locations.")

    loc["zip_code"] = loc["zip_code"].astype(str).str.strip()
    loc = loc[loc["zip_code"].map(lambda z: bool(ZIP_RE.match(z)))]
    loc = loc.drop_duplicates(subset=["activity_code", "quarter", "zip_code"])

    # Beneficiaries (activity-level)
    ben = _read_sql(
        con,
        """
        SELECT
          activity_code,
          quarter,
          households_total,
          renter_households,
          owner_households,
          persons_total,
          housing_units_total,
          jobs_created,
          jobs_retained
        FROM harvey_beneficiaries
        WHERE activity_code IS NOT NULL
          AND activity_code != 'Projected'
        """,
    )

    # Narratives (activity-level)
    narr = _read_sql(
        con,
        """
        SELECT
          activity_code,
          quarter,
          projects_completed,
          projects_underway,
          households_served
        FROM harvey_progress_narratives
        WHERE activity_code IS NOT NULL
          AND activity_code != 'Projected'
        """,
    )

    # Build activity×zip×quarter table (housing only)
    act_zip = acts.merge(loc, on=["activity_code", "quarter"], how="inner")
    if act_zip.empty:
        con.close()
        raise RuntimeError("No overlaps between Harvey Housing activities and extracted ZIPs.")

    act_zip = act_zip.merge(ben, on=["activity_code", "quarter"], how="left").merge(narr, on=["activity_code", "quarter"], how="left")

    # How many ZIPs per activity-quarter?
    zc = act_zip.groupby(["activity_code", "quarter"], dropna=False)["zip_code"].nunique().reset_index(name="n_zips")
    act_zip = act_zip.merge(zc, on=["activity_code", "quarter"], how="left")
    act_zip["n_zips"] = act_zip["n_zips"].fillna(1).astype(int).clip(lower=1)

    # Allocate metrics across ZIPs for each activity-quarter
    alloc_cols = [
        ("total_budget", "alloc_total_budget_usd"),
        ("households_total", "alloc_households_total"),
        ("renter_households", "alloc_renter_households"),
        ("owner_households", "alloc_owner_households"),
        ("persons_total", "alloc_persons_total"),
        ("housing_units_total", "alloc_housing_units_total"),
        ("jobs_created", "alloc_jobs_created"),
        ("jobs_retained", "alloc_jobs_retained"),
        ("projects_completed", "alloc_projects_completed"),
        ("projects_underway", "alloc_projects_underway"),
        ("households_served", "alloc_households_served"),
    ]
    for src, dst in alloc_cols:
        if src not in act_zip.columns:
            act_zip[dst] = pd.NA
            continue
        act_zip[src] = pd.to_numeric(act_zip[src], errors="coerce")
        act_zip[dst] = act_zip[src] / act_zip["n_zips"]

    # ZIP×quarter panel
    panel = (
        act_zip.groupby(["year", "quarter_num", "quarter", "quarter_label", "zip_code"], dropna=False)
        .agg(
            n_activities=("activity_code", pd.Series.nunique),
            n_documents=("document_id", pd.Series.nunique),
            n_responsible_orgs=("responsible_org", pd.Series.nunique),
            n_counties=("county_hint", pd.Series.nunique),
        )
        .reset_index()
    )

    # Attach best-guess county/city labels
    def mode(series: pd.Series) -> Optional[str]:
        s = series.dropna().astype(str).map(lambda x: x.strip()).replace("", pd.NA).dropna()
        if s.empty:
            return None
        return s.value_counts().idxmax()

    geo_labels = (
        act_zip.groupby(["year", "quarter_num", "quarter", "quarter_label", "zip_code"], dropna=False)
        .agg(county_mode=("county_hint", mode), city_mode=("city_hint", mode))
        .reset_index()
    )
    panel = panel.merge(geo_labels, on=["year", "quarter_num", "quarter", "quarter_label", "zip_code"], how="left")

    # Add allocated sums
    for _src, dst in alloc_cols:
        s = (
            act_zip.groupby(["year", "quarter_num", "quarter", "quarter_label", "zip_code"], dropna=False)[dst]
            .sum(min_count=1)
            .reset_index(name=f"sum_{dst}")
        )
        panel = panel.merge(s, on=["year", "quarter_num", "quarter", "quarter_label", "zip_code"], how="left")

    # Status counts per ZIP-quarter (unique activities)
    status_zip = (
        act_zip.pivot_table(
            index=["year", "quarter_num", "quarter", "quarter_label", "zip_code"],
            columns="status",
            values="activity_code",
            aggfunc=pd.Series.nunique,
            fill_value=0,
        )
        .reset_index()
    )
    status_cols = [c for c in status_zip.columns if c not in {"year", "quarter_num", "quarter", "quarter_label", "zip_code"}]
    status_zip = status_zip.rename(columns={c: f"n_status_{str(c).strip().lower().replace(' ', '_')}" for c in status_cols})
    panel = panel.merge(status_zip, on=["year", "quarter_num", "quarter", "quarter_label", "zip_code"], how="left")
    panel = panel.sort_values(["year", "quarter_num", "zip_code"])

    out_panel = EXPORTS_DIR / "harvey_housing_zip_quarter_panel.csv"
    out_quarter_summary = EXPORTS_DIR / "harvey_housing_quarter_summary.csv"
    panel.to_csv(out_panel, index=False)
    quarter_summary.to_csv(out_quarter_summary, index=False)

    # ---- Build report (HTML + PNG charts) ----
    built_at = _now_iso()

    # Determine latest quarter
    latest = quarter_summary.sort_values(["year", "quarter_num"]).tail(1)
    latest_label = str(latest["quarter_label"].iloc[0]) if not latest.empty else "Unknown"
    latest_year = int(latest["year"].iloc[0]) if not latest.empty else None
    latest_qn = int(latest["quarter_num"].iloc[0]) if not latest.empty else None

    panel_latest = panel
    if latest_year is not None and latest_qn is not None:
        panel_latest = panel[(panel["year"] == latest_year) & (panel["quarter_num"] == latest_qn)].copy()

    # Chart 1: total budget over time (overall)
    plt.figure(figsize=(10, 4.2))
    plt.plot(quarter_summary["quarter_label"], quarter_summary["sum_total_budget_usd"], marker="o", linewidth=2, color="#8ab4ff")
    plt.xticks(rotation=45, ha="right")
    plt.title("Harvey Housing – total budget (sum across activities) by quarter")
    plt.ylabel("USD")
    plt.tight_layout()
    chart_total_budget = ASSETS_DIR / "harvey_housing_zip_total_budget_by_quarter.png"
    plt.savefig(chart_total_budget, dpi=160)
    plt.close()

    # Chart 2: unique activities by quarter
    plt.figure(figsize=(10, 4.2))
    plt.plot(quarter_summary["quarter_label"], quarter_summary["n_unique_activities"], marker="o", linewidth=2, color="#9be7ff")
    plt.xticks(rotation=45, ha="right")
    plt.title("Harvey Housing – unique activities by quarter")
    plt.ylabel("Activities")
    plt.tight_layout()
    chart_activities = ASSETS_DIR / "harvey_housing_zip_activities_by_quarter.png"
    plt.savefig(chart_activities, dpi=160)
    plt.close()

    # Chart 3: top ZIPs by allocated budget in latest quarter
    top_zip_budget = panel_latest.sort_values("sum_alloc_total_budget_usd", ascending=False).head(20)
    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=top_zip_budget,
        y="zip_code",
        x="sum_alloc_total_budget_usd",
        color="#8ab4ff",
    )
    plt.title(f"Harvey Housing – top ZIPs by allocated budget ({latest_label})")
    plt.xlabel("Allocated budget (USD; activity budget split across ZIPs)")
    plt.ylabel("ZIP")
    plt.tight_layout()
    chart_top_zips = ASSETS_DIR / "harvey_housing_zip_top_zips_latest_quarter.png"
    plt.savefig(chart_top_zips, dpi=160)
    plt.close()

    # Chart 4: heatmap for top ZIPs over time
    panel_for_heat = panel.copy()
    panel_for_heat["quarter_label"] = panel_for_heat["quarter_label"].astype(str)
    panel_for_heat["val"] = pd.to_numeric(panel_for_heat["sum_alloc_total_budget_usd"], errors="coerce").fillna(0.0)
    top_zips_all = (
        panel_for_heat.groupby("zip_code")["val"]
        .sum()
        .sort_values(ascending=False)
        .head(25)
        .index.tolist()
    )
    heat = panel_for_heat[panel_for_heat["zip_code"].isin(top_zips_all)].pivot_table(
        index="zip_code", columns="quarter_label", values="val", aggfunc="sum", fill_value=0.0
    )
    # Order quarters
    quarter_order = quarter_summary["quarter_label"].astype(str).tolist()
    heat = heat.reindex(columns=quarter_order)
    plt.figure(figsize=(12, 7))
    sns.heatmap(heat, cmap="Blues", cbar_kws={"label": "Allocated budget (USD)"})
    plt.title("Harvey Housing – allocated budget by ZIP over time (top 25 ZIPs)")
    plt.xlabel("Quarter")
    plt.ylabel("ZIP")
    plt.tight_layout()
    chart_heatmap = ASSETS_DIR / "harvey_housing_zip_heatmap.png"
    plt.savefig(chart_heatmap, dpi=160)
    plt.close()

    # HTML report
    report_path = REPORTS_DIR / "harvey_housing_zip_progress_report.html"

    n_zips_total = int(panel["zip_code"].nunique())
    n_rows_panel = int(len(panel))
    n_acts_with_zip = int(act_zip[["activity_code", "quarter"]].drop_duplicates().shape[0])

    # A small table for latest quarter
    latest_table = top_zip_budget[["zip_code", "county_mode", "n_activities", "sum_alloc_total_budget_usd"]].copy()
    latest_table["sum_alloc_total_budget_usd"] = latest_table["sum_alloc_total_budget_usd"].fillna(0.0).map(lambda x: f"${x:,.0f}")
    latest_rows = []
    for r in latest_table.itertuples(index=False):
        latest_rows.append(
            f"<tr><td class='mono'>{html.escape(str(r.zip_code))}</td>"
            f"<td>{html.escape(str(r.county_mode or ''))}</td>"
            f"<td style='text-align:right'>{int(r.n_activities or 0)}</td>"
            f"<td style='text-align:right'>{html.escape(str(r.sum_alloc_total_budget_usd))}</td></tr>"
        )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Harvey Housing – Progress by ZIP</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #101a2e;
      --card: #122042;
      --text: #e9eefc;
      --muted: #b7c3e0;
      --link: #8ab4ff;
      --border: rgba(255,255,255,0.12);
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, var(--bg), #070b14 60%);
      color: var(--text);
    }}
    header {{
      padding: 26px 22px 12px;
      border-bottom: 1px solid var(--border);
      background: radial-gradient(1200px 500px at 20% 0%, rgba(138,180,255,0.20), transparent 60%);
    }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .sub {{ color: var(--muted); font-size: 13px; line-height: 1.4; max-width: 1100px; }}
    main {{ padding: 18px 22px 60px; max-width: 1200px; margin: 0 auto; }}
    a {{ color: var(--link); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin-top: 14px; }}
    .card {{ background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-radius: 12px; padding: 12px 12px; }}
    .card b {{ display:block; font-size: 18px; margin-bottom: 2px; }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }}
    h2 {{ margin: 26px 0 10px; font-size: 15px; }}
    img {{ max-width: 100%; border-radius: 12px; border: 1px solid var(--border); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border-top: 1px solid var(--border); padding: 10px 8px; vertical-align: top; }}
    th {{ text-align: left; font-size: 12px; color: var(--muted); }}
    td {{ font-size: 13px; line-height: 1.35; }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Harvey Housing – Progress by ZIP (Heuristic)</h1>
    <div class="sub">
      Builds a ZIP×quarter panel from Harvey QPR-derived tables in <span class="mono">data/glo_reports.db</span>.
      ZIPs are extracted from text (best-effort). Metrics are allocated across ZIPs when an activity lists multiple ZIPs.
      Built at <span class="mono">{html.escape(built_at)}</span>.
    </div>
    <div class="grid">
      <div class="card"><b>{n_zips_total}</b><div class="small muted">ZIPs with any Harvey Housing coverage</div></div>
      <div class="card"><b>{n_rows_panel}</b><div class="small muted">ZIP×quarter rows</div></div>
      <div class="card"><b>{n_acts_with_zip}</b><div class="small muted">Activity×quarter pairs with ZIPs</div></div>
    </div>
  </header>
  <main>
    <h2>Files</h2>
    <p class="small">
      Panel CSV: <a href="../exports/harvey_housing_zip_quarter_panel.csv">outputs/exports/harvey_housing_zip_quarter_panel.csv</a><br/>
      Quarter summary CSV: <a href="../exports/harvey_housing_quarter_summary.csv">outputs/exports/harvey_housing_quarter_summary.csv</a>
    </p>

    <h2>Overall progress (all Housing activities)</h2>
    <div class="sub">These charts are built from <span class="mono">harvey_activities</span> (not the ZIP panel).</div>
    <div style="margin-top:12px"><img src="assets/harvey_housing_zip_total_budget_by_quarter.png" alt="Total budget by quarter" /></div>
    <div style="margin-top:12px"><img src="assets/harvey_housing_zip_activities_by_quarter.png" alt="Activities by quarter" /></div>

    <h2>Where (ZIPs) – latest quarter snapshot ({html.escape(latest_label)})</h2>
    <div class="sub">
      This uses the ZIP×quarter panel. Budget values are <b>allocated</b> across ZIPs for activities that list multiple ZIPs.
    </div>
    <div style="margin-top:12px"><img src="assets/harvey_housing_zip_top_zips_latest_quarter.png" alt="Top ZIPs" /></div>
    <table>
      <thead>
        <tr>
          <th style="width:90px">ZIP</th>
          <th style="width:180px">County (mode)</th>
          <th style="width:120px; text-align:right">Activities</th>
          <th style="text-align:right">Allocated budget</th>
        </tr>
      </thead>
      <tbody>
        {''.join(latest_rows) if latest_rows else '<tr><td colspan=\"4\" class=\"muted\">No ZIP rows for the latest quarter.</td></tr>'}
      </tbody>
    </table>

    <h2>Top ZIPs over time (heatmap)</h2>
    <div class="sub">Top 25 ZIPs by cumulative allocated budget across all quarters.</div>
    <div style="margin-top:12px"><img src="assets/harvey_housing_zip_heatmap.png" alt="ZIP heatmap" /></div>

    <h2>Notes / limitations</h2>
    <ul class="sub">
      <li>ZIP codes are extracted from QPR narrative/text; many activities do not list ZIPs consistently.</li>
      <li>Allocated ZIP budgets are computed by splitting an activity’s budget equally across its listed ZIP codes for that quarter.</li>
      <li>County/city labels are “mode” hints from extracted text; ZIPs can span multiple counties.</li>
    </ul>
  </main>
</body>
</html>
"""
    report_path.write_text(report_html, encoding="utf-8")

    con.close()

    return {
        "panel_csv": out_panel,
        "quarter_summary_csv": out_quarter_summary,
        "report_html": report_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Harvey Housing ZIP×quarter panel + report")
    ap.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Path to SQLite DB (default: data/glo_reports.db)")
    args = ap.parse_args()

    outputs = build_outputs(Path(args.db))
    print("Wrote:")
    for k, v in outputs.items():
        print(f"  {k:<22} {v.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

