# Harvey Action Plan: Fund Switching Statements (Heuristic Report)

This project includes a **screening report** that scans Harvey Action Plan-related PDFs for text that *looks like* funds were reallocated / reprogrammed / shifted, and extracts nearby snippets for human review.

It is designed to answer questions like:

- “Where do the reports describe moving remaining funds into another project/program?”
- “Do any documents include narrative justifications for reallocations?”

## Outputs

- HTML report (non-technical friendly):
  - `outputs/reports/harvey_action_plan_fund_switch_report.html`
- Data exports (in `outputs/exports/harvey/`):
  - `harvey_action_plan_fund_switch_statements.csv`
  - `harvey_action_plan_fund_switch_doc_summary.csv`
  - `harvey_action_plan_fund_switch_semantic_paragraph_candidates.csv`
  - `harvey_action_plan_fund_switch_semantic_dedup_groups.csv`
  - `harvey_action_plan_fund_switch_bertopic_topics.csv`
  - `harvey_action_plan_fund_switch_bertopic_paragraphs.csv`
  - `harvey_action_plan_fund_switch_justification_timeline_by_topic.csv`
  - `harvey_action_plan_fund_switch_relocation_justification_timeline.csv`

These are also linked from `TEAM_PORTAL.html`.

## Build / rebuild

From the repo root:

- `make harvey-fund-switch`

Or directly:

- `python scripts/build_harvey_action_plan_fund_switch_report.py`

## How it works (plain language)

1. Selects Harvey Action Plan categories in the database (by default):  
   `Harvey_5B_ActionPlan`, `Harvey_57M_ActionPlan`
2. Scans page text for fund-change keywords (e.g., **reallocated**) and keeps nearby snippets.
3. Scores snippets using simple signals (money-like patterns, multiple organizations on page, justification cue words).
4. Writes a CSV + an HTML report with charts to help prioritize review.

### Semantic enhancement (transformers + themes)

In addition to the keyword scan, the report includes a semantic layer that:

- Extracts **paragraph-like chunks** from **narrative** sections (`document_sections` + `section_heading_families`)
- Ranks paragraphs by **embedding similarity** to a set of “fund switch” seed queries
- Clusters candidates into **theme-like groups** and collapses **near-duplicates** across quarters

This produces:

- `harvey_action_plan_fund_switch_semantic_paragraph_candidates.csv` (paragraph-level ranked table)
- `harvey_action_plan_fund_switch_semantic_dedup_groups.csv` (deduplicated group summary)

### BERTopic enhancement (exploratory topic modeling)

The report also includes an exploratory **BERTopic** section that:

- builds a broader pool of narrative paragraphs that *hint* at reallocation/fund switching,
- ranks them by embedding similarity to the same seed queries, then
- fits BERTopic (UMAP + HDBSCAN) to cluster the top-ranked paragraphs into topics.

This produces:

- `harvey_action_plan_fund_switch_bertopic_topics.csv` (topic summary with top terms + example paragraphs)
- `harvey_action_plan_fund_switch_bertopic_paragraphs.csv` (paragraph-level topic assignments + confidence)
- `harvey_action_plan_fund_switch_justification_timeline_by_topic.csv` (quarter-by-quarter timeline of BERTopic themes)
- `harvey_action_plan_fund_switch_relocation_justification_timeline.csv` (quarter-by-quarter timeline for relocation/buyout-related justifications)

## Limitations (important)

- This is **not** a perfect “fund transfer detector.” It is keyword-driven and will miss cases that do not use those keywords.
- It may still contain **false positives**. Treat it as a shortlist for review, not a definitive audit.
- It relies on the PDFs already being ingested into `data/glo_reports.db` (PDF text extraction + NLP entity extraction + money-context extraction improve usefulness).
- The semantic layer depends on `sentence-transformers` model downloads; if the model cannot be loaded, the report will still generate but the semantic section may be skipped.
- BERTopic is exploratory and may produce different clusters if the underlying models/dependencies change. If BERTopic dependencies are missing, the report still builds (with placeholder CSVs) and the BERTopic section may be skipped.

## Disabling BERTopic (if needed)

If you want the report without BERTopic (faster, fewer dependencies):

- `python scripts/build_harvey_action_plan_fund_switch_report.py --no-bertopic`

## If you have the *separate* “State Action Plan + amendments” PDFs

If those PDFs are not already in `data/glo_reports.db`, add them as their own category folder and ingest them:

1. Put PDFs under a new folder (example): `DRGR_Reports/Harvey_State_ActionPlan/`
2. Rebuild the DB text/NLP layers:
   - `python src/pdf_processor.py`
   - `python src/nlp_processor.py`
   - `python src/money_context_extractor.py --build --rebuild`
3. Re-run:
   - `make harvey-fund-switch`
