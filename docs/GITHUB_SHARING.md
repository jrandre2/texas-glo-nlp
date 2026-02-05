# GitHub Sharing Guide

This project contains **large source documents (PDFs)**, a **large SQLite database** (`data/glo_reports.db`), and **large generated artifacts** (some maps/GeoJSON/HTML). A clean GitHub repo should prioritize:

- **Code + documentation** (tracked in git)
- **Small, model-ready exports** (tracked in git where feasible)
- **Large data/artifacts** shared via an external channel (GitHub Releases, an object store, or shared drive)

## What should be in GitHub

- `src/` (pipeline + analysis code)
- `scripts/` (builders, exporters)
- `docs/` (plain-language + technical docs)
- `outputs/model_ready/` (CSV panels and long-form tables designed for modeling/EDA)
- `TEAM_PORTAL.html` (small, single-file portal that links to key outputs)
- `requirements.txt`, `Makefile`, `.env.example`

## What should NOT be in GitHub (by default)

- `DRGR_Reports/` (raw PDFs; large + often licensed/restricted)
- `data/*.db` (SQLite databases are large; treat as build artifacts)
- Large generated outputs (some are >100MB and exceed GitHub limits):
  - `outputs/exports/spatial/spatial_*.html`
  - `outputs/exports/spatial/*_joined.geojson`
  - any other multi-hundred-MB HTML/GeoJSON

These are already covered by `.gitignore` in this repo.

## Recommended sharing workflow

1. **Commit code + docs** (and the small `outputs/model_ready/` tables).
2. Generate share artifacts locally:
   - `make model-ready`
   - `make portal`
   - `make share-bundle` (creates `outputs/share_bundle/` + `outputs/share_bundle.zip`)
   - (Optional) `make clean-macos` to remove `._*` / `.DS_Store` clutter
3. Share large artifacts via one of:
   - **GitHub Releases**: attach a ZIP containing the large HTML maps/GeoJSONs.
   - **Git LFS**: if your org supports it (works well for large HTML/GeoJSON).
   - **Shared drive / S3**: keep a stable “latest outputs” folder.

## Reproducibility note

All model-ready tables in `outputs/model_ready/` are intended to be **regeneratable** from the database:

- Build model-ready datasets: `make model-ready`
- Rebuild the portal: `make portal`

## Share-bundle notes

`make share-bundle` builds a bundle based on what is linked in `TEAM_PORTAL.html` and applies a per-file size cap (default 50MB). By default it **excludes** the very large spatial HTML maps.

If you need those maps in the bundle, run the script directly:

- `python scripts/build_share_bundle.py --zip --include-spatial --max-mb 300`
