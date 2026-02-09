# Glossary

## Programs / Systems

- **CDBG-DR**: Community Development Block Grant -- Disaster Recovery (federal funding).
- **CDBG-MIT**: CDBG -- Mitigation (hazard mitigation funding).
- **DRGR**: Disaster Recovery Grant Reporting (HUD reporting system used for quarterly reporting).
- **LOCCS**: HUD's Line of Credit Control System (often used in drawdown/disbursement language).
- **QPR**: Quarterly Performance Report -- the periodic status report filed through DRGR.

## Time

- **Quarter**: Reporting period like **Q4 2025** (used throughout exports and charts).
- **Latest quarter**: The most recent year/quarter present in `documents` in `data/glo_reports.db`.

## Finance

- **Budget**: Planned/allocated amount (what is intended or approved).
- **Obligated**: Funds committed/assigned to projects or uses.
- **Expended**: Funds spent (expenditures).
- **Drawdown / Disbursed**: Funds drawn from the line of credit / disbursed (often close to cash movement).
- **Unmet need**: The gap between identified recovery needs and available funding.

## Harvey / DRGR Activity Terms

- **Activity**: A single program-level work unit tracked in DRGR (e.g., "Homeowner Assistance -- Harris County"). Each has a status, budget, and optional beneficiary measures.
- **Subrecipient**: An implementing organization that receives CDBG-DR funds through the grantee (Texas GLO) to carry out activities.
- **Activity type**: The normalized category of work (e.g., Acquisition/Buyout, Homeownership Assistance, Rehabilitation/Reconstruction, Infrastructure).
- **Buyout / Acquisition**: Programs that purchase properties in high-risk areas.
- **National objective**: The HUD-required justification for CDBG spending (Low/Mod Income benefit, Urgent Need, Slum/Blight).
- **LMI**: Low-to-Moderate Income -- the most common national objective; activities must benefit LMI households.
- **Accomplishment / Performance measure**: DRGR beneficiary rows showing persons, households, housing units, or jobs served (actual vs. expected).
- **Beneficiary**: A person, household, or entity served by a CDBG-DR activity.

## Spatial / Geographic

- **ZIP (ZCTA)**: ZIP Code Tabulation Area -- the Census approximation of USPS ZIP codes, used for spatial analysis.
- **Census Tract**: A small, relatively permanent statistical subdivision of a county (~1,200-8,000 people).
- **Block Group**: A subdivision of a census tract; the smallest geography for which Census publishes sample data.
- **GEOID**: A standardized geographic identifier (e.g., `48201` for Harris County, `48201310100` for a tract).
- **FIPS code**: Federal Information Processing Standard code for identifying geographic areas. Texas = `48`; Harris County = `201`.
- **County FIPS (3-digit)**: The 3-digit portion identifying a county within a state (e.g., `201` for Harris).
- **H3 Hex**: Uber's hexagonal hierarchical spatial index; used for point-data aggregation in this project (resolution 7).
- **Choropleth**: A map where areas are shaded by the value of a variable (e.g., mention count by county).
- **Boundary GeoJSON**: Vector geometry files for Texas counties, ZIPs, tracts, and block groups stored in `data/boundaries/`.

## NLP Outputs

- **Entity**: A detected thing in text (e.g., COUNTY, PROGRAM, MONEY, DISASTER).
- **Alias / Canonical**: A mapping that groups variations ("Texas General Land Office" -> "Texas GLO") so counts roll up correctly.
- **Canonical form**: The single standardized name chosen to represent a group of aliases.
- **Topic**: A cluster of similar narrative text snippets discovered automatically via embedding-based clustering (used for trend tracking).
- **Section**: A heading-delimited span of pages within a document (e.g., "Executive Summary", "Activity Progress Narrative").
- **Section family**: A taxonomy label for a section heading -- `narrative`, `finance`, `metadata`, `form`, or `table` -- used to filter downstream analyses to narrative-only text.
- **Relation edge**: A pair of entities that appear together in the same sentence (a "mentioned together" signal; not a verified causal relationship).
- **Money mention**: A sentence-level dollar amount extracted from text. This is a *mentions layer* (not an official accounting table).
- **Context label**: The classification assigned to a money mention -- `budget`, `expended`, `obligated`, `drawdown`, or `unknown` -- based on surrounding keywords.

## SEM / Modeling

- **SEM**: Structural Equation Modeling -- a multivariate statistical framework for testing hypothesized causal relationships among latent and observed variables.
- **Latent construct**: An unobserved variable (e.g., "disaster severity") inferred from multiple observed indicators.
- **Indicator**: An observed/measured variable that serves as a proxy for a latent construct.
- **Panel data**: A dataset where the same units (counties, disasters) are observed across multiple time periods (quarters).
- **Coverage**: The percentage of rows in a panel that have non-null (or non-zero) values for a given variable.
- **Provenance**: The source information for an extracted value -- document, page, extraction method, and confidence.
- **Confidence score**: A heuristic 0-1 score indicating extraction reliability (higher = more certain).
- **Signal density**: The number of SEM construct signals present in a panel cell.
- **Construct**: A theoretical variable in an SEM model (e.g., "administrative capacity", "program performance").

## Pipeline / Build

- **Model-ready**: A CSV dataset formatted for direct use in statistical models -- tidy, with stable column names and documented units.
- **Panel (wide-format)**: One row per unit-period (e.g., county x quarter) with variables as columns.
- **Long-format**: One row per observation (e.g., one row per activity, one row per money mention).
- **Manifest**: The `manifest.json` file recording build timestamp, row counts, and quality check results.
- **Quality gate**: An automated check that must pass for a build to succeed (e.g., non-empty output, no large quarter-over-quarter drops).
