.PHONY: help stats pdf-stats nlp-stats link semantic-build harvey harvey-parse harvey-fund-switch harvey-housing-zip harvey-reports spatial sections section-families topics entity-resolve relations money model-ready portal share-bundle clean-macos analyses check

PY := $(shell if [ -x venv/bin/python ]; then echo venv/bin/python; else echo python; fi)

help:
	@echo "Common targets:"
	@echo "  make stats          - print DB snapshot counts"
	@echo "  make pdf-stats      - pdf_processor --stats"
	@echo "  make nlp-stats      - nlp_processor --stats"
	@echo "  make link           - data_linker full run"
	@echo "  make semantic-build - build Chroma semantic index"
	@echo "  make harvey         - parse Harvey + export Sankey/trends"
	@echo "  make spatial        - rebuild spatial tables + export maps"
	@echo "  make sections       - build document section spans"
	@echo "  make section-families - classify section headings into families"
	@echo "  make topics         - fit + export topic clusters"
	@echo "  make entity-resolve - build + export entity aliases"
	@echo "  make relations      - build + export entity relations"
	@echo "  make money          - extract money mentions + context"
	@echo "  make model-ready    - export model-ready CSV panels"
	@echo "  make harvey-reports - build Harvey deliverable reports (HTML + CSV)"
	@echo "  make portal         - build TEAM_PORTAL.html for sharing"
	@echo "  make share-bundle   - build a zip-ready share bundle (portal-linked outputs)"
	@echo "  make clean-macos    - remove .DS_Store / ._ artifacts"
	@echo "  make analyses       - run all NLP analyses"
	@echo "  make check          - compileall on src/"

stats:
	$(PY) src/project_status.py

pdf-stats:
	$(PY) src/pdf_processor.py --stats

nlp-stats:
	$(PY) src/nlp_processor.py --stats

link:
	$(PY) src/data_linker.py

semantic-build:
	$(PY) src/semantic_search.py --build

harvey:
	$(PY) src/financial_parser.py
	$(PY) src/funding_tracker.py --export

harvey-parse:
	$(PY) src/financial_parser.py

harvey-fund-switch:
	$(PY) scripts/build_harvey_action_plan_fund_switch_report.py

harvey-housing-zip: harvey-parse
	$(PY) scripts/build_harvey_housing_zip_progress_report.py

harvey-reports: harvey-fund-switch harvey-housing-zip

spatial:
	$(PY) src/location_extractor.py --rebuild
	$(PY) src/spatial_mapper.py --join --map

sections:
	$(PY) src/section_extractor.py
	$(PY) src/section_extractor.py --export

section-families: sections
	$(PY) src/section_classifier.py --build
	$(PY) src/section_classifier.py --export

topics: section-families
	$(PY) src/topic_model.py --fit --k 40 --families narrative --rebuild
	$(PY) src/topic_model.py --export

entity-resolve:
	$(PY) src/entity_resolution.py --build --rebuild
	$(PY) src/entity_resolution.py --export

relations: section-families entity-resolve
	$(PY) src/relation_extractor.py --rebuild --use-aliases --min-weight 3 --min-org-count 200 --section-families narrative
	$(PY) src/relation_extractor.py --export

money: section-families entity-resolve
	$(PY) src/money_context_extractor.py --build --use-aliases --min-org-count 200 --rebuild
	$(PY) src/money_context_extractor.py --export

model-ready:
	$(PY) scripts/build_model_ready_datasets.py

portal: model-ready harvey-reports
	$(PY) scripts/build_team_portal.py

share-bundle: portal
	$(PY) scripts/build_share_bundle.py --zip

clean-macos:
	$(PY) scripts/clean_macos_artifacts.py --root .

analyses: sections section-families topics entity-resolve relations money

check:
	$(PY) -m compileall -q -x '(^|/)\._' src
