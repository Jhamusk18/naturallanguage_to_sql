VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: install lint type test check demo run ui eval eval-oracle ablation ablation-unpaired db-up db-local db-down

install:
	uv venv --python 3.12 $(VENV)
	uv pip install -e ".[dev,ui]"

lint:
	$(VENV)/bin/ruff check .

type:
	$(VENV)/bin/mypy src

test:
	$(VENV)/bin/pytest -q

check: lint type test

demo:
	$(PYTHON) -m text_to_sql_engine.demo

run:
	$(VENV)/bin/uvicorn text_to_sql_engine.api:app --host 127.0.0.1 --port 8000

ui:
	$(VENV)/bin/streamlit run ui/app.py --server.port 8501

eval:
	$(PYTHON) -m evaluation.run

eval-oracle:
	$(PYTHON) -m evaluation.run --oracle

db-local:
	./scripts/setup_local_postgres.sh

# Controlled ablation. Every arm uses the same questions, model, temperature 0 and
# database; only the prompt (or sample rows) changes. Each arm runs with up to 2
# corrections; because the first attempt of such a run is exactly a corrections-off
# run, each file gives a paired correction OFF vs ON result per question.
# compare.py refuses files from different providers/models, incomplete runs, or
# runs where a fallback model answered. --resume continues a run cut short by a
# free-tier quota. Zero-cost example (Groq free tier):
#   LLM_PROVIDER=groq GROQ_MODEL=openai/gpt-oss-20b GROQ_REASONING_EFFORT=low \
#   LLM_FALLBACK=none make ablation RESULTS=evaluation/results/v2-groq-gpt-oss-20b
RESULTS ?= evaluation/results/latest
EVAL := $(PYTHON) -m evaluation.run --resume

ablation:
	mkdir -p $(RESULTS)
	$(EVAL) --prompt-mode schema_aware --max-corrections 2 --out $(RESULTS)/aware_c2.json
	$(EVAL) --prompt-mode baseline --max-corrections 2 --out $(RESULTS)/names_only_c2.json
	$(EVAL) --prompt-mode columns_only --max-corrections 2 --out $(RESULTS)/columns_only_c2.json
	$(EVAL) --prompt-mode schema_aware --max-corrections 2 --sample-rows 0 \
		--out $(RESULTS)/aware_norows_c2.json
	$(PYTHON) -m evaluation.compare $(RESULTS)/*.json

# Optional cross-check: separate corrections-off runs (costs extra API calls).
ablation-unpaired:
	$(EVAL) --prompt-mode schema_aware --max-corrections 0 --out $(RESULTS)/aware_c0.json
	$(EVAL) --prompt-mode baseline --max-corrections 0 --out $(RESULTS)/names_only_c0.json
	$(PYTHON) -m evaluation.compare $(RESULTS)/*.json

db-up:
	docker compose up -d postgres

db-down:
	docker compose down
