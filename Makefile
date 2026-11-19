# Shortcuts. Every target runs inside the virtual environment.
PY = .venv/bin/python

.PHONY: install test all serve summary

install:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

all:            ## whole pipeline, from EDA to fairness
	$(PY) main.py

summary:
	$(PY) -m src.runlog_summary

serve:
	.venv/bin/uvicorn src.serve:app --port 8000
