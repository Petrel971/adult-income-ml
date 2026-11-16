"""Load config.yaml and expose project paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Return the YAML configuration as a dict."""
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


CFG = load_config()
SEED = CFG["seed"]
MODELS_DIR = ROOT / CFG["paths"]["models"]
FIGURES_DIR = ROOT / CFG["paths"]["figures"]
TABLES_DIR = ROOT / CFG["paths"]["tables"]
RUN_LOG = ROOT / CFG["paths"]["run_log"]

for _d in (MODELS_DIR, FIGURES_DIR, TABLES_DIR, RUN_LOG.parent):
    _d.mkdir(parents=True, exist_ok=True)
