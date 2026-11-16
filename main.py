"""Run the whole project end to end, from raw data to the final model and its audits.

    python main.py            every stage, in order
    python main.py eda tune   only the listed stages

Stages: eda, baselines, tune, compare, explain, calibrate, fairness
"""
import sys

from src import baselines, calibrate, compare, eda, explain, fairness, tune

STAGES = {
    "eda": eda.main,
    "baselines": baselines.main,
    "tune": tune.main,
    "compare": compare.main,
    "explain": explain.main,
    "calibrate": calibrate.main,
    "fairness": fairness.main,
}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(STAGES)
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s): {unknown}. Choose among {list(STAGES)}")
    sys.argv = sys.argv[:1]          # stages read their own optional arguments
    for stage in selected:
        print(f"\n===== {stage} =====")
        STAGES[stage]()
