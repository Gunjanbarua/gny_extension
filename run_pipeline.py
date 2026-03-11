"""
run_pipeline.py
---------------
Top-level entry point for the tree-volume prediction pipeline.

Stages
------
  1  data_preparation  – clean raw field & LiDAR data, build input_data.csv
  2  lmm               – Lasso variable selection + Linear Mixed-Effects Model
  3  random_forest      – RF with spatial CV, hyperparameter tuning, PFI by age
  4  pfi_analysis       – Permutation Feature Importance stratified by TPH
  5  visualization      – all publication-quality figures

Usage
-----
  # Run the full pipeline
  python run_pipeline.py

  # Run only specific stages (comma-separated)
  python run_pipeline.py --stages 1,3,5

  # Run from a particular stage onward
  python run_pipeline.py --from-stage 3
"""

import argparse
import time
from pathlib import Path

# Ensure the project root is on the path when called from another directory
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src import data_preparation, lmm, random_forest, pfi_analysis, visualization


STAGES = {
    1: ("Data Preparation",  data_preparation.run),
    2: ("Linear Mixed-Effects Model", lmm.run),
    3: ("Random Forest",     random_forest.run),
    4: ("PFI by TPH",        pfi_analysis.run),
    5: ("Visualisation",     visualization.run),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tree volume prediction pipeline (RW20 spacing trial)"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stage numbers to run, e.g. '1,3,5'",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=None,
        dest="from_stage",
        help="Run all stages from this number onward",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stages:
        selected = [int(s.strip()) for s in args.stages.split(",")]
    elif args.from_stage:
        selected = [s for s in STAGES if s >= args.from_stage]
    else:
        selected = list(STAGES.keys())

    print("\n" + "=" * 60)
    print("  Tree Volume Prediction Pipeline")
    print(f"  Stages to run: {selected}")
    print("=" * 60)

    total_start = time.time()
    for stage_id in sorted(selected):
        if stage_id not in STAGES:
            print(f"[WARNING] Unknown stage {stage_id} – skipping.")
            continue
        name, fn = STAGES[stage_id]
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"[Stage {stage_id}] '{name}' completed in {elapsed:.1f}s")

    total = time.time() - total_start
    print(f"\nAll selected stages finished in {total:.1f}s.")


if __name__ == "__main__":
    main()
