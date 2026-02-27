"""
ENGINE vNext -- run_experiments.py: Experiment Recreation Script (C1 Audit Fix)

Usage:
    python scripts/run_experiments.py                          # all suites, fresh DB
    python scripts/run_experiments.py --db data/results.db     # custom DB path
    python scripts/run_experiments.py --suites kaprekar        # single suite
    python scripts/run_experiments.py --suites kaprekar paper_b  # multiple suites
    python scripts/run_experiments.py --no-fresh               # append to existing DB

This script recreates results.db from scratch by running all experiment suites
used in Paper A and Paper B. It is the FIRST step of full reproducibility:

    Step 1: python scripts/run_experiments.py              # experiments -> results.db
    Step 2: python scripts/reproduce.py --db data/results.db --bundle  # artifacts -> bundle

Together, these two steps provide end-to-end reproducibility:
  - Step 1 verifies that experimental RESULTS can be recreated from code.
  - Step 2 verifies that ARTIFACTS (manifest, catalogs, LaTeX) are deterministic.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure parent directory (project root) is on sys.path for core module imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment_runner import (
    ExperimentStore, BatchRunner,
    kaprekar_suite, truc1089_suite, paper_b_suite,
)


# =============================================================================
# SUITE REGISTRY
# =============================================================================

AVAILABLE_SUITES = {
    "kaprekar": {
        "fn": kaprekar_suite,
        "label": "Kaprekar d=3..7 (exhaustive)",
        "paper": "A",
    },
    "truc1089": {
        "fn": truc1089_suite,
        "label": "truc_1089 d=3..7 (exhaustive)",
        "paper": "A",
    },
    "paper_b": {
        "fn": paper_b_suite,
        "label": "Paper B representative pipelines (4-digit, exhaustive)",
        "paper": "B",
    },
}

DEFAULT_SUITES = ["kaprekar", "truc1089", "paper_b"]


# =============================================================================
# MAIN
# =============================================================================

def run(db_path: str, suites: list[str], fresh: bool,
        export_json: str | None, export_hashes: str | None) -> int:
    """Run experiment suites and populate results DB.

    Returns exit code: 0 on success, 1 on error.
    """
    # Resolve paths relative to project root
    if not os.path.isabs(db_path):
        db_path = os.path.join(_PROJECT_ROOT, db_path)
    if export_json and not os.path.isabs(export_json):
        export_json = os.path.join(_PROJECT_ROOT, export_json)
    if export_hashes and not os.path.isabs(export_hashes):
        export_hashes = os.path.join(_PROJECT_ROOT, export_hashes)

    # Fresh DB: remove existing
    if fresh and os.path.exists(db_path):
        os.remove(db_path)
        print(f"  Removed existing DB: {db_path}")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    store = ExperimentStore(db_path)
    batch = BatchRunner(store)

    print("=" * 70)
    print("ENGINE vNext — Experiment Runner (C1 Audit Fix)")
    print(f"  Registry:  {batch.registry.short_hash}")
    print(f"  Database:  {db_path}")
    print(f"  Suites:    {', '.join(suites)}")
    print(f"  Fresh DB:  {fresh}")
    print("=" * 70)

    t0 = time.time()
    total_experiments = 0

    for suite_name in suites:
        suite_info = AVAILABLE_SUITES[suite_name]
        print(f"\n-- {suite_info['label']} (Paper {suite_info['paper']}) --\n")
        pipes, doms = suite_info["fn"]()
        results = batch.run_batch(pipes, doms)
        total_experiments += len(results)

    duration = time.time() - t0

    # Summary
    print(f"\n-- Summary --\n")
    print(f"  Experiments run:   {total_experiments}")
    print(f"  Experiments in DB: {store.count()}")
    print(f"  Duration:          {duration:.1f}s")
    print()
    for exp in store.list_experiments():
        print(f"  #{exp['id']:>2}  {exp['pipeline_display']:<50} "
              f"n={exp['num_startpoints']:>8} att={exp['num_attractors']} "
              f"conv={exp['convergence_rate']} H={exp['basin_entropy']}")

    # Export
    if export_json:
        os.makedirs(os.path.dirname(export_json), exist_ok=True)
        n = store.export_json(export_json)
        print(f"\n  Exported {n} experiments to {export_json}")

    if export_hashes:
        os.makedirs(os.path.dirname(export_hashes), exist_ok=True)
        n = store.export_paper_appendix(export_hashes)
        print(f"  Exported {n} verification hashes to {export_hashes}")

    store.close()

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print(f"Next step: python scripts/reproduce.py --db {db_path} --bundle")
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Recreate results.db from scratch by running all experiment suites.",
        epilog="After running this script, use reproduce.py to generate and verify artifacts.",
    )
    parser.add_argument("--db", default="data/results.db",
                        help="Path to results database (default: data/results.db)")
    parser.add_argument("--suites", nargs="*", default=DEFAULT_SUITES,
                        choices=list(AVAILABLE_SUITES.keys()),
                        help=f"Suites to run (default: {' '.join(DEFAULT_SUITES)})")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="Delete existing DB before running (default: True)")
    parser.add_argument("--no-fresh", dest="fresh", action="store_false",
                        help="Append to existing DB instead of fresh start")
    parser.add_argument("--export-json", default="data/results_export.json",
                        help="Export results as JSON (default: data/results_export.json)")
    parser.add_argument("--export-hashes", default="data/paper_b_hashes.json",
                        help="Export verification hashes (default: data/paper_b_hashes.json)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip all exports")

    args = parser.parse_args()

    export_json = None if args.no_export else args.export_json
    export_hashes = None if args.no_export else args.export_hashes

    sys.exit(run(args.db, args.suites, args.fresh, export_json, export_hashes))


if __name__ == "__main__":
    main()
