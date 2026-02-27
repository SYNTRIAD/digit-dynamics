# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research � MIT License
"""
ENGINE vNext — Module M1: Experiment Runner + Result Store

Builds on M0 (pipeline_dsl.py) to provide:
  1. SQLite result store with canonical schema
  2. Batch experiment runner (multiple pipelines × domains)
  3. Deterministic runs (exhaustive or sampled with seed)
  4. JSON export for Paper B appendix hashes
  5. Feature extraction per pipeline result (M2 foundation)

Usage:
    from experiment_runner import ExperimentStore, BatchRunner

    store = ExperimentStore("results.db")
    runner = BatchRunner(store)
    runner.run_batch(pipelines, domains)
    store.export_json("results_export.json")
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from pipeline_dsl import (
    OperationRegistry, Pipeline, PipelineStep, DomainPolicy,
    RunResult, PipelineRunner, WitnessTrace,
    SemanticClass, DSClass, canonical_float,
)


# =============================================================================
# EXPERIMENT STORE (SQLite)
# =============================================================================

SCHEMA_VERSION = "1.0"

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_hash TEXT NOT NULL,
    pipeline_display TEXT NOT NULL,
    pipeline_json TEXT NOT NULL,
    domain_hash TEXT NOT NULL,
    domain_json TEXT NOT NULL,
    op_registry_hash TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'exhaustive',
    sample_size INTEGER,
    sample_seed INTEGER,
    num_startpoints INTEGER NOT NULL,
    num_attractors INTEGER NOT NULL,
    convergence_rate TEXT NOT NULL,
    avg_steps TEXT NOT NULL,
    max_steps INTEGER NOT NULL,
    median_steps TEXT NOT NULL,
    basin_entropy TEXT NOT NULL,
    result_hash TEXT NOT NULL UNIQUE,
    engine_version TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    duration_seconds REAL
);

CREATE TABLE IF NOT EXISTS fixed_points (
    experiment_id INTEGER NOT NULL,
    value INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS cycles (
    experiment_id INTEGER NOT NULL,
    cycle_id INTEGER NOT NULL,
    members TEXT NOT NULL,
    length INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS basin_fractions (
    experiment_id INTEGER NOT NULL,
    attractor TEXT NOT NULL,
    fraction TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS witnesses (
    experiment_id INTEGER NOT NULL,
    start_value INTEGER NOT NULL,
    attractor INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    orbit TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS pipeline_features (
    experiment_id INTEGER NOT NULL UNIQUE,
    num_ops INTEGER NOT NULL,
    has_contractive INTEGER NOT NULL,
    has_expansive INTEGER NOT NULL,
    has_permutation INTEGER NOT NULL,
    ds_class_signature TEXT NOT NULL,
    semantic_signature TEXT NOT NULL,
    mod9_preserving INTEGER NOT NULL,
    contains_complement INTEGER NOT NULL,
    contains_kaprekar INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_exp_pipeline ON experiments(pipeline_hash);
CREATE INDEX IF NOT EXISTS idx_exp_domain ON experiments(domain_hash);
CREATE INDEX IF NOT EXISTS idx_exp_result ON experiments(result_hash);
"""


class ExperimentStore:
    """SQLite-backed store for experiment results."""

    def __init__(self, db_path: str = "results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(_CREATE_TABLES)
        # Set schema version
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )
        self.conn.commit()

    def has_result(self, result_hash: str) -> bool:
        """Check if a result already exists (dedup)."""
        cur = self.conn.execute(
            "SELECT 1 FROM experiments WHERE result_hash = ?", (result_hash,)
        )
        return cur.fetchone() is not None

    def store(self, result: RunResult, mode: str = "exhaustive",
              sample_size: Optional[int] = None, sample_seed: Optional[int] = None,
              duration: Optional[float] = None,
              pipeline: Optional[Pipeline] = None,
              domain: Optional[DomainPolicy] = None,
              registry: Optional[OperationRegistry] = None) -> int:
        """Store a RunResult. Returns experiment_id. Skips if duplicate."""
        if self.has_result(result.sha256):
            cur = self.conn.execute(
                "SELECT id FROM experiments WHERE result_hash = ?", (result.sha256,)
            )
            return cur.fetchone()[0]

        cur = self.conn.execute(
            """INSERT INTO experiments (
                pipeline_hash, pipeline_display, pipeline_json,
                domain_hash, domain_json, op_registry_hash,
                mode, sample_size, sample_seed,
                num_startpoints, num_attractors, convergence_rate,
                avg_steps, max_steps, median_steps, basin_entropy,
                result_hash, engine_version, timestamp, duration_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.pipeline_hash,
                result.pipeline_display,
                pipeline.canonical_json() if pipeline else "{}",
                result.domain_hash,
                domain.canonical_json() if domain else "{}",
                result.op_registry_hash,
                mode,
                sample_size,
                sample_seed,
                result.num_startpoints,
                result.num_attractors,
                canonical_float(result.convergence_rate),
                canonical_float(result.avg_steps),
                result.max_steps,
                canonical_float(result.median_steps),
                canonical_float(result.basin_entropy),
                result.sha256,
                result.engine_version,
                result.timestamp or datetime.now().isoformat(),
                duration,
            ),
        )
        exp_id = cur.lastrowid

        # Fixed points
        for fp in result.fixed_points:
            self.conn.execute(
                "INSERT INTO fixed_points (experiment_id, value) VALUES (?, ?)",
                (exp_id, fp),
            )

        # Cycles
        for i, cycle in enumerate(result.cycles):
            self.conn.execute(
                "INSERT INTO cycles (experiment_id, cycle_id, members, length) VALUES (?, ?, ?, ?)",
                (exp_id, i, json.dumps(sorted(cycle)), len(cycle)),
            )

        # Basin fractions
        for att, frac in result.basin_fractions.items():
            self.conn.execute(
                "INSERT INTO basin_fractions (experiment_id, attractor, fraction) VALUES (?, ?, ?)",
                (exp_id, att, canonical_float(frac)),
            )

        # Witnesses
        for w in result.witnesses:
            self.conn.execute(
                "INSERT INTO witnesses (experiment_id, start_value, attractor, steps, orbit) VALUES (?, ?, ?, ?, ?)",
                (exp_id, w.start, w.attractor, w.steps,
                 json.dumps(w.orbit) if w.orbit else None),
            )

        # Pipeline features (M2 foundation)
        if pipeline and registry:
            feats = extract_pipeline_features(pipeline, registry)
            self.conn.execute(
                """INSERT INTO pipeline_features (
                    experiment_id, num_ops, has_contractive, has_expansive,
                    has_permutation, ds_class_signature, semantic_signature,
                    mod9_preserving, contains_complement, contains_kaprekar
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (exp_id, *feats),
            )

        self.conn.commit()
        return exp_id

    def get_experiment(self, exp_id: int) -> Optional[dict]:
        """Retrieve full experiment by ID."""
        cur = self.conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        exp = dict(zip(cols, row))

        # Add fixed points
        cur2 = self.conn.execute(
            "SELECT value FROM fixed_points WHERE experiment_id = ?", (exp_id,)
        )
        exp["fixed_points"] = sorted(r[0] for r in cur2.fetchall())

        # Add cycles
        cur3 = self.conn.execute(
            "SELECT members, length FROM cycles WHERE experiment_id = ? ORDER BY cycle_id", (exp_id,)
        )
        exp["cycles"] = [{"members": json.loads(r[0]), "length": r[1]} for r in cur3.fetchall()]

        # Add basin fractions
        cur4 = self.conn.execute(
            "SELECT attractor, fraction FROM basin_fractions WHERE experiment_id = ?", (exp_id,)
        )
        exp["basin_fractions"] = {r[0]: r[1] for r in cur4.fetchall()}

        return exp

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM experiments")
        return cur.fetchone()[0]

    def list_experiments(self, limit: int = 50) -> List[dict]:
        """List recent experiments (summary)."""
        cur = self.conn.execute(
            """SELECT id, pipeline_display, domain_hash, num_startpoints,
                      num_attractors, convergence_rate, basin_entropy, result_hash
               FROM experiments ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def export_json(self, path: str, limit: int = 1000):
        """Export all experiments to JSON file."""
        experiments = []
        cur = self.conn.execute("SELECT id FROM experiments ORDER BY id LIMIT ?", (limit,))
        for (exp_id,) in cur.fetchall():
            exp = self.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
        with open(path, 'w') as f:
            json.dump({"schema_version": SCHEMA_VERSION, "experiments": experiments},
                      f, indent=2, sort_keys=True)
        return len(experiments)

    def export_paper_appendix(self, path: str):
        """Export verification hashes for Paper B appendix."""
        cur = self.conn.execute(
            """SELECT pipeline_display, domain_hash, num_startpoints,
                      convergence_rate, basin_entropy, result_hash
               FROM experiments ORDER BY pipeline_display""",
        )
        rows = cur.fetchall()
        reg = OperationRegistry()
        data = {
            "title": "Verification Hashes for Attractor Statistics",
            "generated": datetime.now().isoformat(),
            "op_registry_hash": reg.sha256,
            "op_registry_hash_short": reg.short_hash,
            "results": [],
        }
        for r in rows:
            data["results"].append({
                "pipeline": r[0],
                "domain_hash": r[1],
                "domain_hash_short": r[1][:16],
                "num_inputs": r[2],
                "convergence_rate": r[3],
                "basin_entropy": r[4],
                "result_hash": r[5],
                "result_hash_short": r[5][:16],
            })
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return len(rows)

    def close(self):
        self.conn.close()


# =============================================================================
# PIPELINE FEATURE EXTRACTION (M2 foundation)
# =============================================================================

def extract_pipeline_features(pipeline: Pipeline, registry: OperationRegistry) -> tuple:
    """Extract structural features from a pipeline spec. Returns tuple for DB insert."""
    specs = [registry.get_spec(s.op_name) for s in pipeline.steps]

    num_ops = len(specs)
    has_contractive = int(any(s.ds_class == DSClass.CONTRACTIVE for s in specs))
    has_expansive = int(any(s.ds_class == DSClass.EXPANSIVE for s in specs))
    has_permutation = int(any(s.semantic_class == SemanticClass.PERMUTATION for s in specs))

    ds_sig = "".join(s.ds_class.value for s in specs)
    sem_sig = "|".join(s.semantic_class.value for s in specs)

    mod9 = int(all(s.preserves_mod_b_minus_1 for s in specs))
    has_comp = int(any(s.semantic_class == SemanticClass.COMPLEMENT for s in specs))
    has_kap = int(any(s.name == "kaprekar_step" for s in specs))

    return (num_ops, has_contractive, has_expansive, has_permutation,
            ds_sig, sem_sig, mod9, has_comp, has_kap)


# =============================================================================
# SAMPLED RUNNER (for large domains)
# =============================================================================

class SampledRunner:
    """Run pipeline on a random sample of domain values (deterministic with seed)."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()

    def run_sampled(
        self,
        pipeline: Pipeline,
        domain: DomainPolicy,
        sample_size: int = 10000,
        seed: int = 42,
        max_iter: int = 200,
        store_witnesses: int = 10,
    ) -> Tuple[RunResult, int]:
        """Run pipeline on a random sample. Returns (result, seed)."""
        reg = self.registry
        rng = random.Random(seed)

        lo, hi = domain.range()
        all_values = list(range(lo, hi + 1))
        if domain.exclude_repdigits:
            all_values = [n for n in all_values if not domain.is_repdigit(n)]

        sample = rng.sample(all_values, min(sample_size, len(all_values)))
        sample.sort()

        attractor_counts: Dict[int, int] = {}
        step_counts: List[int] = []
        witness_traces: List[WitnessTrace] = []
        cycles_found: Dict[int, List[int]] = {}
        fixed_points: Set[int] = set()

        for n in sample:
            val = n
            seen = {val: 0}
            orbit = [val]
            converged = False
            for t in range(1, max_iter + 1):
                val = reg.execute_pipeline(pipeline, val)
                orbit.append(val)
                if val in seen:
                    cycle_start = seen[val]
                    cycle = orbit[cycle_start:t]
                    if len(cycle) == 1:
                        fixed_points.add(val)
                    else:
                        ck = min(cycle)
                        if ck not in cycles_found:
                            cycles_found[ck] = sorted(set(cycle))
                    attractor_counts[val] = attractor_counts.get(val, 0) + 1
                    step_counts.append(t)
                    converged = True
                    break
                if val == 0:
                    attractor_counts[0] = attractor_counts.get(0, 0) + 1
                    step_counts.append(t)
                    fixed_points.add(0)
                    converged = True
                    break
                seen[val] = t

            if not converged:
                attractor_counts[-1] = attractor_counts.get(-1, 0) + 1
                step_counts.append(max_iter)

            if len(witness_traces) < store_witnesses:
                witness_traces.append(WitnessTrace(
                    start=n, attractor=val, steps=step_counts[-1],
                    orbit=orbit[:min(20, len(orbit))],
                ))

        total = len(sample)
        if not step_counts:
            step_counts = [0]

        sorted_steps = sorted(step_counts)
        median_steps = sorted_steps[len(sorted_steps) // 2]

        basin_fracs: Dict[str, float] = {}
        for att, cnt in attractor_counts.items():
            basin_fracs[str(att)] = cnt / total if total > 0 else 0.0

        dominant_count = max(attractor_counts.values()) if attractor_counts else 0
        convergence_rate = dominant_count / total if total > 0 else 0.0
        basin_entropy = RunResult.compute_basin_entropy(basin_fracs)

        result = RunResult(
            pipeline_hash=pipeline.sha256,
            pipeline_display=pipeline.display(),
            domain_hash=domain.sha256,
            domain_display=repr(domain),
            op_registry_hash=reg.sha256,
            num_startpoints=total,
            fixed_points=sorted(fixed_points),
            cycles=[sorted(c) for c in cycles_found.values()],
            cycle_lengths=sorted(len(c) for c in cycles_found.values()),
            num_attractors=len(attractor_counts),
            basin_fractions=basin_fracs,
            avg_steps=sum(step_counts) / len(step_counts),
            max_steps=max(step_counts),
            median_steps=float(median_steps),
            convergence_rate=convergence_rate,
            basin_entropy=basin_entropy,
            witnesses=witness_traces,
            timestamp=datetime.now().isoformat(),
        )
        return result, seed


# =============================================================================
# BATCH RUNNER
# =============================================================================

class BatchRunner:
    """Run multiple pipeline × domain combinations and store results."""

    def __init__(self, store: ExperimentStore,
                 registry: Optional[OperationRegistry] = None):
        self.store = store
        self.registry = registry or OperationRegistry()
        self.runner = PipelineRunner(self.registry)
        self.sampled_runner = SampledRunner(self.registry)

    def run_exhaustive(self, pipeline: Pipeline, domain: DomainPolicy,
                       max_iter: int = 200) -> Tuple[int, RunResult]:
        """Run single exhaustive experiment. Returns (exp_id, result)."""
        t0 = time.time()
        result = self.runner.run_exhaustive(pipeline, domain, max_iter=max_iter)
        duration = time.time() - t0

        exp_id = self.store.store(
            result, mode="exhaustive", duration=duration,
            pipeline=pipeline, domain=domain, registry=self.registry,
        )
        return exp_id, result

    def run_sampled(self, pipeline: Pipeline, domain: DomainPolicy,
                    sample_size: int = 10000, seed: int = 42,
                    max_iter: int = 200) -> Tuple[int, RunResult]:
        """Run single sampled experiment."""
        t0 = time.time()
        result, used_seed = self.sampled_runner.run_sampled(
            pipeline, domain, sample_size=sample_size, seed=seed, max_iter=max_iter,
        )
        duration = time.time() - t0

        exp_id = self.store.store(
            result, mode="sampled", sample_size=sample_size,
            sample_seed=used_seed, duration=duration,
            pipeline=pipeline, domain=domain, registry=self.registry,
        )
        return exp_id, result

    def run_batch(
        self,
        pipelines: List[Pipeline],
        domains: List[DomainPolicy],
        mode: str = "exhaustive",
        sample_size: int = 10000,
        seed: int = 42,
        max_iter: int = 200,
    ) -> List[Tuple[int, RunResult]]:
        """Run all pipeline × domain combinations."""
        results = []
        total = len(pipelines) * len(domains)
        i = 0
        for pipe in pipelines:
            for dom in domains:
                i += 1
                print(f"  [{i}/{total}] {pipe.display()} × D{dom.short_hash[:8]}...", end=" ", flush=True)
                if mode == "exhaustive":
                    exp_id, result = self.run_exhaustive(pipe, dom, max_iter=max_iter)
                else:
                    exp_id, result = self.run_sampled(
                        pipe, dom, sample_size=sample_size, seed=seed, max_iter=max_iter,
                    )
                print(f"FPs={result.fixed_points[:3]}{'...' if len(result.fixed_points) > 3 else ''} "
                      f"conv={result.convergence_rate:.4f} H={result.basin_entropy:.4f}")
                results.append((exp_id, result))
        return results


# =============================================================================
# STANDARD EXPERIMENT SUITE
# =============================================================================

def paper_b_suite() -> Tuple[List[Pipeline], List[DomainPolicy]]:
    """The four representative pipelines from Paper B Table 1."""
    reg = OperationRegistry()
    pipelines = [
        Pipeline.parse("digit_pow4 |> truc_1089", registry=reg),
        Pipeline.parse("truc_1089 |> digit_pow4", registry=reg),
        Pipeline.parse("kaprekar_step |> swap_ends", registry=reg),
        Pipeline.parse("kaprekar_step |> sort_asc |> truc_1089 |> kaprekar_step", registry=reg),
    ]
    domains = [
        DomainPolicy(base=10, digit_length=4, exclude_repdigits=True),
    ]
    return pipelines, domains


def kaprekar_suite() -> Tuple[List[Pipeline], List[DomainPolicy]]:
    """Kaprekar across digit lengths 3-7."""
    reg = OperationRegistry()
    pipelines = [Pipeline.parse("kaprekar_step", registry=reg)]
    domains = [
        DomainPolicy(base=10, digit_length=k, exclude_repdigits=True)
        for k in range(3, 8)
    ]
    return pipelines, domains


def truc1089_suite() -> Tuple[List[Pipeline], List[DomainPolicy]]:
    """1089-trick across digit lengths 3-7."""
    reg = OperationRegistry()
    pipelines = [Pipeline.parse("truc_1089", registry=reg)]
    domains = [
        DomainPolicy(base=10, digit_length=k)
        for k in range(3, 8)
    ]
    return pipelines, domains


# =============================================================================
# MAIN: Run standard suites
# =============================================================================

if __name__ == "__main__":
    import sys

    db_path = "results.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = ExperimentStore(db_path)
    batch = BatchRunner(store)

    print("=" * 70)
    print("ENGINE vNext — M1: Experiment Runner")
    print(f"Registry: {batch.registry.short_hash}")
    print(f"Database: {db_path}")
    print("=" * 70)

    # ── Suite 1: Kaprekar d=3..7 ──
    print("\n── Suite 1: Kaprekar d=3..7 (exhaustive) ──\n")
    pipes, doms = kaprekar_suite()
    batch.run_batch(pipes, doms)

    # ── Suite 2: 1089-trick d=3..7 ──
    print("\n── Suite 2: truc_1089 d=3..7 (exhaustive) ──\n")
    pipes, doms = truc1089_suite()
    batch.run_batch(pipes, doms)

    # ── Suite 3: Paper B representative pipelines ──
    print("\n── Suite 3: Paper B pipelines (4-digit, exhaustive) ──\n")
    pipes, doms = paper_b_suite()
    batch.run_batch(pipes, doms)

    # ── Summary ──
    print(f"\n── Summary ──\n")
    print(f"  Experiments stored: {store.count()}")
    for exp in store.list_experiments():
        print(f"  #{exp['id']:>2}  {exp['pipeline_display']:<50} "
              f"n={exp['num_startpoints']:>8} att={exp['num_attractors']} "
              f"conv={exp['convergence_rate']} H={exp['basin_entropy']}")

    # ── Export ──
    n = store.export_json("results_export.json")
    print(f"\n  Exported {n} experiments to results_export.json")

    n = store.export_paper_appendix("paper_b_hashes.json")
    print(f"  Exported {n} verification hashes to paper_b_hashes.json")

    store.close()
    print("\n" + "=" * 70)
    print("M1 COMPLETE")
    print("=" * 70)
