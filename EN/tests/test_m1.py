"""
Tests for Engine vNext M1: Experiment Runner + Result Store

Test categories:
  1. Store CRUD: create, read, dedup, count
  2. Batch runner: exhaustive + sampled
  3. Determinism: same seed → same result
  4. Export: JSON + paper appendix
  5. Feature extraction: pipeline structural features
  6. Schema integrity: foreign keys, indexes
"""

import json
import os
import tempfile
import pytest

from pipeline_dsl import (
    OperationRegistry, Pipeline, DomainPolicy, RunResult, PipelineRunner,
    SemanticClass, DSClass, canonical_float,
)
from experiment_runner import (
    ExperimentStore, BatchRunner, SampledRunner,
    extract_pipeline_features, paper_b_suite, kaprekar_suite,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reg():
    return OperationRegistry()

@pytest.fixture
def tmp_db():
    """Temporary database that gets cleaned up."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)
    # WAL/SHM files
    for ext in ("-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)

@pytest.fixture
def store(tmp_db):
    s = ExperimentStore(tmp_db)
    yield s
    s.close()

@pytest.fixture
def batch(store, reg):
    return BatchRunner(store, registry=reg)


# =============================================================================
# 1. STORE CRUD
# =============================================================================

@pytest.mark.integration
class TestStoreCRUD:

    def test_empty_store(self, store):
        assert store.count() == 0

    def test_store_and_retrieve(self, store, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
        runner = PipelineRunner(reg)
        result = runner.run_exhaustive(pipe, domain)

        exp_id = store.store(result, pipeline=pipe, domain=domain, registry=reg)
        assert exp_id == 1
        assert store.count() == 1

        exp = store.get_experiment(exp_id)
        assert exp is not None
        assert exp["pipeline_hash"] == pipe.sha256
        assert exp["domain_hash"] == domain.sha256
        assert 6174 in exp["fixed_points"]

    def test_dedup(self, store, reg):
        """Same result stored twice → same experiment_id."""
        pipe = Pipeline.parse("digit_sum", registry=reg)
        domain = DomainPolicy(base=10, digit_length=2)
        runner = PipelineRunner(reg)
        result = runner.run_exhaustive(pipe, domain)

        id1 = store.store(result, pipeline=pipe, domain=domain, registry=reg)
        id2 = store.store(result, pipeline=pipe, domain=domain, registry=reg)
        assert id1 == id2
        assert store.count() == 1

    def test_multiple_experiments(self, store, reg):
        runner = PipelineRunner(reg)
        for name in ["kaprekar_step", "truc_1089", "digit_sum"]:
            pipe = Pipeline.parse(name, registry=reg)
            domain = DomainPolicy(base=10, digit_length=3)
            result = runner.run_exhaustive(pipe, domain)
            store.store(result, pipeline=pipe, domain=domain, registry=reg)
        assert store.count() == 3

    def test_list_experiments(self, store, reg):
        runner = PipelineRunner(reg)
        pipe = Pipeline.parse("reverse", registry=reg)
        domain = DomainPolicy(base=10, digit_length=2)
        result = runner.run_exhaustive(pipe, domain)
        store.store(result, pipeline=pipe, domain=domain, registry=reg)

        listing = store.list_experiments()
        assert len(listing) == 1
        assert listing[0]["pipeline_display"] == "reverse"

    def test_nonexistent_experiment(self, store):
        assert store.get_experiment(999) is None


# =============================================================================
# 2. BATCH RUNNER
# =============================================================================

@pytest.mark.integration
class TestBatchRunner:

    def test_exhaustive_run(self, batch, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
        exp_id, result = batch.run_exhaustive(pipe, domain)
        assert exp_id >= 1
        assert 6174 in result.fixed_points

    def test_sampled_run(self, batch, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
        exp_id, result = batch.run_sampled(pipe, domain, sample_size=500, seed=42)
        assert exp_id >= 1
        assert result.num_startpoints == 500
        assert 6174 in result.fixed_points

    def test_batch_run(self, batch, reg):
        pipes = [
            Pipeline.parse("kaprekar_step", registry=reg),
            Pipeline.parse("digit_sum", registry=reg),
        ]
        doms = [DomainPolicy(base=10, digit_length=3)]
        results = batch.run_batch(pipes, doms)
        assert len(results) == 2
        assert batch.store.count() == 2

    def test_batch_stores_features(self, batch, reg):
        pipe = Pipeline.parse("kaprekar_step |> digit_sum", registry=reg)
        domain = DomainPolicy(base=10, digit_length=3)
        exp_id, _ = batch.run_exhaustive(pipe, domain)

        cur = batch.store.conn.execute(
            "SELECT * FROM pipeline_features WHERE experiment_id = ?", (exp_id,)
        )
        row = cur.fetchone()
        assert row is not None
        # num_ops = 2
        assert row[1] == 2


# =============================================================================
# 3. DETERMINISM
# =============================================================================

@pytest.mark.integration
class TestDeterminism:

    def test_sampled_same_seed(self, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
        sr = SampledRunner(reg)
        r1, _ = sr.run_sampled(pipe, domain, sample_size=200, seed=123)
        r2, _ = sr.run_sampled(pipe, domain, sample_size=200, seed=123)
        assert r1.fixed_points == r2.fixed_points
        assert r1.num_startpoints == r2.num_startpoints
        assert canonical_float(r1.convergence_rate) == canonical_float(r2.convergence_rate)

    def test_sampled_different_seed(self, reg):
        pipe = Pipeline.parse("truc_1089", registry=reg)
        domain = DomainPolicy(base=10, digit_length=5)
        sr = SampledRunner(reg)
        r1, _ = sr.run_sampled(pipe, domain, sample_size=200, seed=1)
        r2, _ = sr.run_sampled(pipe, domain, sample_size=200, seed=2)
        # Different seeds may produce different results (not guaranteed, but likely)
        # At minimum, both should have valid structure
        assert r1.num_startpoints == 200
        assert r2.num_startpoints == 200

    def test_exhaustive_deterministic(self, reg):
        pipe = Pipeline.parse("digit_sum", registry=reg)
        domain = DomainPolicy(base=10, digit_length=2)
        runner = PipelineRunner(reg)
        r1 = runner.run_exhaustive(pipe, domain)
        r2 = runner.run_exhaustive(pipe, domain)
        assert r1.sha256 == r2.sha256


# =============================================================================
# 4. EXPORT
# =============================================================================

@pytest.mark.integration
class TestExport:

    def test_json_export(self, batch, reg, tmp_db):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=3)
        batch.run_exhaustive(pipe, domain)

        export_path = tmp_db + ".export.json"
        try:
            n = batch.store.export_json(export_path)
            assert n == 1
            with open(export_path) as f:
                data = json.load(f)
            assert "experiments" in data
            assert len(data["experiments"]) == 1
            assert "fixed_points" in data["experiments"][0]
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)

    def test_paper_appendix_export(self, batch, reg, tmp_db):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
        batch.run_exhaustive(pipe, domain)

        appendix_path = tmp_db + ".appendix.json"
        try:
            n = batch.store.export_paper_appendix(appendix_path)
            assert n == 1
            with open(appendix_path) as f:
                data = json.load(f)
            assert "op_registry_hash" in data
            assert len(data["results"]) == 1
            assert "result_hash" in data["results"][0]
        finally:
            if os.path.exists(appendix_path):
                os.remove(appendix_path)


# =============================================================================
# 5. FEATURE EXTRACTION
# =============================================================================

@pytest.mark.unit
class TestFeatureExtraction:

    def test_kaprekar_features(self, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        feats = extract_pipeline_features(pipe, reg)
        num_ops, has_c, has_e, has_p, ds_sig, sem_sig, mod9, has_comp, has_kap = feats
        assert num_ops == 1
        assert has_kap == 1

    def test_composed_features(self, reg):
        pipe = Pipeline.parse("complement_9 |> reverse |> digit_sum", registry=reg)
        feats = extract_pipeline_features(pipe, reg)
        num_ops, has_c, has_e, has_p, ds_sig, sem_sig, mod9, has_comp, has_kap = feats
        assert num_ops == 3
        assert has_comp == 1
        assert has_p == 1  # reverse is permutation
        assert has_c == 1  # digit_sum is contractive

    def test_permutation_only(self, reg):
        pipe = Pipeline.parse("reverse |> sort_asc |> rotate_left", registry=reg)
        feats = extract_pipeline_features(pipe, reg)
        num_ops, has_c, has_e, has_p, ds_sig, sem_sig, mod9, has_comp, has_kap = feats
        assert num_ops == 3
        assert has_p == 1
        assert has_c == 0
        assert has_e == 0

    def test_semantic_signature(self, reg):
        pipe = Pipeline.parse("kaprekar_step |> digit_sum", registry=reg)
        feats = extract_pipeline_features(pipe, reg)
        sem_sig = feats[5]
        assert "subtractive" in sem_sig
        assert "aggregate" in sem_sig


# =============================================================================
# 6. SCHEMA INTEGRITY
# =============================================================================

@pytest.mark.integration
class TestSchemaIntegrity:

    def test_schema_version_stored(self, store):
        cur = store.conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "1.0"

    def test_foreign_keys_enabled(self, store):
        cur = store.conn.execute("PRAGMA foreign_keys")
        assert cur.fetchone()[0] == 1

    def test_indexes_exist(self, store):
        cur = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_exp_%'"
        )
        indexes = {r[0] for r in cur.fetchall()}
        assert "idx_exp_pipeline" in indexes
        assert "idx_exp_domain" in indexes
        assert "idx_exp_result" in indexes

    def test_basin_fractions_stored(self, store, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
        runner = PipelineRunner(reg)
        result = runner.run_exhaustive(pipe, domain)
        exp_id = store.store(result, pipeline=pipe, domain=domain, registry=reg)

        cur = store.conn.execute(
            "SELECT COUNT(*) FROM basin_fractions WHERE experiment_id = ?", (exp_id,)
        )
        assert cur.fetchone()[0] > 0

    def test_witnesses_stored(self, store, reg):
        pipe = Pipeline.parse("digit_sum", registry=reg)
        domain = DomainPolicy(base=10, digit_length=2)
        runner = PipelineRunner(reg)
        result = runner.run_exhaustive(pipe, domain)
        exp_id = store.store(result, pipeline=pipe, domain=domain, registry=reg)

        cur = store.conn.execute(
            "SELECT COUNT(*) FROM witnesses WHERE experiment_id = ?", (exp_id,)
        )
        assert cur.fetchone()[0] > 0


# =============================================================================
# 7. SUITES
# =============================================================================

@pytest.mark.unit
class TestSuites:

    def test_kaprekar_suite_produces_valid(self, reg):
        pipes, doms = kaprekar_suite()
        assert len(pipes) == 1
        assert len(doms) == 5
        assert pipes[0].display() == "kaprekar_step"
        assert doms[0].digit_length == 3
        assert doms[-1].digit_length == 7

    def test_paper_b_suite_produces_valid(self, reg):
        pipes, doms = paper_b_suite()
        assert len(pipes) == 4
        assert len(doms) == 1
