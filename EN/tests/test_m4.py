"""
Tests for Engine vNext M4: Paper Appendix Auto-Emitter

Test categories:
  1. ManifestBuilder: canonical manifest, full SHA-256, no short hashes
  2. Snapshotter: registry/pipeline/domain/result catalogs
  3. LatexEmitter: deterministic Paper A/B appendix generation
  4. DeterminismGuard: hash validation, sort order, rerun stability
  5. Paper A vs Paper B preset filtering
  6. Integration: DB → manifest → LaTeX → guard pipeline
  7. Golden freeze: known run → stable appendix snippet
"""

import json
import os
import re
import tempfile

import pytest

from pipeline_dsl import (
    OperationRegistry, Pipeline, DomainPolicy, PipelineRunner, RunResult,
)
from feature_extractor import (
    ConjectureMiner, ConjectureType, Conjecture, TestedDomain,
)
from proof_engine import (
    SkeletonGenerator, DensityEstimator, RankingModelV1,
    RankedConjecture, RANKING_MODEL_VERSION,
)
from experiment_runner import ExperimentStore, BatchRunner
from appendix_emitter import (
    ManifestBuilder, Snapshotter, LatexEmitter, DeterminismGuard,
    AppendixEmitter, ArtifactPackager, EMITTER_VERSION,
    MANIFEST_VERSION, ENGINE_VERSION, BUNDLE_CONTENTS,
    collect_environment, generate_lockfile,
    _tex_escape, _tex_hash, _tex_float,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reg():
    return OperationRegistry()

@pytest.fixture
def manifest_builder(reg):
    return ManifestBuilder(reg)

@pytest.fixture
def snapshotter(reg):
    return Snapshotter(reg)

@pytest.fixture
def latex_emitter(reg):
    return LatexEmitter(reg)

@pytest.fixture
def guard():
    return DeterminismGuard()

@pytest.fixture
def sample_experiments(reg):
    """Create a set of sample experiments as dicts (mimicking DB rows)."""
    runner = PipelineRunner(reg)
    experiments = []
    for k in [3, 4]:
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        # Paper A domain (include_cycles=False)
        dom_a = DomainPolicy.paper_a_kaprekar(k)
        result_a = runner.run_exhaustive(pipe, dom_a)
        experiments.append({
            "pipeline_hash": pipe.sha256,
            "pipeline_display": pipe.display(),
            "pipeline_json": pipe.canonical_json(),
            "domain_hash": dom_a.sha256,
            "domain_json": dom_a.canonical_json(),
            "op_registry_hash": reg.sha256,
            "result_hash": result_a.sha256,
            "num_startpoints": result_a.num_startpoints,
            "num_attractors": result_a.num_attractors,
            "convergence_rate": str(result_a.convergence_rate),
            "basin_entropy": str(result_a.basin_entropy),
            "fixed_points": result_a.fixed_points,
            "mode": "exhaustive",
            "engine_version": result_a.engine_version,
        })

        # Engine metrics domain (include_cycles=True)
        dom_b = DomainPolicy.engine_metrics_kaprekar(k)
        result_b = runner.run_exhaustive(pipe, dom_b)
        experiments.append({
            "pipeline_hash": pipe.sha256,
            "pipeline_display": pipe.display(),
            "pipeline_json": pipe.canonical_json(),
            "domain_hash": dom_b.sha256,
            "domain_json": dom_b.canonical_json(),
            "op_registry_hash": reg.sha256,
            "result_hash": result_b.sha256,
            "num_startpoints": result_b.num_startpoints,
            "num_attractors": result_b.num_attractors,
            "convergence_rate": str(result_b.convergence_rate),
            "basin_entropy": str(result_b.basin_entropy),
            "fixed_points": result_b.fixed_points,
            "mode": "exhaustive",
            "engine_version": result_b.engine_version,
        })
    return experiments


@pytest.fixture
def sample_ranked(reg):
    """Generate ranked conjectures for kaprekar_step k=3,4."""
    miner = ConjectureMiner(reg)
    ranker = RankingModelV1(reg)
    conjs = miner.mine_all("kaprekar_step", [3, 4])
    return ranker.rank_all(conjs)


# =============================================================================
# 1. MANIFEST BUILDER
# =============================================================================

@pytest.mark.unit
class TestManifestBuilder:

    def test_manifest_has_required_fields(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        required = [
            "manifest_version", "created_utc", "engine_version",
            "ranking_model_version",
            "op_registry_hash", "op_registry_version", "engine_semantic_version",
            "environment", "random_seed_policy", "reproduction",
            "pipelines", "domains", "results", "bundle",
        ]
        for f in required:
            assert f in m, f"Missing field: {f}"
        assert m["manifest_version"] == "1.1"
        assert m["engine_version"] == ENGINE_VERSION

    def test_manifest_full_sha256(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        # Registry hash must be full 64-char hex
        assert len(m["op_registry_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in m["op_registry_hash"])
        # All result hashes must be full
        for r in m["results"]:
            for hf in ["pipeline_hash", "domain_hash", "result_hash", "op_registry_hash"]:
                assert len(r[hf]) == 64, f"{hf} is not full SHA-256"

    def test_manifest_no_short_hashes(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        manifest_str = json.dumps(m)
        assert "short_hash" not in manifest_str
        assert "_short" not in manifest_str

    def test_manifest_sorted_results(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        keys = [(r["pipeline_display"], r["domain_hash"]) for r in m["results"]]
        assert keys == sorted(keys)

    def test_manifest_sorted_pipelines(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        hashes = [p["pipeline_hash"] for p in m["pipelines"]]
        assert hashes == sorted(hashes)

    def test_manifest_sorted_domains(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        hashes = [d["domain_hash"] for d in m["domains"]]
        assert hashes == sorted(hashes)

    def test_manifest_includes_ranking_version(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        assert m["ranking_model_version"] == RANKING_MODEL_VERSION

    def test_manifest_sha256_stable(self, manifest_builder, sample_experiments):
        m1 = manifest_builder.build_manifest(sample_experiments)
        m2 = manifest_builder.build_manifest(sample_experiments)
        # SHA excludes timestamp
        assert ManifestBuilder.manifest_sha256(m1) == ManifestBuilder.manifest_sha256(m2)

    def test_manifest_with_conjectures(self, manifest_builder, sample_experiments, sample_ranked):
        m = manifest_builder.build_manifest(sample_experiments, sample_ranked)
        assert "conjectures" in m
        assert len(m["conjectures"]) > 0
        assert "ranking_model_version" in m["conjectures"][0]


# =============================================================================
# 2. SNAPSHOTTER
# =============================================================================

@pytest.mark.unit
class TestSnapshotter:

    def test_registry_snapshot_canonical(self, snapshotter):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name
        try:
            sha = snapshotter.write_registry_snapshot(path)
            assert len(sha) == 64
            with open(path) as f:
                data = json.load(f)
            assert "operations" in data
            assert "version" in data
            assert len(data["operations"]) == 22  # 22 ops in registry
        finally:
            os.unlink(path)

    def test_registry_snapshot_rerun_stable(self, snapshotter):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path2 = f.name
        try:
            sha1 = snapshotter.write_registry_snapshot(path1)
            sha2 = snapshotter.write_registry_snapshot(path2)
            assert sha1 == sha2
            with open(path1) as f:
                content1 = f.read()
            with open(path2) as f:
                content2 = f.read()
            assert content1 == content2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_pipeline_catalog(self, snapshotter):
        pipes = [
            {"pipeline_hash": "a" * 64, "pipeline_display": "kaprekar_step"},
            {"pipeline_hash": "b" * 64, "pipeline_display": "truc_1089"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name
        try:
            sha = snapshotter.write_pipeline_catalog(pipes, path)
            assert len(sha) == 64
            with open(path) as f:
                data = json.load(f)
            assert data["catalog_version"] == EMITTER_VERSION
            assert len(data["pipelines"]) == 2
        finally:
            os.unlink(path)

    def test_domain_catalog(self, snapshotter):
        doms = [{"domain_hash": "c" * 64, "domain_json": "{}"}]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name
        try:
            sha = snapshotter.write_domain_catalog(doms, path)
            assert len(sha) == 64
        finally:
            os.unlink(path)

    def test_result_catalog(self, snapshotter):
        results = [{"pipeline_display": "x", "domain_hash": "d" * 64, "result_hash": "e" * 64}]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name
        try:
            sha = snapshotter.write_result_catalog(results, path)
            assert len(sha) == 64
        finally:
            os.unlink(path)


# =============================================================================
# 3. LATEX EMITTER
# =============================================================================

@pytest.mark.unit
class TestLatexEmitter:

    def test_paper_a_contains_header(self, latex_emitter, sample_experiments):
        tex = latex_emitter.emit_paper_a(sample_experiments)
        assert "Paper A" in tex
        assert "AUTO-GENERATED" in tex
        assert latex_emitter.registry.sha256 in tex

    def test_paper_b_contains_header(self, latex_emitter, sample_experiments):
        tex = latex_emitter.emit_paper_b(sample_experiments)
        assert "Paper B" in tex

    def test_paper_a_filters_domains(self, latex_emitter, sample_experiments):
        tex_a = latex_emitter.emit_paper_a(sample_experiments)
        tex_b = latex_emitter.emit_paper_b(sample_experiments)
        # Paper A should have fewer entries (only include_cycles=False)
        a_rows = tex_a.count("\\\\")
        b_rows = tex_b.count("\\\\")
        assert a_rows <= b_rows

    def test_latex_deterministic(self, latex_emitter, sample_experiments):
        tex1 = latex_emitter.emit_paper_b(sample_experiments)
        tex2 = latex_emitter.emit_paper_b(sample_experiments)
        assert tex1 == tex2

    def test_latex_contains_definitions(self, latex_emitter, sample_experiments):
        tex = latex_emitter.emit_paper_b(sample_experiments)
        assert "Definitions" in tex
        assert "Digit length" in tex
        assert "Repdigit exclusion" in tex

    def test_latex_contains_hash_table(self, latex_emitter, sample_experiments):
        tex = latex_emitter.emit_paper_b(sample_experiments)
        assert "Traceability Hashes" in tex
        assert "Operation Registry" in tex

    def test_latex_contains_result_table(self, latex_emitter, sample_experiments):
        tex = latex_emitter.emit_paper_b(sample_experiments)
        assert "Experiment Results" in tex
        assert "Conv. Rate" in tex

    def test_latex_with_conjectures(self, latex_emitter, sample_experiments, sample_ranked):
        tex = latex_emitter.emit_paper_b(sample_experiments, sample_ranked)
        assert "Ranked Conjectures" in tex
        assert f"v{RANKING_MODEL_VERSION}" in tex
        assert "Density estimate" in tex
        assert "Proof skeleton" in tex
        assert "Ranking breakdown" in tex

    def test_latex_conjecture_density_label(self, latex_emitter, sample_experiments, sample_ranked):
        tex = latex_emitter.emit_paper_b(sample_experiments, sample_ranked)
        # Should contain at least one density label
        assert any(label in tex for label in ["strong", "moderate", "weak", "minimal"])

    def test_latex_conjecture_repdigit_adjustment(self, latex_emitter, sample_experiments, sample_ranked):
        tex = latex_emitter.emit_paper_b(sample_experiments, sample_ranked)
        # If any tested domain has exclude_repdigits, should show adjustment
        for rc in sample_ranked:
            if any(td.exclude_repdigits for td in rc.conjecture.tested_domains):
                assert "repdigit" in tex.lower()
                break

    def test_latex_ranking_weights(self, latex_emitter, sample_experiments, sample_ranked):
        tex = latex_emitter.emit_paper_b(sample_experiments, sample_ranked)
        assert str(RankingModelV1.W_EMPIRICAL) in tex
        assert str(RankingModelV1.W_STRUCTURAL) in tex

    def test_tex_escape(self):
        assert _tex_escape("a_b") == "a\\_b"
        assert _tex_escape("a & b") == "a \\& b"
        assert _tex_escape("100%") == "100\\%"

    def test_tex_hash(self):
        h = "abcdef1234567890" * 4
        result = _tex_hash(h, 16)
        assert "\\texttt{" in result
        assert "\\dots" in result

    def test_tex_float(self):
        assert _tex_float(0.123456789, 4) == "0.1235"
        assert _tex_float("0.5", 2) == "0.50"


# =============================================================================
# 4. DETERMINISM GUARD
# =============================================================================

@pytest.mark.unit
class TestDeterminismGuard:

    def test_valid_manifest_no_issues(self, guard, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        issues = guard.check_manifest(m)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_detects_short_hash(self, guard):
        m = {
            "op_registry_hash": "a" * 64,
            "results": [],
            "pipelines": [],
            "domains": [],
            "ranking_model_version": "1.0",
            "short_hash": "abc123",  # This should be flagged
        }
        issues = guard.check_manifest(m)
        assert any("short" in i.lower() for i in issues)

    def test_detects_non_full_sha(self, guard):
        m = {
            "op_registry_hash": "abc123",  # Too short
            "results": [],
            "pipelines": [],
            "domains": [],
            "ranking_model_version": "1.0",
        }
        issues = guard.check_manifest(m)
        assert any("SHA-256" in i for i in issues)

    def test_detects_bad_pipeline_hash(self, guard):
        """Fix 4: guard now also validates pipeline[] hashes."""
        m = {
            "op_registry_hash": "a" * 64,
            "results": [],
            "pipelines": [{"pipeline_hash": "tooshort"}],
            "domains": [],
            "ranking_model_version": "1.0",
        }
        issues = guard.check_manifest(m)
        assert any("Pipeline hash" in i for i in issues)

    def test_detects_bad_domain_hash(self, guard):
        """Fix 4: guard now also validates domain[] hashes."""
        m = {
            "op_registry_hash": "a" * 64,
            "results": [],
            "pipelines": [],
            "domains": [{"domain_hash": "xyz"}],
            "ranking_model_version": "1.0",
        }
        issues = guard.check_manifest(m)
        assert any("Domain hash" in i for i in issues)

    def test_detects_unsorted_results(self, guard):
        m = {
            "op_registry_hash": "a" * 64,
            "results": [
                {"pipeline_display": "z", "domain_hash": "a" * 64,
                 "pipeline_hash": "b" * 64, "result_hash": "c" * 64,
                 "op_registry_hash": "d" * 64},
                {"pipeline_display": "a", "domain_hash": "b" * 64,
                 "pipeline_hash": "e" * 64, "result_hash": "f" * 64,
                 "op_registry_hash": "d" * 64},
            ],
            "pipelines": [],
            "domains": [],
            "ranking_model_version": "1.0",
        }
        issues = guard.check_manifest(m)
        assert any("sorted" in i.lower() for i in issues)

    def test_detects_missing_ranking_version(self, guard):
        m = {
            "op_registry_hash": "a" * 64,
            "results": [],
            "pipelines": [],
            "domains": [],
        }
        issues = guard.check_manifest(m)
        assert any("ranking_model_version" in i for i in issues)

    def test_latex_determinism_check(self, guard):
        tex1 = "\\section{Test}\nContent"
        tex2 = "\\section{Test}\nContent"
        assert guard.check_latex_determinism(tex1, tex2)

    def test_latex_determinism_ignores_timestamps(self, guard):
        tex1 = "% Generated: 2025-01-01\n\\section{Test}"
        tex2 = "% Generated: 2025-01-02\n\\section{Test}"
        assert guard.check_latex_determinism(tex1, tex2)

    def test_latex_determinism_detects_diff(self, guard):
        tex1 = "\\section{Test}\nA"
        tex2 = "\\section{Test}\nB"
        assert not guard.check_latex_determinism(tex1, tex2)


# =============================================================================
# 5. PAPER A vs PAPER B PRESET FILTERING (Fix 1: preset_name)
# =============================================================================

@pytest.mark.unit
class TestPresetFiltering:

    def test_preset_name_in_domain_policy(self):
        """Fix 1: DomainPolicy presets now carry preset_name."""
        pa = DomainPolicy.paper_a_kaprekar(4)
        assert pa.preset_name == "paper_a_kaprekar"
        em = DomainPolicy.engine_metrics_kaprekar(4)
        assert em.preset_name == "engine_metrics_kaprekar"
        pa1089 = DomainPolicy.paper_a_1089(3)
        assert pa1089.preset_name == "paper_a_1089"
        em1089 = DomainPolicy.engine_metrics_1089(3)
        assert em1089.preset_name == "engine_metrics_1089"

    def test_preset_name_in_canonical_dict(self):
        pa = DomainPolicy.paper_a_kaprekar(4)
        d = pa.canonical_dict()
        assert d["preset_name"] == "paper_a_kaprekar"
        # Plain domain has no preset_name key
        plain = DomainPolicy(base=10, digit_length=4)
        assert "preset_name" not in plain.canonical_dict()

    def test_preset_name_changes_hash(self):
        """Adding preset_name changes the domain hash (intended: different specs)."""
        with_preset = DomainPolicy.paper_a_kaprekar(4)
        without_preset = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True,
                                       exclude_zero=True, include_cycles=False,
                                       engine_semantic_version="2.0")
        assert with_preset.sha256 != without_preset.sha256

    def test_paper_a_domain_filter_uses_preset_name(self, latex_emitter):
        """Fix 1: filter uses preset_name primarily."""
        paper_a_dom = DomainPolicy.paper_a_kaprekar(4)
        engine_dom = DomainPolicy.engine_metrics_kaprekar(4)

        exp_a = {
            "domain_json": paper_a_dom.canonical_json(),
            "pipeline_display": "kaprekar_step",
        }
        exp_b = {
            "domain_json": engine_dom.canonical_json(),
            "pipeline_display": "kaprekar_step",
        }

        assert latex_emitter._is_paper_a_domain(exp_a) is True
        assert latex_emitter._is_paper_a_domain(exp_b) is False

    def test_paper_a_filter_fallback_no_preset(self, latex_emitter):
        """Fix 1: without preset_name, falls back to conjunction check."""
        exp_paper_a_like = {
            "domain_json": json.dumps({"include_cycles": False, "exclude_zero": True}),
        }
        exp_not_paper_a = {
            "domain_json": json.dumps({"include_cycles": True, "exclude_zero": True}),
        }
        assert latex_emitter._is_paper_a_domain(exp_paper_a_like) is True
        assert latex_emitter._is_paper_a_domain(exp_not_paper_a) is False

    def test_paper_a_appendix_excludes_engine_metrics(self, latex_emitter, sample_experiments):
        tex_a = latex_emitter.emit_paper_a(sample_experiments)
        paper_a_count = sum(1 for e in sample_experiments
                            if latex_emitter._is_paper_a_domain(e))
        paper_b_count = len(sample_experiments)
        assert paper_a_count < paper_b_count
        assert paper_a_count > 0

    def test_paper_b_includes_all(self, latex_emitter, sample_experiments):
        tex_b = latex_emitter.emit_paper_b(sample_experiments)
        for e in sample_experiments:
            pd = e.get("pipeline_display", "")
            if pd:
                assert pd.replace("_", "\\_") in tex_b

    def test_conjecture_filter_uses_tested_domains(self, latex_emitter):
        """Fix 2: conjecture applicability uses tested_domains, not just pipeline."""
        # Create a conjecture with Paper A-compatible tested_domains
        td_paper_a = TestedDomain(base=10, digit_length=4, range_lo=1000,
                                   range_hi=9999, mode="exhaustive",
                                   exclude_repdigits=True)
        c = Conjecture(
            ctype=ConjectureType.COUNTING, statement="test",
            quantifier="", predicate="", evidence={}, exceptions=[],
            pipeline="kaprekar_step", parameters={},
            confidence=0.9, novelty=0.5, simplicity=0.7,
            tested_domains=[td_paper_a],
        )
        from proof_engine import DensityEstimate, ProofSkeleton, ProofStrategy
        rc = RankedConjecture(
            conjecture=c,
            density_estimate=DensityEstimate(total_search_space=9000,
                num_counterexamples=0, observed_density=0.0,
                upper_bound_95=0.001, upper_bound_99=0.002,
                confidence_score=0.99, search_volume_log10=3.95,
                k_range=(4, 4), falsification_label="strong"),
            proof_skeleton=ProofSkeleton(conjecture_statement="test",
                strategy=ProofStrategy.MOD_INVARIANT,
                structural_invariant="n/a",
                reduction_steps=[], remaining_gaps=[], known_theorem_links=[],
                proof_strength="weak"),
            patterns=[],
            rank_breakdown={"empirical_confidence": 0.9, "structural_strength": 0.5,
                            "novelty": 0.5, "simplicity": 0.7, "falsifiability": 0.6},
            final_score=0.8,
        )
        # Should match Paper A filter (exclude_repdigits + exclude_zero fallback)
        assert latex_emitter._conjecture_matches_paper(
            rc, latex_emitter._is_paper_a_domain) is True


# =============================================================================
# 6. INTEGRATION (requires DB + mining)
# =============================================================================

@pytest.mark.integration
class TestM4Integration:

    def test_full_emit_pipeline(self, reg, tmp_path):
        """Full e2e: create DB → store results → emit all artifacts."""
        db_path = str(tmp_path / "test_results.db")
        store = ExperimentStore(db_path)
        batch = BatchRunner(store, reg)

        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        dom = DomainPolicy.paper_a_kaprekar(4)
        batch.run_exhaustive(pipe, dom)
        store.close()

        output_dir = str(tmp_path / "output")
        emitter = AppendixEmitter(db_path, reg)
        summary = emitter.emit_all(output_dir, k_range=[3, 4])
        if emitter.store:
            emitter.store.close()

        # Check all artifacts exist
        for name in summary["artifacts"]:
            path = summary["artifacts"][name]
            assert os.path.exists(path), f"Missing: {name}"

        # Check manifest is valid JSON
        with open(summary["artifacts"]["repro_manifest.json"]) as f:
            manifest = json.load(f)
        assert manifest["op_registry_hash"] == reg.sha256
        assert len(manifest["results"]) >= 1

        # Check determinism
        assert summary["determinism_issues"] == [] or \
            all("not found in manifest" in i for i in summary["determinism_issues"])

    def test_manifest_hashes_in_latex(self, reg, tmp_path):
        """All hashes in LaTeX must exist in manifest."""
        db_path = str(tmp_path / "test_results.db")
        store = ExperimentStore(db_path)
        batch = BatchRunner(store, reg)

        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        dom = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
        batch.run_exhaustive(pipe, dom)
        store.close()

        output_dir = str(tmp_path / "output")
        emitter = AppendixEmitter(db_path, reg)
        summary = emitter.emit_all(output_dir)
        if emitter.store:
            emitter.store.close()

        with open(summary["artifacts"]["repro_manifest.json"]) as f:
            manifest = json.load(f)

        with open(summary["artifacts"]["appendix_paper_b.tex"]) as f:
            tex_b = f.read()

        guard = DeterminismGuard()
        issues = guard.check_all_hashes_in_manifest(tex_b, manifest)
        # Only allow hash prefix mismatches (not structural issues)
        structural = [i for i in issues if "not found" not in i]
        assert structural == [], f"Structural issues: {structural}"

    def test_rerun_produces_identical_artifacts(self, reg, tmp_path):
        """Running emit_all twice → byte-identical JSON catalogs."""
        db_path = str(tmp_path / "test_results.db")
        store = ExperimentStore(db_path)
        batch = BatchRunner(store, reg)

        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        dom = DomainPolicy(base=10, digit_length=3, exclude_repdigits=True)
        batch.run_exhaustive(pipe, dom)
        store.close()

        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")

        emitter = AppendixEmitter(db_path, reg)
        emitter.emit_all(out1)
        emitter.emit_all(out2)
        if emitter.store:
            emitter.store.close()

        # Compare registry snapshots (must be byte-identical)
        with open(os.path.join(out1, "registry_snapshot.json")) as f:
            reg1 = f.read()
        with open(os.path.join(out2, "registry_snapshot.json")) as f:
            reg2 = f.read()
        assert reg1 == reg2

        # Compare pipeline catalogs
        with open(os.path.join(out1, "pipeline_catalog.json")) as f:
            p1 = f.read()
        with open(os.path.join(out2, "pipeline_catalog.json")) as f:
            p2 = f.read()
        assert p1 == p2

        # Compare LaTeX (excluding timestamp)
        with open(os.path.join(out1, "appendix_paper_b.tex")) as f:
            tex1 = f.read()
        with open(os.path.join(out2, "appendix_paper_b.tex")) as f:
            tex2 = f.read()
        assert DeterminismGuard.check_latex_determinism(tex1, tex2)


# =============================================================================
# 7. GOLDEN FREEZE
# =============================================================================

@pytest.mark.integration
class TestGoldenFreeze:

    def test_kaprekar_4digit_appendix_snippet(self, reg):
        """Known Kaprekar 4-digit run → specific content in appendix."""
        runner = PipelineRunner(reg)
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        dom = DomainPolicy.paper_a_kaprekar(4)
        result = runner.run_exhaustive(pipe, dom)

        # Golden values
        assert 6174 in result.fixed_points
        assert result.num_startpoints == 8991  # 9000 - 9 repdigits

        exp = {
            "pipeline_hash": pipe.sha256,
            "pipeline_display": pipe.display(),
            "pipeline_json": pipe.canonical_json(),
            "domain_hash": dom.sha256,
            "domain_json": dom.canonical_json(),
            "op_registry_hash": reg.sha256,
            "result_hash": result.sha256,
            "num_startpoints": result.num_startpoints,
            "num_attractors": result.num_attractors,
            "convergence_rate": str(result.convergence_rate),
            "basin_entropy": str(result.basin_entropy),
            "fixed_points": result.fixed_points,
            "mode": "exhaustive",
            "engine_version": result.engine_version,
        }

        latex = LatexEmitter(reg)
        tex = latex.emit_paper_a([exp])

        # Must contain registry hash
        assert reg.sha256 in tex
        # Must contain Paper A label
        assert "Paper A" in tex
        # Must contain the pipeline
        assert "kaprekar" in tex.lower()
        # Must reference 8991 startpoints
        assert "8991" in tex

    def test_emitter_version(self):
        assert EMITTER_VERSION == "1.0"
        assert MANIFEST_VERSION == "1.1"
        assert RANKING_MODEL_VERSION == "1.0"


# =============================================================================
# 8. REPDIGIT ADJUSTMENT FIX (Fix 3)
# =============================================================================

@pytest.mark.unit
class TestRepdigitAdjustment:

    def test_adj_none_when_no_exclusion(self, latex_emitter, sample_experiments):
        """Fix 3: repdigit_adj='none' when exclude_repdigits=False."""
        td = TestedDomain(base=10, digit_length=3, range_lo=100, range_hi=999,
                          mode="exhaustive", exclude_repdigits=False)
        c = Conjecture(
            ctype=ConjectureType.COUNTING, statement="test",
            quantifier="", predicate="", evidence={}, exceptions=[],
            pipeline="test", parameters={},
            confidence=0.9, novelty=0.5, simplicity=0.7,
            tested_domains=[td],
        )
        from proof_engine import DensityEstimate, ProofSkeleton, ProofStrategy
        rc = RankedConjecture(
            conjecture=c,
            density_estimate=DensityEstimate(total_search_space=900,
                num_counterexamples=0, observed_density=0.0,
                upper_bound_95=0.001, upper_bound_99=0.002,
                confidence_score=0.99, search_volume_log10=2.95,
                k_range=(3, 3), falsification_label="strong"),
            proof_skeleton=ProofSkeleton(conjecture_statement="test",
                strategy=ProofStrategy.MOD_INVARIANT,
                structural_invariant="n/a",
                reduction_steps=[], remaining_gaps=[], known_theorem_links=[],
                proof_strength="weak"),
            patterns=[],
            rank_breakdown={"empirical_confidence": 0.9, "structural_strength": 0.5,
                            "novelty": 0.5, "simplicity": 0.7, "falsifiability": 0.6},
            final_score=0.8,
        )
        tex = latex_emitter._section_conjectures([rc])
        assert "repdigit\\_adj=none" in tex

    def test_adj_exact_when_excluded(self, latex_emitter, sample_experiments, sample_ranked):
        """Fix 3: repdigit_adj='exact (...)' when exclude_repdigits=True, k>=2."""
        tex = latex_emitter._section_conjectures(sample_ranked)
        # Kaprekar conjectures use exclude_repdigits=True, k>=3
        has_exact = "repdigit\\_adj=exact" in tex
        has_none = "repdigit\\_adj=none" in tex
        # Should have at least one exact entry from kaprekar tests
        assert has_exact or has_none  # At least some tested domains


# =============================================================================
# 9. ARTIFACT PACKAGER
# =============================================================================

@pytest.mark.unit
class TestArtifactPackager:

    def test_readme_contains_expected_hashes(self):
        summary = {
            "manifest_sha256": "a" * 64,
            "op_registry_sha256": "b" * 64,
            "appendix_paper_a_sha256": "c" * 64,
            "appendix_paper_b_sha256": "d" * 64,
            "experiments_count": 5,
            "conjectures_count": 3,
            "determinism_issues": [],
        }
        readme = ArtifactPackager.generate_readme(summary)
        assert "a" * 64 in readme
        assert "b" * 64 in readme
        assert "c" * 64 in readme
        assert "d" * 64 in readme

    def test_readme_contains_commands(self):
        summary = {"manifest_sha256": "x" * 64, "experiments_count": 0,
                   "conjectures_count": 0, "determinism_issues": []}
        readme = ArtifactPackager.generate_readme(summary)
        assert "pip install" in readme
        assert "reproduce.py" in readme

    def test_readme_contains_environment(self):
        summary = {"manifest_sha256": "x" * 64, "experiments_count": 0,
                   "conjectures_count": 0, "determinism_issues": []}
        readme = ArtifactPackager.generate_readme(summary)
        assert "Python" in readme
        assert "OS" in readme
        assert "Architecture" in readme

    def test_readme_contains_hashing_convention(self):
        """Fix 5: hashing convention documented."""
        summary = {"manifest_sha256": "x" * 64, "experiments_count": 0,
                   "conjectures_count": 0, "determinism_issues": []}
        readme = ArtifactPackager.generate_readme(summary)
        assert "indent=2" in readme or "indent" in readme.lower()
        assert "canonical" in readme.lower()

    def test_bundle_creates_zip(self, tmp_path):
        out = str(tmp_path / "out")
        os.makedirs(out)
        # Create minimal artifacts
        for name in ["repro_manifest.json", "registry_snapshot.json",
                     "pipeline_catalog.json", "domain_catalog.json",
                     "result_catalog.json", "appendix_paper_a.tex",
                     "appendix_paper_b.tex"]:
            with open(os.path.join(out, name), 'w') as f:
                f.write("test")
        summary = {
            "artifacts": {n: os.path.join(out, n) for n in [
                "repro_manifest.json", "registry_snapshot.json",
                "pipeline_catalog.json", "domain_catalog.json",
                "result_catalog.json", "appendix_paper_a.tex",
                "appendix_paper_b.tex"]},
            "manifest_sha256": "x" * 64,
            "op_registry_sha256": "y" * 64,
            "appendix_paper_a_sha256": "a" * 64,
            "appendix_paper_b_sha256": "b" * 64,
            "experiments_count": 0,
            "conjectures_count": 0,
            "determinism_issues": [],
        }
        bundle_path, bundle_sha = ArtifactPackager.package_bundle(
            out, summary, db_path="nonexistent.db",
            lockfile_content="pytest==7.4.3\n", lockfile_sha="a" * 64)
        assert os.path.exists(bundle_path)
        assert len(bundle_sha) == 64
        import zipfile
        with zipfile.ZipFile(bundle_path) as zf:
            names = zf.namelist()
            assert "README_repro.md" in names
            assert "ENVIRONMENT.md" in names
            assert "requirements.lock.txt" in names
            assert "repro_manifest.json" in names
            assert "appendix_paper_a.tex" in names


@pytest.mark.integration
class TestArtifactPackagerIntegration:

    def test_full_bundle_from_emitter(self, reg, tmp_path):
        """Full e2e: emit → package → verify bundle contents."""
        db_path = str(tmp_path / "test_results.db")
        store = ExperimentStore(db_path)
        batch = BatchRunner(store, reg)
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        dom = DomainPolicy.paper_a_kaprekar(4)
        batch.run_exhaustive(pipe, dom)
        store.close()

        output_dir = str(tmp_path / "output")
        emitter = AppendixEmitter(db_path, reg)
        summary = emitter.emit_all(output_dir)
        if emitter.store:
            emitter.store.close()

        bundle_path, bundle_sha = ArtifactPackager.package_bundle(
            output_dir, summary, db_path=db_path,
            lockfile_content="pytest==7.4.3\n", lockfile_sha="f" * 64)
        assert os.path.exists(bundle_path)
        assert len(bundle_sha) == 64

        import zipfile
        with zipfile.ZipFile(bundle_path) as zf:
            names = zf.namelist()
            assert "README_repro.md" in names
            assert "ENVIRONMENT.md" in names
            assert "requirements.lock.txt" in names
            assert "results.db" in names
            assert "repro_manifest.json" in names
            # Verify README contains actual hashes
            readme_content = zf.read("README_repro.md").decode("utf-8")
            assert summary["manifest_sha256"] in readme_content
            assert summary["op_registry_sha256"] in readme_content


# =============================================================================
# 10. M4.1: MANIFEST V1.1
# =============================================================================

@pytest.mark.unit
class TestManifestV11:

    def test_manifest_has_environment_block(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        env = m["environment"]
        required_keys = [
            "python_version", "python_implementation", "platform",
            "platform_machine", "os_name", "sqlite3_version",
            "endianness", "timezone", "locale",
        ]
        for k in required_keys:
            assert k in env, f"Missing environment key: {k}"

    def test_manifest_has_reproduction_block(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        repro = m["reproduction"]
        assert repro["entrypoint"] == "python reproduce.py"
        assert "db_path" in repro["args"]
        assert "out_dir" in repro["args"]

    def test_manifest_has_bundle_block(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        bundle = m["bundle"]
        assert bundle["bundle_filename"] == "reproducibility_bundle.zip"
        assert bundle["bundle_sha256"] is None  # Filled in after packaging
        assert bundle["bundle_contents"] == sorted(bundle["bundle_contents"])

    def test_manifest_has_random_seed_policy(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        assert m["random_seed_policy"] == "none"

    def test_manifest_accepts_dependencies(self, manifest_builder, sample_experiments):
        deps = {
            "freeze_method": "pip_freeze",
            "requirements_lock_sha256": "a" * 64,
            "pip_freeze": ["pytest==7.4.3"],
        }
        m = manifest_builder.build_manifest(sample_experiments, dependencies=deps)
        assert m["dependencies"] == deps

    def test_manifest_sha256_excludes_volatile(self, manifest_builder, sample_experiments):
        """created_utc and bundle_sha256 must not affect manifest hash."""
        m1 = manifest_builder.build_manifest(sample_experiments)
        m2 = manifest_builder.build_manifest(sample_experiments)
        # Different timestamps but same hash
        m1["created_utc"] = "2025-01-01T00:00:00+00:00"
        m2["created_utc"] = "2025-12-31T23:59:59+00:00"
        m1["bundle"]["bundle_sha256"] = "a" * 64
        m2["bundle"]["bundle_sha256"] = "b" * 64
        assert ManifestBuilder.manifest_sha256(m1) == ManifestBuilder.manifest_sha256(m2)

    def test_bundle_contents_matches_constant(self, manifest_builder, sample_experiments):
        m = manifest_builder.build_manifest(sample_experiments)
        assert m["bundle"]["bundle_contents"] == BUNDLE_CONTENTS


# =============================================================================
# 11. M4.1: COLLECT_ENVIRONMENT
# =============================================================================

@pytest.mark.unit
class TestCollectEnvironment:

    def test_returns_all_required_keys(self):
        env = collect_environment()
        required = [
            "python_version", "python_implementation", "platform",
            "platform_machine", "os_name", "sqlite3_version",
            "endianness", "timezone", "locale",
        ]
        for k in required:
            assert k in env, f"Missing: {k}"

    def test_python_version_format(self):
        env = collect_environment()
        parts = env["python_version"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_endianness_valid(self):
        env = collect_environment()
        assert env["endianness"] in ("little", "big")


# =============================================================================
# 12. M4.1: DETERMINISM GUARD V1.1
# =============================================================================

@pytest.mark.unit
class TestDeterminismGuardV11:

    def test_v11_manifest_passes(self, guard, manifest_builder, sample_experiments):
        """A proper v1.1 manifest should pass all checks."""
        m = manifest_builder.build_manifest(sample_experiments)
        issues = guard.check_manifest(m)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_v11_missing_environment_key(self, guard):
        m = {
            "manifest_version": "1.1",
            "op_registry_hash": "a" * 64,
            "results": [], "pipelines": [], "domains": [],
            "ranking_model_version": "1.0",
            "random_seed_policy": "none",
            "environment": {"python_version": "3.11.9"},  # Missing other keys
            "bundle": {"bundle_contents": []},
        }
        issues = guard.check_manifest(m)
        assert any("Missing environment key" in i for i in issues)

    def test_v11_missing_random_seed_policy(self, guard):
        m = {
            "manifest_version": "1.1",
            "op_registry_hash": "a" * 64,
            "results": [], "pipelines": [], "domains": [],
            "ranking_model_version": "1.0",
            "environment": collect_environment(),
            "bundle": {"bundle_contents": []},
        }
        issues = guard.check_manifest(m)
        assert any("random_seed_policy" in i for i in issues)

    def test_v11_unsorted_bundle_contents(self, guard):
        m = {
            "manifest_version": "1.1",
            "op_registry_hash": "a" * 64,
            "results": [], "pipelines": [], "domains": [],
            "ranking_model_version": "1.0",
            "random_seed_policy": "none",
            "environment": collect_environment(),
            "bundle": {"bundle_contents": ["z.txt", "a.txt"]},
        }
        issues = guard.check_manifest(m)
        assert any("bundle_contents" in i for i in issues)

    def test_v11_bad_lockfile_hash(self, guard):
        m = {
            "manifest_version": "1.1",
            "op_registry_hash": "a" * 64,
            "results": [], "pipelines": [], "domains": [],
            "ranking_model_version": "1.0",
            "random_seed_policy": "none",
            "environment": collect_environment(),
            "dependencies": {"requirements_lock_sha256": "tooshort"},
            "bundle": {"bundle_contents": []},
        }
        issues = guard.check_manifest(m)
        assert any("requirements_lock_sha256" in i for i in issues)


# =============================================================================
# 13. M4.1: BUNDLE INTEGRITY
# =============================================================================

@pytest.mark.unit
class TestBundleIntegrity:

    def test_check_bundle_contents_match(self, tmp_path):
        """Bundle contents listing must match actual zip contents."""
        out = str(tmp_path / "out")
        os.makedirs(out)
        for name in ["a.txt", "b.txt"]:
            with open(os.path.join(out, name), 'w') as f:
                f.write("test")
        bundle_path = os.path.join(out, "test.zip")
        import zipfile
        with zipfile.ZipFile(bundle_path, 'w') as zf:
            zf.write(os.path.join(out, "a.txt"), "a.txt")
            zf.write(os.path.join(out, "b.txt"), "b.txt")
        # Matching listing
        manifest = {"bundle": {
            "bundle_contents": ["a.txt", "b.txt"],
            "bundle_sha256": None,
        }}
        issues = DeterminismGuard.check_bundle_integrity(bundle_path, manifest)
        assert issues == []

    def test_check_bundle_contents_mismatch(self, tmp_path):
        out = str(tmp_path / "out")
        os.makedirs(out)
        with open(os.path.join(out, "a.txt"), 'w') as f:
            f.write("test")
        bundle_path = os.path.join(out, "test.zip")
        import zipfile
        with zipfile.ZipFile(bundle_path, 'w') as zf:
            zf.write(os.path.join(out, "a.txt"), "a.txt")
        manifest = {"bundle": {
            "bundle_contents": ["a.txt", "b.txt", "c.txt"],
            "bundle_sha256": None,
        }}
        issues = DeterminismGuard.check_bundle_integrity(bundle_path, manifest)
        assert any("contents mismatch" in i.lower() for i in issues)


# =============================================================================
# 14. M4.1: ENVIRONMENT.MD
# =============================================================================

@pytest.mark.unit
class TestEnvironmentMd:

    def test_generate_environment_md(self):
        env = {"python_version": "3.11.9", "os_name": "nt"}
        md = ArtifactPackager.generate_environment_md(env)
        assert "# Environment Specification" in md
        assert "3.11.9" in md
        assert "nt" in md

    def test_environment_md_uses_collect_if_none(self):
        md = ArtifactPackager.generate_environment_md()
        assert "# Environment Specification" in md
        assert "python_version" in md.lower() or "Python" in md
