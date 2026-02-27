# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research � MIT License
"""
ENGINE vNext — Module M4: Paper Appendix Auto-Emitter

Generates review-proof, hash-stable LaTeX appendices + JSON manifests from
results.db, M0–M3 metadata, and registry/pipeline/domain specs.

Output artifacts:
  1. appendix_paper_a.tex  — strict Paper A presets only
  2. appendix_paper_b.tex  — engine metrics domains
  3. repro_manifest.json   — canonical, full SHA-256, no short hashes
  4. registry_snapshot.json — all 22 OperationSpecs canonical
  5. pipeline_catalog.json  — all used pipelines + canonical JSON + sha256
  6. domain_catalog.json    — all used DomainPolicies/presets + sha256
  7. result_catalog.json    — RunResult hashes + key metrics

Core principles:
  - Canonical formatting: deterministic sort, fixed decimal, fixed section order
  - Traceability chain: registry → pipeline → domain → result → conjecture
  - Paper A vs Paper B domain separation enforced by DomainPolicy.preset_name
  - Ranking model version + density labels + proof skeletons embedded

Hashing convention:
  JSON catalog files are written with indent=2 for human readability.
  Content hashes are always computed on the compact canonical form
  (sort_keys=True, separators=(',',':')). File bytes may differ from
  the hash input; only the canonical no-whitespace JSON determines the hash.

Usage:
    from appendix_emitter import AppendixEmitter
    emitter = AppendixEmitter("results.db")
    emitter.emit_all("output/")
"""

from __future__ import annotations

import hashlib
import json
import locale
import os
import platform
import re
import sqlite3
import subprocess
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pipeline_dsl import (
    OperationRegistry, Pipeline, DomainPolicy, RunResult,
    PipelineRunner, canonical_float, FLOAT_PRECISION,
)
from feature_extractor import (
    Conjecture, ConjectureType, ConjectureMiner, TestedDomain,
)
from proof_engine import (
    SkeletonGenerator, DensityEstimator, PatternCompressor,
    RankingModelV1, RankedConjecture, RANKING_MODEL_VERSION,
)
from experiment_runner import ExperimentStore


# =============================================================================
# CONSTANTS
# =============================================================================

EMITTER_VERSION = "1.0"
MANIFEST_VERSION = "1.1"
ENGINE_VERSION = "vNext-M4.1"
LATEX_FLOAT_DIGITS = 6   # Decimal places in LaTeX tables

BUNDLE_CONTENTS = sorted([
    "README_repro.md",
    "ENVIRONMENT.md",
    "repro_manifest.json",
    "registry_snapshot.json",
    "pipeline_catalog.json",
    "domain_catalog.json",
    "result_catalog.json",
    "appendix_paper_a.tex",
    "appendix_paper_b.tex",
    "requirements.lock.txt",
    "reproduce.py",
    "run_experiments.py",
    "results.db",
])


def collect_environment() -> dict:
    """Collect deterministic environment metadata for manifest."""
    try:
        loc = locale.getlocale()[0] or "unknown"
    except Exception:
        loc = "unknown"
    try:
        sqlite_ver = sqlite3.sqlite_version
    except Exception:
        sqlite_ver = "unknown"
    return {
        "python_version": (f"{sys.version_info.major}."
                           f"{sys.version_info.minor}."
                           f"{sys.version_info.micro}"),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "os_name": os.name,
        "sqlite3_version": sqlite_ver,
        "endianness": sys.byteorder,
        "timezone": "UTC",
        "locale": loc,
    }


def generate_lockfile(output_path: str) -> Tuple[str, str]:
    """Generate requirements.lock.txt via pip freeze.
    Returns (content_string, sha256_of_content)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True, text=True, timeout=30,
        )
        content = result.stdout.strip() + "\n"
    except Exception:
        content = "# pip freeze unavailable\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    sha = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return content, sha


# =============================================================================
# 1. MANIFEST BUILDER
# =============================================================================

class ManifestBuilder:
    """Collect all pipelines, domains, results, and M3 metadata into a
    canonical, reproducible manifest."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()

    def build_manifest(
        self,
        experiments: List[dict],
        ranked_conjectures: Optional[List[RankedConjecture]] = None,
        dependencies: Optional[dict] = None,
    ) -> dict:
        """Build the full reproducibility manifest (canonical JSON)."""
        pipelines_seen: Dict[str, dict] = {}
        domains_seen: Dict[str, dict] = {}
        results: List[dict] = []

        for exp in sorted(experiments, key=lambda e: (
                e.get("pipeline_display", ""), e.get("domain_hash", ""))):
            pipe_hash = exp.get("pipeline_hash", "")
            dom_hash = exp.get("domain_hash", "")

            if pipe_hash and pipe_hash not in pipelines_seen:
                pipelines_seen[pipe_hash] = {
                    "pipeline_hash": pipe_hash,
                    "pipeline_display": exp.get("pipeline_display", ""),
                    "pipeline_json": exp.get("pipeline_json", "{}"),
                }

            if dom_hash and dom_hash not in domains_seen:
                domains_seen[dom_hash] = {
                    "domain_hash": dom_hash,
                    "domain_json": exp.get("domain_json", "{}"),
                }

            results.append({
                "pipeline_hash": pipe_hash,
                "pipeline_display": exp.get("pipeline_display", ""),
                "domain_hash": dom_hash,
                "op_registry_hash": exp.get("op_registry_hash", ""),
                "result_hash": exp.get("result_hash", ""),
                "num_startpoints": exp.get("num_startpoints", 0),
                "num_attractors": exp.get("num_attractors", 0),
                "convergence_rate": exp.get("convergence_rate", ""),
                "basin_entropy": exp.get("basin_entropy", ""),
                "fixed_points": exp.get("fixed_points", []),
                "mode": exp.get("mode", "exhaustive"),
                "engine_version": exp.get("engine_version", ""),
            })

        # Conjecture rankings (M3)
        conjecture_entries = []
        if ranked_conjectures:
            for rc in ranked_conjectures:
                conjecture_entries.append(rc.to_dict())

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "engine_version": ENGINE_VERSION,
            "ranking_model_version": RANKING_MODEL_VERSION,
            "op_registry_hash": self.registry.sha256,
            "op_registry_version": self.registry.VERSION,
            "engine_semantic_version": "2.0",
            "environment": collect_environment(),
            "random_seed_policy": "none",
            "reproduction": {
                "entrypoint": "python reproduce.py",
                "args": {"db_path": "results.db", "out_dir": "repro_out"},
            },
            "pipelines": [pipelines_seen[k] for k in sorted(pipelines_seen.keys())],
            "domains": [domains_seen[k] for k in sorted(domains_seen.keys())],
            "results": results,
            "conjectures": conjecture_entries,
        }
        if dependencies:
            manifest["dependencies"] = dependencies
        manifest["bundle"] = {
            "bundle_filename": "reproducibility_bundle.zip",
            "bundle_sha256": None,
            "bundle_contents": BUNDLE_CONTENTS,
        }
        return manifest

    @staticmethod
    def manifest_sha256(manifest: dict) -> str:
        """Compute SHA-256 of the manifest (excluding volatile fields).
        Excluded: created_utc (timestamp), bundle.bundle_sha256 (circular)."""
        m = dict(manifest)
        m.pop("generated", None)
        m.pop("created_utc", None)
        if "bundle" in m:
            b = dict(m["bundle"])
            b.pop("bundle_sha256", None)
            m["bundle"] = b
        canonical = json.dumps(m, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


# =============================================================================
# 2. SNAPSHOTTER
# =============================================================================

class Snapshotter:
    """Write canonical JSON catalog files for registry, pipelines, domains, results."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()

    def write_registry_snapshot(self, path: str) -> str:
        """Write all OperationSpecs as canonical JSON. Returns SHA-256."""
        data = self.registry.canonical_dict()
        data["snapshot_version"] = EMITTER_VERSION
        canonical = json.dumps(data, sort_keys=True, indent=2)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(canonical)
        return self.registry.sha256

    def write_pipeline_catalog(self, pipelines: List[dict], path: str) -> str:
        """Write pipeline catalog. Returns content SHA-256."""
        catalog = {
            "catalog_version": EMITTER_VERSION,
            "pipelines": sorted(pipelines, key=lambda p: p.get("pipeline_hash", "")),
        }
        return self._write_canonical(catalog, path)

    def write_domain_catalog(self, domains: List[dict], path: str) -> str:
        """Write domain catalog. Returns content SHA-256."""
        catalog = {
            "catalog_version": EMITTER_VERSION,
            "domains": sorted(domains, key=lambda d: d.get("domain_hash", "")),
        }
        return self._write_canonical(catalog, path)

    def write_result_catalog(self, results: List[dict], path: str) -> str:
        """Write result catalog. Returns content SHA-256."""
        catalog = {
            "catalog_version": EMITTER_VERSION,
            "results": sorted(results, key=lambda r: (
                r.get("pipeline_display", ""), r.get("domain_hash", ""))),
        }
        return self._write_canonical(catalog, path)

    @staticmethod
    def _write_canonical(data: dict, path: str) -> str:
        """Write canonical JSON and return its SHA-256."""
        canonical = json.dumps(data, sort_keys=True, indent=2)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(canonical)
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()


# =============================================================================
# 3. LATEX EMITTER
# =============================================================================

def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    for ch in ['&', '%', '$', '#', '_', '{', '}', '~', '^']:
        s = s.replace(ch, '\\' + ch)
    return s


def _tex_hash(h: str, short: int = 16) -> str:
    """Format hash for LaTeX: short prefix in monospace."""
    return f"\\texttt{{{h[:short]}\\dots}}"


def _tex_float(x, digits: int = LATEX_FLOAT_DIGITS) -> str:
    """Format float for LaTeX with fixed decimal precision."""
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x
    return format(x, f".{digits}f")


class LatexEmitter:
    """Render deterministic LaTeX appendices for Paper A and Paper B."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()

    def emit_paper_a(
        self,
        experiments: List[dict],
        ranked_conjectures: Optional[List[RankedConjecture]] = None,
        title: str = "Appendix: Reproducibility Data (Paper A)",
    ) -> str:
        """Generate Paper A appendix LaTeX source.
        Paper A: strict domains (exclude_repdigits, no cycles, Paper A presets)."""
        return self._emit_appendix(
            experiments, ranked_conjectures, title,
            paper_label="A",
            filter_fn=self._is_paper_a_domain,
        )

    def emit_paper_b(
        self,
        experiments: List[dict],
        ranked_conjectures: Optional[List[RankedConjecture]] = None,
        title: str = "Appendix: Reproducibility Data (Paper B)",
    ) -> str:
        """Generate Paper B appendix LaTeX source.
        Paper B: engine metrics domains (includes cycles, broader domain)."""
        return self._emit_appendix(
            experiments, ranked_conjectures, title,
            paper_label="B",
            filter_fn=None,  # No filter — Paper B includes all
        )

    def _emit_appendix(
        self,
        experiments: List[dict],
        ranked_conjectures: Optional[List[RankedConjecture]],
        title: str,
        paper_label: str,
        filter_fn=None,
    ) -> str:
        """Core appendix renderer."""
        if filter_fn:
            experiments = [e for e in experiments if filter_fn(e)]

        experiments = sorted(experiments, key=lambda e: (
            e.get("pipeline_display", ""), e.get("domain_hash", "")))

        sections = []
        sections.append(self._section_header(title, paper_label))
        sections.append(self._section_definitions())
        sections.append(self._section_hash_table(experiments))
        sections.append(self._section_result_table(experiments))

        if ranked_conjectures:
            applicable = ranked_conjectures
            if filter_fn:
                applicable = [rc for rc in ranked_conjectures
                              if self._conjecture_matches_paper(rc, filter_fn)]
            if applicable:
                sections.append(self._section_conjectures(applicable))

        sections.append(self._section_footer())
        return "\n\n".join(sections)

    # ── Section renderers ──

    def _section_header(self, title: str, paper_label: str) -> str:
        reg = self.registry
        return (
            f"% AUTO-GENERATED by Engine vNext M4 (v{EMITTER_VERSION})\n"
            f"% Paper {paper_label} Appendix\n"
            f"% Ranking Model v{RANKING_MODEL_VERSION}\n"
            f"% Registry SHA-256: {reg.sha256}\n"
            f"% DO NOT EDIT — regenerate with appendix_emitter.py\n"
            f"\n"
            f"\\section*{{{_tex_escape(title)}}}\n"
            f"\\label{{appendix:{paper_label.lower()}}}\n"
            f"\n"
            f"This appendix was auto-generated by Engine vNext M4.\n"
            f"All hashes are full SHA-256. Operation registry version: "
            f"\\texttt{{{reg.VERSION}}}.\n"
            f"Ranking model version: \\texttt{{{RANKING_MODEL_VERSION}}}."
        )

    def _section_definitions(self) -> str:
        lines = [
            "\\subsection*{Definitions \\& Policies}",
            "\\begin{itemize}",
            "  \\item \\textbf{Digit length $k$}: "
            "Input domain $D(k) = \\{n \\in \\mathbb{N} : 10^{k-1} \\le n < 10^k\\}$.",
            "  \\item \\textbf{Repdigit exclusion}: "
            "$D^*(k) = D(k) \\setminus \\{n : \\text{all digits equal}\\}$.",
            "  \\item \\textbf{Leading zeros}: "
            "Inputs always have $k$ significant digits; internal operations may pad.",
            "  \\item \\textbf{Convergence rate}: "
            "Fraction of domain converging to dominant attractor.",
            "  \\item \\textbf{Basin entropy}: "
            "$H = -\\sum_a p_a \\log_2 p_a$ over attractor basin fractions.",
            "\\end{itemize}",
        ]
        return "\n".join(lines)

    def _section_hash_table(self, experiments: List[dict]) -> str:
        reg_hash = self.registry.sha256
        lines = [
            "\\subsection*{Traceability Hashes}",
            "",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Operation Registry \\& Pipeline Hashes}",
            "\\label{tab:hashes}",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "Component & SHA-256 \\\\",
            "\\midrule",
            f"Operation Registry & {_tex_hash(reg_hash, 32)} \\\\",
        ]

        # Unique pipelines
        seen_pipes: Dict[str, str] = {}
        for e in experiments:
            ph = e.get("pipeline_hash", "")
            pd = e.get("pipeline_display", "")
            if ph and ph not in seen_pipes:
                seen_pipes[ph] = pd

        for ph in sorted(seen_pipes.keys()):
            pd = _tex_escape(seen_pipes[ph])
            lines.append(f"Pipeline: {pd} & {_tex_hash(ph, 32)} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)

    def _section_result_table(self, experiments: List[dict]) -> str:
        lines = [
            "\\subsection*{Experiment Results}",
            "",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Attractor Statistics by Pipeline \\& Domain}",
            "\\label{tab:results}",
            "\\begin{tabular}{llrrrl}",
            "\\toprule",
            "Pipeline & Domain Hash & $|D|$ & Conv. Rate & $H$ & Result Hash \\\\",
            "\\midrule",
        ]

        for e in experiments:
            pd = _tex_escape(e.get("pipeline_display", ""))
            dh = _tex_hash(e.get("domain_hash", ""), 12)
            n = e.get("num_startpoints", 0)
            cr = _tex_float(e.get("convergence_rate", "0"), 4)
            he = _tex_float(e.get("basin_entropy", "0"), 4)
            rh = _tex_hash(e.get("result_hash", ""), 12)
            lines.append(f"{pd} & {dh} & {n} & {cr} & {he} & {rh} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)

    def _section_conjectures(self, ranked: List[RankedConjecture]) -> str:
        lines = [
            "\\subsection*{Ranked Conjectures}",
            f"Ranking Model v{RANKING_MODEL_VERSION}: "
            f"$w_{{\\text{{emp}}}}={RankingModelV1.W_EMPIRICAL}$, "
            f"$w_{{\\text{{str}}}}={RankingModelV1.W_STRUCTURAL}$, "
            f"$w_{{\\text{{nov}}}}={RankingModelV1.W_NOVELTY}$, "
            f"$w_{{\\text{{sim}}}}={RankingModelV1.W_SIMPLICITY}$, "
            f"$w_{{\\text{{fal}}}}={RankingModelV1.W_FALSIFIABILITY}$.",
            "",
        ]

        for i, rc in enumerate(ranked, 1):
            c = rc.conjecture
            d = rc.density_estimate
            s = rc.proof_skeleton

            lines.append(f"\\paragraph{{Conjecture {i}}} "
                         f"(score $= {rc.final_score:.4f}$, "
                         f"density label: \\texttt{{{d.falsification_label}}})")
            lines.append("")

            # Statement
            lines.append(f"\\textit{{{_tex_escape(c.statement)}}}")
            lines.append("")

            # Tested domains
            if c.tested_domains:
                lines.append("\\textbf{Tested domains:}")
                lines.append("\\begin{itemize}")
                for td in c.tested_domains:
                    if not td.exclude_repdigits:
                        adj = "none"
                    elif td.digit_length >= 2:
                        adj = f"exact (repdigits excluded: {td.base - 1} values)"
                    else:
                        adj = "approximate"
                    lines.append(
                        f"  \\item base={td.base}, $k={td.digit_length}$, "
                        f"range=$[{td.range_lo}, {td.range_hi}]$, "
                        f"mode=\\texttt{{{td.mode}}}, "
                        f"repdigit\\_adj={adj}")
                lines.append("\\end{itemize}")
                lines.append("")

            # Density
            lines.append(f"\\textbf{{Density estimate:}} "
                         f"$N = {d.total_search_space}$, "
                         f"CEs $= {d.num_counterexamples}$, "
                         f"$\\hat{{p}}_{{95}} < {d.upper_bound_95:.2e}$, "
                         f"$\\log_{{10}} N = {d.search_volume_log10:.1f}$.")
            lines.append("")

            # Proof skeleton
            lines.append(f"\\textbf{{Proof skeleton}} "
                         f"(strategy: \\texttt{{{s.strategy.value}}}, "
                         f"strength: \\texttt{{{s.proof_strength}}}):")
            if s.reduction_steps:
                lines.append("\\begin{enumerate}")
                for step in s.reduction_steps:
                    status_mark = {"proven": "$\\checkmark$",
                                   "claimed": "$\\sim$",
                                   "gap": "$\\square$"}.get(step.status, "?")
                    lines.append(f"  \\item [{status_mark}] "
                                 f"{_tex_escape(step.description)}")
                lines.append("\\end{enumerate}")
            if s.remaining_gaps:
                lines.append("\\textbf{Remaining gaps:}")
                lines.append("\\begin{itemize}")
                for gap in s.remaining_gaps:
                    lines.append(f"  \\item {_tex_escape(gap)}")
                lines.append("\\end{itemize}")
            if s.known_theorem_links:
                lines.append("\\textbf{Known theorem links:} "
                             + "; ".join(_tex_escape(t) for t in s.known_theorem_links)
                             + ".")

            # Ranking breakdown
            lines.append("")
            bd = rc.rank_breakdown
            lines.append(f"\\textbf{{Ranking breakdown:}} "
                         f"emp$={bd['empirical_confidence']:.3f}$, "
                         f"str$={bd['structural_strength']:.3f}$, "
                         f"nov$={bd['novelty']:.3f}$, "
                         f"sim$={bd['simplicity']:.3f}$, "
                         f"fal$={bd['falsifiability']:.3f}$.")
            lines.append("")

        return "\n".join(lines)

    def _section_footer(self) -> str:
        return (
            f"\\vspace{{1em}}\n"
            f"\\noindent\\textit{{Generated by Engine vNext M4 "
            f"(emitter v{EMITTER_VERSION}, ranking v{RANKING_MODEL_VERSION}).}}"
        )

    @staticmethod
    def _is_paper_a_domain(exp: dict) -> bool:
        """Check if experiment uses Paper A domain definition.
        Primary: check preset_name starts with 'paper_a_'.
        Fallback: conjunction of include_cycles=False + exclude_zero=True."""
        dj = exp.get("domain_json", "{}")
        try:
            d = json.loads(dj) if isinstance(dj, str) else dj
        except (json.JSONDecodeError, TypeError):
            return False
        pn = d.get("preset_name", "")
        if pn:
            return pn.startswith("paper_a_")
        return (d.get("include_cycles") is False
                and d.get("exclude_zero") is True)

    @staticmethod
    def _conjecture_matches_paper(rc: RankedConjecture, filter_fn) -> bool:
        """Check if a ranked conjecture belongs to a paper via its tested_domains."""
        for td in rc.conjecture.tested_domains:
            td_exp = {
                "domain_json": json.dumps({
                    "include_cycles": False,
                    "exclude_zero": True,
                    "exclude_repdigits": td.exclude_repdigits,
                    "preset_name": "",
                }),
            }
            if filter_fn(td_exp):
                return True
        # Fallback: if no tested_domains, allow through
        return len(rc.conjecture.tested_domains) == 0


# =============================================================================
# 4. DETERMINISM GUARD
# =============================================================================

class DeterminismGuard:
    """Verify determinism and traceability of emitted artifacts."""

    @staticmethod
    def _is_full_sha256(val: str) -> bool:
        return len(val) == 64 and all(c in "0123456789abcdef" for c in val)

    @staticmethod
    def check_manifest(manifest: dict) -> List[str]:
        """Return list of issues found (empty = all good)."""
        issues = []

        # 1. Top-level hashes must be full SHA-256 (64 hex chars)
        for field_name in ["op_registry_hash"]:
            val = manifest.get(field_name, "")
            if not DeterminismGuard._is_full_sha256(val):
                issues.append(f"Manifest '{field_name}' is not full SHA-256: {val!r}")

        # 1b. Pipeline catalog hashes
        for p in manifest.get("pipelines", []):
            val = p.get("pipeline_hash", "")
            if val and not DeterminismGuard._is_full_sha256(val):
                issues.append(f"Pipeline hash is not full SHA-256: {val[:20]}...")

        # 1c. Domain catalog hashes
        for d in manifest.get("domains", []):
            val = d.get("domain_hash", "")
            if val and not DeterminismGuard._is_full_sha256(val):
                issues.append(f"Domain hash is not full SHA-256: {val[:20]}...")

        # 1d. Result hashes
        for r in manifest.get("results", []):
            for hf in ["pipeline_hash", "domain_hash", "result_hash", "op_registry_hash"]:
                val = r.get(hf, "")
                if val and not DeterminismGuard._is_full_sha256(val):
                    issues.append(f"Result '{hf}' is not full SHA-256: {val[:20]}...")

        # 2. Results sorted by (pipeline_display, domain_hash)
        results = manifest.get("results", [])
        sort_keys = [(r.get("pipeline_display", ""), r.get("domain_hash", ""))
                     for r in results]
        if sort_keys != sorted(sort_keys):
            issues.append("Results are not sorted by (pipeline_display, domain_hash)")

        # 3. Pipelines sorted by hash
        pipes = manifest.get("pipelines", [])
        pipe_hashes = [p.get("pipeline_hash", "") for p in pipes]
        if pipe_hashes != sorted(pipe_hashes):
            issues.append("Pipelines not sorted by hash")

        # 4. Domains sorted by hash
        doms = manifest.get("domains", [])
        dom_hashes = [d.get("domain_hash", "") for d in doms]
        if dom_hashes != sorted(dom_hashes):
            issues.append("Domains not sorted by hash")

        # 5. Ranking model version present
        if "ranking_model_version" not in manifest:
            issues.append("Missing ranking_model_version in manifest")

        # 6. No short hashes leaked (check for any 16-char hex that's not part of full)
        manifest_str = json.dumps(manifest, sort_keys=True)
        for key in ["_short", "short_hash"]:
            if key in manifest_str:
                issues.append(f"Short hash field '{key}' found in manifest")

        # 7. Manifest v1.1 checks
        if manifest.get("manifest_version") == "1.1":
            env = manifest.get("environment", {})
            required_env_keys = [
                "python_version", "python_implementation", "platform",
                "platform_machine", "os_name", "sqlite3_version",
                "endianness", "timezone", "locale",
            ]
            for key in required_env_keys:
                if key not in env:
                    issues.append(f"Missing environment key: {key}")

            if "random_seed_policy" not in manifest:
                issues.append("Missing random_seed_policy in manifest v1.1")

            deps = manifest.get("dependencies", {})
            if deps:
                lock_sha = deps.get("requirements_lock_sha256", "")
                if lock_sha and not DeterminismGuard._is_full_sha256(lock_sha):
                    issues.append(
                        f"dependencies.requirements_lock_sha256 is not full SHA-256")

            bundle = manifest.get("bundle", {})
            bc = bundle.get("bundle_contents", [])
            if bc != sorted(bc):
                issues.append("bundle.bundle_contents is not sorted")

        return issues

    @staticmethod
    def check_latex_determinism(tex1: str, tex2: str) -> bool:
        """Check that two LaTeX renders are identical (excluding timestamp comments)."""
        def strip_timestamps(tex: str) -> str:
            lines = tex.split('\n')
            return '\n'.join(l for l in lines if not l.strip().startswith('% Generated:'))
        return strip_timestamps(tex1) == strip_timestamps(tex2)

    @staticmethod
    def check_bundle_integrity(
        bundle_path: str, manifest: dict,
    ) -> List[str]:
        """Verify bundle zip integrity against manifest."""
        issues = []
        bundle_info = manifest.get("bundle", {})

        # Check bundle sha256
        expected_sha = bundle_info.get("bundle_sha256")
        if expected_sha and expected_sha != "null":
            with open(bundle_path, 'rb') as f:
                actual_sha = hashlib.sha256(f.read()).hexdigest()
            if actual_sha != expected_sha:
                issues.append(
                    f"Bundle SHA-256 mismatch: expected "
                    f"{expected_sha[:16]}... got {actual_sha[:16]}...")

        # Check contents listing
        expected_contents = sorted(bundle_info.get("bundle_contents", []))
        with zipfile.ZipFile(bundle_path) as zf:
            actual_contents = sorted(zf.namelist())
        if expected_contents and expected_contents != actual_contents:
            issues.append(
                f"Bundle contents mismatch: expected "
                f"{len(expected_contents)} files, got {len(actual_contents)}")

        # Check lockfile hash if present
        deps = manifest.get("dependencies", {})
        lock_sha = deps.get("requirements_lock_sha256")
        if lock_sha:
            with zipfile.ZipFile(bundle_path) as zf:
                if "requirements.lock.txt" in zf.namelist():
                    lock_bytes = zf.read("requirements.lock.txt")
                    actual_lock_sha = hashlib.sha256(lock_bytes).hexdigest()
                    if actual_lock_sha != lock_sha:
                        issues.append("Lockfile SHA-256 mismatch in bundle")
                else:
                    issues.append("requirements.lock.txt missing from bundle")

        return issues

    @staticmethod
    def check_all_hashes_in_manifest(tex: str, manifest: dict) -> List[str]:
        """Verify all hashes referenced in LaTeX exist in the manifest."""
        issues = []
        # Extract all hex strings of length >= 12 from LaTeX
        hex_pattern = re.compile(r'[0-9a-f]{12,64}')
        tex_hashes = set(hex_pattern.findall(tex))

        # Collect all hashes from manifest
        manifest_hashes = set()
        manifest_hashes.add(manifest.get("op_registry_hash", ""))
        for p in manifest.get("pipelines", []):
            manifest_hashes.add(p.get("pipeline_hash", ""))
        for d in manifest.get("domains", []):
            manifest_hashes.add(d.get("domain_hash", ""))
        for r in manifest.get("results", []):
            for hf in ["pipeline_hash", "domain_hash", "result_hash", "op_registry_hash"]:
                manifest_hashes.add(r.get(hf, ""))

        # Check: every hash prefix in LaTeX must be a prefix of some manifest hash
        for th in tex_hashes:
            if not any(mh.startswith(th) for mh in manifest_hashes if mh):
                # Could be a false positive (e.g., hex in table cell)
                # Only flag if it looks like a hash reference
                if len(th) >= 16:
                    issues.append(f"LaTeX hash prefix '{th[:20]}...' not found in manifest")

        return issues


# =============================================================================
# 5. MAIN EMITTER FACADE
# =============================================================================

class AppendixEmitter:
    """One-command facade: DB → LaTeX + JSON manifests + snapshots."""

    def __init__(self, db_path: str = "results.db",
                 registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()
        self.store = ExperimentStore(db_path) if os.path.exists(db_path) else None
        self.manifest_builder = ManifestBuilder(self.registry)
        self.snapshotter = Snapshotter(self.registry)
        self.latex = LatexEmitter(self.registry)
        self.guard = DeterminismGuard()

    def emit_all(
        self,
        output_dir: str,
        conjectures: Optional[List[Conjecture]] = None,
        k_range: Optional[List[int]] = None,
        dependencies: Optional[dict] = None,
    ) -> dict:
        """Emit all artifacts to output_dir. Returns summary dict."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Collect experiments from DB
        experiments = []
        if self.store:
            cur = self.store.conn.execute("SELECT id FROM experiments ORDER BY id")
            for (eid,) in cur.fetchall():
                exp = self.store.get_experiment(eid)
                if exp:
                    experiments.append(exp)

        # 2. Mine + rank conjectures (if not provided)
        ranked = []
        if conjectures:
            ranker = RankingModelV1(self.registry)
            ranked = ranker.rank_all(conjectures)
        elif k_range:
            miner = ConjectureMiner(self.registry)
            ranker = RankingModelV1(self.registry)
            for pipe_name in ["kaprekar_step", "truc_1089"]:
                try:
                    conjs = miner.mine_all(pipe_name, k_range)
                    ranked.extend(ranker.rank_all(conjs))
                except Exception:
                    pass
            ranked.sort(key=lambda r: r.final_score, reverse=True)

        # 3. Build manifest
        manifest = self.manifest_builder.build_manifest(
            experiments, ranked or None, dependencies=dependencies)
        manifest_path = os.path.join(output_dir, "repro_manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, sort_keys=True, indent=2)

        # 4. Snapshots
        reg_path = os.path.join(output_dir, "registry_snapshot.json")
        self.snapshotter.write_registry_snapshot(reg_path)

        pipe_path = os.path.join(output_dir, "pipeline_catalog.json")
        self.snapshotter.write_pipeline_catalog(
            manifest.get("pipelines", []), pipe_path)

        dom_path = os.path.join(output_dir, "domain_catalog.json")
        self.snapshotter.write_domain_catalog(
            manifest.get("domains", []), dom_path)

        res_path = os.path.join(output_dir, "result_catalog.json")
        self.snapshotter.write_result_catalog(
            manifest.get("results", []), res_path)

        # 5. LaTeX appendices
        tex_a = self.latex.emit_paper_a(experiments, ranked or None)
        tex_a_path = os.path.join(output_dir, "appendix_paper_a.tex")
        with open(tex_a_path, 'w', encoding='utf-8') as f:
            f.write(tex_a)

        tex_b = self.latex.emit_paper_b(experiments, ranked or None)
        tex_b_path = os.path.join(output_dir, "appendix_paper_b.tex")
        with open(tex_b_path, 'w', encoding='utf-8') as f:
            f.write(tex_b)

        # 6. Determinism guard
        issues = self.guard.check_manifest(manifest)
        tex_issues_a = self.guard.check_all_hashes_in_manifest(tex_a, manifest)
        tex_issues_b = self.guard.check_all_hashes_in_manifest(tex_b, manifest)

        tex_a_sha = hashlib.sha256(tex_a.encode('utf-8')).hexdigest()
        tex_b_sha = hashlib.sha256(tex_b.encode('utf-8')).hexdigest()

        summary = {
            "artifacts": {
                "repro_manifest.json": manifest_path,
                "registry_snapshot.json": reg_path,
                "pipeline_catalog.json": pipe_path,
                "domain_catalog.json": dom_path,
                "result_catalog.json": res_path,
                "appendix_paper_a.tex": tex_a_path,
                "appendix_paper_b.tex": tex_b_path,
            },
            "experiments_count": len(experiments),
            "conjectures_count": len(ranked),
            "manifest_sha256": ManifestBuilder.manifest_sha256(manifest),
            "op_registry_sha256": self.registry.sha256,
            "appendix_paper_a_sha256": tex_a_sha,
            "appendix_paper_b_sha256": tex_b_sha,
            "determinism_issues": issues + tex_issues_a + tex_issues_b,
        }
        return summary


# =============================================================================
# 6. ARTIFACT PACKAGER
# =============================================================================

class ArtifactPackager:
    """Package all M4.1 artifacts into a reproducibility bundle (zip + README).

    Produces:
      - README_repro.md      (reviewer instructions)
      - ENVIRONMENT.md        (environment snapshot)
      - requirements.lock.txt (pip freeze)
      - reproducibility_bundle.zip  (all of the above + core artifacts + DB)
    """

    @staticmethod
    def generate_environment_md(env: Optional[dict] = None) -> str:
        """Generate ENVIRONMENT.md from environment dict."""
        env = env or collect_environment()
        lines = [
            "# Environment Specification",
            "",
            "This file records the exact environment used to generate "
            "these reproducibility artifacts.",
            "",
        ]
        for k in sorted(env.keys()):
            lines.append(f"- **{k}**: `{env[k]}`")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def generate_readme(summary: dict, db_path: str = "results.db") -> str:
        """Generate README_repro.md — one-page reviewer reproducibility guide."""
        env = summary.get("environment", collect_environment())
        py_ver = env.get("python_version", "unknown")

        lines = [
            "# Reproducibility Bundle",
            "",
            "## Purpose",
            "",
            f"This bundle contains all artifacts needed to independently verify "
            f"the computational results presented in the paper. Generated by "
            f"Engine vNext M4.1 (emitter v{EMITTER_VERSION}, "
            f"manifest v{MANIFEST_VERSION}, "
            f"ranking model v{RANKING_MODEL_VERSION}).",
            "",
            "## System Requirements",
            "",
            f"- **Python**: {py_ver} (exact version used for generation)",
            f"- **OS tested**: {env.get('platform', 'unknown')}",
            f"- **Architecture**: {env.get('platform_machine', 'unknown')}",
            "- **Hardware**: CPU only, no GPU required",
            "",
            "## Install",
            "",
            "```bash",
            "python -m venv .venv",
            "# Windows:",
            ".venv\\Scripts\\activate",
            "# Linux/macOS:",
            "# source .venv/bin/activate",
            "",
            "pip install -r requirements.lock.txt",
            "```",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python reproduce.py --db results.db --out repro_out --bundle",
            "```",
            "",
            "This single command will:",
            "1. Verify the environment and dependencies",
            "2. Run the full M4 emitter pipeline (DB -> manifest + catalogs + LaTeX)",
            "3. Run the DeterminismGuard (rerun verification)",
            "4. Package the reproducibility bundle",
            "5. Print the **FINAL MANIFEST SHA256** for verification",
            "",
            "## Verify",
            "",
            "Success means:",
            "- The final manifest SHA-256 matches the expected hash below",
            "- All appendix hashes match",
            "- Zero determinism issues reported",
            "",
            "## Expected SHA-256 Hashes",
            "",
            "```",
            f"manifest (content hash): {summary.get('manifest_sha256', 'N/A')}",
            f"op_registry:             {summary.get('op_registry_sha256', 'N/A')}",
            f"appendix_paper_a.tex:    "
            f"{summary.get('appendix_paper_a_sha256', 'N/A')}",
            f"appendix_paper_b.tex:    "
            f"{summary.get('appendix_paper_b_sha256', 'N/A')}",
            "```",
            "",
            "## Contents",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `repro_manifest.json` | Canonical v1.1 manifest with full SHA-256 |",
            "| `registry_snapshot.json` | All 22 OperationSpecs (canonical JSON) |",
            "| `pipeline_catalog.json` | All used pipelines + SHA-256 |",
            "| `domain_catalog.json` | All used DomainPolicies + SHA-256 |",
            "| `result_catalog.json` | RunResult hashes + key metrics |",
            "| `appendix_paper_a.tex` | Paper A appendix (strict domains only) |",
            "| `appendix_paper_b.tex` | Paper B appendix (all engine metrics) |",
            "| `requirements.lock.txt` | Locked pip dependencies |",
            "| `ENVIRONMENT.md` | Environment snapshot |",
            "| `reproduce.py` | One-command reproduction script |",
            "| `results.db` | SQLite experiment database |",
            "| `README_repro.md` | This file |",
            "",
            "## Hashing Convention",
            "",
            "JSON catalog files are written with `indent=2` for readability.",
            "Content hashes are computed on the compact canonical form",
            "(`json.dumps(data, sort_keys=True, separators=(',',':'))`).",
            "File bytes may differ from the hash input; only the canonical",
            "no-whitespace JSON determines the content hash.",
            "",
            "The manifest SHA-256 excludes the `created_utc` timestamp and",
            "`bundle.bundle_sha256` (circular dependency) from the hash input.",
            "",
            "## Known Limitations",
            "",
            "- CPU only; no GPU or CUDA dependency.",
            "- **Cross-platform float precision:** Floating-point results (convergence",
            "  rates, basin entropy, Lyapunov bounds) may differ at the last 1-2",
            "  decimal places across platforms (x86 vs ARM, different compilers).",
            "  All integer results (fixed points, attractor counts) are exact.",
            "  Canonical floats are rounded to 12 significant digits to mitigate this.",
            "- SQLite version differences may cause byte-level DB differences",
            "  but content hashes remain stable.",
            "- `PYTHONHASHSEED=0` is recommended for dict iteration order",
            "  (all critical paths use explicit sorting, but this adds safety).",
            "",
            "## Expected Runtime",
            "",
            f"- **Dev test suite** (M0-M4, unit+integration): ~7 seconds",
            f"- **Full exhaustive suite** (all k-ranges): ~20+ minutes",
            f"- **M4 artifact generation**: ~3 seconds",
            "",
            "## Statistics",
            "",
            f"- Experiments in DB: {summary.get('experiments_count', 0)}",
            f"- Ranked conjectures: {summary.get('conjectures_count', 0)}",
            f"- Determinism issues: {len(summary.get('determinism_issues', []))}",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def package_bundle(
        output_dir: str,
        summary: dict,
        manifest: Optional[dict] = None,
        db_path: str = "results.db",
        reproduce_py_path: Optional[str] = None,
        lockfile_content: Optional[str] = None,
        lockfile_sha: Optional[str] = None,
        bundle_name: str = "reproducibility_bundle.zip",
    ) -> Tuple[str, str]:
        """Create reproducibility_bundle.zip with all artifacts.
        Returns (bundle_path, bundle_sha256).

        If lockfile_content/lockfile_sha are not provided, generates via pip freeze.
        If reproduce_py_path is not provided, looks for reproduce.py next to this file.
        """
        env = summary.get("environment", collect_environment())

        # 1. Generate lockfile
        lock_path = os.path.join(output_dir, "requirements.lock.txt")
        if lockfile_content is not None and lockfile_sha is not None:
            with open(lock_path, 'w', encoding='utf-8') as f:
                f.write(lockfile_content)
        else:
            lockfile_content, lockfile_sha = generate_lockfile(lock_path)

        # 2. ENVIRONMENT.md
        env_md = ArtifactPackager.generate_environment_md(env)
        env_path = os.path.join(output_dir, "ENVIRONMENT.md")
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_md)

        # 3. README_repro.md
        readme = ArtifactPackager.generate_readme(summary, db_path)
        readme_path = os.path.join(output_dir, "README_repro.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme)

        # 4. Locate reproduce.py and run_experiments.py
        _project_root = os.path.dirname(os.path.abspath(__file__))
        if reproduce_py_path is None:
            reproduce_py_path = os.path.join(_project_root, "scripts", "reproduce.py")
            if not os.path.exists(reproduce_py_path):
                reproduce_py_path = os.path.join(_project_root, "reproduce.py")
        run_exp_path = os.path.join(_project_root, "scripts", "run_experiments.py")

        # 5. Create zip
        bundle_path = os.path.join(output_dir, bundle_name)
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(readme_path, "README_repro.md")
            zf.write(env_path, "ENVIRONMENT.md")
            zf.write(lock_path, "requirements.lock.txt")
            for name, path in summary.get("artifacts", {}).items():
                if os.path.exists(path):
                    zf.write(path, name)
            if os.path.exists(db_path):
                zf.write(db_path, "results.db")
            if os.path.exists(reproduce_py_path):
                zf.write(reproduce_py_path, "reproduce.py")
            if os.path.exists(run_exp_path):
                zf.write(run_exp_path, "run_experiments.py")

        # 6. Compute bundle SHA-256
        with open(bundle_path, 'rb') as f:
            bundle_sha = hashlib.sha256(f.read()).hexdigest()

        # 7. Update manifest on disk with bundle_sha256 (if manifest provided)
        if manifest is not None:
            manifest.setdefault("bundle", {})
            manifest["bundle"]["bundle_sha256"] = bundle_sha
            manifest_path = summary.get("artifacts", {}).get("repro_manifest.json")
            if manifest_path:
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, sort_keys=True, indent=2)

        return bundle_path, bundle_sha


# =============================================================================
# MAIN: Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENGINE vNext — M4: Paper Appendix Auto-Emitter")
    print("=" * 70)

    output_dir = "output_m4"
    emitter = AppendixEmitter("results.db")
    summary = emitter.emit_all(output_dir, k_range=[3, 4, 5])

    print(f"\n  Artifacts generated in: {output_dir}/")
    for name, path in summary["artifacts"].items():
        print(f"    {name}")
    print(f"\n  Experiments: {summary['experiments_count']}")
    print(f"  Conjectures: {summary['conjectures_count']}")
    print(f"  Manifest SHA: {summary['manifest_sha256'][:32]}...")
    print(f"  Registry SHA: {summary['op_registry_sha256'][:32]}...")

    if summary["determinism_issues"]:
        print(f"\n  ⚠ Determinism issues:")
        for issue in summary["determinism_issues"]:
            print(f"    - {issue}")
    else:
        print(f"\n  ✓ No determinism issues found")

    if emitter.store:
        emitter.store.close()

    print("\n" + "=" * 70)
    print("M4 COMPLETE")
    print("=" * 70)
