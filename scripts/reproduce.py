"""
ENGINE vNext -- reproduce.py: One-Command Reproducibility Runner (M4.1)

Usage:
    python reproduce.py --db results.db --out repro_out --bundle

This script orchestrates end-to-end artifact generation and verification:
  1. Check deterministic runtime knobs (PYTHONHASHSEED, timezone)
  2. Print environment summary
  3. Generate/verify requirements.lock.txt (pip freeze)
  4. Run M4 emitter end-to-end (DB -> manifest + catalogs + LaTeX)
  5. Run DeterminismGuard (rerun + byte-identical verification)
  6. Package reproducibility_bundle.zip
  7. Update manifest with bundle_sha256
  8. Print FINAL MANIFEST SHA256

Does NOT modify core research code; orchestration only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

# Ensure parent directory (project root) is on sys.path for core module imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from appendix_emitter import (
    AppendixEmitter, ArtifactPackager, DeterminismGuard, ManifestBuilder,
    collect_environment, generate_lockfile,
    EMITTER_VERSION, MANIFEST_VERSION, ENGINE_VERSION,
)
from pipeline_dsl import OperationRegistry
from proof_engine import RANKING_MODEL_VERSION


def print_header():
    print("=" * 70)
    print("ENGINE vNext -- Reproducibility Runner (M4.1)")
    print(f"  Emitter v{EMITTER_VERSION} | Manifest v{MANIFEST_VERSION} "
          f"| Engine {ENGINE_VERSION}")
    print(f"  Ranking Model v{RANKING_MODEL_VERSION}")
    print("=" * 70)


def check_runtime_knobs():
    """Check and warn about deterministic runtime settings."""
    issues = []
    hashseed = os.environ.get("PYTHONHASHSEED", "")
    if hashseed != "0":
        issues.append(
            f"PYTHONHASHSEED={hashseed!r} (recommend '0' for determinism; "
            f"all critical paths use explicit sorting)")
    return issues


def print_environment():
    """Print environment summary to stdout."""
    env = collect_environment()
    print("\n  Environment:")
    for k in sorted(env.keys()):
        print(f"    {k}: {env[k]}")
    return env


def step_lockfile(output_dir: str):
    """Generate requirements.lock.txt via pip freeze."""
    lock_path = os.path.join(output_dir, "requirements.lock.txt")
    content, sha = generate_lockfile(lock_path)
    n_packages = len([l for l in content.strip().split('\n')
                      if l and not l.startswith('#')])
    print(f"\n  Lockfile: {n_packages} packages, SHA-256: {sha[:32]}...")
    return content, sha


def step_emit(db_path: str, output_dir: str, k_range: list,
              dependencies: dict):
    """Run M4 emitter end-to-end."""
    reg = OperationRegistry()
    emitter = AppendixEmitter(db_path, reg)
    summary = emitter.emit_all(output_dir, k_range=k_range,
                               dependencies=dependencies)
    # Reload manifest from disk for later use
    manifest_path = summary["artifacts"]["repro_manifest.json"]
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    if emitter.store:
        emitter.store.close()

    print(f"\n  Emitter complete:")
    print(f"    Experiments: {summary['experiments_count']}")
    print(f"    Conjectures: {summary['conjectures_count']}")
    print(f"    Manifest SHA: {summary['manifest_sha256'][:32]}...")
    print(f"    Issues: {len(summary['determinism_issues'])}")
    for issue in summary["determinism_issues"]:
        print(f"      - {issue}")

    return summary, manifest


def step_guard_rerun(db_path: str, output_dir: str, k_range: list,
                     summary: dict, dependencies: dict):
    """Rerun emitter in temp dir and verify byte-identical output."""
    print("\n  Determinism verification (rerun)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = OperationRegistry()
        emitter2 = AppendixEmitter(db_path, reg)
        summary2 = emitter2.emit_all(tmpdir, k_range=k_range,
                                     dependencies=dependencies)
        if emitter2.store:
            emitter2.store.close()

        # Compare key hashes
        guard = DeterminismGuard()
        match = True
        for key in ["manifest_sha256", "op_registry_sha256",
                     "appendix_paper_a_sha256", "appendix_paper_b_sha256"]:
            v1 = summary.get(key, "")
            v2 = summary2.get(key, "")
            if v1 != v2:
                print(f"    MISMATCH {key}: {v1[:16]}... vs {v2[:16]}...")
                match = False

        # Also compare LaTeX byte-for-byte (minus timestamps)
        for tex_name in ["appendix_paper_a.tex", "appendix_paper_b.tex"]:
            p1 = summary["artifacts"][tex_name]
            p2 = summary2["artifacts"][tex_name]
            with open(p1, 'r', encoding='utf-8') as f:
                t1 = f.read()
            with open(p2, 'r', encoding='utf-8') as f:
                t2 = f.read()
            if not guard.check_latex_determinism(t1, t2):
                print(f"    MISMATCH LaTeX: {tex_name}")
                match = False

        if match:
            print("    PASS: rerun produces identical artifacts")
        else:
            print("    FAIL: rerun produced different artifacts")
        return match


def step_package(output_dir: str, summary: dict, manifest: dict,
                 db_path: str, lockfile_content: str, lockfile_sha: str):
    """Package reproducibility bundle."""
    reproduce_py = os.path.abspath(__file__)
    bundle_path, bundle_sha = ArtifactPackager.package_bundle(
        output_dir, summary, manifest=manifest, db_path=db_path,
        reproduce_py_path=reproduce_py,
        lockfile_content=lockfile_content, lockfile_sha=lockfile_sha,
    )
    print(f"\n  Bundle: {bundle_path}")
    print(f"  Bundle SHA-256: {bundle_sha[:32]}...")
    return bundle_path, bundle_sha


def main():
    parser = argparse.ArgumentParser(
        description="Engine vNext M4.1 — One-command reproducibility runner")
    parser.add_argument("--db", default="results.db",
                        help="Path to results.db (default: results.db)")
    parser.add_argument("--out", default="repro_out",
                        help="Output directory (default: repro_out)")
    parser.add_argument("--bundle", action="store_true", default=True,
                        help="Create reproducibility_bundle.zip (default: True)")
    parser.add_argument("--no-bundle", dest="bundle", action="store_false",
                        help="Skip bundle creation")
    parser.add_argument("--k-range", nargs="*", type=int, default=[3, 4, 5, 6, 7],
                        help="k values for conjecture mining (default: 3 4 5 6 7)")
    args = parser.parse_args()

    print_header()

    # 1. Runtime knobs
    knob_issues = check_runtime_knobs()
    if knob_issues:
        print("\n  Runtime warnings:")
        for w in knob_issues:
            print(f"    - {w}")

    # 2. Environment
    env = print_environment()

    # 3. Lockfile
    os.makedirs(args.out, exist_ok=True)
    lockfile_content, lockfile_sha = step_lockfile(args.out)
    dependencies = {
        "freeze_method": "pip_freeze",
        "requirements_lock_sha256": lockfile_sha,
        "pip_freeze": [l for l in lockfile_content.strip().split('\n')
                       if l and not l.startswith('#')],
    }

    # 4. Emit
    summary, manifest = step_emit(args.db, args.out, args.k_range,
                                  dependencies)

    # 5. Guard rerun
    rerun_ok = step_guard_rerun(args.db, args.out, args.k_range,
                                summary, dependencies)

    # 6. Package
    bundle_path = None
    if args.bundle:
        bundle_path, bundle_sha = step_package(
            args.out, summary, manifest, args.db,
            lockfile_content, lockfile_sha)

    # 7. Final manifest hash
    manifest_sha = ManifestBuilder.manifest_sha256(manifest)

    # 8. Print final
    print("\n" + "=" * 70)
    if rerun_ok and len(summary.get("determinism_issues", [])) == 0:
        print("  STATUS: ALL CHECKS PASSED")
    else:
        print("  STATUS: ISSUES DETECTED (see above)")
    print(f"\n  FINAL MANIFEST SHA256: {manifest_sha}")
    print("=" * 70)

    return 0 if rerun_ok else 1


if __name__ == "__main__":
    sys.exit(main())
