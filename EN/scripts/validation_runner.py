#!/usr/bin/env python3
# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research — MIT License
"""
Auditable Re-Validation Runner for digit-dynamics.

Runs a 4-step validation session with hash-chain audit trail:
  1. Backward compatibility: all existing M0-M4 tests must pass
  2. Base-10 hash comparison: paper_b_hashes.json must reproduce
  3. Multi-base verification: Kaprekar & 1089 constants across bases
  4. Seal & certificate: HMAC-SHA256 certificate + audit trail

Output:
  EN/validation/YYYY-MM-DD_HHMMSS/
    validation_certificate.json
    validation_session.jsonl
    validation_summary.md
    paper_b_hash_comparison.json
    multibase_results.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure src/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_EN_DIR = _SCRIPT_DIR.parent
_SRC_DIR = _EN_DIR / "src"
sys.path.insert(0, str(_SRC_DIR))

from pipeline_dsl import (
    BaseDigitOps, OperationRegistry, Pipeline, DomainPolicy, PipelineRunner,
)
from hash_chain import chain_init, chain_log, chain_seal, chain_verify, export_jsonl


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _run_pytest(test_files: list[str], mark_filter: str = "not exhaustive") -> dict:
    """Run pytest on given files, return {passed, failed, errors, output}."""
    cmd = [
        sys.executable, "-m", "pytest",
        *test_files,
        "-q", "--tb=short",
        "-m", mark_filter,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_EN_DIR), env=env,
        timeout=300,
    )
    output = result.stdout + result.stderr

    # Parse pytest summary line
    passed = failed = errors = 0
    for line in output.splitlines():
        if "passed" in line:
            import re
            m = re.search(r"(\d+) passed", line)
            if m:
                passed = int(m.group(1))
            m = re.search(r"(\d+) failed", line)
            if m:
                failed = int(m.group(1))
            m = re.search(r"(\d+) error", line)
            if m:
                errors = int(m.group(1))

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "returncode": result.returncode,
        "output_tail": output[-500:] if len(output) > 500 else output,
    }


def step1_backward_compat(session_id: str) -> bool:
    """Step 1: Run all existing M0-M4 tests."""
    print("\n=== Step 1: Backward Compatibility Check ===")

    test_files = [
        f"tests/test_m{i}.py" for i in range(5)
    ]
    result = _run_pytest(test_files)

    chain_log(session_id, "backward_compat", {
        "test_files": test_files,
        "passed": result["passed"],
        "failed": result["failed"],
        "errors": result["errors"],
        "status": "PASS" if result["returncode"] == 0 else "FAIL",
    })

    if result["returncode"] != 0:
        print(f"  FAIL: {result['failed']} failures, {result['errors']} errors")
        print(f"  {result['output_tail']}")
        return False

    print(f"  PASS: {result['passed']} tests passed")
    return True


def step2_base10_hashes(session_id: str) -> dict:
    """Step 2: Reproduce paper_b_hashes.json and compare."""
    print("\n=== Step 2: Base-10 Hash Comparison ===")

    hashes_file = _EN_DIR / "data" / "paper_b_hashes.json"
    with open(hashes_file) as f:
        paper_data = json.load(f)

    reg = OperationRegistry()
    runner = PipelineRunner(reg)
    comparisons = []
    all_match = True

    for exp in paper_data["results"]:
        pipe = Pipeline.parse(exp["pipeline"], registry=reg)
        # Reconstruct domain from num_inputs and domain_hash
        # We need to figure out digit_length from num_inputs
        domain = _reconstruct_domain(exp, reg)
        if domain is None:
            comparisons.append({
                "pipeline": exp["pipeline"],
                "status": "SKIP",
                "reason": "Could not reconstruct domain",
            })
            continue

        result = runner.run_exhaustive(pipe, domain)
        match = result.short_hash == exp["result_hash"]

        comparisons.append({
            "pipeline": exp["pipeline"],
            "domain_hash": exp["domain_hash"],
            "expected_hash": exp["result_hash"],
            "computed_hash": result.short_hash,
            "match": match,
        })

        if not match:
            all_match = False
            print(f"  MISMATCH: {exp['pipeline']} — expected {exp['result_hash']}, got {result.short_hash}")
        else:
            print(f"  OK: {exp['pipeline']} ({exp['num_inputs']} inputs)")

    chain_log(session_id, "base10_hash_comparison", {
        "num_experiments": len(paper_data["results"]),
        "num_matched": sum(1 for c in comparisons if c.get("match")),
        "num_mismatched": sum(1 for c in comparisons if c.get("match") is False),
        "num_skipped": sum(1 for c in comparisons if c.get("status") == "SKIP"),
        "all_match": all_match,
        "op_registry_hash": reg.short_hash,
    })

    return {"comparisons": comparisons, "all_match": all_match}


def _reconstruct_domain(exp: dict, reg: OperationRegistry) -> DomainPolicy | None:
    """Reconstruct DomainPolicy from paper_b experiment entry."""
    num_inputs = exp["num_inputs"]
    # Known domain sizes for base-10 k-digit with exclude_repdigits=True:
    # k=3: 891 (900-9), k=4: 8991, k=5: 89991, k=6: 899991, k=7: 8999991
    # Without exclude_repdigits:
    # k=3: 900, k=4: 9000, k=5: 90000, k=6: 900000, k=7: 9000000
    domain_map = {
        891: (3, True), 8991: (4, True), 89991: (5, True),
        899991: (6, True), 8999991: (7, True),
        900: (3, False), 9000: (4, False), 90000: (5, False),
        900000: (6, False), 9000000: (7, False),
    }

    if num_inputs in domain_map:
        k, exclude_rep = domain_map[num_inputs]
        return DomainPolicy(base=10, digit_length=k, exclude_repdigits=exclude_rep)

    return None


def step3_multibase_verification(session_id: str) -> dict:
    """Step 3: Multi-base mathematical verification."""
    print("\n=== Step 3: Multi-Base Verification ===")

    results = {}

    # 3a. Kaprekar constants for even bases (3-digit)
    print("  3a. Kaprekar 3-digit constants...")
    kaprekar_results = []
    for base in [4, 6, 8, 10, 12, 16]:
        expected = (base // 2) * (base * base - 1)
        ops = BaseDigitOps(base)
        actual = ops.kaprekar_step(expected)
        ok = actual == expected
        kaprekar_results.append({
            "base": base, "expected": expected, "actual": actual,
            "formula": f"({base}//2)*({base}²-1)", "match": ok,
        })
        print(f"    Base {base:2d}: K₃ = {expected:6d}  {'OK' if ok else 'FAIL'}")

    results["kaprekar_3digit"] = kaprekar_results

    # 3b. 1089 analogues
    print("  3b. 1089 analogues...")
    truc_results = []
    for base in [6, 8, 10, 12]:
        expected = (base - 1) * (base + 1) ** 2
        ops = BaseDigitOps(base)
        test_input = (base - 1) * base * base
        actual = ops.truc_1089(test_input)
        ok = actual == expected
        truc_results.append({
            "base": base, "expected": expected, "actual": actual,
            "test_input": test_input,
            "formula": f"({base}-1)*({base}+1)²", "match": ok,
        })
        print(f"    Base {base:2d}: T₃ = {expected:6d}  {'OK' if ok else 'FAIL'}")

    results["truc_1089_analogues"] = truc_results

    # 3c. Complement involution (stochastic)
    print("  3c. Complement involution (stochastic)...")
    complement_results = []
    import random
    random.seed(42)  # reproducible
    for base in [6, 8, 10, 12, 16]:
        ops = BaseDigitOps(base)
        tested = 0
        passed = 0
        for _ in range(100):
            n = random.randint(1, base ** 4 - 1)
            digits = ops.to_digits(n)
            if digits[0] < base - 1:  # No leading-zero issue
                tested += 1
                if ops.complement(ops.complement(n)) == n:
                    passed += 1
        complement_results.append({
            "base": base, "tested": tested, "passed": passed,
            "rate": passed / tested if tested > 0 else 0,
        })
        print(f"    Base {base:2d}: {passed}/{tested} involution checks OK")

    results["complement_involution"] = complement_results

    # 3d. Exhaustive Kaprekar base-6 3-digit
    print("  3d. Exhaustive Kaprekar base-6 3-digit...")
    reg = OperationRegistry()
    runner = PipelineRunner(reg)
    pipe = Pipeline.parse("kaprekar_step", registry=reg)
    domain = DomainPolicy(base=6, digit_length=3, exclude_repdigits=True)
    run_result = runner.run_exhaustive(pipe, domain)
    results["kaprekar_base6_3d"] = {
        "fixed_points": run_result.fixed_points,
        "num_attractors": run_result.num_attractors,
        "convergence_rate": run_result.convergence_rate,
        "result_hash": run_result.short_hash,
    }
    print(f"    FPs: {run_result.fixed_points}, conv: {run_result.convergence_rate:.4f}")

    chain_log(session_id, "multibase_verification", {
        "kaprekar_all_match": all(r["match"] for r in kaprekar_results),
        "truc_all_match": all(r["match"] for r in truc_results),
        "complement_all_pass": all(r["rate"] == 1.0 for r in complement_results),
        "base6_3d_fp": run_result.fixed_points,
    })

    return results


def step4_seal_and_export(session_id: str, output_dir: Path, **extra_data) -> dict:
    """Step 4: Seal hash chain and write all output files."""
    print("\n=== Step 4: Seal & Certificate ===")

    certificate = chain_seal(session_id)
    jsonl = export_jsonl(session_id)

    # Verify our own certificate
    verify = chain_verify(certificate)
    certificate["self_verification"] = verify

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    cert_path = output_dir / "validation_certificate.json"
    with open(cert_path, "w") as f:
        json.dump(certificate, f, indent=2, sort_keys=True)
    print(f"  Certificate: {cert_path}")

    log_path = output_dir / "validation_session.jsonl"
    with open(log_path, "w") as f:
        f.write(jsonl)
    print(f"  Audit log:   {log_path}")

    if "hash_comparison" in extra_data:
        with open(output_dir / "paper_b_hash_comparison.json", "w") as f:
            json.dump(extra_data["hash_comparison"], f, indent=2)

    if "multibase_results" in extra_data:
        with open(output_dir / "multibase_results.json", "w") as f:
            json.dump(extra_data["multibase_results"], f, indent=2, default=str)

    # Write summary
    summary = _generate_summary(certificate, extra_data)
    with open(output_dir / "validation_summary.md", "w") as f:
        f.write(summary)
    print(f"  Summary:     {output_dir / 'validation_summary.md'}")

    return certificate


def _generate_summary(certificate: dict, extra_data: dict) -> str:
    """Generate human-readable validation summary."""
    lines = [
        "# Digit-Dynamics Validation Summary",
        "",
        f"**Session**: {certificate['session_id']}",
        f"**Created**: {certificate['created']}",
        f"**Sealed**: {certificate['sealed']}",
        f"**Chain entries**: {certificate['num_entries']}",
        f"**Certificate HMAC**: `{certificate['hmac_sha256'][:32]}...`",
        f"**Self-verification**: {certificate['self_verification']['details']}",
        "",
        "## Results",
        "",
    ]

    if "backward_compat" in extra_data:
        bc = extra_data["backward_compat"]
        lines.append(f"### Step 1: Backward Compatibility")
        lines.append(f"- Status: **{'PASS' if bc else 'FAIL'}**")
        lines.append("")

    if "hash_comparison" in extra_data:
        hc = extra_data["hash_comparison"]
        n_match = sum(1 for c in hc["comparisons"] if c.get("match"))
        n_total = len(hc["comparisons"])
        lines.append(f"### Step 2: Base-10 Hash Comparison")
        lines.append(f"- Matched: {n_match}/{n_total}")
        lines.append(f"- All match: **{hc['all_match']}**")
        lines.append("")

    if "multibase_results" in extra_data:
        mb = extra_data["multibase_results"]
        lines.append("### Step 3: Multi-Base Verification")
        if "kaprekar_3digit" in mb:
            ok = sum(1 for r in mb["kaprekar_3digit"] if r["match"])
            lines.append(f"- Kaprekar 3-digit: {ok}/{len(mb['kaprekar_3digit'])} bases verified")
        if "truc_1089_analogues" in mb:
            ok = sum(1 for r in mb["truc_1089_analogues"] if r["match"])
            lines.append(f"- 1089 analogues: {ok}/{len(mb['truc_1089_analogues'])} bases verified")
        if "complement_involution" in mb:
            ok = sum(1 for r in mb["complement_involution"] if r["rate"] == 1.0)
            lines.append(f"- Complement involution: {ok}/{len(mb['complement_involution'])} bases passed")
        if "kaprekar_base6_3d" in mb:
            fp = mb["kaprekar_base6_3d"]["fixed_points"]
            lines.append(f"- Kaprekar base-6 3-digit FPs: {fp}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by digit-dynamics validation runner*")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("DIGIT-DYNAMICS AUDITABLE RE-VALIDATION")
    print("=" * 70)

    ts = _timestamp()
    output_dir = _EN_DIR / "validation" / ts

    # Init hash chain
    init = chain_init(f"validation-{ts}")
    session_id = init["session_id"]
    print(f"Session: {session_id}")
    print(f"Genesis: {init['genesis_hash'][:32]}...")

    extra_data = {}

    # Step 1
    bc_ok = step1_backward_compat(session_id)
    extra_data["backward_compat"] = bc_ok
    if not bc_ok:
        print("\n*** GATE FAILED: Backward compatibility broken. Aborting. ***")
        chain_log(session_id, "abort", {"reason": "backward_compat_failed"})
        step4_seal_and_export(session_id, output_dir, **extra_data)
        sys.exit(1)

    # Step 2
    hash_result = step2_base10_hashes(session_id)
    extra_data["hash_comparison"] = hash_result
    if not hash_result["all_match"]:
        print("\n*** WARNING: Some base-10 hashes did not match. Continuing. ***")

    # Step 3
    multibase = step3_multibase_verification(session_id)
    extra_data["multibase_results"] = multibase

    # Step 4
    cert = step4_seal_and_export(session_id, output_dir, **extra_data)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print(f"Certificate: {cert['hmac_sha256'][:32]}...")
    print(f"Output dir:  {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
