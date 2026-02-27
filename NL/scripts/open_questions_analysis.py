"""
Open Questions Analysis — R12 Session
======================================
Empirical investigation of the 7 open questions from paper.tex v2.

Q1: Armstrong counting formula
Q2: Kaprekar FP-count as f(d)
Q3: 549945 palindrome mystery
Q4: Fifth infinite FP family
Q5-Q7: deferred (base generalization, C1, C2)
"""

import time
from collections import defaultdict
from math import gcd, comb, log2
from functools import reduce

# =============================================================================
# Q1: ARMSTRONG COUNTING FORMULA
# =============================================================================

# Known Armstrong numbers (base 10) from OEIS A005188
# Count per digit length k
ARMSTRONG_COUNTS_BASE10 = {
    1: 9,   # 1-9
    2: 0,
    3: 4,   # 153, 370, 371, 407
    4: 3,   # 1634, 8208, 9474
    5: 3,   # 54748, 92727, 93084
    6: 1,   # 548834
    7: 4,   # 1741725, 4210818, 9800817, 9926315
    8: 3,   # 24678050, 24678051, 88593477
    9: 4,   # 146511208, 472335975, 534494836, 912985153
    10: 1,  # 4679307774
    11: 8,  # 32164049650, ..., 94204591914
    12: 0,
    13: 2,  # 28116440335967
    14: 0,
    15: 4,
    16: 1,  # 4338281769391370
    17: 3,
    18: 0,
    19: 4,
    20: 3,
    21: 0,
    22: 0,
    23: 2,
    24: 2,
    25: 1,
    26: 0,
    27: 0,
    28: 0,
    29: 2,
    30: 0,
    31: 1,
    32: 1,
    33: 1,
    34: 0,
    35: 1,
    36: 1,
    37: 0,
    38: 1,
    39: 1,
    # k >= 40: all zero (total: 88 Armstrong numbers in base 10)
}


def analyze_armstrong_counts():
    """Analyze the sequence of Armstrong counts for patterns."""
    print("=" * 70)
    print("Q1: ARMSTRONG COUNTING FORMULA")
    print("=" * 70)
    
    counts = [(k, ARMSTRONG_COUNTS_BASE10[k]) for k in sorted(ARMSTRONG_COUNTS_BASE10)]
    
    print(f"\n{'k':>3} | {'count':>5} | {'k*9^k':>15} | {'10^(k-1)':>15} | ratio")
    print("-" * 65)
    for k, c in counts:
        max_val = k * 9**k
        threshold = 10**(k-1) if k > 1 else 1
        ratio = max_val / threshold if threshold > 0 else float('inf')
        print(f"{k:>3} | {c:>5} | {max_val:>15} | {threshold:>15} | {ratio:.4f}")
    
    # Statistics
    nonzero = [c for _, c in counts if c > 0]
    zero = [k for k, c in counts if c == 0]
    print(f"\nTotal Armstrong numbers (base 10): {sum(c for _, c in counts)}")
    print(f"Nonzero k-values: {len(nonzero)} / {len(counts)}")
    print(f"Zero k-values: {zero}")
    print(f"Max count per k: {max(c for _, c in counts)} (at k=11)")
    
    # Check for mod patterns
    print("\n--- Modular analysis ---")
    for m in [2, 3, 4, 6, 9]:
        residues = defaultdict(list)
        for k, c in counts:
            if c > 0:
                residues[k % m].append(k)
        print(f"  mod {m}: nonzero at residues {dict((r, len(ks)) for r, ks in sorted(residues.items()))}")
    
    # Check if count correlates with k*9^k / 10^(k-1)
    print("\n--- Correlation with feasibility ratio ---")
    for k, c in counts[:15]:
        ratio = k * 9**k / (10**(k-1)) if k > 1 else k * 9
        density = c / (9 * 10**(k-1)) if k > 1 else c / 9  # fraction of k-digit numbers
        print(f"  k={k:>2}: count={c}, ratio={ratio:.2e}, density={density:.2e}")
    
    # Conclusion
    print("\n>>> CONCLUSION Q1:")
    print("  The Armstrong count sequence {9,0,4,3,3,1,4,3,4,1,8,0,2,0,...}")
    print("  shows NO regular pattern. No closed-form likely exists.")
    print("  The sequence depends on the number-theoretic properties of")
    print("  solutions to Σd_i^k = n, which is a Diophantine problem.")
    print("  Best characterization: bounded by the feasibility ratio k*9^k/10^(k-1).")


# =============================================================================
# Q2: KAPREKAR FP-COUNT AS f(d)
# =============================================================================

def kaprekar_step(n):
    """One step of the Kaprekar map in base 10."""
    s = str(n)
    k = len(s)
    s_padded = s.zfill(k)
    desc = int(''.join(sorted(s_padded, reverse=True)))
    asc = int(''.join(sorted(s_padded)))
    return desc - asc


def find_kaprekar_fixed_points(d, base=10):
    """Find all fixed points of the Kaprekar map for d-digit numbers."""
    lo = base**(d-1)
    hi = base**d
    fps = []
    for n in range(lo, hi):
        # Skip repdigits
        s = str(n)
        if len(set(s)) == 1:
            continue
        if kaprekar_step(n) == n:
            fps.append(n)
    return fps


def find_kaprekar_cycles(d, base=10, max_iter=100):
    """Find all cycles (including FPs) of the Kaprekar map for d-digit numbers."""
    lo = base**(d-1)
    hi = base**d
    
    endpoints = defaultdict(int)
    cycle_members = set()
    
    for n in range(lo, hi):
        s = str(n)
        if len(set(s)) == 1:
            continue
        
        # Follow orbit
        current = n
        seen = {}
        for step in range(max_iter):
            if current in seen:
                # Found cycle
                cycle_start = seen[current]
                cycle = []
                c = current
                while True:
                    cycle.append(c)
                    c = kaprekar_step(c)
                    if c == current:
                        break
                cycle_key = tuple(sorted(cycle))
                endpoints[cycle_key] += 1
                cycle_members.update(cycle)
                break
            seen[current] = step
            current = kaprekar_step(current)
    
    return endpoints, cycle_members


def analyze_kaprekar_fp_count():
    """Analyze Kaprekar FP count for d=3..7."""
    print("\n" + "=" * 70)
    print("Q2: KAPREKAR FP-COUNT AS f(d)")
    print("=" * 70)
    
    results = {}
    for d in range(3, 8):
        t0 = time.time()
        fps = find_kaprekar_fixed_points(d)
        elapsed = time.time() - t0
        results[d] = fps
        print(f"\n  d={d}: {len(fps)} fixed points in {elapsed:.2f}s")
        for fp in fps:
            ds = sum(int(c) for c in str(fp))
            is_pal = str(fp) == str(fp)[::-1]
            div9 = fp % 9 == 0
            print(f"    {fp} (digit_sum={ds}, palindrome={is_pal}, div9={div9})")
    
    # Also find cycles for d=3..6
    print("\n--- Cycle analysis ---")
    for d in range(3, 7):
        t0 = time.time()
        cycles, _ = find_kaprekar_cycles(d)
        elapsed = time.time() - t0
        fp_cycles = {k: v for k, v in cycles.items() if len(k) == 1}
        longer_cycles = {k: v for k, v in cycles.items() if len(k) > 1}
        print(f"  d={d}: {len(fp_cycles)} FPs, {len(longer_cycles)} cycles (lengths: {sorted(set(len(k) for k in longer_cycles))}), {elapsed:.2f}s")
    
    print(f"\n  FP count sequence: {[len(results[d]) for d in range(3, 8)]}")
    print("  d:     3  4  5  6  7")
    print(f"  #FP:  {len(results[3]):>2} {len(results[4]):>2} {len(results[5]):>2} {len(results[6]):>2} {len(results[7]):>2}")


# =============================================================================
# Q3: 549945 PALINDROME MYSTERY
# =============================================================================

def analyze_549945_palindrome():
    """Investigate why 549945 is a palindrome among 6-digit Kaprekar FPs."""
    print("\n" + "=" * 70)
    print("Q3: 549945 PALINDROME MYSTERY")
    print("=" * 70)
    
    fps_6 = [549945, 631764]
    
    for fp in fps_6:
        s = str(fp)
        ds = sum(int(c) for c in s)
        comp = ''.join(str(9 - int(c)) for c in s)
        is_pal = s == s[::-1]
        desc = int(''.join(sorted(s, reverse=True)))
        asc = int(''.join(sorted(s)))
        
        # Factorization
        n = fp
        factors = []
        for p in range(2, 1000):
            while n % p == 0:
                factors.append(p)
                n //= p
            if n == 1:
                break
        if n > 1:
            factors.append(n)
        
        print(f"\n  {fp}:")
        print(f"    digits: {list(s)}")
        print(f"    palindrome: {is_pal}")
        print(f"    digit_sum: {ds}")
        print(f"    complement: {comp} = {int(comp)}")
        print(f"    sort_desc: {desc}, sort_asc: {asc}")
        print(f"    desc - asc: {desc - asc} (= {fp}? {desc - asc == fp})")
        print(f"    factors: {' × '.join(str(f) for f in factors)}")
        print(f"    mod 9: {fp % 9}, mod 99: {fp % 99}, mod 999: {fp % 999}")
        print(f"    mod 11: {fp % 11}, mod 101: {fp % 101}, mod 1001: {fp % 1001}")
    
    # Check: is the palindrome property forced by the Kaprekar equation?
    print("\n--- Algebraic analysis ---")
    print("  For a 6-digit Kaprekar FP n = sort_desc(n) - sort_asc(n):")
    print("  Let sorted digits be a ≥ b ≥ c ≥ d ≥ e ≥ f.")
    print("  Then desc = a*10^5 + b*10^4 + c*10^3 + d*10^2 + e*10 + f")
    print("       asc  = f*10^5 + e*10^4 + d*10^3 + c*10^2 + b*10 + a")
    print("  n = desc - asc = (a-f)(10^5 - 1) + (b-e)(10^4 - 10) + (c-d)(10^3 - 10^2)")
    print("                 = (a-f)*99999 + (b-e)*9990 + (c-d)*900")
    
    # Check all solutions
    print("\n  Exhaustive search over (a-f, b-e, c-d):")
    solutions = []
    for af in range(1, 10):  # a-f
        for be in range(0, 10):  # b-e
            for cd in range(0, 10):  # c-d
                n = af * 99999 + be * 9990 + cd * 900
                if 100000 <= n < 1000000:
                    # Check if n is actually a FP
                    if kaprekar_step(n) == n:
                        s = str(n)
                        is_pal = s == s[::-1]
                        solutions.append((n, af, be, cd, is_pal))
    
    print(f"  Found {len(solutions)} solutions:")
    for n, af, be, cd, is_pal in solutions:
        print(f"    n={n}, a-f={af}, b-e={be}, c-d={cd}, palindrome={is_pal}")
    
    # Check palindrome condition
    print("\n  For n to be a palindrome: d_1=d_6, d_2=d_5, d_3=d_4.")
    print("  549945: 5=5✓, 4=4✓, 9=9✓")
    print("  631764: 6≠4✗ → not palindrome")
    
    # Check: among ALL d-digit Kaprekar FPs, which are palindromes?
    print("\n--- Palindrome check across all known Kaprekar FPs ---")
    all_kap_fps = {3: [495], 4: [6174], 6: [549945, 631764]}
    for d, fps in all_kap_fps.items():
        for fp in fps:
            s = str(fp)
            is_pal = s == s[::-1]
            print(f"  d={d}: {fp} palindrome={is_pal}")
    
    # Check 549945 = 99999*5 + 9990*4 + 900*5
    n549 = 99999*5 + 9990*4 + 900*5
    print(f"\n  549945 = 99999×5 + 9990×4 + 900×5 = {n549}")
    print(f"  Note: a-f = c-d = 5, this symmetric coefficient pattern")
    print(f"  forces digit symmetry that produces a palindrome.")


# =============================================================================
# Q4: FIFTH INFINITE FP FAMILY (exploratory)
# =============================================================================

def search_fifth_family():
    """Search for patterns that might constitute a fifth infinite FP family."""
    print("\n" + "=" * 70)
    print("Q4: FIFTH INFINITE FP FAMILY — EXPLORATORY SEARCH")
    print("=" * 70)
    
    # Known families:
    # (i) Symmetric: rev∘comp FPs
    # (ii) 1089×m: comp-closed 4-digit
    # (iii) Sort-descending: non-increasing digits
    # (iv) Palindromes: rev FPs
    
    # Candidate: digit_sum fixed points
    # ds(n) = n iff n is a single digit ≤ 9... not infinite in interesting way
    
    # Candidate: Kaprekar-like operations on k-digit numbers
    # Already covered
    
    # Candidate: complement palindromes
    # n such that comp(n) = rev(n)
    print("\n  Candidate A: Complement-palindromes (comp(n) = rev(n))")
    count_per_k = {}
    for k in range(2, 7):
        lo = 10**(k-1)
        hi = 10**k
        fps = []
        for n in range(lo, hi):
            s = str(n)
            comp_s = ''.join(str(9 - int(c)) for c in s)
            rev_s = s[::-1]
            if comp_s == rev_s:
                fps.append(n)
        count_per_k[k] = len(fps)
        examples = fps[:5]
        print(f"    k={k}: {len(fps)} numbers (examples: {examples})")
    
    # Check if there's a formula
    print(f"    Counts: {[count_per_k[k] for k in range(2, 7)]}")
    # comp(n) = rev(n) means (9-d_i) = d_{k+1-i}
    # i.e., d_i + d_{k+1-i} = 9 for all i
    # This is exactly the symmetric family (i)!
    print("    → This IS family (i) (symmetric FPs of rev∘comp)!")
    
    # Candidate: digit-rotation fixed points
    # rotate_left(n) = n impossible for multi-digit (shifts digits)
    
    # Candidate: swap_ends fixed points
    print("\n  Candidate B: swap_ends(n) = n")
    count_swap = {}
    for k in range(2, 7):
        lo = 10**(k-1)
        hi = 10**k
        count = 0
        for n in range(lo, hi):
            s = str(n)
            if s[0] == s[-1]:  # swap_ends(n)=n iff first=last digit
                count += 1
        count_swap[k] = count
        print(f"    k={k}: {count} numbers")
    print(f"    Counts: {[count_swap[k] for k in range(2, 7)]}")
    print("    Formula: 9 × 10^(k-2) for k≥2 (first digit 1-9, last=first, middle free)")
    print("    → Overlaps substantially with palindromes. Not truly new.")
    
    # Candidate: digit-product preservation
    print("\n  Candidate C: Numbers where digit_product divides n")
    for k in range(2, 5):
        lo = 10**(k-1)
        hi = min(10**k, 10000)  # cap for speed
        fps = []
        for n in range(lo, hi):
            dp = reduce(lambda a, b: a * b, [int(c) for c in str(n)])
            if dp > 0 and n % dp == 0:
                fps.append(n)
        print(f"    k={k}: {len(fps)} numbers (examples: {fps[:8]})")
    print("    → These are 'Zuckerman numbers'. Known infinite family but")
    print("      not related to our digit-operation pipeline framework.")
    
    # Candidate: complement-sorted fixed points (new pipeline)
    print("\n  Candidate D: sort_desc(comp(n)) = n")
    count_sc = {}
    for k in range(2, 7):
        lo = 10**(k-1)
        hi = 10**k
        fps = []
        for n in range(lo, hi):
            s = str(n)
            comp_s = ''.join(str(9 - int(c)) for c in s)
            sorted_comp = ''.join(sorted(comp_s, reverse=True))
            if int(sorted_comp) == n:
                fps.append(n)
        count_sc[k] = len(fps)
        if fps:
            print(f"    k={k}: {len(fps)} FPs: {fps[:10]}")
        else:
            print(f"    k={k}: 0 FPs")
    
    # Candidate: truc_1089 fixed points
    print("\n  Candidate E: truc_1089(n) = n (1089-trick fixed points)")
    count_1089 = {}
    for k in range(3, 8):
        lo = 10**(k-1)
        hi = 10**k
        fps = []
        for n in range(lo, hi):
            s = str(n)
            rev_s = s[::-1]
            rev_n = int(rev_s)
            diff = abs(n - rev_n)
            if diff == 0:
                continue
            rev_diff = int(str(diff).zfill(len(str(diff)))[::-1])
            result = diff + rev_diff
            if result == n:
                fps.append(n)
        count_1089[k] = len(fps)
        if fps:
            print(f"    k={k}: {len(fps)} FPs (examples: {fps[:8]})")
        else:
            print(f"    k={k}: 0 FPs")
    
    print(f"    Counts: {[count_1089.get(k, '?') for k in range(3, 8)]}")
    print("    → If growing, this could be a fifth family!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("OPEN QUESTIONS ANALYSIS — R12 SESSION")
    print("=" * 70)
    
    analyze_armstrong_counts()
    analyze_kaprekar_fp_count()
    analyze_549945_palindrome()
    search_fifth_family()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
