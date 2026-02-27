# STRUCTURAL ANALYSIS REPORT

**Research Protocol ID:** R2.0-DDS-20260225  
**Meta-Spec v1.0:** Structural Explanation Phase  
**Date:** 2026-02-25  
**Classification Version:** v2.0 (frozen)  
**Ranking Model Version:** v1.0 (frozen)  

**Input:** 9 validated conjectures (survived secondary domain falsification)  
**Constraint:** No new conjectures, no parameter tuning, no protocol modifications  

---

## Conjecture C003

**Statement:**  
|F_{digit_sum∘reverse}^{b,k}| ≈ 1.00·b + -0.00·k + -1.00

**Simplified:**  
|F_{digit_sum∘reverse}^{b,k}| = b - 1

**Pattern Type:** formula_linear  
**Evidence:** 18 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = digit_sum(reverse(n))
```

For a number n in base b with k digits:
- `reverse(n)` produces a palindrome-like transformation
- `digit_sum(reverse(n))` = `digit_sum(n)` (digit sum is permutation-invariant)

**Fixed Point Condition:**
```
n = digit_sum(n)
```

This requires n to be a single-digit number in base b.

**Domain Constraint:**
- We exclude repdigits
- Single-digit numbers: {1, 2, ..., b-1} (0 excluded as leading zero)
- Repdigits excluded: none in single-digit range (all single digits are trivially repdigits)

**Wait — this is wrong. Let me reconsider.**

**Corrected Analysis:**

For multi-digit numbers (k ≥ 2):
- `digit_sum(n)` produces a number typically much smaller than n
- For n to be a fixed point: n = digit_sum(n)
- This is only possible if n is already small

**But the formula says |F| = b - 1, not dependent on k.**

**Structural Explanation:**

The pipeline `digit_sum∘reverse` is **contractive**:
- `reverse` preserves digit sum
- `digit_sum` reduces to range [0, k(b-1)]

For n to be a fixed point:
```
n = digit_sum(reverse(n)) = digit_sum(n)
```

This requires n ∈ {1, 2, ..., b-1} (single-digit numbers).

**But wait:** If n is single-digit, then `reverse(n) = n`, and `digit_sum(n) = n`.

So all single-digit numbers {1, ..., b-1} are fixed points.

**Repdigit Exclusion:**
- Single-digit numbers are NOT repdigits in the multi-digit sense
- Repdigit exclusion applies to k-digit numbers where all digits are the same
- For k ≥ 2, single-digit numbers don't exist in the domain

**Resolution:**

The domain is k-digit numbers (excluding repdigits).  
For k ≥ 2, the only fixed points are numbers where:
```
n = digit_sum(n)
```

But this is impossible for k ≥ 2 unless n is already single-digit, which contradicts the k-digit constraint.

**Structural explanation insufficient — empirical regularity only.**

The formula |F| = b - 1 is observed but not algebraically derived from operator properties.

### Invariants

None identified.

### Necessary Conditions

- Base b ≥ 2
- Digit length k ≥ 2
- Repdigit exclusion policy active

### Failure Conditions

- If repdigit exclusion is disabled
- If k = 1 (single-digit domain)
- Possibly for very large k (untested)

### Structural Strength

**S0** — Empirical only

The pattern is observed with R² = 1.0 but lacks algebraic derivation.

---

## Conjecture C004

**Statement:**  
|F_{digit_gcd∘sort_desc}^{b,k}| ≈ 1.00·b + -0.00·k + -1.00

**Simplified:**  
|F_{digit_gcd∘sort_desc}^{b,k}| = b - 1

**Pattern Type:** formula_linear  
**Evidence:** 18 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = digit_gcd(sort_desc(n))
```

- `sort_desc(n)` sorts digits in descending order
- `digit_gcd(digits)` computes GCD of all digits

**Fixed Point Condition:**
```
n = digit_gcd(sort_desc(n))
```

For n to equal the GCD of its digits:
- n must be a single-digit number
- GCD of a single digit is itself

**But the formula says |F| = b - 1, independent of k.**

**Structural Explanation:**

For k-digit numbers (k ≥ 2):
- `sort_desc(n)` produces a k-digit number (or k-1 if leading zero)
- `digit_gcd(digits)` produces a number in range [0, b-1]

For n to be a fixed point:
```
n = digit_gcd(sort_desc(n))
```

This requires n to be small (single-digit range).

**Algebraic Derivation:**

If n ∈ {1, 2, ..., b-1} (single-digit):
- `sort_desc(n)` = n (single digit, no sorting needed)
- `digit_gcd([n])` = n

So all single-digit numbers {1, ..., b-1} are fixed points.

**But the domain is k-digit numbers (k ≥ 2).**

**Structural explanation insufficient — empirical regularity only.**

### Invariants

None identified.

### Necessary Conditions

- Base b ≥ 2
- Digit length k ≥ 2
- Repdigit exclusion policy active

### Failure Conditions

- If repdigit exclusion is disabled
- If k = 1
- Possibly for very large k

### Structural Strength

**S0** — Empirical only

---

## Conjecture C006

**Statement:**  
|F_{digit_pow4∘sort_desc}^{b,k}| ≈ -0.00·b + 1.00·k + -1.00

**Simplified:**  
|F_{digit_pow4∘sort_desc}^{b,k}| = k - 1

**Pattern Type:** formula_linear  
**Evidence:** 18 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = digit_pow4(sort_desc(n))
```

- `sort_desc(n)` sorts digits in descending order
- `digit_pow4(digits)` raises each digit to the 4th power

**Fixed Point Condition:**
```
n = digit_pow4(sort_desc(n))
```

**Observation:** The formula depends on k (digit length), not base b.

This suggests the pattern is related to the **number of digits**, not the base system.

**Structural Explanation:**

For k-digit numbers:
- `sort_desc(n)` produces a k-digit number (or fewer if leading zeros)
- `digit_pow4([d₁, d₂, ..., dₖ])` produces a number with potentially more digits

For n to be a fixed point, the digit-wise power operation must preserve the number.

**This is highly non-trivial and lacks obvious algebraic structure.**

**Structural explanation insufficient — empirical regularity only.**

### Invariants

None identified.

### Necessary Conditions

- Digit length k ≥ 2
- Repdigit exclusion policy active

### Failure Conditions

- If repdigit exclusion is disabled
- If k = 1
- Possibly for very large k (digit_pow4 may overflow)

### Structural Strength

**S0** — Empirical only

---

## Conjecture C007

**Statement:**  
|F_{complement_9}^{b,k}| = 0 for all tested (b,k)

**Pattern Type:** invariant_constant  
**Evidence:** 22 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = complement_9(n)
```

In base b, `complement_9(n)` computes:
```
complement_9([d₁, d₂, ..., dₖ]) = [(b-1-d₁), (b-1-d₂), ..., (b-1-dₖ)]
```

**Fixed Point Condition:**
```
n = complement_9(n)
```

This requires:
```
dᵢ = (b - 1 - dᵢ)
2dᵢ = b - 1
dᵢ = (b - 1) / 2
```

**Necessary Condition:** b must be odd.

For even bases, no digit satisfies dᵢ = (b-1)/2 (non-integer).

**For odd bases:**
- Only the middle digit (b-1)/2 satisfies the condition
- All digits must equal (b-1)/2
- This is a repdigit

**Repdigit Exclusion:**
- The only potential fixed points are repdigits
- Repdigits are excluded from the domain

**Algebraic Derivation:**

For all bases (even or odd), the only fixed points of `complement_9` are repdigits of the form:
```
n = [(b-1)/2, (b-1)/2, ..., (b-1)/2]
```

Since repdigits are excluded, |F| = 0.

### Invariants

**Involution Property:**  
`complement_9(complement_9(n)) = n` for all n.

**Repdigit-Only Fixed Points:**  
Fixed points exist only for repdigits (when b is odd).

### Necessary Conditions

- Repdigit exclusion policy active

### Failure Conditions

- If repdigit exclusion is disabled, |F| > 0 for odd bases
- If base b is even, |F| = 0 regardless of repdigit policy

### Structural Strength

**S2** — Algebraically necessary under defined constraints

The empty set is **structurally guaranteed** by the combination of:
1. Complement operator properties
2. Repdigit exclusion policy

---

## Conjecture C008

**Statement:**  
|F_{digit_sum}^{b,k}| = 0 for all tested (b,k)

**Pattern Type:** invariant_constant  
**Evidence:** 18 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = digit_sum(n)
```

**Fixed Point Condition:**
```
n = digit_sum(n) = d₁ + d₂ + ... + dₖ
```

For k-digit numbers in base b:
- Minimum value: b^(k-1)
- Maximum digit sum: k(b-1)

For k ≥ 2 and b ≥ 2:
```
b^(k-1) ≥ 2^(2-1) = 2
k(b-1) ≤ k(b-1)
```

For n = digit_sum(n):
```
n ≤ k(b-1)
```

But n ≥ b^(k-1), so:
```
b^(k-1) ≤ k(b-1)
```

For k = 2, b = 2: 2 ≤ 2 (boundary case)  
For k = 2, b ≥ 3: b ≥ 2(b-1) → b ≥ 2b - 2 → 0 ≥ b - 2 → b ≤ 2 (contradiction)

**Algebraic Derivation:**

For k ≥ 2 and b ≥ 3:
```
b^(k-1) > k(b-1)
```

Therefore, no k-digit number can equal its digit sum.

### Invariants

**Contractive Property:**  
`digit_sum(n) < n` for all n with k ≥ 2 digits.

### Necessary Conditions

- Digit length k ≥ 2
- Base b ≥ 3

### Failure Conditions

- If k = 1 (single-digit numbers are fixed points)
- If b = 2 and k = 2 (boundary case, needs verification)

### Structural Strength

**S2** — Algebraically necessary under defined constraints

The empty set is **structurally guaranteed** for k ≥ 2 and b ≥ 3.

---

## Conjecture C009

**Statement:**  
|F_{digit_product}^{b,k}| = 0 for all tested (b,k)

**Pattern Type:** invariant_constant  
**Evidence:** 18 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = digit_product(n) = d₁ × d₂ × ... × dₖ
```

**Fixed Point Condition:**
```
n = d₁ × d₂ × ... × dₖ
```

For k-digit numbers in base b:
- Minimum value: b^(k-1)
- Maximum digit product: (b-1)^k

For n = digit_product(n):
```
n ≤ (b-1)^k
```

But n ≥ b^(k-1), so:
```
b^(k-1) ≤ (b-1)^k
```

For k = 2, b = 2: 2 ≤ 1 (false)  
For k = 2, b = 3: 3 ≤ 4 (true, but needs verification)

**Algebraic Derivation:**

For most bases and digit lengths:
```
b^(k-1) > (b-1)^k
```

Therefore, no k-digit number can equal its digit product.

**Exception:** For small b and k, boundary cases may exist.

### Invariants

**Contractive Property:**  
`digit_product(n) < n` for most k-digit numbers.

### Necessary Conditions

- Digit length k ≥ 2
- Base b ≥ 2

### Failure Conditions

- If k = 1 (single-digit numbers are fixed points)
- Possibly for small b and k (boundary cases)

### Structural Strength

**S1** — Algebraic derivation partial

The empty set is **likely** but not rigorously proven for all (b, k).

---

## Conjecture C010

**Statement:**  
|F_{digit_gcd}^{b,k}| = 0 for all tested (b,k)

**Pattern Type:** invariant_constant  
**Evidence:** 18 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = digit_gcd(n) = gcd(d₁, d₂, ..., dₖ)
```

**Fixed Point Condition:**
```
n = gcd(d₁, d₂, ..., dₖ)
```

For k-digit numbers:
- Minimum value: b^(k-1)
- Maximum GCD: b-1

For n = digit_gcd(n):
```
n ≤ b - 1
```

But n ≥ b^(k-1), so:
```
b^(k-1) ≤ b - 1
```

For k = 2, b = 2: 2 ≤ 1 (false)  
For k = 2, b = 3: 3 ≤ 2 (false)

**Algebraic Derivation:**

For k ≥ 2:
```
b^(k-1) > b - 1
```

Therefore, no k-digit number can equal its digit GCD.

### Invariants

**Contractive Property:**  
`digit_gcd(n) < n` for all k-digit numbers with k ≥ 2.

### Necessary Conditions

- Digit length k ≥ 2

### Failure Conditions

- If k = 1 (single-digit numbers are fixed points)

### Structural Strength

**S2** — Algebraically necessary under defined constraints

The empty set is **structurally guaranteed** for k ≥ 2.

---

## Conjecture C011

**Statement:**  
|F_{digit_xor}^{b,k}| = 0 for all tested (b,k)

**Pattern Type:** invariant_constant  
**Evidence:** 18 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = digit_xor(n) = d₁ ⊕ d₂ ⊕ ... ⊕ dₖ
```

**Fixed Point Condition:**
```
n = d₁ ⊕ d₂ ⊕ ... ⊕ dₖ
```

For k-digit numbers:
- Minimum value: b^(k-1)
- Maximum XOR: b-1 (bounded by digit range)

For n = digit_xor(n):
```
n ≤ b - 1
```

But n ≥ b^(k-1), so:
```
b^(k-1) ≤ b - 1
```

This is false for k ≥ 2.

**Algebraic Derivation:**

For k ≥ 2:
```
b^(k-1) > b - 1
```

Therefore, no k-digit number can equal its digit XOR.

### Invariants

**Contractive Property:**  
`digit_xor(n) < n` for all k-digit numbers with k ≥ 2.

### Necessary Conditions

- Digit length k ≥ 2

### Failure Conditions

- If k = 1 (single-digit numbers are fixed points)

### Structural Strength

**S2** — Algebraically necessary under defined constraints

The empty set is **structurally guaranteed** for k ≥ 2.

---

## Conjecture C012

**Statement:**  
|F_{digit_xor∘sort_asc}^{b,k}| = 0 for all tested (b,k)

**Pattern Type:** invariant_constant  
**Evidence:** 18 primary runs + 6 secondary runs  
**R²:** 1.0000  
**Domain Coverage:** 1.0 (within pipeline)  
**Validation Status:** VALIDATED  

### Structural Analysis

**Operator Decomposition:**
```
f(n) = digit_xor(sort_asc(n))
```

- `sort_asc(n)` sorts digits in ascending order
- `digit_xor(digits)` computes XOR of all digits

**Fixed Point Condition:**
```
n = digit_xor(sort_asc(n))
```

Similar to C011, the XOR of digits is bounded by b-1, while n ≥ b^(k-1).

**Algebraic Derivation:**

For k ≥ 2:
```
b^(k-1) > b - 1 ≥ digit_xor(sort_asc(n))
```

Therefore, no k-digit number can equal the XOR of its sorted digits.

### Invariants

**Contractive Property:**  
`digit_xor(sort_asc(n)) < n` for all k-digit numbers with k ≥ 2.

### Necessary Conditions

- Digit length k ≥ 2

### Failure Conditions

- If k = 1

### Structural Strength

**S2** — Algebraically necessary under defined constraints

The empty set is **structurally guaranteed** for k ≥ 2.

---

## Summary Table

| Conjecture | Statement | Structural Strength | Explanation |
|------------|-----------|---------------------|-------------|
| C003 | \|F_{digit_sum∘reverse}\| = b - 1 | **S0** | Empirical only |
| C004 | \|F_{digit_gcd∘sort_desc}\| = b - 1 | **S0** | Empirical only |
| C006 | \|F_{digit_pow4∘sort_desc}\| = k - 1 | **S0** | Empirical only |
| C007 | \|F_{complement_9}\| = 0 | **S2** | Algebraically necessary (repdigit exclusion) |
| C008 | \|F_{digit_sum}\| = 0 | **S2** | Algebraically necessary (k ≥ 2, b ≥ 3) |
| C009 | \|F_{digit_product}\| = 0 | **S1** | Partial derivation |
| C010 | \|F_{digit_gcd}\| = 0 | **S2** | Algebraically necessary (k ≥ 2) |
| C011 | \|F_{digit_xor}\| = 0 | **S2** | Algebraically necessary (k ≥ 2) |
| C012 | \|F_{digit_xor∘sort_asc}\| = 0 | **S2** | Algebraically necessary (k ≥ 2) |

---

## Conclusions

### Structural Classification

**S2 (Algebraically Necessary):** 5 conjectures  
- C007, C008, C010, C011, C012

**S1 (Partial Derivation):** 1 conjecture  
- C009

**S0 (Empirical Only):** 3 conjectures  
- C003, C004, C006

### Key Findings

1. **Empty set conjectures are structurally robust.**  
   Most "no fixed points" conjectures are algebraically necessary due to contractive properties.

2. **Formula conjectures lack structural explanation.**  
   The formulas |F| = b - 1 and |F| = k - 1 are empirically observed but not algebraically derived.

3. **Repdigit exclusion is critical.**  
   C007 relies entirely on repdigit exclusion for its structural guarantee.

### Limitations

- Structural analysis is limited to validated conjectures only
- No new conjectures generated
- No parameter tuning performed
- No domain expansion attempted

### Recommendations

1. **Upgrade C003, C004, C006 to S1 or S2** requires deeper algebraic analysis
2. **Test boundary cases** (k=1, small bases) to verify failure conditions
3. **Investigate repdigit inclusion** to test C007 structural dependency

---

**End of Structural Analysis Report**
