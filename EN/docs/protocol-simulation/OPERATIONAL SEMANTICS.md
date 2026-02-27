# OPERATIONAL SEMANTICS

**Research Protocol ID:** R2.0-DDS-20260225  
**Engine Semantic Version:** 2.0 → **2.1** (projection semantics formalized)  
**Date:** 2026-02-25  

This document formalizes the operational semantics of digit-dynamical pipelines under fixed-length projection dynamics.

---

## 1. State Space

### Domain
For base $b$ and digit length $k$:

$$
D(b,k) = \{n \in \mathbb{N} \mid b^{k-1} \leq n \leq b^k - 1\}
$$

This is the set of all $k$-digit numbers in base $b$ (exact digit length).

### Digit Representation
For any $n \in \mathbb{N}$:

$$
d(n) = [d_{k-1}, d_{k-2}, \ldots, d_1, d_0] \in \{0, \ldots, b-1\}^k
$$

where $n = \sum_{i=0}^{k-1} d_i \cdot b^i$.

---

## 2. Canonicalization Operator

### Definition

The **canonicalization operator** $\text{canon}_k : \mathbb{N} \to D(b,k)$ projects any natural number into the $k$-digit representation via zero-padding:

$$
\text{canon}_k(x) = \text{from\_digits}(\text{to\_digits}(x, k))
$$

where:
- $\text{to\_digits}(x, k)$ converts $x$ to a digit vector of length $k$ (zero-padded if necessary)
- $\text{from\_digits}(d)$ converts digit vector $d$ back to a natural number

### Example

For $b=10$, $k=2$:

$$
\text{canon}_2(1) = \text{from\_digits}([0, 1]) = 01_{10} = 1
$$

But when composed with reverse:

$$
\text{reverse}(\text{canon}_2(1)) = \text{reverse}([0, 1]) = [1, 0] = 10
$$

---

## 3. Pipeline Evaluation Semantics

### Formal Definition

For a pipeline $O_1 \circ O_2 \circ \cdots \circ O_m$, the evaluation proceeds as:

$$
\begin{align}
x_0 &:= n \\
x_i &:= \text{from\_digits}\big(\text{canon}_k(O_i(x_{i-1}))\big) \quad \text{for } i = 1, \ldots, m \\
f(n) &:= x_m
\end{align}
$$

**Key insight:** After each operator $O_i$, the result is **projected back into $k$-digit representation** via $\text{canon}_k$.

### Implicit Projection

This means every pipeline is actually:

$$
\boxed{f = O_m \circ P_k \circ O_{m-1} \circ P_k \circ \cdots \circ P_k \circ O_1}
$$

where $P_k = \text{canon}_k$ is the projection operator.

---

## 4. Crucial Consequences

### 4.1 Non-Commutativity

**Without projection:**
$$
\text{digit\_sum} \circ \text{reverse} = \text{reverse} \circ \text{digit\_sum}
$$
(because digit sum is permutation-invariant)

**With projection:**
$$
\text{digit\_sum} \circ P_k \circ \text{reverse} \neq \text{reverse} \circ P_k \circ \text{digit\_sum}
$$

**Example:** For $n=10$, $b=10$, $k=2$:
- $\text{digit\_sum}(\text{reverse}(10)) = \text{digit\_sum}(1) = 1 \neq 10$
- $\text{reverse}(P_2(\text{digit\_sum}(10))) = \text{reverse}(P_2(1)) = \text{reverse}([0,1]) = 10$ ✓

### 4.2 Operator Algebra Not Closed

The set of digit operators is **not closed** under composition with $P_k$:

- Pure operators: $\{\text{reverse}, \text{sort}, \text{digit\_sum}, \ldots\}$
- Projected operators: $\{O \circ P_k \mid O \in \text{operators}\}$

The projection $P_k$ introduces **domain-dependent dynamics** that change the algebraic structure.

### 4.3 Intermediate Values Outside Domain

Operators like `digit_sum`, `digit_gcd`, `digit_xor` produce values in $[0, k(b-1)]$, which is typically $\ll b^{k-1}$.

**Without $P_k$:** These values would be outside $D(b,k)$.  
**With $P_k$:** They are zero-padded back into $D(b,k)$.

### 4.4 Projected Dynamical System

This semantics is analogous to **dynamics on a quotient space**:

$$
f : D(b,k) / \sim \to D(b,k) / \sim
$$

where $x \sim y$ if $\text{canon}_k(x) = \text{canon}_k(y)$.

---

## 5. Impact on Structural Analysis

### 5.1 Contractive Operators

An operator $O$ is **contractive** if:

$$
O(n) < n \quad \text{for all } n \in D(b,k)
$$

**Examples:** `digit_sum`, `digit_gcd`, `digit_xor`

**Under projection:** Contractive operators become **projection-dependent**:
- $O(n)$ may be small (e.g., single-digit)
- $P_k(O(n))$ pads it back to $k$ digits
- Composition with other operators creates **emergent fixed points**

### 5.2 Symmetry-Driven Fixed Points

Operators like `reverse` create **symmetry** under projection:

$$
\text{reverse}(P_k(d)) = d \cdot b^{k-1} \quad \text{for single-digit } d
$$

This is the mechanism behind C003's fixed points.

---

## 6. Semantic Version Update

### Engine Semantic Version

**Previous:** `engine_semantic_version = "2.0"` (undocumented projection)  
**Current:** `engine_semantic_version = "2.1"` (formalized projection semantics)

### Manifest Hash Impact

**Manifest hashes remain unchanged** because the projection semantics were always present in the executor. This document **formalizes** what was already implemented.

### Audit Note

> "We discovered implicit projection semantics during structural analysis. The projection operator $P_k$ (zero-padding to $k$ digits) was always present in the executor but not formally documented. This document formalizes the operational semantics, and previous structural interpretations have been updated accordingly."

---

## 7. Implications for Conjectures

### 7.1 Empty Set Conjectures (S2)

Conjectures like C007-C012 remain **S2 (algebraically necessary)** because:

$$
O(n) < b^{k-1} \quad \Rightarrow \quad O(n) \neq n
$$

The projection $P_k$ does not change this inequality.

### 7.2 Formula Conjectures (S0 → S3)

Conjectures like C003 were **S0 (empirical only)** under the assumption of pure composition.

**Under projection semantics:** They become **S3 (algebraically necessary)** because the fixed points are **structurally derived** from the projection dynamics.

---

## 8. Recommendations

### 8.1 For Future Research

1. **Explicitly document $P_k$** in all pipeline specifications
2. **Distinguish pure vs. projected operators** in operator taxonomy
3. **Investigate projection-free semantics** (variable-length digits) as alternative
4. **Formalize quotient space dynamics** for theoretical analysis

### 8.2 For Current Research 2.0

1. **Update all conjecture statements** to reference projection semantics
2. **Re-classify conjectures** under correct structural strength (S0 → S3 where applicable)
3. **Bump engine_semantic_version to 2.1** in all manifests
4. **Add this document to reproducibility bundle**

---

**End of Operational Semantics Document**
