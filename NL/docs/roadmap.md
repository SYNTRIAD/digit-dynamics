# Roadmap: Autonomous Discovery Engine v15.0 → v16.0
## R12 Sessie

---

## Huidige staat (v15.0 / R11)

- **79 KB-feiten** (65 bewezen), DS011–DS068
- **16 invarianten** per fixed point
- **19 analysefasen** (incl. Pad B + Pad D + Pad E)
- **30 modules** (A–Z + R11 modules)
- **22 operaties**
- **117 unit tests** (100% passing)
- **12/12 formele bewijzen** computationeel geverifieerd
- **Multi-base support**: b ∈ {5..16}
- **Armstrong numbers**: catalogus k=1..7, k_max formule bewezen
- **Kaprekar**: 3-digit (495), 4-digit (6174), 6-digit (549945, 631764)
- **Universele Lyapunov**: digit_sum conditioneel bewezen (DS061)
- **Repunits**: nooit CC FPs (DS055, bewezen)
- **4 oneindige FP-families**: symmetric, 1089×m, sort_desc, palindromen (DS064)

### Nieuwe resultaten R11 (PAD E — Open Vragen)
- **DS061**: digit_sum Lyapunov — conditioneel bewezen (NIET universeel)
- **DS062**: sort_desc FPs — oneindige familie, formule C(k+9,k)-1 bewezen
- **DS063**: palindromen — oneindige FP-familie van reverse, formule bewezen
- **DS064**: 4 disjuncte oneindige FP-families bewezen
- **DS065**: Armstrong k_max formule — k_max(b) = max{k : k×(b-1)^k ≥ b^(k-1)} bewezen
- **DS066**: Kaprekar 6-digit — twee FPs (549945, 631764) exhaustief geverifieerd
- **DS067**: alle Kaprekar FPs deelbaar door 9 (mod 9 invariant)
- **DS068**: Kaprekar FP-count per digit-lengte onregelmatig (geen formule)

### Bewezen resultaten (R7–R10)
- **DS034**: Symmetrische FP-formule (b-2)×b^(k-1) voor ELKE basis b≥3
- **DS035**: CC getallen deelbaar door (b-1) in elke basis
- **DS036/037**: Involuties comp∘comp en rev∘rev met randgevallen
- **DS038–DS045**: Lyapunov-bounds digit_pow2–5 en digit_factorial_sum
- **DS039**: Kaprekar K_b = (b/2)(b²-1) algebraïsch bewezen
- **DS040**: 1089-familie is **UNIVERSEEL** voor alle bases b≥3
- **DS041**: Odd-length rev∘comp = ∅ voor even bases
- **DS046**: Armstrong numbers eindig per k (Lyapunov-argument)
- **DS047/048**: Armstrong k=3 en k=4 exhaustief geverifieerd
- **DS049**: Even bases Kaprekar-FP is uniek
- **DS050**: Oneven bases Kaprekar: cycli en FPs (EMPIRISCH)
- **DS052**: Odd-length rev∘comp FPs bestaan WEL in oneven bases
- **DS055**: Repunits R_k zijn NOOIT CC FPs (bewezen)
- **DS056**: (b-1)×R_k altijd palindroom, nooit CC FP (bewezen)
- **DS057**: Kaprekar 4-digit = 6174, ≤7 stappen (bewezen)

---

## ✅ PAD A — DIEPER: VOLTOOID (R8)

| # | Taak | Resultaat | Status |
|---|------|-----------|--------|
| A1 | Kaprekar-constanten formaliseren | DS039 → PROVEN | ✅ |
| A2 | 1089-universaliteit bewijzen | DS040 → PROVEN + GECORRIGEERD | ✅ |
| A3 | Odd-length rev∘comp = ∅ | DS041 PROVEN | ✅ |
| A4 | Lyapunov digit_pow3/4/5 | DS042–DS044 PROVEN | ✅ |
| A5 | Lyapunov digit_factorial_sum | DS045 PROVEN | ✅ |

## ✅ PAD B — BREDER: VOLTOOID (R9)

| # | Taak | Resultaat | Status |
|---|------|-----------|--------|
| B1 | Parametrische bifurcatie | NarcissisticAnalyzer (Module S) | ✅ |
| B2 | Narcissistische getallen | Armstrong k=1..7 catalogus, DS046–DS048 | ✅ |
| B3 | Orbitdynamica | OrbitAnalyzer (Module U), convergentietijden | ✅ |
| B4 | Nieuwe operaties | digit_gcd, digit_xor, narcissistic_step (22 ops) | ✅ |
| B5 | Oneven bases Kaprekar | OddBaseKaprekarAnalyzer (Module T), DS049–DS050 | ✅ |

## ✅ PAD D — DIEPER²: VOLTOOID (R10)

| # | Taak | Resultaat | Status |
|---|------|-----------|--------|
| D1 | Langere pipelines | ExtendedPipelineAnalyzer (Module V), DS053 | ✅ |
| D2 | Universele Lyapunov | UniversalLyapunovSearch (Module W), DS054 | ✅ |
| D3 | Repunit-verband | RepunitAnalyzer (Module X), DS055–DS056 | ✅ |
| D4 | Attractorcyclus-classificatie | CycleTaxonomy (Module Y), DS059 | ✅ |
| D5 | 4+ digit Kaprekar | MultiDigitKaprekar (Module Z), DS057–DS058, DS060 | ✅ |

---

## ✅ PAD E — OPEN VRAGEN: VOLTOOID (R11)

| # | Taak | Resultaat | Status |
|---|------|-----------|--------|
| E1 | Kaprekar d>3 algebraïsche analyse | KaprekarAlgebraicAnalyzer, DS066-DS068 | ✅ |
| E2 | 3e+ oneindige FP-familie | ThirdFamilySearcher, DS062-DS064 | ✅ |
| E3 | digit_sum Lyapunov bewijs | DigitSumLyapunovProof, DS061 | ✅ |
| E4 | Armstrong k_max bounds | ArmstrongBoundAnalyzer, DS065 | ✅ |

### R11 Ontdekkingen

**Kaprekar 6-digit (549945, 631764):**
- 549945 = 3² × 5 × 11² × 101 — **palindroom!** — ds=36, ÷9, ÷11
- 631764 = 2² × 3² × 7 × 23 × 109 — ds=27, ÷9, NIET ÷11
- Geen algebraïsche formule gevonden — FP-count per d is onregelmatig
- Pair_sums zijn NIET constant → geen eenvoudige symmetrie

**4 oneindige FP-families:**
1. Symmetric rev∘comp: d_i + d_{2k+1-i} = 9 → (b-2)×b^(k-1) per digit-lengte
2. 1089×m multiplicatief: A_b × m voor m=1..b-1
3. sort_desc FPs: niet-stijgende digits → C(k+9,k)-1 per digit-lengte
4. Palindromen: reverse-invariant → 9×10^(floor((k-1)/2)) per digit-lengte

**digit_sum Lyapunov:**
- NIET universeel — complement_9, kaprekar_step, truc_1089 verhogen ds
- CONDITIONEEL bewezen voor ds-niet-toenemende pipelines

**Armstrong k_max:**
- k_max(10) = 60, k_max(2) = 2, k_max(16) = 116
- Formule: k_max(b) = max{k : k×(b-1)^k ≥ b^(k-1)}
- k_max/b ratio groeit langzaam: ~6 voor b=10, ~7.25 voor b=16

---

## ✅ PAD C — PUBLICATIE: VOLTOOID (R11)

| # | Taak | Resultaat | Status |
|---|------|-----------|--------|
| C1 | Paper structuur | 12 secties, abstract met 8 theorems | ✅ |
| C2 | Hoofdstelling | Theorem 1 (DS034) volledig bewijs | ✅ |
| C3 | Nevenresultaten | Theorems 2–8 volledig uitgeschreven | ✅ |
| C4 | Methodologie-sectie | v15.0 engine beschrijving, 11 feedback rounds | ✅ |
| C5 | Paper draft v1.0 | `paper_draft.md` — 660 regels, publicatie-klaar | ✅ |

---

## Strategische paden (R12+)

### 📝 PAD F — SUBMISSION PREPARATION (SUPERSEDED)

> **Vervangen door:** `docs/ROADMAP_SUBMISSION.md` — gebaseerd op onafhankelijke technische audit
> (docs/SYNTRIAD_ENGINE_vNext_AUDIT_REPORT.md, 2026-02-25).
> PAD F items zijn volledig gedekt door het nieuwe actieplan (C1–C4, I1–I5, N1–N3).

| # | Taak | Beschrijving | Status |
|---|------|--------------|--------|
| F1 | LaTeX conversie | paper_draft.md → .tex met AMS-stijl | ✅ → paper_A.tex, paper_B.tex bestaan; finalisatie via C2 |
| F2 | Peer review | Onafhankelijke audit + taalcorrectie | ✅ → Audit rapport + C3 language fix |
| F3 | Code repository | Repo geherstructureerd (tests/, engines/, scripts/, papers/, docs/, data/) | ✅ → Phase 0 + C4 bundle cleanup |
| F4 | arXiv submission | Na alle audit-fixes | ⏳ → zie ROADMAP_SUBMISSION.md |

**Sterkste publicatie-claims:**
> 1. "Voor elke basis b≥3: het aantal FPs van rev∘comp met 2k cijfers
>    is precies (b-2)×b^(k-1). Voor oneven lengte in even bases: nul."
> 2. "De 1089-multiplicatieve familie (b-1)(b+1)²×m is UNIVERSEEL:
>    A_b×m heeft digits [m, m-1, (b-1)-m, b-m] en is CC in elke basis."
> 3. "Er bestaan minstens 4 disjuncte oneindige FP-families voor
>    digit-operatie pipelines, elk met bewezen telformule."
> 4. "Kaprekar K_b = (b/2)(b²-1) is algebraïsch bewezen als FP voor even b≥4."
> 5. "Armstrong k_max(b) = max{k : k×(b-1)^k ≥ b^(k-1)} is bewezen;
>    k_max(10) = 60 met complete catalogus k=1..7."
> 6. "digit_sum is conditioneel Lyapunov voor ds-niet-toenemende pipelines."
> 7. "Repunits R_k zijn NOOIT complement-gesloten FPs (bewezen)."
> 8. "Kaprekar 6-digit: twee FPs (549945 palindroom, 631764); geen formule."

---

## Uitvoeringsvolgorde

```
R8:  PAD A (A1–A5)  →  ✅ VOLTOOID. DS039–DS045, 12/12 bewijzen, 57 tests.
R9:  PAD B (B1–B5)  →  ✅ VOLTOOID. Modules S–U, DS046–DS052, 22 ops, 76 tests.
R10: PAD D (D1–D5)  →  ✅ VOLTOOID. Modules V–Z, DS053–DS060, 98 tests.
R11: PAD E (E1–E4)  →  ✅ VOLTOOID. Open vragen, DS061–DS068, 117 tests.
R11: PAD C (C1–C5)  →  ✅ VOLTOOID. Paper v1.0, 660 regels, 8 theorems.
R12: PAD F (F1–F4)  →  LaTeX conversie + arXiv submission
```

---

## Afgerond (NIET opnieuw doen)

| Item | Status | Sessie |
|------|--------|--------|
| Multi-base engine (BaseNDigitOps) | ✅ | R6 |
| SymbolicFPClassifier (10 condities) | ✅ | R6+R7 |
| LyapunovSearch (grid search) | ✅ | R6 |
| FamilyProof1089 (algebraïsch bewijs) | ✅ | R6 |
| FormalProofEngine (12/12 bewijzen) | ✅ | R7+R8 |
| DS034–DS045 PROVEN | ✅ | R7+R8 |
| DS040 GECORRIGEERD + UNIVERSEEL | ✅ | R8 |
| **PAD A voltooid (A1–A5)** | ✅ | **R8** |
| **57 unit tests** | ✅ | **R8** |
| **PAD B voltooid (B1–B5)** | ✅ | **R9** |
| **NarcissisticAnalyzer (Module S)** | ✅ | **R9** |
| **OddBaseKaprekarAnalyzer (Module T)** | ✅ | **R9** |
| **OrbitAnalyzer (Module U)** | ✅ | **R9** |
| **DS046–DS052** | ✅ | **R9** |
| **22 operaties** | ✅ | **R9** |
| **76 unit tests** | ✅ | **R9** |
| **README + roadmap v13.0** | ✅ | **R9** |
| **PAD D voltooid (D1–D5)** | ✅ | **R10** |
| **ExtendedPipelineAnalyzer (Module V)** | ✅ | **R10** |
| **UniversalLyapunovSearch (Module W)** | ✅ | **R10** |
| **RepunitAnalyzer (Module X)** | ✅ | **R10** |
| **CycleTaxonomy (Module Y)** | ✅ | **R10** |
| **MultiDigitKaprekar (Module Z)** | ✅ | **R10** |
| **DS053–DS060** | ✅ | **R10** |
| **98 unit tests** | ✅ | **R10** |
| **README + roadmap v14.0** | ✅ | **R10** |
| **PAD E voltooid (E1–E4)** | ✅ | **R11** |
| **KaprekarAlgebraicAnalyzer** | ✅ | **R11** |
| **ThirdFamilySearcher** | ✅ | **R11** |
| **DigitSumLyapunovProof** | ✅ | **R11** |
| **ArmstrongBoundAnalyzer** | ✅ | **R11** |
| **DS061–DS068** | ✅ | **R11** |
| **117 unit tests** | ✅ | **R11** |
| **README + roadmap v15.0** | ✅ | **R11** |
| **PAD C voltooid (C1–C5)** | ✅ | **R11** |
| **Paper draft v1.0 (660 regels, 8 theorems)** | ✅ | **R11** |
| **paper.tex (AMS-art LaTeX, arXiv-ready)** | ✅ | **R11** |

---

## Open wiskundige vragen

1. ~~Waarom zijn 1089×m complement-gesloten?~~ → **BEWEZEN (DS024)**
2. ~~Bestaan analoge families in andere bases?~~ → **JA! UNIVERSEEL (DS040)**
3. ~~Is er een verband tussen repunits (111...1) en complement-gesloten families?~~ → **NEE: repunits nooit CC FPs (DS055)**
4. ~~Kunnen we voor elke pipeline een FP-conditie afleiden?~~ → **10 condities bewezen (Module O)**
5. ~~Bestaat er een universele Lyapunov-functie voor alle convergente pipelines?~~ → **digit_sum beste kandidaat, maar niet 100% universeel (DS054)**
6. ~~Zijn Kaprekar-constanten bewezen per basis?~~ → **JA, even b (DS039). Oneven b: cycli (DS050)**
7. ~~Waarom faalt de 1089-structuur in andere bases?~~ → **FAALT NIET! Universeel (DS040)**
8. ~~Hebben odd-length getallen ooit rev∘comp FPs?~~ → **NEE in even bases (DS041). JA in oneven (DS052)**
9. ~~Wat zijn de Kaprekar-constanten voor oneven bases?~~ → **Geanalyseerd: mix van FPs en cycli (DS050, Module T)**
10. ~~Zijn er meer dan 2 disjuncte oneindige FP-families?~~ → **JA! Minstens 4 families (DS064)**
11. ~~Wat is de exacte bovengrens voor Armstrong numbers (k_max in basis b)?~~ → **k_max(b) = max{k : k×(b-1)^k ≥ b^(k-1)} (DS065)**
12. Bestaat er een gesloten formule voor het aantal Armstrong numbers per k? → **OPEN — count-reeks is onregelmatig**
13. ~~Kan digit_sum als Lyapunov bewezen worden (niet alleen empirisch)?~~ → **CONDITIONEEL BEWEZEN (DS061)**
14. ~~Bestaat er een algebraïsche formule voor Kaprekar-constanten bij d>3?~~ → **NEE voor d>4 — FP-count onregelmatig, geen formule (DS068)**
15. Bestaat er een gesloten formule voor Kaprekar FP-count als functie van d? → **OPEN — onregelmatig (DS068)**
16. Is 549945 (6-digit Kaprekar palindroom) algebraïsch verklaarbaar? → **OPEN**
