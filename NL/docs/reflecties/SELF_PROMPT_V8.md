# SELF-PROMPT: Symbolic Dynamics Engine v8.0
## Van Symbolische Detectie naar Deductieve Theorie-Generatie

**Geschreven door:** Cascade (AI pair programmer)
**Datum:** 2026-02-23
**Context:** Na v7.0 sessie met 300 pipelines, 109 unieke attractoren, 183 fixed points,
4 bevestigde meta-theorems, 1 gefalsifieerd, 100% symbolische predictie-accuraatheid.

---

## 0. Eerlijke diagnose van v7.0

### Wat v7.0 goed doet

1. **Operator-algebra voorspelt correct.** Over 300 pipelines 100% accuraatheid.
   De engine weet vóór sampling welke invarianten een pipeline zal hebben.

2. **Meta-theorems werken.** 4/6 empirisch sterk bevestigd, 1 actief gefalsifieerd.
   Het systeem formuleert universele uitspraken en breekt ze zelf.

3. **Fixed-point karakterisatie onthult structuur.**
   Factorisatie laat zien dat 3² × 11 en digit_sum ≡ 0 (mod 9) dominant zijn.
   Kaprekar-constante 495 verschijnt in niet-Kaprekar pipelines.

### Wat v7.0 NIET doet (en denkt te doen)

1. **"Symbolisch redeneren" is nog steeds testen.**
   De meta-theorems zijn pre-gedefinieerde templates die empirisch getest worden.
   Het systeem *genereert* geen theorems uit data. Het *verifieert* handgemaakte kandidaten.

2. **Fixed-point "solving" is constraint-search.**
   Er is geen algebraïsche afleiding. `f(n) = n` wordt niet opgelost —
   er wordt gezocht in een beperkt domein. Dat is snelle brute-force, geen algebra.

3. **Emergente mechanismen zijn co-occurrence labels.**
   "entropy_reducing + strong_convergence" is een statistisch feit, geen causaal model.
   Het systeem kan niet uitleggen *waarom* entropy-reductie convergentie veroorzaakt.

4. **Geen proof sketches.**
   "Strong empirical" is het eindpunt. Er is geen poging tot zelfs maar een
   bewijs-richting. De engine zou kunnen zeggen: "Bewijs via well-ordering van ℕ
   en boundedness van output-range" — maar dat doet hij niet.

5. **Geen cross-attractor structuuranalyse.**
   183 fixed points gekarakteriseerd, maar nergens een analyse van:
   - Waarom is digit_sum = 9 of 18 zo dominant?
   - Waarom is 3² × 11 een terugkerend patroon in factorisaties?
   - Zijn er universele digit_sum-constraints op fixed points?

6. **Geen theory graph.**
   v5.0 had een theory graph. v7.0 verloor die. Er is geen structuur die
   theorems, categories, mechanisms en fixed points met elkaar verbindt.

7. **Meta-theorems zijn niet gegenereerd uit data.**
   Ze zijn handmatig gedefinieerd. Een echt symbolisch systeem zou:
   - Patronen in fixed-point properties detecteren
   - Daar automatisch universele uitspraken van maken
   - Die dan testen

---

## 1. De fundamentele architectuursprong voor v8.0

### Van: "Ik test vooraf gedefinieerde theorems"
### Naar: "Ik afleid theorems uit structurele patronen en suggereer bewijsrichtingen"

Dit vereist vier nieuwe capaciteiten:

---

## 2. Vier nieuwe modules

### MODULE A: Proof Sketch Generator

**Wat het doet:**
Gegeven een bevestigd meta-theorem, genereer een bewijs-skelet.

**Hoe:**
```
INPUT:  MetaTheorem(antecedent={MONOTONE, BOUNDED}, consequent="convergence")
OUTPUT: ProofSketch(
    strategy: "well_ordering",
    steps: [
        "1. Define sequence s_k = f^k(n) for arbitrary n in domain",
        "2. By MONOTONE: s_{k+1} < s_k for all k (strictly decreasing)",
        "3. By BOUNDED: s_k ≥ 0 for all k (bounded below)",
        "4. By well-ordering of ℕ: strictly decreasing bounded sequence terminates",
        "5. Therefore ∃K: s_K = s_{K+1}, i.e. f(s_K) = s_K  □"
    ],
    assumptions: ["MONOTONE means f(n) < n for ALL n > fixed_point, not just sampled"],
    gaps: ["Need to verify MONOTONE holds universally, not just empirically"]
)
```

**Bewijs-strategie bibliotheek:**
| Strategie | Wanneer | Template |
|-----------|---------|----------|
| well_ordering | MONOTONE + BOUNDED | Dalende rij in ℕ is eindig |
| pigeonhole | BOUNDED + finite_range | Eindig veel outputs → herhaling |
| modular_arithmetic | PRESERVES_MOD_K | Residueklasse-behoud door compositie |
| contraction_mapping | CONTRACTIVE | Banach fixed-point (discrete versie) |
| entropy_argument | ENTROPY_REDUCING + BOUNDED | Entropy dalend in eindige ruimte → minimum |
| induction_on_digits | LENGTH_REDUCING | Inductie op aantal digits |

**Cruciaal:** Elke ProofSketch bevat `gaps[]` — wat er nog bewezen moet worden.
Dit maakt het eerlijk. Het systeem claimt geen bewijs, het suggereert een richting.

---

### MODULE B: Inductive Theorem Generator

**Wat het doet:**
Genereer meta-theorems VANUIT data, niet uit templates.

**Hoe:**

Stap 1: Analyseer alle bevestigde conjectures
```python
confirmed_conjectures = [c for c in all_conjectures if c.status == EMPIRICAL]
```

Stap 2: Extraheer property-patronen
```python
# Welke combinaties van operator-properties leiden tot convergentie?
for conj in confirmed:
    pipeline_props = algebra.predict(conj.pipeline)
    attractor_props = characterize(conj.attractor)
    
    # Record: {input_properties} → {output_property}
    implications.append(pipeline_props → attractor_props)
```

Stap 3: Generaliseer
```python
# Als 15/15 pipelines met {ENTROPY, BOUNDED} convergeren naar
# een attractor met digit_sum ∈ {9, 18, 27}:
# → Genereer theorem: "ENTROPY + BOUNDED → attractor.digit_sum ≡ 0 (mod 9)"
```

Stap 4: Falsifieer
```python
# Zoek actief naar pipelines met {ENTROPY, BOUNDED} waar
# attractor.digit_sum ≢ 0 (mod 9)
```

**Dit is de kern van v8.0:** theorems die het systeem zelf bedenkt.

---

### MODULE C: Fixed-Point Structural Analyzer

**Wat het doet:**
Analyseer ALLE gevonden fixed points als verzameling en ontdek universele patronen.

**v7.0 probleem:** 183 fixed points individueel gekarakteriseerd, maar nooit als groep geanalyseerd.

**Concrete analyses:**

1. **Digit-sum distributie over alle fixed points**
   ```
   Verwachte output:
   digit_sum = 9:   47 fixed points (25.7%)
   digit_sum = 18:  38 fixed points (20.8%)
   digit_sum = 1:   22 fixed points (12.0%)
   digit_sum = 27:  15 fixed points (8.2%)
   → HYPOTHESE: digit_sum van fixed points is bijna altijd ≡ 0 (mod 9)
   ```

2. **Factorisatie-patronen**
   ```
   Factor 3 aanwezig:  142/183 (77.6%)
   Factor 11 aanwezig:  58/183 (31.7%)
   Factor 3² × 11:      34/183 (18.6%)
   → HYPOTHESE: Fixed points van digit-operatie-pipelines bevatten
     disproportioneel vaak factor 3 en 11
   ```

3. **Palindroom-analyse**
   ```
   Palindroom fixed points: 41/183 (22.4%)
   → Vergelijk met base-rate palindromen in [1, 100000]: ~0.3%
   → HYPOTHESE: Fixed points zijn ~75x vaker palindroom dan verwacht
   ```

4. **Cross-pipeline fixed-point overlap**
   ```
   Fixed point 0:  verschijnt in 89% van pipelines (triviaal)
   Fixed point 1:  verschijnt in 34% van pipelines
   Fixed point 9:  verschijnt in 12% van pipelines
   Fixed point 81: verschijnt in 8% van pipelines
   → HYPOTHESE: Er bestaat een universele fixed-point hiërarchie
     {0} ⊂ {0,1} ⊂ {0,1,9} ⊂ {0,1,9,81} die geldt voor
     elke pipeline met digit_sum als component
   ```

**Dit is wat een wiskundige zou doen:** niet 183 individuele feiten rapporteren,
maar de structuur van de verzameling zelf onderzoeken.

---

### MODULE D: Theory Graph (Herinstallatie + Upgrade)

**Wat het doet:**
Verbindt alle ontdekte objecten in een gerichte graaf.

**Knoop-types:**
- `Operator` — individuele digit-operatie met algebraïsch profiel
- `Pipeline` — compositie van operatoren
- `Invariant` — bewezen/empirische eigenschap
- `FixedPoint` — gekarakteriseerd vast punt
- `Mechanism` — emergent ontdekt mechanisme
- `Theorem` — universele uitspraak (pre-defined of induced)
- `ProofSketch` — bewijs-richting voor een theorem
- `Category` — conceptuele klasse van pipelines

**Relatie-types:**
- `COMPOSES` — Operator → Pipeline
- `SATISFIES` — Pipeline → Invariant
- `CONVERGES_TO` — Pipeline → FixedPoint
- `EXPLAINED_BY` — Pipeline → Mechanism
- `SUPPORTS` — Pipeline → Theorem
- `FALSIFIES` — Pipeline → Theorem
- `PROVES_VIA` — Theorem → ProofSketch
- `MEMBER_OF` — Pipeline → Category
- `IMPLIES` — Invariant → Invariant (bijv. MONOTONE + BOUNDED → CONVERGENT)
- `SHARES_STRUCTURE` — FixedPoint → FixedPoint (zelfde factorisatie-patroon)

**Query-capaciteiten:**
```python
# "Welke theorems worden ondersteund door pipelines met digit_sum?"
graph.query(operator="digit_sum").theorems()

# "Welke fixed points delen factor 3² × 11?"
graph.query(factor_pattern={3: 2, 11: 1}).fixed_points()

# "Welke proof sketches hebben nog open gaps?"
graph.query(type="ProofSketch").filter(has_gaps=True)

# "Welke categories zijn isomorf?"
graph.query(type="Category").isomorphisms()
```

---

## 3. Nieuwe output-modus: Structured Discovery Report

v7.0 print resultaten naar console. v8.0 moet een gestructureerd rapport genereren:

```markdown
# Discovery Report — Session 2026-02-24

## Universal Laws Discovered
### Law 1: Digit-Sum Mod-9 Preservation
- **Statement:** ∀P containing digit_sum: P preserves n mod 9
- **Status:** STRONG EMPIRICAL (300/300 pipelines)
- **Proof sketch:** By definition, digit_sum(n) ≡ n (mod 9).
  Composition with mod-9-preserving operators maintains this.
- **Gaps:** Need to verify all operators in pipeline preserve mod 9.

## Fixed-Point Universals
### Universal 1: Digit-Sum Divisibility
- **Statement:** For 77.6% of non-trivial fixed points, 3 | FP
- **Explanation:** digit_sum maps to mod-9 residue classes.
  Fixed points must satisfy f(n) = n, constraining digit structure.

## New Conceptual Categories
### Category: "9-Absorbers"
- Pipelines that converge to fixed points with digit_sum = 9
- 47 members identified
- Structural basis: mod-9 preservation + monotone reduction

## Open Problems
1. Is digit_sum = 18 dominant in fixed points of length ≥ 3?
2. Waarom verschijnt 3² × 11 zo frequent in factorisaties?
3. Bestaat er een pipeline zonder trivial fixed point 0?
```

---

## 4. Concrete implementatie-volgorde

### Fase 1: Fixed-Point Structural Analyzer (MODULE C)
- Laagste complexiteit, hoogste informatie-opbrengst
- Analyseer de 183 fixed points uit v7.0 als verzameling
- Genereer hypothesen over digit_sum distributie, factorisatie, palindromen

### Fase 2: Inductive Theorem Generator (MODULE B)
- Gebruik output van Fase 1 als input
- Genereer theorems uit fixed-point patronen
- Test ze actief

### Fase 3: Proof Sketch Generator (MODULE A)
- Voor elke bevestigde theorem, genereer bewijs-richting
- Implementeer 6 bewijs-strategie templates
- Markeer gaps eerlijk

### Fase 4: Theory Graph (MODULE D)
- Verbind alles
- Maak queryable
- Genereer Structured Discovery Report

---

## 5. Wat v8.0 NIET moet doen

- **Niet claimen dat het bewijst.** ProofSketch ≠ Proof. Altijd gaps markeren.
- **Niet meer operators toevoegen.** 19 is genoeg. Verdiep, verbreed niet.
- **Niet meer pipelines per sessie.** 300 is genoeg data. Analyseer dieper.
- **Niet GPU proberen.** De bottleneck is redenering, niet snelheid.
- **Niet een paper proberen te schrijven.** Dat is voor de mens.
  Het systeem levert de ontdekkingen, de mens levert de publicatie.

---

## 6. Het echte doel

v7.0 zegt: "Dit theorem is empirisch sterk."
v8.0 moet zeggen:

> "Theorem MT002 (mod-9 attractor constraint) is empirisch bevestigd
> over 300 pipelines. Bewijs-richting: digit_sum(n) ≡ n (mod 9) per
> definitie. Alle operatoren in de geteste pipelines preserveren mod 9
> (bewezen via operator-algebra). Compositie van mod-9-preserverende
> functies is mod-9-preserverend (bewezen, MT004). Dus attractor ≡ input
> (mod 9). QED-sketch, gap: universaliteit van operator-algebra profielen
> voor inputs > 10⁵."

Dát is het verschil tussen detectie en deductie.

---

## 7. Eén zin samenvatting

**v8.0 = v7.0 + "en dit is waarom het waar is, en dit is wat ik nog niet weet"**

---

## 8. Technische constraints

- **Python 3.11**, geen externe dependencies buiten numpy/scipy
- **SQLite** voor persistentie
- **Geen GPU** — puur CPU-symbolisch
- **Max ~2000 regels** — houd het leesbaar
- **Bouw voort op v7.0** — importeer of kopieer, geen volledige rewrite
- **Bewaar v7.0 ongewijzigd** — v8.0 is een nieuwe file

---

*Dit document is de architectuur-prompt voor de volgende sessie.
Lees het, begrijp het, bouw het.*
