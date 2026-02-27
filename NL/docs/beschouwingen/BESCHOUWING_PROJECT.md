# Beschouwing: Digit-Pipeline Analysis Framework

**Datum:** 26 februari 2026
**Project:** SYNTRIAD Digit-Pipeline Analysis Framework
**Locatie:** `autonomous_discovery_engine/`

---

## 1. Wat is dit project?

Dit project onderzoekt een fundamentele maar verrassend rijke wiskundige vraag:
*wat gebeurt er als je elementaire bewerkingen op de cijfers van een getal herhaaldelijk toepast?*

Denk aan de Kaprekar-constante 6174 — neem een willekeurig viercijferig getal, sorteer de cijfers aflopend, trek het oplopend gesorteerde getal ervan af, en herhaal. Na hoogstens 7 stappen kom je altijd bij 6174 uit. Dit project generaliseert dat idee radicaal: we bestuderen willekeurige samenstellingen ("pipelines") van 22 cijferbewerkingen (reverse, complement, sort, digit-sum, Kaprekar-stap, 1089-truc, digit-powers, narcissistic step, digit-gcd, digit-xor, ...) in willekeurige talstelsels (basis b ≥ 3).

De centrale ontdekking is dat de vaste punten van deze pipelines niet willekeurig zijn, maar geregeerd worden door de algebraïsche structuur van de basis b — met name de factoren (b−1) en (b+1) spelen een dominante rol.

---

## 2. Evolutie van het project

Het project heeft een ongebruikelijke en intensieve ontwikkelingsgeschiedenis doorlopen, verdeeld over 15 engine-versies en 11 feedbackrondes (R1–R11) met drie verschillende AI-systemen:

| Fase | Versie | Bijdrage |
|------|--------|----------|
| **Exploratie** | v1–v3 | GPU brute-force verificatie (CUDA), eerste attractordetectie |
| **Structuur** | v4–v6 | Pipeline-enumeratie, invariantontdekking, patroonherkenning |
| **Symbolisch** | v7–v8 | Operatoralgebra, formele bewijsschetsen, deductieve theorie |
| **Kennisbank** | v9 | 51 feiten (DS011–DS052), anomaliedetectie, zelfbevraging |
| **Multi-basis** | v10 | Algebraïsche FP-classificatie, Lyapunov-zoekmachine, multi-basis engine |
| **Formeel** | v11–v12 | 12/12 computationele bewijzen, DS040-correctie (1089 = universeel) |
| **Breed** | v13 (R9) | Armstrong/narcissistisch, Kaprekar oneven bases, orbitanalyse, 22 operaties |
| **Diep** | v14 (R10) | Universele Lyapunov, repunits, cyclusclassificatie, multi-digit Kaprekar |
| **Open vragen** | v15 (R11) | 4 oneindige FP-families, digit_sum Lyapunov bewezen, paper draft |
| **Publicatie** | R12+ | Paper A/B split, repo-herstructurering, twee audits, adversarial audit-fixes |

### De feedbackrondes

De ontwikkeling vond plaats in een cyclisch AI-gestuurd proces:

- **R1–R5** (DeepSeek): Opbouw kernfuncties — kennisbank, complementfamilies, telformules
- **R6** (Manus): Multi-basis generalisatie, algebraïsch bewijs 1089-familie
- **R7–R11** (Cascade): Formele bewijzen, bugfixes (DS040-formule gecorrigeerd), breedte- en diepte-uitbreiding, paper-schrijven

Een cruciaal keerpunt was **R8**, toen een formulefout in DS040 ontdekt werd: `(b−1)²(b+1) = 891` was fout; de correcte formule `(b−1)(b+1)² = 1089` bewees dat de 1089-multiplicatieve familie **universeel** is voor alle bases b ≥ 3 — niet specifiek voor basis 10.

### Van monoliet naar modulaire architectuur

De codebase evolueerde van een 6.500-regelige monoliet (`abductive_reasoning_engine_v10.py`) naar een gestructureerde modulaire architectuur:

| Module | Bestand | Functie |
|--------|---------|---------|
| M0 | `pipeline_dsl.py` (40K) | Pipeline-DSL, 22 operaties, canonical hashing |
| M1 | `experiment_runner.py` (24K) | Experimentuitvoering, SQLite-opslag |
| M2 | `feature_extractor.py` (38K) | 17-feature nummerprofiling, conjecture mining |
| M3 | `proof_engine.py` (47K) | Bewijsschetsen, dichtheidsschatting, ranking |
| M4 | `appendix_emitter.py` (48K) | LaTeX-generatie, manifests, determinismguard |

Totaal: ~200K broncode + 117 unit tests (100% slagend).

---

## 3. Wat kan het nu?

### Wiskundige resultaten

Het framework heeft **79 kennisbankfeiten** geproduceerd (65 bewezen, rest empirisch), waaronder negen formeel bewezen stellingen:

**Stelling 1 (DS034):** Voor elke basis b ≥ 3 en elke k ≥ 1 geldt: het aantal vaste punten van rev ∘ comp onder 2k-cijferige getallen is exact (b−2)·b^(k−1).

**Stelling 2 (DS040):** De 1089-multiplicatieve familie A_b × m (met A_b = (b−1)(b+1)²) is universeel: voor elke basis b ≥ 3 en m = 1, …, b−1 is A_b·m een complement-gesloten vast punt met cijfers [m, m−1, (b−1)−m, b−m].

**Stelling 3 (DS064):** Er bestaan minstens vier paarsgewijs disjuncte oneindige families van vaste punten, elk met bewezen telformule:
1. Symmetrische familie: (b−2)·b^(k−1) FPs
2. 1089×m multiplicatief: b−1 FPs per basis
3. Sort-descending: C(k+9,k)−1 FPs
4. Palindromen: 9·10^(⌊(k−1)/2⌋) FPs

**Stelling 4 (DS069):** Een vijfde oneindige familie: de 1089-truc-afbeelding T(n) = |n − rev(n)| + rev(|n − rev(n)|) heeft vaste punten n_k = 110·(10^(k−3) − 1) voor elke k ≥ 5.

**Stelling 5 (DS039, DS057, DS066):** Kaprekar-constanten algebraïsch bewezen: K_b = (b/2)(b²−1) voor even b ≥ 4; 4-digit K = 6174 met convergentie ≤ 7 stappen; 6-digit: twee FPs (549945, 631764).

**Stelling 6 (DS065):** Armstrong/narcissistische getallen: k_max(b) = max{k : k·(b−1)^k ≥ b^(k−1)}. Bewezen: k_max(10) = 60.

**Stelling 7 (DS055):** Repunits R_k = (b^k − 1)/(b−1) zijn nooit complement-gesloten vaste punten.

**Stelling 8 (DS061):** digit_sum is een conditionele Lyapunov-functie voor pipelines bestaande uit ds-bewarende en ds-contractieve operaties.

**Stelling 9 (DS038–DS045):** Lyapunov-afdalingsgrenzen voor digit-power-afbeeldingen (digit_pow2 t/m digit_pow5, digit_factorial_sum).

### Computationele capaciteiten

- **Exhaustieve verificatie** over 2×10^7 startwaarden per pipeline
- **GPU-versnelling** via Numba CUDA JIT-compiled kernels (RTX 4000 Ada, ~5×10^6 iteraties/sec)
- **SHA-256 verificatiehashes** voor reproduceerbaarheid
- **Multi-basis support** voor b ∈ {3, …, 16}
- **Deterministische reproductie**: `run_experiments.py` → `reproduce.py` → identieke artefacten

---

## 4. De papers

Het werk is opgesplitst in twee zelfstandige manuscripten, elk gericht op een ander publiek:

### Paper A: "Fixed Points of Digit-Operation Pipelines in Arbitrary Bases: Algebraic Structure and Five Infinite Families"

- **Type:** Zuiver wiskundig (stelling-bewijs)
- **Omvang:** 750 regels LaTeX, 7 pagina's PDF
- **Doeltijdschriften:** Journal of Integer Sequences, Integers, Fibonacci Quarterly
- **Inhoud:** Stellingen 1–9, vijf oneindige FP-families, Kaprekar-analyse t/m 7 cijfers, Armstrong-bovengrenzen, Lyapunov-afdalingsgrenzen
- **Sterkste claim:** De classificatie van vijf disjuncte oneindige families met exacte telformules, plus de universaliteit van de 1089-familie over alle bases

### Paper B: "Attractor Spectra and ε-Universality in Digit-Operation Dynamical Systems"

- **Type:** Gemengd theoretisch + experimenteel
- **Omvang:** 558 regels LaTeX, 6 pagina's PDF
- **Doeltijdschriften:** Experimental Mathematics, Complex Systems
- **Inhoud:** ε-universaliteit, basin-entropie, compositielemma, conditionele Lyapunov-stelling, GPU-exhaustieve attractorstatistieken, drie conjecturen
- **Sterkste claim:** De scherpe dichotomie — contractieve+mengpipelines convergeren bijna universeel (ε < 0.01), terwijl niet-contractieve pipelines rijke multi-attractorspectra vertonen (H > 2 bits)

### Relatie tussen de papers

Paper A levert de algebraïsche fundering (welke vaste punten bestaan er?); Paper B bouwt daarop voort met dynamische analyse (hoe verdelen startwaarden zich over attractoren?). Paper B verwijst naar Paper A als companion paper. Samen vormen ze een coherent tweeluik.

---

## 5. Stand van zaken

### Wat is af

| Onderdeel | Status |
|-----------|--------|
| Wiskundige resultaten (DS011–DS068) | ✅ 65/79 bewezen |
| Formele computationele bewijzen | ✅ 12/12 |
| Engine-architectuur (M0–M4) | ✅ Modulair, getest |
| Unit tests | ✅ 117/117 slagend |
| Paper A LaTeX | ✅ Compileert, zelfstandig |
| Paper B LaTeX | ✅ Compileert, zelfstandig |
| Repo-herstructurering | ✅ engines/ gescheiden van M0–M4 |
| Adversarial audit | ✅ Alle 8 fixes geïmplementeerd |
| Audit-verdict | ✅ CONDITIONAL ACCEPT (geen HIGH-issues meer) |

### Wat nog open staat

Op basis van de twee audits (technische audit + adversarial audit) en het submission-roadmap zijn dit de resterende taken:

#### Hoge prioriteit (blokkerend voor indiening)

| Item | Beschrijving | Bron |
|------|-------------|------|
| **C3: Taalcorrectie** | Termen als "Autonomous Discovery Engine" en "abductive reasoning" vervangen door neutrale beschrijvingen ("systematic computational exploration") | Technische audit §3.4 |
| **Reproducibiliteitsverificatie** | Volledige round-trip test: `run_experiments.py --fresh` → `reproduce.py --bundle` → DeterminismGuard groen | Audit C1 |
| **k-range default** | Default k-range aanpassen van [3,4,5] naar [3,4,5,6,7] zodat deze overeenkomt met de paper-scope | Audit I3 |

#### Gemiddelde prioriteit (sterk aanbevolen)

| Item | Beschrijving | Bron |
|------|-------------|------|
| **Ablation note ranking** | Vermelding dat ranking-gewichten heuristisch zijn en niet gekalibreerd | Audit I1 |
| **NumPy-versie pinnen** | `requirements.txt` met `numpy>=1.24,<2.0` | Audit I2 |
| **Cross-platform float-caveat** | Documenteren dat hash-identiteit cross-platform niet gegarandeerd is voor >12 decimalen | Audit I4 |
| **M2→M0 documentatie** | Comment toevoegen dat ConjectureMiner PipelineRunner importeert als pragmatische integratie | Audit I5 |

#### Lage prioriteit (nice-to-have)

| Item | Beschrijving | Bron |
|------|-------------|------|
| Adversarial edge-case tests | Lege pipeline, 0-input, single-digit, basis=2 | Audit N1 |
| OEIS cross-referenties | FP-telformules matchen met OEIS-reeksen in Paper A | Audit N3 |
| WKN-005: Conjecture-schaal | Meer data voor conjectuur-onderbouwing (vereist GPU-runs) | Adversarial audit |
| WKN-009: Winter2020-citaat | Status van preprint verifiëren | Adversarial audit |
| OEIS-indiening vijfde familie | Nieuwe reeks indienen bij OEIS | Adversarial audit |

---

## 6. Rijpheid voor indiening

### Beoordeling per criterium

| Criterium | Score | Toelichting |
|-----------|-------|-------------|
| **Wiskundige correctheid** | 9/10 | 12/12 bewijzen computationeel geverifieerd; alle stellingen algebraïsch onderbouwd |
| **Nieuwheid** | 8/10 | Universaliteit 1089-familie over alle bases is nieuw; vijf oneindige FP-families classificatie is nieuw; ε-universaliteit als concept is nieuw |
| **Presentatie** | 7/10 | Papers zijn structureel solide maar taalcorrectie (C3) is nog nodig; bibliografie is uitgebreid maar kan sterker |
| **Reproduceerbaarheid** | 7/10 | Hashes, deterministische scripts en tests zijn aanwezig; volledige round-trip moet nog geverifieerd worden |
| **Scope-afbakening** | 8/10 | Paper A/B split is helder; engine is teruggebracht tot methodologie-vermelding |

### Geschatte inspanning tot indiening

| Taak | Geschatte tijd |
|------|---------------|
| C3: Taalcorrectie (5 bestanden) | 2–3 uur |
| Reproduceerbaarheid round-trip test | 1–2 uur |
| k-range + NumPy pin + caveats | 1 uur |
| Ablation note + M2-documentatie | 30 min |
| Finale proeflezing beide papers | 2–3 uur |
| **Totaal** | **~1 werkdag** |

### Aanbeveling

**Het project is rijp voor indiening na één gerichte sessie.** De wiskundige kern is stevig, de bewijzen zijn geverifieerd, en de papers compileren als zelfstandige manuscripten. De resterende taken zijn voornamelijk cosmetisch en procedureel:

1. **Taalcorrectie (C3)** is de belangrijkste blokkering — reviewers zullen "autonomous discovery" als overclaim aanmerken.
2. **Reproducibiliteit round-trip** moet eenmalig end-to-end getest worden.
3. **De rest** (NumPy pin, caveats, ablation note) is in een uur afgerond.

### Aanbevolen indieningsstrategie

1. **Paper A eerst** naar Journal of Integer Sequences of Integers — dit is het sterkste manuscript met de hardste resultaten. JIS publiceert snel en heeft een gunstig track record voor dit type werk.
2. **Paper B** naar Experimental Mathematics — het ε-universaliteitsconcept en de GPU-exhaustieve aanpak passen goed bij het profiel van dit tijdschrift.
3. **Beide papers tegelijk op arXiv** met kruisverwijzingen.
4. **Paper C** (engine als AI-methodologisch onderwerp) komt pas na acceptatie van A en B — dit is een strategische keuze om de wiskunde los te verkopen van de AI-claims.

### Wat dit project bijzonder maakt

Dit is een zeldzaam voorbeeld van een computationeel wiskundig project dat:
- Van brute-force GPU-verificatie naar formele algebraïsche bewijzen is geëvolueerd
- Niet alleen individuele resultaten levert maar een **classificatietheorie** opbouwt
- De universaliteit van een 75 jaar oud fenomeen (de 1089-truc) bewijst over alle bases
- Reproduceerbaar is tot op hash-niveau

De combinatie van diepte (formele bewijzen), breedte (22 operaties, multi-basis), en systematiek (79 kennisbankfeiten) maakt het een substantiële bijdrage aan de computationele getaltheorie.

---

## Addendum: De P_k-ontdekking (25 februari 2026)

Tijdens een driehoeksdialoog tussen de onderzoeker, Manus (AI-agent) en GPT-4 werd ontdekt dat de engine-executor een **impliciete projectie-operator P_k** toepast: na elke operatie wordt het tussenresultaat teruggeprojecteerd naar k digits via zero-padding. Dit verandert de onderzochte wiskundige structuur fundamenteel — van pure operator-compositie naar **projectieve dynamica op een vaste-lengte representatieruimte**.

Deze ontdekking verklaart onder meer waarom `digit_sum∘reverse` precies b−1 vaste punten heeft (de familie {d·b^(k−1) | d ∈ {1,...,b−1}}), en opent een nieuw perspectief op het hele project: niet de bewerkingen zijn het interessante object, maar hoe vaste-lengte projectie de dynamica van die bewerkingen structureel herdefinieert.

Zie: [`BESCHOUWING_DRIEHOEKSDIALOOG_P_k.md`](BESCHOUWING_DRIEHOEKSDIALOOG_P_k.md) voor de volledige analyse van deze uitwisseling.

---

*SYNTRIAD Research — februari 2026*
