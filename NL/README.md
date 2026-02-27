# SYNTRIAD Digit-Dynamics Discovery Engine — Nederlandse Documentatie

**Volledige technische referentie voor het digit-dynamics onderzoeksproject.**

[← Terug naar root](../README.md) | [English version →](../EN/README.md)

---

## Inhoud

- [Overzicht](#overzicht)
- [Installatie](#installatie)
- [Mappenstructuur](#mappenstructuur)
- [Onderzoeksengines (v1–v15)](#onderzoeksengines-v1v15)
- [Reproduceerbaarheidsinfrastructuur (M0–M4.1)](#reproduceerbaarheidsinfrastructuur-m0m41)
- [Code Uitvoeren](#code-uitvoeren)
- [Testsuite](#testsuite)
- [Papers](#papers)
- [Wiskundige Resultaten](#wiskundige-resultaten)
- [Kennisbank](#kennisbank)
- [Onderzoeksproces](#onderzoeksproces)
- [Resultaten Reproduceren](#resultaten-reproduceren)
- [Probleemoplossing](#probleemoplossing)

---

## Overzicht

Dit project onderzoekt vastepuntstructuur in samengestelde dynamische systemen van cijferoperaties. Gegeven een pipeline van cijferoperaties (reverse, complement, sort, digit-sum, Kaprekar-stap, 1089-truc, etc.) die iteratief op natuurlijke getallen worden toegepast, identificeert het systeem welke getallen vaste punten zijn, classificeert ze algebraïsch, en bewijst structurele resultaten die gelden over alle getalbases b ≥ 3.

De codebase heeft twee sporen:
1. **Onderzoeksengines** (v1–v15): Exploratieve, enkel-bestand engines gebruikt voor ontdekking en vermoedengeneratie.
2. **M0–M4.1 modules**: Modulaire, deterministische infrastructuur voor reproduceerbare resultaten en paper-appendixgeneratie.

---

## Installatie

### Vereisten
- Python 3.10+
- NumPy (enige externe afhankelijkheid voor de kernengines)
- pytest (voor de testsuite)

### Opzet

```bash
cd NL
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Optioneel
- CUDA toolkit + RTX GPU (voor GPU-versnelde exhaustieve verificatie in v1–v2 engines)
- LaTeX-distributie met amsart (voor het compileren van papers)

---

## Mappenstructuur

```
NL/
├── README.md                  # Dit bestand
├── requirements.txt           # Python-afhankelijkheden
├── LICENSE                    # MIT-licentie
│
├── engines/                   # Onderzoeksengines (exploratiespoor)
│   ├── gpu_attractor_verification.py    # v1.0 — GPU brute-force
│   ├── gpu_rigorous_analysis.py         # v2.0 — Methodologische verfijning
│   ├── autonomous_discovery_engine_v4.py # v4.0 — Zelfgenererende pipelines
│   ├── meta_symmetry_engine_v5.py       # v5.0 — Operator-embeddings
│   ├── invariant_discovery_engine_v6.py # v6.0 — Structurele abstractie
│   ├── symbolic_dynamics_engine_v7.py   # v7.0 — Operatoralgebra
│   ├── deductive_theory_engine_v8.py    # v8.0 — Bewijsschetsen
│   └── research_engine_v15.py           # v15.0 — Huidig (30 modules)
│
├── src/                       # Reproduceerbaarheidsinfrastructuur (M0–M4.1)
│   ├── pipeline_dsl.py        # M0: Canonieke semantiek & hashing
│   ├── experiment_runner.py   # M1: SQLite-opslag & batch-runner
│   ├── feature_extractor.py   # M2: Kenmerkextractie & vermoedenmining
│   ├── proof_engine.py        # M3: Bewijsskeletten & rangschikkingsmodel
│   ├── appendix_emitter.py    # M4: LaTeX-appendix & manifestgeneratie
│   ├── reproduce.py           # M4.1: Eén-commando reproduceerbaarheidsrunner
│   └── conftest.py            # Pytest-configuratie
│
├── tests/                     # Testsuite
│   ├── test_m0.py             # M0 tests: parsing, hashing, operaties
│   ├── test_m1.py             # M1 tests: opslag, batch-runner, export
│   ├── test_m2.py             # M2 tests: kenmerken, orbits, vermoedens
│   ├── test_m3.py             # M3 tests: skeletten, dichtheid, rangschikking
│   ├── test_m4.py             # M4 tests: manifest, LaTeX, determinisme
│   └── test_engine.py         # Legacy tests voor onderzoeksengines
│
├── papers/                    # LaTeX-manuscripten
│   ├── paper_A.tex            # Paper A: Zuivere wiskunde (9 theorema's)
│   ├── paper_A.pdf            # Gecompileerde Paper A
│   ├── paper_B.tex            # Paper B: Experimentele wiskunde
│   ├── paper_B.pdf            # Gecompileerde Paper B
│   └── paper.tex              # Gecombineerd werkdocument
│
├── data/                      # Gegenereerde data & databases
│   ├── results.db             # SQLite-experimentdatabase
│   ├── results_export.json    # JSON-export van alle experimenten
│   └── paper_b_hashes.json    # Verificatiehashes voor Paper B
│
└── docs/                      # Uitgebreide documentatie
    ├── WAT_WE_ONTDEKTEN.md
    ├── EVOLUTIE_VAN_SCRIPTS_NAAR_REDENEREN.md
    ├── PUBLICATIESTRATEGIE.md
    ├── FORMEEL_VERIFICATIERAPPORT.md
    └── REFLECTIE_V8.md / REFLECTIE_V10.md
```

---

## Onderzoeksengines (v1–v15)

De onderzoeksengines vormen het *ontdekkingsspoor* — enkel-bestand scripts die progressief evolueerden:

| Versie | Engine | Regels | Kerncapaciteit |
|--------|--------|--------|----------------|
| v1.0 | gpu_attractor_verification | ~500 | CUDA exhaustieve verificatie |
| v2.0 | gpu_rigorous_analysis | ~550 | State-space bounding, cyclusdetectie |
| v4.0 | autonomous_discovery_engine | ~900 | Zelfgenererende pipeline-exploratie |
| v5.0 | meta_symmetry_engine | ~1.300 | Operator-embeddings, meta-learning |
| v6.0 | invariant_discovery_engine | ~1.500 | Structurele abstractie, vermoedengeneratie |
| v7.0 | symbolic_dynamics_engine | ~1.400 | Operatoralgebra, 100% symbolische voorspelling |
| v8.0 | deductive_theory_engine | ~1.800 | Bewijsschetsen, geïnduceerde theorema's |
| v15.0 | research_engine_v15 | ~6.500 | 30 modules, 6 redeneerlagen, 83 KB-feiten |

**De huidige onderzoeksengine draaien:**

```bash
python engines/research_engine_v15.py
```

Dit draait een volledige sessie: initialiseert 22 cijferoperaties, laadt de kennisbank (83 feiten), voert multi-base analyse uit, en geeft resultaten weer op stdout.

**Let op:** De onderzoeksengines gebruiken `random` voor stochastische pipelinegeneratie. Resultaten variëren tussen runs. Gebruik de M0–M4 infrastructuur voor deterministische, reproduceerbare resultaten.

---

## Reproduceerbaarheidsinfrastructuur (M0–M4.1)

De M0–M4.1 modules vormen het *formaliseringsspoor* — ontworpen voor deterministische, hash-geverifieerde reproduceerbaarheid:

### M0: Canonieke Semantiek (pipeline_dsl.py)

Het fundament. Biedt:
- **Laag A (Semantisch)**: `OperationSpec`, `Pipeline`, `DomainPolicy` — bevroren dataklassen, zuivere data
- **Laag B (Executie)**: `OperationExecutor`, `PipelineRunner` — implementaties, strikt gescheiden van Laag A

Kernontwerp: Pipeline-identiteit wordt bepaald door canonieke JSON → SHA-256, nooit door stringrepresentatie. Witruimte, scheidingstekenkeuze (`|>`, `->`, `>>`), en opmaak zijn irrelevant.

```python
from pipeline_dsl import OperationRegistry, Pipeline, DomainPolicy, PipelineRunner

reg = OperationRegistry()  # 22 operaties met volledige metadata
pipe = Pipeline.parse("kaprekar_step |> digit_pow4 |> digit_sum", registry=reg)
domain = DomainPolicy.paper_a_kaprekar(k=4)

runner = PipelineRunner(reg)
result = runner.run_exhaustive(pipe, domain)
print(f"Vaste punten: {result.fixed_points}")
print(f"Resultaathash: {result.sha256}")
```

### M1: Experiment Runner (experiment_runner.py)

SQLite-gebaseerde experimentopslag met:
- Geversioneerd schema (`SCHEMA_VERSION = "1.0"`)
- Batch-runner voor meerdere pipelines × domeinen
- JSON-export met volledige hashketens

### M2: Kenmerkextractor (feature_extractor.py)

Per-getal, per-orbit en per-pipeline kenmerkextractie:
- 17 getalkenmerken (digit_sum, palindroom, sorteerbaarheid, entropie, ...)
- Orbitanalyse (contractieratio, transiente lengte, cyclusdetectie)
- 6 vermoedentypes: TELLING, MODULAIR, MONOTONICITEIT, UNIVERSALITEIT, STRUCTUUR, INVARIANT
- Delta-debugging falsificatie-engine

### M3: Bewijsengine (proof_engine.py)

Structurele redeneerlaag:
- **Bewijsskeletgenerator**: Identificeert bewijsstrategie (MOD_INVARIANT, BOUNDING, COUNTING_RECURRENCE, ...), reductiestappen en resterende gaten
- **Tegenvoorbeelddichtheidsschatter**: Clopper-Pearson grenzen, Regel van Drie, gekalibreerde betrouwbaarheid
- **Patrooncompressor**: Detecteert affiene, polynomiale, modulaire en recurrentiepatronen
- **Vermoedenmutator**: Generalisatie, overdracht, versterking, verzwakking
- **Rangschikkingsmodel v1.0**: Expliciete gewichten (empirisch 0,30, structureel 0,25, nieuwheid 0,20, eenvoud 0,15, falsifieerbaarheid 0,10), geversioneerd en gelogd

### M4: Appendix-emitter (appendix_emitter.py)

Genereert review-bestendige artefacten:
- Paper A / Paper B appendix-LaTeX (domeingescheiden)
- Canonieke JSON-manifesten en -catalogi
- DeterminismGuard: herhaalde uitvoering met byte-identieke controle
- ArtifactPackager: zip-bundel met README, omgevingssnapshot, lockbestand

### M4.1: Reproduceerbaarheidsrunner (reproduce.py)

Eén-commando orkestratie:

```bash
python src/reproduce.py --db data/results.db --out repro_out --bundle
```

---

## Code Uitvoeren

### Snelle verificatie

```bash
# Draai alle unit + integratie tests (~7 seconden)
pytest tests/ -v -m "not exhaustive"

# Draai alleen M0 tests (snelst, ~1 seconde)
pytest tests/test_m0.py -v

# Draai exhaustieve tests (20+ minuten, alle k-bereiken)
pytest tests/ -v -m exhaustive
```

### Paper-appendices genereren

```bash
python src/reproduce.py --db data/results.db --out repro_out --bundle
```

Dit produceert:
- `repro_out/appendix_paper_a.tex`
- `repro_out/appendix_paper_b.tex`
- `repro_out/repro_manifest.json`
- `repro_out/reproducibility_bundle.zip`

### Een specifiek experiment draaien

```python
from src.pipeline_dsl import OperationRegistry, Pipeline, DomainPolicy, PipelineRunner

reg = OperationRegistry()
pipe = Pipeline.parse("truc_1089 |> digit_pow4", registry=reg)
domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
runner = PipelineRunner(reg)
result = runner.run_exhaustive(pipe, domain)

print(f"Attractoren: {result.num_attractors}")
print(f"Vaste punten: {result.fixed_points}")
print(f"Convergentiesnelheid: {result.convergence_rate:.6f}")
print(f"Basinentropie: {result.basin_entropy:.6f}")
print(f"SHA-256: {result.sha256}")
```

---

## Testsuite

| Suite | Bestand | Tests | Dekking | Looptijd |
|-------|---------|-------|---------|----------|
| M0: Canonicalisering | test_m0.py | ~50 | Parsing, hashing, operaties, golden freezes | ~1s |
| M1: Experimentopslag | test_m1.py | ~30 | Opslag, batch-runner, JSON-export | ~2s |
| M2: Kenmerkextractie | test_m2.py | ~30 | Getalkenmerken, orbits, vermoedens | ~2s |
| M3: Bewijsengine | test_m3.py | ~50 | Skeletten, dichtheid, patronen, rangschikking | ~2s |
| M4: Appendix-emitter | test_m4.py | ~80 | Manifest, LaTeX, determinisme, integratie | ~3s |
| Legacy: Onderzoeksengines | test_engine.py | ~98 | Operaties, KB-feiten, multi-base, bewijzen | ~10s |
| **Totaal** | | **~338** | | **~20s** |

```bash
# Volledige suite
pytest tests/ -v

# Met dekkingsrapport
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Papers

### Paper A: "Fixed Points of Digit-Operation Pipelines in Arbitrary Bases"

Zuivere wiskundepaper. Bevat 9 theorema's met algebraïsche bewijzen:

1. **Symmetrische VP-telling**: rev∘comp produceert (b−2)·b^(k−1) vaste punten onder 2k-cijferige getallen
2. **Universele 1089-familie**: A_b = (b−1)(b+1)² voor alle bases b ≥ 3
3. **Vier oneindige families**: Expliciete telformules, paarsgewijs disjunct
4. **Vijfde familie**: 1089-truc vaste punten n_k = 110·(10^(k−3)−1) voor k ≥ 5
5. **Kaprekar-constanten**: K_b = (b/2)(b²−1) voor even bases
6. **Armstrong-bovengrens**: k_max(b) ≤ ⌊b·log(b)/log(b−1)⌋ + 1
7. **Repunit-uitsluiting**: Repunit-veelvouden zijn geen vaste punten van rev∘comp

Doeljournaal: Journal of Integer Sequences of Integers.

### Paper B: "Attractor Spectra and ε-Universality in Digit-Operation Dynamical Systems"

Experimentele wiskundepaper. Introduceert:
- **ε-universaliteit**: Kwantitatieve maat voor attractordominantie
- **Basinentropie**: Shannon-entropie van de basinfractieverdeling
- **Compositielemma**: ε-grens voor samengestelde pipelines
- **Conditioneel Lyapunov-theorema**: Convergentiegaranties voor operaties in klasse P ∪ C
- **Drie vermoedens** (C1: basinentropie-monotoniciteit, C2: asymptotische ε-universaliteit, C3: attractoraantalgroei)

Doeljournaal: Experimental Mathematics.

### Compileren

```bash
cd papers
pdflatex paper_A.tex
pdflatex paper_B.tex
```

---

## Wiskundige Resultaten

### De 22 Cijferoperaties

| Operatie | Notatie | Klasse | ds-klasse |
|----------|---------|--------|-----------|
| reverse | rev(n) | Permutatie | P |
| complement_9 | comp(n) | Cijfergewijze afbeelding | P |
| digit_sum | ds(n) | Aggregaat | C |
| digit_product | dp(n) | Aggregaat | C |
| digit_pow2–5 | dp_k(n) | Aggregaat | X |
| sort_asc / sort_desc | sort↑/↓(n) | Permutatie | P |
| kaprekar_step | kap(n) | Subtractief | C |
| truc_1089 | T(n) | Gemengd | C |
| add_reverse | n + rev(n) | Gemengd | X |
| sub_reverse | |n − rev(n)| | Gemengd | C |
| swap_ends | swap(n) | Permutatie | P |
| rotate_left/right | rot(n) | Permutatie | P |
| digit_factorial_sum | dfs(n) | Aggregaat | X |
| digit_gcd | dgcd(n) | Aggregaat | C |
| digit_xor | dxor(n) | Aggregaat | C |
| collatz | col(n) | Aritmetisch | X |

Klassen: P = digit-sum behoudend, C = contractief, X = expansief.

### Het Algebraïsche Kerninzicht

Vaste punten van cijferoperatiepipelines worden bepaald door de algebraïsche structuur van de getalbase:
- **10 ≡ 1 (mod 9)** → digit_sum behoudt residu mod 9 → factor-3 verrijking in vaste punten
- **10 ≡ −1 (mod 11)** → alternerende digit-sum-structuur → factor-11 verrijking
- **(3 × 11)² = 1089** → het universele resonantiepunt waar beide structuren kruisen
- Generaliseert: **A_b = (b−1)(b+1)²** voor elke base b ≥ 3

### Kennisbank (83 Feiten)

De onderzoeksengine onderhoudt een kennisbank van 83 feiten (DS011–DS072):
- 72 bewezen (algebraïsch of exhaustief bewijs)
- 11 vermoed (sterke empirische onderbouwing)

Omvattend: complement-gesloten families, symmetrische VP-tellingen, Kaprekar-constanten, 1089-universaliteit, Lyapunov-dalingsgrenzen, Armstrong-grenzen, repunit-uitsluiting, orbitanalyse.

---

## Onderzoeksproces

### Multi-Agent Samenwerking

| Ronde | Agent | Focus |
|-------|-------|-------|
| R1–R5 | DeepSeek | Wiskundige consultatie, vermoedenverfijning |
| R6 | Manus | Multi-base engine, bulk-implementatie |
| R7–R8 | Claude/Cascade | Formele bewijsverificatie (12/12) |
| R9 | Claude/Cascade | Armstrong, Kaprekar oneven bases, orbitanalyse |
| R10 | Claude/Cascade | Universele Lyapunov, repunits, cyclustaxonomie |
| R11 | Claude/Cascade | Open vragen, vijfde familie, publicatievoorbereiding |

De menselijke onderzoeker (R. Havenaar) stuurde alle fasen, identificeerde de algebraïsche structuren, en maakte de conceptuele verbindingen.

### Zelfcorrectievoorbeelden

De epistemologische gezondheid van het systeem wordt gedemonstreerd door autonome correcties:
- **R5**: Engine detecteerde dat DeepSeek's voorspelling (9×10^(k−1) VP's) fout was → gecorrigeerd naar 8×10^(k−1) (uitsluiting voorloopnullen)
- **R8**: Engine ontdekte formulefout in DS040: (b−1)²(b+1) → (b−1)(b+1)², algebraïsch geverifieerd
- **v7.0**: Engine falsifieerde eigen meta-theorema "Monotoon+Begrensd → convergentie" met concreet tegenvoorbeeld

---

## Resultaten Reproduceren

### Volledige reproduceerbaarheidspipeline

```bash
# 1. Stel deterministische hashseed in (aanbevolen)
export PYTHONHASHSEED=0

# 2. Draai reproduceerbaarheidsrunner
python src/reproduce.py --db data/results.db --out repro_out --bundle

# 3. Verifieer output
# De laatste regel toont FINAL MANIFEST SHA256
# Dit moet overeenkomen met de hash gepubliceerd in de paper-appendix
```

### Wat reproduce.py doet

1. Controleert runtime-determinismeknobs (PYTHONHASHSEED)
2. Toont omgevingssamenvatting
3. Genereert `requirements.lock.txt` (pip freeze)
4. Draait M4-emitter: DB → manifest + catalogi + LaTeX-appendices
5. Draait DeterminismGuard: heruitvoering in tijdelijke map, byte-voor-byte vergelijking
6. Verpakt `reproducibility_bundle.zip`
7. Toont uiteindelijke manifest SHA-256

### Hashketen

```
OperationRegistry.sha256
    └── Pipeline.sha256 (canonieke JSON van operatiereeks)
        └── DomainPolicy.sha256 (base, digit_length, uitsluitingen)
            └── RunResult.sha256 (alle numerieke resultaten, vaste precisie)
                └── Manifest.sha256 (alles hierboven + omgeving)
```

Elke schakel in de keten is deterministisch binnen dezelfde Python-versie en hetzelfde platform.

---

## Probleemoplossing

**`ModuleNotFoundError: No module named 'pipeline_dsl'`**
→ Zorg dat je vanuit de `NL/src/` map draait, of voeg het toe aan `PYTHONPATH`:
```bash
export PYTHONPATH=NL/src:$PYTHONPATH
```

**`FileNotFoundError: results.db`**
→ De experimentdatabase moet bestaan vóór het draaien van `reproduce.py`. Deze is meegeleverd in de repository onder `data/results.db`.

**Andere manifesthash bij herdraaiing**
→ Controleer: (1) Zelfde Python-versie? (2) Zelfde NumPy-versie? (3) PYTHONHASHSEED=0? Float-opmaak op het 12e decimaal kan verschillen tussen platforms.

**LaTeX-compilatiefouten**
→ Paper A vereist de `amsart` documentklasse. Installeer een volledige TeX-distributie (TeX Live of MikTeX).

---

*SYNTRIAD Research — februari 2026*
