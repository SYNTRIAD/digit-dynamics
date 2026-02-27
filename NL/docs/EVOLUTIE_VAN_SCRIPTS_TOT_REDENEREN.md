# Van Scripts tot Redeneren

*De evolutie van een brute-force calculator naar een gelaagd redeneersysteem*

---

## Overzicht

Dit document beschrijft hoe een reeks Python-scripts voor het doorrekenen van cijferbewerkingen zich ontwikkelde tot een autonoom onderzoekssysteem met zes lagen van redeneren. Het bredere Mahler-project loopt circa vijf weken. Maar de autonome discovery engine -- van eerste GPU-script tot een systeem met zes redeneringslagen, 30 modules, formele bewijzen en de P_k-ontdekking -- ontstond in **drie dagen** (23-26 februari 2026). Het is het verhaal van een machine die leerde -- niet in de AI-marketingbetekenis van het woord, maar in de wetenschappelijke: van rekenen naar verklaren naar bewijzen.

De evolutie verliep in vier fasen:

1. **Fase I: Rekenen** — GPU brute-force, miljoenen getallen doorrekenen
2. **Fase II: Ontdekken** — Patronen herkennen, pipelines genereren, hypotheses formuleren
3. **Fase III: Begrijpen** — Algebraïsche structuur, bewijsschetsen, kennisbank
4. **Fase IV: Verantwoorden** — Formele modules, reproduceerbaar protocol, publicatie-rijp

---

## De Projectstructuur

```
mahler-analysis/
├── syntriad_extreme_experiments/       ← Fase 0: Mahler-numeriek
├── syntriad_theoretical_experiments/   ← Fase 0: Theoretische verkenning
└── symmetry_discovery/                 ← Fase I-IV
    ├── symmetry_discovery_engine.py    ← v1.0: De eerste engine
    ├── run_discovery.py                ← Launcher
    ├── quick_research.py               ← Snelle iteratie
    ├── scoring_engine_v2.py            ← Trivialiteitsfilter
    ├── meta_discovery_engine.py        ← v2.0: Zelf-verbeterend
    ├── autonomous_researcher.py        ← v3.0: Dynamische operaties
    ├── gpu_symmetry_hunter.py          ← GPU CUDA kernels
    ├── gpu_deep_researcher.py          ← GPU + zelf-adaptatie
    ├── gpu_creative_research.py        ← GPU creatieve operaties
    ├── extended_research_session.py    ← v4.0: Langdurig autonoom
    ├── extended_research_session_v5.py ← v5.0: Cycli + genetisch
    ├── extended_research_session_v6.py ← v6.0: ML-versneld
    └── autonomous_discovery_engine/    ← Fase III-IV
        ├── engines/                    ← Research prototypes
        │   ├── autonomous_discovery_engine_v4.py   (36 KB)
        │   ├── meta_symmetry_engine_v5.py          (48 KB)
        │   ├── invariant_discovery_engine_v6.py     (56 KB)
        │   ├── symbolic_dynamics_engine_v7.py       (54 KB)
        │   ├── deductive_theory_engine_v8.py        (73 KB)
        │   ├── abductive_reasoning_engine_v9.py    (105 KB)
        │   └── abductive_reasoning_engine_v10.py   (289 KB)
        ├── pipeline_dsl.py        ← M0: Canonieke semantiek
        ├── experiment_runner.py   ← M1: Experiment + opslag
        ├── feature_extractor.py   ← M2: Feature mining
        ├── proof_engine.py        ← M3: Bewijsschetsen
        └── appendix_emitter.py    ← M4: Publicatie-artefacten
```

---

## Fase I: Rekenen (23 februari 2026, dag 1)

### Het startpunt: `symmetry_discovery_engine.py` (v1.0)

Het allereerste script. Een klassieke brute-force engine die:

- **22 operaties** definieert als Python-klassen: `reverse`, `digit_sum`, `sort_desc`, `kaprekar_step`, `truc_1089`, `complement_9`, enzovoort
- **Pipelines** bouwt door operaties te combineren (bijv. `kaprekar_step |> sort_desc`)
- Een **evolutionair algoritme** gebruikt om interessante pipelines te vinden
- Resultaten opslaat in een **SQLite database**

Op dit niveau is het systeem een calculator. Het past bewerkingen toe, kijkt of er iets convergeert, en slaat het op. Geen hypotheses, geen verklaring, geen structuur.

**Wat het kon:** Miljoenen getallen doorrekenen en convergentiepunten vinden.
**Wat het niet kon:** Begrijpen waarom iets convergeert.

### GPU-versnelling: `gpu_symmetry_hunter.py`

De eerste grote versnelling. Alle kernoperaties worden herschreven als **CUDA device functions** voor de RTX 4000 Ada:

```python
@cuda.jit(device=True)
def gpu_reverse(n: int64) -> int64:
    result = 0
    temp = n if n > 0 else -n
    while temp > 0:
        result = result * 10 + (temp % 10)
        temp //= 10
    return result
```

Throughput: ~120-150 miljoen samples per seconde. Dit maakte exhaustieve verificatie mogelijk -- niet steekproefsgewijs, maar elk getal in het domein.

### `gpu_deep_researcher.py` (v3.0)

De GPU-engine wordt zelf-adapterend: het systeem past zijn zoekstrategie aan op basis van wat het vindt. Als een pipeline interessant is, worden varianten automatisch gegenereerd en getest.

### Scoring en filtering: `scoring_engine_v2.py`

Een cruciaal probleem werd zichtbaar: de engine vond veel **triviale** attractoren (0-9). Een getal als 7 is een "vast punt" van `digit_sum` -- maar dat is triviaal. De scoring engine v2 filtert deze eruit en beloont:

- Niet-triviale waarden (>10)
- Bekende wiskundige constanten (6174, 1089)
- Palindromen, priemgetallen, perfecte machten
- Nieuwe, onbekende attractoren (novelty bonus)

Dit was de eerste stap naar **oordeel**: niet alles wat convergeert is interessant.

### Extended Research Sessions (v4, v5, v6)

Drie generaties van langdurende (5+ minuten) autonome sessies:

| Versie | Nieuwe capaciteit | Kernverbetering |
|--------|-------------------|-----------------|
| **v4** | Langdurig autonoom onderzoek | Bouwt voort op eerdere ontdekkingen |
| **v5** | Cyclusdetectie + genetische mutatie | Vindt cycli, niet alleen vaste punten |
| **v6** | ML Success Predictor + CPU parallelisatie | Voorspelt welke pipelines succesvol zullen zijn |

v6.0 is het eindpunt van Fase I. Het systeem kan nu snel en slim zoeken. Maar het begrijpt nog niets.

---

## Fase II: Ontdekken (23-24 februari 2026, dag 1-2)

De sprong van `symmetry_discovery/` naar `autonomous_discovery_engine/engines/` markeert een fundamentele verschuiving. Het systeem gaat van "patronen vinden" naar "patronen begrijpen."

### v4.0: `autonomous_discovery_engine_v4.py` (36 KB)

De eerste "echte" autonome onderzoeker. Nieuwe capaciteiten:

- **Basin-of-attraction analyse** -- niet alleen waar convergeert het, maar hoe groot is het aantrekkingsgebied?
- **Exceptie-identificatie** -- welke getallen gedragen zich anders dan verwacht?
- **Hypotheseformulering** -- het systeem stelt zelf vragen: "klopt het dat alle vaste punten deelbaar zijn door 9?"
- **Algebraische reducties** -- detectie dat bepaalde operator-combinaties equivalent zijn

### v5.0: `meta_symmetry_engine_v5.py` (48 KB)

De meta-learning sprong. Het systeem leert over zichzelf:

- **Operator embeddings** -- operaties worden gerepresenteerd als vectoren, zodat gelijksoortige operaties dicht bij elkaar liggen
- **Theory Graph** -- een graaf die relaties legt tussen ontdekkingen ("deze attractor verschijnt in drie verschillende pipelines")
- **Entropie als maat** -- Shannon-entropie van de cijferverdeling als fundamentele eigenschap van getallen
- **Zelf-reflectie** -- het systeem evalueert zijn eigen zoekstrategie en past die aan

Dit was het moment waarop het systeem niet meer alleen rekende, maar begon na te denken over wat het aan het doen was.

### v6.0: `invariant_discovery_engine_v6.py` (56 KB)

De abstractie-sprong. Drie nieuwe lagen:

- **Laag 2: Structurele Abstractie** -- het systeem zoekt niet meer naar specifieke getallen maar naar *klassen* van getallen die een eigenschap delen
- **Laag 3: Symbolische Redenering** -- conjectures worden eerste-klas objecten, met een sterkte-classificatie en actieve falsificatie
- **Cross-domain isomorfismen** -- het systeem herkent dat patronen in basis 10 soms ook gelden in basis 12

Hier ontstond het woord "conjecture" in het systeem. Niet meer "ik vond een patroon", maar "ik beweer dat dit altijd geldt, en ik zoek actief naar een tegenvoorbeeld."

---

## Fase III: Begrijpen (24-25 februari 2026, dag 2-3)

### v7.0: `symbolic_dynamics_engine_v7.py` (54 KB)

De overgang van statistisch naar symbolisch. Vier fundamentele upgrades:

1. **Operator Algebra** -- formele eigenschappen van operaties (commutatief? idempotent? contractief?) worden symbolisch vastgelegd en gebruikt om voorspellingen te doen *voor* je iets berekent
2. **Fixed-Point Solver** -- in plaats van zoeken naar vaste punten, de vergelijking f(n)=n *oplossen*
3. **Meta-Theorem Generator** -- universele uitspraken over klassen van operaties, met actieve falsificatie
4. **Emergente mechanismen** -- clustering van co-occurrence patronen

De operator-algebra bereikte 100% predictie-accuraatheid over 300 pipelines. Het systeem wist *vooraf* welke invarianten een pipeline zou hebben -- zonder te berekenen.

### v8.0: `deductive_theory_engine_v8.py` (73 KB)

De deductieve sprong. Vier nieuwe modules:

- **MODULE A: Proof Sketch Generator** -- gegeven een bevestigd patroon, genereer een bewijs-richting
- **MODULE B: Inductive Theorem Generator** -- leid theorema's *af uit data* in plaats van ze te testen
- **MODULE C: Fixed-Point Structural Analyzer** -- analyseer de verzameling vaste punten als geheel
- **MODULE D: Theory Graph** -- verbind alle ontdekte objecten in een samenhangend netwerk

Het kernprincipe van v8.0 staat letterlijk in de code: *"en dit is waarom het waar is, en dit is wat ik nog niet weet."*

De eerlijke zelfreflectie (`REFLECTION_V8.md`) na deze sessie was cruciaal. Het systeem identificeerde wat echt significant was -- het 3^2 x 11 patroon in 22% van alle niet-triviale vaste punten -- en wat minder indrukwekkend was dan het leek. En het concludeerde: "v8.0 weet *wat* waar is, maar niet *waarom*."

### v9.0: `abductive_reasoning_engine_v9.py` (105 KB)

De abductieve sprong -- van "wat is waar" naar "waarom is het waar." Vijf nieuwe modules:

- **MODULE E: Knowledge Base** -- 34 bewezen wiskundige feiten als axioma's, niet als observaties
- **MODULE F: Causal Chain Constructor** -- bouwt verklaringsketens ("vaste punten zijn deelbaar door 3 *omdat* digit_sum invariant is mod 9 *omdat* 10 = 1 mod 9")
- **MODULE G: Surprise Detector** -- signaleert anomalieen ("1089 verschijnt in pipelines die niets met truc_1089 te maken hebben -- waarom?")
- **MODULE H: Gap Closure Loop** -- bewezen feiten sluiten automatisch gaten in bewijsschetsen
- **MODULE I: Self-Questioner** -- na elke ontdekking: "waarom?" en "wat volgt hieruit?"

Dit was de eerste keer dat het systeem iets deed wat op begrip lijkt. Niet alleen patronen herkennen, maar verklaren *waarom* die patronen bestaan.

### v10-v15: `abductive_reasoning_engine_v10.py` (289 KB, ~6500 regels)

Het volledige systeem. 30 modules (A-Z plus 4 extra), gegroeid over 11 feedbackrondes (R1-R11) met drie verschillende AI-agenten (DeepSeek R1-R5, Manus R6, Cascade R7-R11).

Nieuwe modules in R6-R11:

| Module | Naam | Functie |
|--------|------|---------|
| **N** | Multi-Base Engine | Generalisatie naar basis 8, 10, 12, 16 |
| **O** | Symbolic FP Classifier | Automatische algebraische FP-condities |
| **P** | Lyapunov Search | Dalende functies voor convergentiebewijs |
| **Q** | 1089-Family Proof | Algebraisch bewijs complement-geslotenheid |
| **R** | Formal Proof Engine | 12/12 computationeel geverifieerde bewijzen |
| **S** | Narcissistic Analyzer | Armstrong numbers, bifurcatie |
| **T** | Odd-Base Kaprekar | Kaprekar-dynamica in oneven bases |
| **U** | Orbit Analyzer | Convergentietijd, cycluslengte |
| **V** | Extended Pipeline | 5+ operatie-pipelines, FP-saturatie |
| **W** | Universal Lyapunov | Universele Lyapunov-functie zoektocht |
| **X** | Repunit Analyzer | Repunit-verband met CC-families |
| **Y** | Cycle Taxonomy | Attractorcyclus-classificatie |
| **Z** | Multi-Digit Kaprekar | 4+ digit Kaprekar-dynamica |

Resultaat: 9 stellingen, 5 oneindige families van vaste punten, 83 KB-feiten, 117 tests.

---

## Fase IV: Verantwoorden (25-26 februari 2026, dag 3-4)

### De M0-M4 refactoring

Het monolithische v10-bestand (6500 regels) werd ontleed in vijf schone, modulaire componenten -- de "submission codebase":

| Module | Bestand | Functie | Omvang |
|--------|---------|---------|--------|
| **M0** | `pipeline_dsl.py` | Canonieke semantiek + reproduceerbare hashing | 40 KB |
| **M1** | `experiment_runner.py` | Experiment-uitvoering + SQLite opslag | 24 KB |
| **M2** | `feature_extractor.py` | Feature-extractie + conjecture mining | 38 KB |
| **M3** | `proof_engine.py` | Bewijsschetsen + densiteitsschatting + ranking | 47 KB |
| **M4** | `appendix_emitter.py` | Deterministische artefact-generatie + bundling | 48 KB |

De architectuur van M0 is bijzonder. Het introduceert een strikt onderscheid tussen:

- **Layer A (Semantisch)** -- pure data: welke operaties bestaan, hoe een pipeline is samengesteld, wat het domeinbeleid is
- **Layer B (Executie)** -- implementaties: hoe operaties worden uitgevoerd

Dit onderscheid maakt het mogelijk om pipelines te analyseren *als data* (voor symbolische analyse, conjecture mining) zonder ze uit te voeren.

### Research 2.0: Het Manus Protocol

Een AI-agent (Manus) voerde een volledig geprotocolleerd onderzoek uit met het M0-M4 framework:

- **630 experimentele runs** (35 pipelines x 6 bases x 3 cijferlengtes)
- **28 conjectures** gemined, elk met R^2 = 1.0
- **Falsificatie op secundair domein** -- 9 van 10 overleefden, 1 terecht gefalsifieerd
- **Structurele analyse** -- poging tot algebraische verklaring
- **Manifest hashes** voor volledige reproduceerbaarheid

### De P_k-ontdekking

Tijdens de structurele analyse van Research 2.0 werd een fundamenteel inzicht ontdekt: de engine past impliciet een **projectie-operator P_k** toe na elke operatie -- zero-padding naar k digits. Dit verandert het wiskundige object van pure operator-compositie naar projectieve dynamica.

(Zie `WAT_WE_ONTDEKTEN.md` en `BESCHOUWING_DRIEHOEKSDIALOOG_P_k.md` voor details.)

---

## De Zes Lagen van Redeneren

Het eindresultaat is een systeem met zes expliciete redeneringslagen plus een meta-laag:

```
LAAG 6 ──── Multi-base Generalisatie ─────── "Geldt dit in ELKE basis?"
LAAG 5 ──── Abductief Redeneren ──────────── "WAAROM is dit waar?"
LAAG 4 ──── Deductieve Theorie ───────────── "WAT VOLGT hieruit?"
LAAG 3 ──── Symbolische Redenering ──────── "WAT VOORSPEL ik?"
LAAG 2 ──── Operator Algebra + Kennisbank ── "WAT WEET ik zeker?"
LAAG 1 ──── Empirische Dynamica ──────────── "WAT ZIE ik?"
META   ──── Homeostatische Zelfregulatie ─── "Functioneer ik goed?"
```

### Laag 1: Empirische Dynamica

**Vraag:** "Wat gebeurt er als ik deze bewerking herhaal?"

Detecteert attractoren (vaste punten en cycli) door getallen te samplen of exhaustief te doorlopen. Dit is de zintuiglaag -- het systeem observeert, maar interpreteert niet.

**Voorbeeld:** "Als ik digit_sum herhaaldelijk toepas op 9876, kom ik uit op 9."

### Laag 2: Operator Algebra + Kennisbank

**Vraag:** "Wat weet ik *zeker* over deze operaties?"

Een bibliotheek van bewezen feiten en formele eigenschappen. De 83 KB-feiten vormen een axiomatisch fundament:

- `digit_sum(n) = n (mod 9)` -- bewezen getaltheorie
- `reverse` bewaart de cijfer-multiset -- bewezen
- `complement_9` is een involutie -- bewezen

Plus formele operator-eigenschappen: is deze operatie contractief? Bewaart ze de cijferlengte? Is ze commutatief met andere operaties?

### Laag 3: Symbolische Redenering

**Vraag:** "Kan ik *voorspellen* wat er zal gebeuren, zonder te rekenen?"

De operator-algebra combineert bekende eigenschappen om het gedrag van een pipeline te voorspellen *voor* die wordt uitgevoerd. Bereikte 100% accuraatheid over 300 pipelines.

De Fixed-Point Solver lost f(n)=n op via constraint-analyse in plaats van brute-force search.

### Laag 4: Deductieve Theorie

**Vraag:** "Wat *volgt logisch* uit wat ik weet?"

Genereert bewijsschetsen voor bevestigde patronen. Leidt nieuwe stellingen af uit combinaties van bekende feiten. Onderhoudt een Theory Graph die alle ontdekte objecten verbindt.

**Voorbeeld:** "Omdat digit_sum mod 9-invariant is, en omdat pipelines zonder groei-operaties bounded zijn, moet elke bounded pipeline die digit_sum bevat convergeren naar een waarde < 9."

### Laag 5: Abductief Redeneren

**Vraag:** "Waarom is dit *zo* en niet anders?"

De meest geavanceerde laag. Zoekt naar de *beste verklaring* voor een observatie:

- **Causale ketens:** "Vaste punten zijn deelbaar door 3 *omdat* ..."
- **Surprise-detectie:** "1089 verschijnt waar het niet hoort -- onderzoek!"
- **Self-questioning:** "Waarom is factor 11 dominant? Wat mis ik?"
- **Gap closure:** "Feit DS013 sluit het gat in bewijs PS002."

### Laag 6: Multi-base Generalisatie

**Vraag:** "Is dit specifiek voor basis 10, of universeel?"

Vertaalt alle resultaten naar willekeurige bases. De resonantiefactoren 9 en 11 in basis 10 worden (b-1) en (b+1) in basis b. De 1089-constante generaliseert (deels).

### Meta-laag: Homeostatische Zelfregulatie

**Vraag:** "Functioneer ik goed?"

De Self-Prompt/Reflection cyclus: na elke onderzoekssessie schrijft het systeem een eerlijke zelfreflectie ("wat is echt, wat is ruis") en genereert een prompt voor de volgende versie. Dit stuurt de evolutie.

---

## De Zelfsturende Cyclus

Het meest ongewone aspect van het project is de manier waarop de engine zijn eigen evolutie stuurt. Na elke onderzoekssessie worden twee documenten gegenereerd:

1. **REFLECTION** -- eerlijke analyse: wat is echt significant, wat is een artefact, wat mist er?
2. **SELF_PROMPT** -- concrete instructies voor de volgende versie

Dit creeerde een feedbacklus:

```
v7.0 draait → REFLECTION_V8.md:
  "v7.0 detecteert patronen maar begrijpt ze niet.
   100% symbolische predictie is indrukwekkend,
   maar er zijn geen proof sketches."
                    ↓
SELF_PROMPT_V8.md:
  "Bouw MODULE A: Proof Sketch Generator.
   Bouw MODULE B: Inductive Theorem Generator."
                    ↓
v8.0 draait → REFLECTION na v8.0:
  "Het 3^2 x 11 patroon is echt significant.
   Maar v8.0 weet WAT waar is, niet WAAROM."
                    ↓
v9.0: Abductief redeneren, kennisbank, causale ketens
```

Drie AI-systemen namen deel aan deze cyclus:
- **DeepSeek** (R1-R5): wiskundige consultatie
- **Manus** (R6): bulk-implementatie en protocol-uitvoering
- **Cascade/Claude** (R7-R11): formele bewijzen en architecturele sprongen

De menselijke onderzoeker orkestreerde de cyclus, koos de volgorde, en maakte de conceptuele sprongen die geen van de AI-systemen zelfstandig kon maken.

---

## Wat dit benadert

Is dit "echt redeneren"? Dat hangt af van je definitie. Maar het systeem doet iets dat er sterk op lijkt:

1. **Het observeert** (Laag 1) -- en onderscheidt triviale van niet-triviale observaties
2. **Het classificeert** (Laag 2) -- en bouwt een bibliotheek van bewezen feiten
3. **Het voorspelt** (Laag 3) -- en bereikt 100% accuraatheid op voorspellingen
4. **Het leidt af** (Laag 4) -- en genereert bewijsschetsen
5. **Het verklaart** (Laag 5) -- en bouwt causale ketens
6. **Het generaliseert** (Laag 6) -- en test of resultaten universeel zijn
7. **Het reflecteert** (Meta) -- en stuurt zijn eigen verbetering

Geen van deze lagen is op zichzelf bijzonder. Wat bijzonder is, is de **gelaagde compositie**: elke laag bouwt voort op de vorige, en de meta-laag stuurt de evolutie van het geheel. Dat is geen redeneren in de filosofische zin. Maar het is ook niet meer "gewoon rekenen."

De meest eerlijke beschrijving: het systeem benadert wat wetenschappelijk onderzoek is -- een gestructureerd proces van observeren, verklaren, bewijzen, falsifieren, en reflecteren -- zonder dat er een menselijke wetenschapper aan elk beslismoment te pas hoeft te komen.

De grens werd zichtbaar bij de P_k-ontdekking. Het systeem kon correct rapporteren dat het een patroon niet kon verklaren ("Structural explanation insufficient"). Maar het kon niet zelfstandig de stap maken van "ik begrijp het niet" naar "laat me eens kijken hoe de implementatie precies werkt." Die stap -- van inhoudelijke onmacht naar implementatie-inspectie -- vereiste menselijke interventie.

Dat is misschien de eerlijkste conclusie: het systeem redenert *binnen* een kader. Maar het kader zelf veranderen -- dat is nog steeds mensenwerk.

---

## Bestandsregister

### Fase I: Rekenen

| Bestand | Omvang | Kernbijdrage |
|---------|--------|--------------|
| `symmetry_discovery_engine.py` | 30 KB | Eerste engine, 22 operaties, evolutionair algoritme |
| `run_discovery.py` | 5 KB | Unified launcher (demo/explore/gpu/meta modes) |
| `quick_research.py` | 6 KB | Snelle iteratieve ontdekking |
| `meta_discovery_engine.py` | 35 KB | Zelf-verbeterend, nieuwe operaties dynamisch |
| `autonomous_researcher.py` | 26 KB | Dynamische operatie-generatie via templates |
| `gpu_symmetry_hunter.py` | 24 KB | CUDA kernels, 150M samples/sec |
| `gpu_deep_researcher.py` | 31 KB | GPU + zelf-adaptatie |
| `gpu_creative_research.py` | 21 KB | GPU creatieve operaties |
| `scoring_engine_v2.py` | 17 KB | Trivialiteitsfilter, property bonuses |
| `extended_research_session.py` | 25 KB | v4: Langdurig autonoom |
| `extended_research_session_v5.py` | 42 KB | v5: Cycli, genetische mutatie |
| `extended_research_session_v6.py` | 35 KB | v6: ML predictor, CPU parallel |

### Fase II-III: Ontdekken en Begrijpen (engines/)

| Bestand | Omvang | Redeneringslaag |
|---------|--------|-----------------|
| `autonomous_discovery_engine_v4.py` | 36 KB | Exploratie + hypothesevorming |
| `meta_symmetry_engine_v5.py` | 48 KB | Meta-learning + theory graph |
| `invariant_discovery_engine_v6.py` | 56 KB | Structurele abstractie + conjectures |
| `symbolic_dynamics_engine_v7.py` | 54 KB | Operator algebra + FP solver |
| `deductive_theory_engine_v8.py` | 73 KB | Bewijsschetsen + inductieve theorema's |
| `abductive_reasoning_engine_v9.py` | 105 KB | Causale ketens + zelf-vraagstelling |
| `abductive_reasoning_engine_v10.py` | 289 KB | 30 modules, 83 KB-feiten, 12 bewijzen |

### Fase IV: Verantwoorden (M0-M4)

| Bestand | Omvang | Publicatiefunctie |
|---------|--------|-------------------|
| `pipeline_dsl.py` | 40 KB | Canonieke semantiek, SHA-256 hashing |
| `experiment_runner.py` | 24 KB | Deterministische runs, SQLite schema |
| `feature_extractor.py` | 38 KB | Conjecture mining, falsificatie-engine |
| `proof_engine.py` | 47 KB | Bewijsschetsen, ranking model v1.0 |
| `appendix_emitter.py` | 48 KB | LaTeX appendices, JSON manifests |

### Totale omvang

- **Fase I scripts:** ~270 KB Python
- **Engine prototypes (v4-v15):** ~661 KB Python
- **M0-M4 submission code:** ~197 KB Python
- **Tests:** ~143 KB Python
- **Documentatie:** ~380 KB Markdown
- **Totaal:** ~1.6 MB broncode + documentatie

---

*SYNTRIAD Research -- februari 2026*
