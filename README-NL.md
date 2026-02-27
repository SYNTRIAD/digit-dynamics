# SYNTRIAD Digit-Dynamics Discovery Engine

**Systematische computationele exploratie van vastepunt-structuur in dynamische systemen van cijferoperaties.**

[![Licentie: MIT](https://img.shields.io/badge/Licentie-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

Een computationele engine voor het verkennen, classificeren en verifiëren van algebraïsche structuur in samengestelde cijferoperatiesystemen. Geëvolueerd door 15 versies en 11 menselijk gestuurde onderzoekssessies, waarbij multi-agent AI-samenwerking gecombineerd werd met algebraïsch redeneren om 9 theorema's te identificeren en computationeel te verifiëren over willekeurige getalbases (b ≥ 3).

![Convergentiepatroon](assets/convergence-pattern.png)

---

## 🎯 Kernontdekking: De P_k Projectie

**Het Probleem:** Wanneer je cijferoperaties (reverse, digit-sum, sort) herhaaldelijk op getallen toepast, krimpen ze doorgaans tot enkelvoudige cijfers. Wiskundig oninteressant.

**Het Inzicht:** Voeg **vaste-lengte padding** toe (de P_k projectie) en er ontstaat rijke algebraïsche structuur:
- 9 theorema's met algebraïsche bewijzen, exhaustief geverifieerd
- 5 oneindige vasteputfamilies met expliciete telformules
- Universele patronen over **alle getalbases** (b ≥ 3)
- Resonantiestructuur bepaald door base-aritmetiek: (b−1)(b+1)² = 1089 in base 10

**Het diepere punt:** De vaste punten zijn geen toevalligheden — ze worden afgedwongen door de algebraïsche structuur van positionele getalsystemen. Specifiek bepalen 10 ≡ 1 (mod 9) en 10 ≡ −1 (mod 11) welke getallen herhaalde cijferoperaties overleven.

Lees meer: [Wat We Ontdekten](NL/docs/WAT_WE_ONTDEKTEN.md) | [Emergentie-essay](assets/emergence-mechanisms.md)

---

## 📚 Repository Structuur

Deze repository bevat zowel Engelse als Nederlandse versies:

### **[→ English Version (EN/)](EN/)**
Complete documentation, papers, code, and research artifacts in English.

### **[→ Nederlandse Versie (NL/)](NL/)**  
Volledige documentatie, papers, code en onderzoeksartefacten in het Nederlands.

### **[→ Assets](assets/)**
Visualisaties en essays over universele patronen.

---

## 🚀 Snel Starten

```bash
# Nederlandse versie
cd NL
pip install -r requirements.txt

# Draai de onderzoeksengine (v15, laatste versie)
python engines/research_engine_v15.py

# Draai de reproduceerbaarheidspipeline (M0-M4 modules)
python src/reproduce.py --db results.db --out repro_out --bundle

# Draai alle tests
pytest tests/ -v
```

Zie [NL/README.md](NL/README.md) voor volledige documentatie.

---

## 📄 Publicaties

### Papers (arXiv in voorbereiding)
- **Paper A:** "Fixed Points of Digit-Operation Pipelines in Arbitrary Bases"  
  Zuivere wiskunde — 9 theorema's, 5 oneindige families, multi-base generalisatie
  
- **Paper B:** "Attractor Spectra and ε-Universality in Digit-Operation Dynamical Systems"  
  Experimentele wiskunde — nieuw ε-universaliteitskader, exhaustieve verificatie over 10⁷ inputs

### Kandidaat OEIS-rij
- **a(n) = 110×(10^(n+1) − 1)** voor n ≥ 1  
  Vaste punten van de 1089-truc-afbeelding voor (n+4)-cijferige getallen

---

## 🧬 De Evolutie

De engine evolueerde door 15 versies over 11 feedbackrondes, gestuurd door een menselijke onderzoeker die drie AI-systemen orkestreerde:

| Fase | Versies | Wat Veranderde |
|------|---------|----------------|
| **Rekenen** | v1–v2 | GPU brute-force verificatie, exhaustieve attractordetectie |
| **Verkennen** | v4–v6 | Operatoralgebra, invariantontdekking, symbolische voorspelling |
| **Begrijpen** | v7–v9 | Kennisbank (83 feiten), causale ketens, zelf-bevraging |
| **Verifiëren** | v10–v15 | Formele bewijzen (12/12), multi-base generalisatie, open vragen |
| **Formaliseren** | M0–M4 | Canonieke hashing, deterministische reproduceerbaarheid, paper-appendices |

De progressie: *observeren → classificeren → voorspellen → bewijzen*.

Volledig verhaal: [Evolutie van Scripts naar Redeneren](NL/docs/EVOLUTIE_VAN_SCRIPTS_NAAR_REDENEREN.md)

---

## 🎨 Visualisaties & Essays

### [Convergentiepatroon](assets/convergence-pattern.png)
Hoge-resolutie visualisatie van vastepuntclustering in de cijferoperatieruimte.

### [De Mechanica van Emergentie](assets/emergence-mechanisms.md)
Essay dat onderzoekt hoe eenvoudige regels complexe structuur creëren over vijf systemen — van moleculen tot cultuur. Toont cijferdynamica als instantie van universele emergentiepatronen.

---

## 🔬 Kernresultaten

### Wiskundige Resultaten (Paper A)

| Theorema | Stelling | Scope |
|----------|----------|-------|
| Symmetrische VP-telling | (b−2) · b^(k−1) symmetrische vaste punten onder 2k-cijferige getallen | Alle bases b ≥ 3 |
| Universele 1089-familie | A_b = (b−1)(b+1)² generaliseert 1089 naar elke base | Alle bases b ≥ 3 |
| Vier oneindige families | Expliciete telformules, paarsgewijs disjunct | Base 10 |
| Vijfde familie (1089-truc) | n_k = 110 · (10^(k−3) − 1) voor k ≥ 5 | Base 10 |
| Kaprekar-constanten | K_b = (b/2)(b²−1) voor even bases; 495 en 6174 algebraïsch | Bases b ≥ 4 |
| Armstrong-bovengrens | k_max(b) ≤ ⌊b · log(b) / log(b − 1)⌋ + 1 | Alle bases b ≥ 3 |
| Conditionele Lyapunov | Digit-sum-daling voor operaties in klasse P ∪ C | Alle bases |

### Computationele Verificatie

- 260 unittests over M0–M4 modules (deterministische infrastructuur)
- 98 legacy tests over onderzoeksengines (v4–v15)
- 12/12 algebraïsche bewijzen computationeel geverifieerd
- Exhaustieve verificatie over alle k-cijferige inputs voor k = 3…7
- Canonieke SHA-256 hashketen: register → pipeline → domein → resultaat

---

## 🤖 Multi-Agent Onderzoeksproces

Dit project gebruikte een tripartite samenwerkingsmodel:

| Rol | Agent | Bijdrage |
|-----|-------|----------|
| **Menselijke Visionair** | R. Havenaar | Onderzoeksrichting, conceptuele sprongen, orkestratie, algebraïsch inzicht |
| **Wiskundige Consultant** | DeepSeek (R1–R5) | Diep wiskundig redeneren, verfijning van vermoedens |
| **Implementatie & Schaling** | Manus (R6) | Bulk-implementatie, multi-base engine, protocoluitvoering |
| **Formele Bewijzen & Architectuur** | Claude/Cascade (R7–R11) | Bewijsverificatie, M0–M4 architectuur, publicatievoorbereiding |

De menselijke onderzoeker stuurde elke onderzoeksfase, identificeerde de algebraïsche structuren, en maakte de conceptuele sprongen die cijferoperaties verbonden met modulaire aritmetiek. De AI-systemen voerden uit, verifieerden en formaliseerden.

---

## 🏗️ Architectuur

De codebase heeft twee sporen:

### Onderzoeksengine (v15)
Enkel-bestand exploratie-engine (~6.500 regels). Bevat 30 modules verspreid over 6 redeneerlagen — van empirische dynamica tot abductief redeneren. Gebruikt voor ontdekking en vermoedengeneratie.

### Reproduceerbaarheidsinfrastructuur (M0–M4.1)
Modulaire, deterministische, publicatiekwaliteit codebase:

| Module | Functie | Regels |
|--------|---------|--------|
| **M0** (pipeline_dsl.py) | Canonieke semantiek, operatieregister, SHA-256 identiteit | ~1.050 |
| **M1** (experiment_runner.py) | SQLite resultaatopslag, batchuitvoering, JSON-export | ~640 |
| **M2** (feature_extractor.py) | Getalkenmerken, orbitanalyse, vermoedenmining | ~900 |
| **M3** (proof_engine.py) | Bewijsskeletten, dichtheidsschatting, rangschikkingsmodel v1.0 | ~1.160 |
| **M4** (appendix_emitter.py) | LaTeX-appendixgeneratie, manifest, reproduceerbaarheidsbundel | ~1.170 |

Kernbeslissing in het ontwerp: **Laag A (semantisch) / Laag B (executie) scheiding** in M0. Pipelinespecificaties zijn zuivere data — inspecteerbaar, hashbaar en onafhankelijk van implementatie.

---

## 📖 Citeren

Als je dit werk gebruikt, citeer dan:

```bibtex
@misc{syntriad2026digit,
  title={Algebraic Structure of Fixed Points in Composed Digit-Operation Dynamical Systems},
  author={Havenaar, Remco and SYNTRIAD Research},
  year={2026},
  note={Computationele exploratie van cijferoperatiepipelines over willekeurige bases},
  url={https://github.com/SYNTRIAD/digit-dynamics}
}
```

---

## 📜 Licentie

MIT-licentie — zie [LICENSE](NL/LICENSE) voor details.

---

## 🔗 Links

- **Papers:** [EN/papers/](EN/papers/)
- **Documentatie:** [NL/docs/](NL/docs/)
- **Broncode:** [NL/src/](NL/src/) (M0–M4 modules)
- **Onderzoeksengines:** [NL/engines/](NL/engines/) (v1–v15)
- **Reproduceerbaarheid:** [NL/src/reproduce.py](NL/src/reproduce.py)
- **SYNTRIAD Research:** [syntriad.com](https://syntriad.com)

---

*SYNTRIAD Research — februari 2026*
