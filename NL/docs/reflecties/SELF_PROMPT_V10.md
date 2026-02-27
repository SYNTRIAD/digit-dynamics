# SELF_PROMPT_V10.md — R7-sessie (2026-02-25)

## Sessie-doel: Formaliseren en Bewijzen

De R6-sessie heeft de theorie verfijnd en een cruciale fout in de formule voor symmetrische FPs aan het licht gebracht. De R7-sessie moet zich richten op het formaliseren van deze nieuwe inzichten en het dieper graven in de onverwachte resultaten. De focus verschuift van brede exploratie naar diep, formeel bewijswerk.

Implementeer de volgende vier prioriteiten (P5-P8) als uitbreidingen op `abductive_reasoning_engine_v10.py`. Lever een `v11.0` op, inclusief de run-output, nieuwe KB-feiten (DS034+), en een `REFLECTION_V11.md`.

### P5: Formule-correctie voor Symmetrische FPs (Module R)

**Probleem:** De formule `(b-2)×b^(k-1)` voor het aantal symmetrische FPs van `rev∘comp` is incorrect (systematische `+1` afwijking).
**Doel:** Leid de correcte, algemene formule af voor het aantal symmetrische FPs van `rev∘comp` in basis `b` voor `2k` digits.

**Implementatie:**
1.  Creëer een nieuwe module `SymmetricFPFormula` (Module R).
2.  Implementeer een functie `count_symmetric_fps_bruteforce(base, k)` die het exacte aantal telt voor kleine `k`.
3.  Implementeer een functie `derive_formula(base, k)` die de correcte formule algebraïsch afleidt. Houd rekening met de randgevallen:
    *   `d_1 = 0` (niet toegestaan)
    *   `d_1 = b-1` (leidt tot `d_{2k}=0`, wat een leading zero geeft na complement, maar het getal `(b-1)0...` kan zelf een FP zijn).
    *   Het geval `b` is even vs. oneven.
4.  Voeg een nieuwe fase toe aan de `run_research_session` die de afgeleide formule verifieert tegen de brute-force telling voor `k=1, 2, 3` en `b=8, 10, 12`.
5.  Documenteer de correcte formule en het bewijs in een nieuw KB-feit (DS034).

### P6: Generalisatie van de 1089-familie (Module S)

**Probleem:** De 1089-familie lijkt uniek voor basis 10. De theoretische voorspelling `(b-1)²×(b+1)` is incorrect.
**Doel:** Onderzoek waarom de 1089-familie zo uniek is voor basis 10.

**Implementatie:**
1.  Creëer een nieuwe module `Family1089Generalizer` (Module S).
2.  Implementeer een functie `analyze_kaprekar_analog(base)` die de 3-digit Kaprekar-constante in basis `b` vindt en analyseert (is het complement-gesloten? heeft het een vergelijkbare structuur?).
3.  Implementeer een functie `find_true_analog(base)` die systematisch zoekt naar een 4-digit getal `N` in basis `b` zodanig dat `N×m` complement-gesloten is voor `m=1..b-1`.
4.  Voeg een fase toe die deze analyse uitvoert voor `b=8, 12, 16` en de resultaten rapporteert.
5.  Formuleer een hypothese (DS035) over de voorwaarden waaronder een `1089-analoog` kan bestaan (bv. vereist het dat `b-1` en `b+1` specifieke eigenschappen hebben?).

### P7: Algebraïsch Bewijs voor Empirische FP-condities (Module T)

**Probleem:** De Symbolic FP Classifier vond een nieuwe, empirische conditie: "FPs van `truc_1089 → digit_sum → digit_product` zijn palindromen".
**Doel:** Bewijs (of weerleg) deze stelling algebraïsch.

**Implementatie:**
1.  Creëer een nieuwe module `EmpiricalProofEngine` (Module T).
2.  Implementeer een functie `prove_palindrome_fp_conjecture(pipeline)`.
3.  De functie moet de pipeline symbolisch analyseren. Laat `n` een palindroom zijn. Volg de transformatie:
    *   `truc_1089(n)`: Wat is het effect op een palindroom?
    *   `digit_sum(...)`: Wat is de digit sum van het resultaat?
    *   `digit_product(...)`: Wat is het digit product daarvan?
    *   Toon aan dat het eindresultaat weer `n` is, of vind een tegenvoorbeeld.
4.  Voeg een fase toe die dit bewijs uitvoert en het resultaat (bewezen, weerlegd, of open) rapporteert en vastlegt in een KB-feit (DS036).

### P8: Lyapunov-functie Verificatie (Module U)

**Probleem:** De Lyapunov-zoeker vond `L(n) = 1×digit_count + 2×hamming` als dalende functie voor de `rotate_right → sort_asc → truc_1089` pipeline.
**Doel:** Probeer algebraïsch te bewijzen dat deze functie inderdaad strikt dalend is.

**Implementatie:**
1.  Creëer een nieuwe module `LyapunovVerifier` (Module U).
2.  Implementeer een functie `verify_lyapunov_decrease(pipeline, L)`.
3.  Analyseer het effect van elke operatie in de pipeline op de componenten van `L` (digit_count en hamming weight).
    *   `rotate_right(n)`: `digit_count` en `hamming` blijven gelijk.
    *   `sort_asc(n)`: `digit_count` en `hamming` blijven gelijk.
    *   `truc_1089(n)`: Dit is de cruciale stap. Analyseer hoe `truc_1089` de `digit_count` en `hamming` van een getal beïnvloedt. Is `L(truc_1089(n)) < L(n)`?
4.  Voeg een fase toe die deze verificatie uitvoert en het resultaat (bewezen, weerlegd, of open) rapporteert en vastlegt in een KB-feit (DS037).

Lever alle nieuwe en bijgewerkte bestanden op in een ZIP-archief genaamd `symbolic_dynamics_engine_v11.zip`zip`.
