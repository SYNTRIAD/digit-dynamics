# Hoe een implementatiedetail een wiskundig systeem creëert

*Over cijferbewerkingen, vaste-lengte projectie, en waarom de interessante wiskunde zit waar je hem niet verwacht*

---

## Het beginpunt

We onderzoeken wat er gebeurt als je bewerkingen uitvoert op de cijfers van een getal.

Neem een getal als 83. Tel de cijfers op: 8 + 3 = 11. Draai de cijfers om: 38. Sorteer ze aflopend: 83. Neem het complement in basis 10: 16. Dit soort bewerkingen — simpel, elementair, op basisschoolniveau uit te leggen — kun je combineren tot "pipelines": eerst dit, dan dat, dan dat.

De centrale vraag is: **welke getallen blijven onveranderd?**

Als je een bewerking toepast en hetzelfde getal terugkrijgt, heb je een *vast punt*. Het beroemdste voorbeeld is 6174 — de Kaprekar-constante. Neem een willekeurig viercijferig getal, sorteer de cijfers aflopend, trek het oplopend gesorteerde getal ervan af, en herhaal. Na hoogstens zeven stappen kom je altijd bij 6174 uit.

Ons project generaliseert dit idee. We bestuderen 22 verschillende cijferbewerkingen, in willekeurige combinaties, in elk talstelsel (niet alleen basis 10). We hebben daar inmiddels negen stellingen over bewezen, vijf oneindige families van vaste punten geclassificeerd, en meer dan 20 miljoen startwaarden doorgerekend.

Maar toen ontdekten we iets dat al onze aannames veranderde.

---

## De ontdekking

We lieten een AI-systeem (Manus) 630 experimenten draaien over 35 pipelines. Een externe reviewer (GPT-4) controleerde de resultaten. En daar zat een vreemd patroon.

De pipeline "tel cijfers op" had nul vaste punten. Logisch — de cijfersom van 83 is 11, en 11 ≠ 83. Geen enkel meercijferig getal is gelijk aan de som van zijn eigen cijfers.

Maar de pipeline "tel cijfers op, draai dan om" had precies **b − 1** vaste punten in basis b. In basis 10: negen stuks. Dat is vreemd. Omdraaien verandert de cijfersom niet (de som van 8 en 3 is dezelfde als de som van 3 en 8). Dus waarom zou de combinatie wél vaste punten hebben als de cijfersom alleen dat niet heeft?

Het antwoord bleek niet in de wiskunde te zitten, maar in de implementatie.

---

## De verborgen stap

Ons systeem werkt met getallen van een **vaste lengte** — bijvoorbeeld precies twee cijfers. En na elke bewerking wordt het resultaat teruggebracht naar die vaste lengte door voorloopnullen toe te voegen.

Kijk wat er dan gebeurt met het getal 10:

```
Stap 1: cijfersom van 10 = 1
Stap 2: 1 heeft maar één cijfer → aanvullen tot twee cijfers → 01
Stap 3: omdraaien → 10
```

We beginnen met 10 en eindigen met 10. Vast punt.

Hetzelfde geldt voor 20, 30, 40, ... tot en met 90. Dat zijn er negen — precies b − 1 in basis 10.

**Zonder die aanvulstap zou dit niet werken.** De cijfersom geeft een klein getal terug. Het systeem duwt dat kleine getal terug naar de vaste lengte. En het omdraaien van een getal met een voorloopnul creëert een getal dat toevallig precies de juiste cijfersom heeft.

Dat is geen bug. Dat is een *projectie*.

---

## Wat projectie doet

Stel je voor dat je in een kamer bent die precies twee meter bij twee meter is. Je gooit een bal. De bal zou normaal door de muur vliegen, maar in plaats daarvan kaatst hij terug. De kamer *dwingt* de bal om binnen de ruimte te blijven.

Dat is wat onze vaste-lengte aanvulling doet. Het resultaat van een bewerking kan buiten de "kamer" van k-cijferige getallen vallen. De projectie duwt het terug.

En net zoals de bal in de kamer andere banen volgt dan een bal in open ruimte, zo volgen onze getallen andere patronen dan ze zouden doen zonder projectie.

---

## Waarom dit alles verandert

**Zonder projectie** is ons systeem saai. De cijfersom krimpt alles:

```
83 → 11 → 2 → 2 → 2 → ...
```

Einde verhaal. Alles convergert naar een enkel cijfer.

**Met projectie** wordt het systeem structureel rijk:

```
83 → 11 → 011 → 110 → ...
```

De aanvulling met nullen creëert nieuwe informatie. Er ontstaan:
- **Vaste punten** die zonder projectie niet bestaan
- **Symmetrieën** die door de vaste lengte worden afgedwongen
- **Attractoren** — getallen waar alles naartoe stroomt
- **Families** — niet losse vaste punten maar hele reeksen, algebraïsch verklaarbaar

Het verschil is fundamenteel. Zonder projectie bestudeer je functies die krimpen. Met projectie bestudeer je een **gesloten dynamisch systeem** — een systeem dat in zichzelf terugkeert, met een eigen structuur.

---

## De wiskundige kern (voor wie het wil)

Voor het getal d·b^(k−1) (bijvoorbeeld 30 in basis 10 met k=2):

1. **Cijfersom**: de cijfers zijn d, 0, 0, ..., 0. Som = d.
2. **Projectie**: d is een enkel cijfer. Aanvullen tot k cijfers → 0, 0, ..., d.
3. **Omdraaien**: d, 0, 0, ..., 0.
4. **Terug naar getal**: d·b^(k−1).

Terug bij af. Vast punt. En dit werkt voor elke d van 1 tot b−1.

Bovendien kun je bewijzen dat er **geen andere** vaste punten zijn. Als de cijfersom groter is dan b−1, dan is het getal na projectie en omdraaien altijd kleiner dan het oorspronkelijke getal. Dus de enige oplossingen zijn precies die ene familie.

Dat is geen empirisch patroon. Dat is een algebraïsch noodzakelijk resultaat — een stelling.

---

## Wat dit zegt over ons project

We dachten dat we cijferbewerkingen bestudeerden.

In werkelijkheid bestuderen we **cijferbewerkingen onder vaste-lengte projectie**. Dat is een ander wiskundig object. De projectie is geen implementatiedetail — het is de kern van het systeem.

Dit heeft consequenties:

- **Operatoren zijn niet meer commutatief.** "Eerst optellen, dan omdraaien" geeft een ander resultaat dan "eerst omdraaien, dan optellen" — niet vanwege de bewerkingen zelf, maar vanwege de projectiestap ertussen.
- **Nieuwe structuren ontstaan.** Vaste-punt-families die puur algebraïsch volgen uit de projectie-eigenschappen.
- **Het systeem wordt rijker in plaats van armer.** Waar pure cijferbewerkingen alles laten krimpen, creëert projectie een gesloten wereld met eigen dynamica.

---

## Hoe we dit ontdekten

Niet door meer te rekenen. Maar door een discrepantie serieus te nemen.

Een AI-agent (Manus) draaide honderden experimenten en rapporteerde eerlijk: "ik zie het patroon maar kan het niet verklaren." Een reviewer (GPT-4) vond een logische fout in de verklaring. Dat leidde tot inspectie van de code. En daar bleek de projectiestap te zitten — impliciet, ongedocumenteerd, maar bepalend.

We formaliseerden die stap. We heranalyseerden de resultaten. En we zagen dat patronen die eerst toevallig leken, structureel noodzakelijk waren.

Dat is hoe computationeel onderzoek volwassen wordt. Niet door méér te rekenen, maar door te begrijpen wat je eigenlijk aan het berekenen bent.

---

## De kern in één zin

De interessante wiskunde zit niet in de bewerking. Die zit in de projectie.

---

*SYNTRIAD Research — februari 2026*
