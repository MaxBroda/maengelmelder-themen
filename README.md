# Themenextraktion aus dem Mängelmelder Dresden

Portfolio-Projekt im Kurs *Projekt: Data Analysis* (DLBDSEDA02_D), IU Internationale Hochschule.

Aus den Freitexten der Meldungen an den [Mängelmelder Dresden](https://buergerbeteiligung.sachsen.de/portal/dresden/beteiligung/themen/1024377) werden mit NLP-Verfahren die häufigsten Themen extrahiert. Verglichen werden Bag-of-Words und TF-IDF als Vektorisierung sowie LDA und NMF als Themenmodelle.

## Setup

Voraussetzung: Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` enthält das spaCy-Modell `de_core_news_sm`. Ein separater `spacy download` ist nicht nötig.

## Daten laden

```bash
python -m maengelmelder.fetch
```

Das Skript ruft alle Meldungen seit 02/2022 über die Open311-Schnittstelle ab und schreibt sie nach `data/raw/requests.csv`. Die Rohdaten sind nicht versioniert. Da laufend neue Meldungen hinzukommen, liefert ein späterer Abruf mehr Zeilen.

Die Schnittstelle gibt pro Anfrage höchstens 1.000 Meldungen zurück und ignoriert Paging-Parameter. Der Client fragt deshalb monatsweise ab und halbiert ein Zeitfenster so lange, bis es weniger als 1.000 Treffer enthält.

## Vorverarbeitung

```bash
python -m maengelmelder.preprocess
```

Schreibt `data/processed/requests_clean.csv` mit einer Spalte `tokens` (Lemmata, durch Leerzeichen getrennt). Die Schritte:

1. Reguläre Ausdrücke entfernen URLs, E-Mail-Adressen, Postleitzahlen, Straßennamen mit Hausnummer und Grußformeln.
2. Texte, die danach identisch sind, werden nur einmal behalten.
3. spaCy (`de_core_news_sm`) tokenisiert und bestimmt die Wortart. Behalten werden Substantive, Adjektive und Verben ohne Stoppwörter.
4. `simplemma` bildet die Grundform. Die spaCy-Modelle lemmatisieren deutsche Plurale unzuverlässig („Laternen“ → „Latern“).
5. Die Lemmata werden kleingeschrieben. Eine eigene Stoppwortliste entfernt Wörter ohne Thema („bitte“, „siehe“, „Ecke“, „Woche“).
6. Meldungen mit weniger als drei Lemmata entfallen.

## Vektorisierung

```bash
python -m maengelmelder.vectorize
```

Erzeugt aus den Lemmata eine Bag-of-Words- und eine TF-IDF-Matrix mit denselben Einstellungen: Unigramme und Bigramme, `min_df=5`, `max_df=0.5`. Ausgegeben werden die Größe des Vokabulars, die stärksten Begriffe beider Verfahren und die Begriffe, deren Rang sich durch die TF-IDF-Gewichtung am stärksten verschiebt.

## Wahl der Themenzahl k

```bash
python -m maengelmelder.select_k
```

Trainiert LDA (auf Bag-of-Words) und NMF (auf TF-IDF) für k = 4, 6, …, 30. Je k gibt es drei Läufe, jeder auf einer anderen Zufallsstichprobe von 80 % der Meldungen. Laufzeit: etwa 10 Minuten. Bewertet werden:

| Kennzahl | Frage |
|---|---|
| Kohärenz c_v, c_npmi | Kommen die Top-Wörter eines Themas in den Meldungen gemeinsam vor? |
| Stabilität | Finden die drei Läufe dieselben Themen? Gemessen über die Jaccard-Ähnlichkeit der Top-10-Wörter, Themen per Hungarian-Algorithmus zugeordnet (vereinfacht nach Greene et al., 2014). |
| Topic Diversity | Welcher Anteil der Top-25-Wörter ist über alle Themen eindeutig (Dieng et al., 2020)? |
| NMI | Wie stark stimmen die Themen mit `service_name` überein? „Sonstiges“ ist ausgenommen. |

Die Themenmodelle arbeiten nur mit Einzelwörtern. Mit Bigrammen füllen Varianten wie „laterne defekt“ und „defekt laterne“ die Top-Wörter und treiben die Kohärenz künstlich hoch.

Ergebnis: `results/k_selection.csv` und `results/k_selection.png`. Gewählt wurde NMF mit k = 10. Bei k = 8–10 sind Kohärenz und Stabilität am höchsten, ab k = 12 fällt die Kohärenz deutlich. Gegenüber k = 8 trennt k = 10 Sperrmüll und Wertstoffcontainer als eigene Themen ab.

## Themen und Unterthemen

```bash
python -m maengelmelder.analyze
```

Trainiert NMF und LDA mit k = 10 auf allen Meldungen. Laufzeit: etwa 30 Sekunden. Ergebnisse in `results/`:

| Datei | Inhalt |
|---|---|
| `topics_nmf.csv`, `topics_lda.csv` | Anteil, Top-Wörter und Beispielmeldungen je Thema |
| `crosstab_nmf.csv`, `crosstab_nmf.png` | Anteil der NMF-Themen je `service_name` |
| `subtopics.csv` | Unterthemen je Kategorie (NMF, Kategorien ab 500 Meldungen) |
| `subtopics_k.csv` | Kennzahlen zur Wahl von k je Kategorie |

Für die Unterthemen gilt dieselbe Logik wie bei den Hauptthemen: Unter den k = 2–8 mit einer Stabilität von mindestens 0,75 zählt die höchste Kohärenz c_v. Erreicht kein k die Schwelle, wird das stabilste k genommen und die Kategorie als `stabil = False` markiert.

## Tests

```bash
python -m pytest
```

## Struktur

```
maengelmelder/   Python-Paket (Datenabruf, später Vorverarbeitung und Modelle)
notebooks/       Analyse und Auswertung
data/raw/        Rohdaten aus der API (nicht versioniert)
data/processed/  bereinigte Texte (nicht versioniert)
tests/
```
