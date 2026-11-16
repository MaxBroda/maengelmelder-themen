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
