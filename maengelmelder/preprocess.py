"""Bereinigt die Meldungstexte und speichert Lemmata für die Vektorisierung.

    python -m maengelmelder.preprocess [--raw data/raw/requests.csv] [--out data/processed/requests_clean.csv]
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import simplemma
import spacy
from spacy.language import Language

from maengelmelder.export import write_csv

_STREET_SUFFIX = r"(?i:straße|strasse|str\.|gasse|allee|ufer)"
_PLACE_SUFFIX = r"(?i:weg|platz|ring|damm|markt)"


class TextCleaner:
    # Wörter wie "Radweg" oder "Spielplatz" sind Inhalt, keine Adresse. Deshalb
    # gelten -weg/-platz/-ring nur mit Bindestrich-Namen oder Hausnummer als Adresse.
    _PATTERNS = [
        r"https?://\S+|www\.\S+",
        r"\S+@\S+\.\S+",
        r"\b0\d{4}\s+Dresden\b|\b0\d{4}\b",
        rf"\b[A-ZÄÖÜ][\wäöüß-]*{_STREET_SUFFIX}(?:\s*\d+\s*[a-z]?\b)?",
        rf"\b[A-ZÄÖÜ][\wäöüß]*(?:er|sche[nr]?)\s+(?:Straße|Strasse|Str\.|Gasse|Allee|Weg|Platz|Ring|Ufer)(?:\s*\d+\s*[a-z]?\b)?",
        rf"\b[A-ZÄÖÜ][\wäöüß]*-[\wäöüß-]*{_PLACE_SUFFIX}\b(?:\s*\d+\s*[a-z]?\b)?",
        rf"\b[A-ZÄÖÜ][\wäöüß]*{_PLACE_SUFFIX}\s+\d+\s*[a-z]?\b",
        r"(?i:\b(?:mit\s+)?(?:freundlichen|besten|vielen|viele|liebe)\s+grü(?:ß|ss)en?\b)",
        r"(?i:\bsehr\s+geehrte\w*(?:\s+damen\s+und\s+herren)?\b)",
        r"(?i:\b(?:vielen\s+)?dank(?:e|eschön)?\b|\bmfg\b|\blg\b|\bhallo\b|\bguten\s+(?:tag|morgen|abend)\b)",
    ]

    def __init__(self) -> None:
        self._pattern = re.compile("|".join(f"(?:{p})" for p in self._PATTERNS))

    def clean(self, text: str) -> str:
        text = self._pattern.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()


class Lemmatizer:
    # Wörter, die quer durch alle Meldungen stehen und kein Thema tragen:
    # Bitten, Verweise auf Fotos, Orts- und Zeitangaben, allgemeine Verben.
    DOMAIN_STOPWORDS = frozenset({
        "bitte", "bitten", "entfernen", "beseitigen", "beräumen", "beräumung", "siehe",
        "sehen", "bild", "foto", "anhang", "dresden", "stadt", "melden", "meldung",
        "liegen", "geben", "stehen", "befinden", "lassen", "stellen", "bestehen",
        "seit", "mal", "schon", "leider", "wieder", "immer", "gerade", "heute",
        "ecke", "höhe", "richtung", "bereich", "stelle", "seite", "ort", "gesamt",
        "tag", "woche", "monat", "jahr", "gruß",
    })
    CONTENT_POS = frozenset({"NOUN", "ADJ", "VERB"})

    def __init__(self, nlp: Language, lemmatize: Callable[[str], str], min_length: int = 3) -> None:
        self._nlp = nlp
        self._lemmatize = lemmatize
        self._min_length = min_length

    def lemmas(self, texts: Iterable[str], batch_size: int = 500) -> list[list[str]]:
        return [self._content_lemmas(doc) for doc in self._nlp.pipe(texts, batch_size=batch_size)]

    def _content_lemmas(self, doc) -> list[str]:
        # Erst nach der Lemmatisierung kleinschreiben: Wortart-Tagger und
        # Lemmatisierer erkennen Substantive an der Großschreibung.
        lemmas = (
            self._lemmatize(token.text).lower()
            for token in doc
            if token.is_alpha and not token.is_stop and token.pos_ in self.CONTENT_POS
        )
        return [
            lemma for lemma in lemmas
            if len(lemma) >= self._min_length and lemma not in self.DOMAIN_STOPWORDS
        ]


class Preprocessor:
    def __init__(self, cleaner: TextCleaner, lemmatizer: Lemmatizer, min_tokens: int = 3) -> None:
        self._cleaner = cleaner
        self._lemmatizer = lemmatizer
        self._min_tokens = min_tokens

    def run(self, raw: pd.DataFrame) -> pd.DataFrame:
        frame = raw[["service_request_id", "service_name", "requested_datetime", "description"]].copy()
        frame["text_clean"] = frame["description"].fillna("").map(self._cleaner.clean)
        frame = frame.drop_duplicates(subset="text_clean")
        tokens = self._lemmatizer.lemmas(frame["text_clean"])
        frame["tokens"] = [" ".join(t) for t in tokens]
        frame["n_tokens"] = [len(t) for t in tokens]
        return frame[frame["n_tokens"] >= self._min_tokens].reset_index(drop=True)


def load_nlp() -> Language:
    return spacy.load("de_core_news_sm", exclude=["parser", "ner", "lemmatizer"])


def german_lemma(word: str) -> str:
    # Die spaCy-Modelle (sm/md/lg) lemmatisieren deutsche Plurale unzuverlässig
    # ("Laternen" -> "Latern"); simplemma arbeitet mit einem Wörterbuch.
    return simplemma.lemmatize(word, lang="de", greedy=True)


def default_lemmatizer() -> Lemmatizer:
    return Lemmatizer(load_nlp(), german_lemma)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/requests.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/requests_clean.csv"))
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    preprocessor = Preprocessor(TextCleaner(), default_lemmatizer())
    clean = preprocessor.run(raw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(clean, args.out)
    print(f"{len(raw)} Meldungen -> {len(clean)} nach Bereinigung -> {args.out}")


if __name__ == "__main__":
    main()
