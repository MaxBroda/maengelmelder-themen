"""Wandelt die Lemmata in Bag-of-Words- und TF-IDF-Matrizen um und vergleicht beide.

    python -m maengelmelder.vectorize [--clean data/processed/requests_clean.csv]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass(frozen=True)
class DocumentTermMatrix:
    name: str
    matrix: csr_matrix
    vocabulary: np.ndarray

    @property
    def n_terms(self) -> int:
        return self.matrix.shape[1]

    @property
    def density(self) -> float:
        rows, cols = self.matrix.shape
        return self.matrix.nnz / (rows * cols)

    @property
    def n_bigrams(self) -> int:
        return int(sum(" " in term for term in self.vocabulary))

    def top_terms(self, n: int = 20) -> pd.Series:
        return self.ranks().nsmallest(n)

    def ranks(self) -> pd.Series:
        weights = np.asarray(self.matrix.sum(axis=0)).ravel()
        return pd.Series(weights, index=self.vocabulary).rank(ascending=False, method="first").astype(int)


class CorpusVectorizer:
    """Gemeinsame Einstellungen für beide Verfahren, damit der Vergleich fair bleibt.

    min_df: Ein Begriff muss in mindestens so vielen Meldungen vorkommen.
    max_df: Anteil der Meldungen, ab dem ein Begriff als zu allgemein gilt.
    """

    def __init__(self, min_df: int = 5, max_df: float = 0.5, ngram_range: tuple[int, int] = (1, 2)) -> None:
        # Die Texte sind bereits lemmatisiert und kleingeschrieben.
        self._params = dict(
            token_pattern=r"\S+", lowercase=False, min_df=min_df, max_df=max_df, ngram_range=ngram_range
        )

    def bag_of_words(self, documents: Sequence[str]) -> DocumentTermMatrix:
        return self._fit("Bag-of-Words", CountVectorizer(**self._params), documents)

    def tfidf(self, documents: Sequence[str]) -> DocumentTermMatrix:
        return self._fit("TF-IDF", TfidfVectorizer(**self._params), documents)

    @staticmethod
    def _fit(name: str, vectorizer: CountVectorizer, documents: Sequence[str]) -> DocumentTermMatrix:
        matrix = vectorizer.fit_transform(documents).tocsr()
        return DocumentTermMatrix(name, matrix, vectorizer.get_feature_names_out())


def compare(matrices: Sequence[DocumentTermMatrix], n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        {
            m.name: {"Begriffe": m.n_terms, "davon Bigramme": m.n_bigrams, "Dichte": f"{m.density:.4%}"}
            for m in matrices
        }
    )
    top = pd.DataFrame({m.name: m.top_terms(n).index for m in matrices}, index=range(1, n + 1))
    return summary, top


def rank_shift(bow: DocumentTermMatrix, tfidf: DocumentTermMatrix, within: int = 100, n: int = 10) -> pd.DataFrame:
    """Begriffe, deren Rang sich durch die TF-IDF-Gewichtung am stärksten verschiebt.

    Positiver Wert: TF-IDF stuft den Begriff höher ein als die reine Zählung.
    """
    ranks = pd.DataFrame({bow.name: bow.ranks(), tfidf.name: tfidf.ranks()})
    ranks = ranks[ranks.min(axis=1) <= within]
    ranks["Verschiebung"] = ranks[bow.name] - ranks[tfidf.name]
    ranks = ranks.sort_values("Verschiebung")
    return pd.concat([ranks.tail(n)[::-1], ranks.head(n)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", type=Path, default=Path("data/processed/requests_clean.csv"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    documents = pd.read_csv(args.clean)["tokens"].tolist()
    vectorizer = CorpusVectorizer()
    bow, tfidf = vectorizer.bag_of_words(documents), vectorizer.tfidf(documents)

    summary, top = compare([bow, tfidf], n=args.top)
    print(f"{len(documents)} Meldungen\n\n{summary}\n\nStärkste Begriffe (Summe über alle Meldungen):\n{top}")

    print(f"\nStärkste Rangverschiebung unter den Top 100 (oben: aufgewertet, unten: abgewertet):\n{rank_shift(bow, tfidf)}")


if __name__ == "__main__":
    main()
