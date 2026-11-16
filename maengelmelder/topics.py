from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import normalized_mutual_info_score

from maengelmelder.vectorize import DocumentTermMatrix


@dataclass(frozen=True)
class FittedTopics:
    components: np.ndarray
    doc_topics: np.ndarray
    vocabulary: np.ndarray

    @property
    def k(self) -> int:
        return self.components.shape[0]

    def top_terms(self, n: int = 10) -> list[list[str]]:
        return [list(self.vocabulary[row.argsort()[::-1][:n]]) for row in self.components]

    def dominant_topic(self) -> np.ndarray:
        return self.doc_topics.argmax(axis=1)


@dataclass(frozen=True)
class TopicModelSpec:
    """Ein Themenmodell zusammen mit der Matrix, auf der es trainiert wird."""

    name: str
    data: DocumentTermMatrix
    build: Callable[[int, int], object]

    def fit(self, k: int, seed: int, rows: np.ndarray | None = None) -> FittedTopics:
        model = self.build(k, seed)
        train = self.data.matrix if rows is None else self.data.matrix[rows]
        model.fit(train)
        doc_topics = model.transform(self.data.matrix)
        return FittedTopics(model.components_, doc_topics, self.data.vocabulary)


def lda_spec(bow: DocumentTermMatrix, max_iter: int = 20) -> TopicModelSpec:
    return TopicModelSpec(
        "LDA",
        bow,
        lambda k, seed: LatentDirichletAllocation(
            n_components=k, random_state=seed, learning_method="batch", max_iter=max_iter
        ),
    )


def nmf_spec(tfidf: DocumentTermMatrix, max_iter: int = 400) -> TopicModelSpec:
    return TopicModelSpec(
        "NMF",
        tfidf,
        lambda k, seed: NMF(n_components=k, random_state=seed, init="nndsvda", max_iter=max_iter),
    )


class TopicEvaluator:
    """Kennzahlen zur Wahl der Themenzahl.

    categories: service_name je Meldung; None-Einträge (z. B. "Sonstiges")
    fließen nicht in die NMI ein, weil sie keine inhaltliche Kategorie sind.
    """

    def __init__(self, texts: Sequence[list[str]], categories: Sequence[str | None], top_n: int = 10) -> None:
        self._texts = list(texts)
        self._dictionary = Dictionary(self._texts)
        self._labels = np.asarray(categories, dtype=object)
        self._labelled = np.array([c is not None for c in categories])
        self._top_n = top_n

    def coherence(self, topics: FittedTopics, measure: str) -> float:
        model = CoherenceModel(
            topics=topics.top_terms(self._top_n),
            texts=self._texts,
            dictionary=self._dictionary,
            coherence=measure,
            processes=1,
        )
        return float(model.get_coherence())

    @staticmethod
    def diversity(topics: FittedTopics, top_n: int = 25) -> float:
        """Anteil eindeutiger Wörter unter den Top-n aller Themen (Dieng et al., 2020)."""
        terms = [term for topic in topics.top_terms(top_n) for term in topic]
        return len(set(terms)) / len(terms)

    def nmi(self, topics: FittedTopics) -> float:
        return float(
            normalized_mutual_info_score(self._labels[self._labelled], topics.dominant_topic()[self._labelled])
        )

    def stability(self, runs: Sequence[FittedTopics]) -> float:
        """Mittlere Übereinstimmung der Themen zwischen Läufen (vereinfacht nach Greene et al., 2014).

        Je Paar von Läufen werden die Themen per Hungarian-Algorithmus einander
        zugeordnet und die Jaccard-Ähnlichkeit ihrer Top-Wörter gemittelt. 1.0 heißt:
        Jeder Lauf findet dieselben Themen.
        """
        return float(np.mean([self._agreement(a, b) for a, b in combinations(runs, 2)]))

    def _agreement(self, a: FittedTopics, b: FittedTopics) -> float:
        sets_a = [set(t) for t in a.top_terms(self._top_n)]
        sets_b = [set(t) for t in b.top_terms(self._top_n)]
        similarity = np.array([[len(x & y) / len(x | y) for y in sets_b] for x in sets_a])
        rows, cols = linear_sum_assignment(similarity, maximize=True)
        return float(similarity[rows, cols].mean())


def subsample(n_docs: int, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_docs, size=int(n_docs * fraction), replace=False))
