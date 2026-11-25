"""Trainiert die finalen Themenmodelle und schreibt Tabellen und Grafiken nach results/.

    python -m maengelmelder.analyze [--k 10]

- topics_nmf.csv, topics_lda.csv: Anteil, Top-Wörter und Beispielmeldungen je Thema
- crosstab_nmf.csv/.png: Verteilung der NMF-Themen je service_name
- subtopics.csv, subtopics_k.csv: Unterthemen je Kategorie (NMF) und die Kennzahlen zur Wahl ihres k
- results.xlsx: alle Tabellen zusätzlich als Blätter einer Excel-Datei
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maengelmelder.export import ResultWriter
from maengelmelder.select_k import KSelection
from maengelmelder.topics import FittedTopics, TopicEvaluator, lda_spec, nmf_spec
from maengelmelder.vectorize import CorpusVectorizer


def topic_table(topics: FittedTopics, descriptions: pd.Series, n_terms: int = 10, n_examples: int = 2) -> pd.DataFrame:
    dominant = topics.dominant_topic()
    rows = []
    for topic, terms in enumerate(topics.top_terms(n_terms)):
        strongest = np.argsort(topics.doc_topics[:, topic])[::-1][:n_examples]
        rows.append({
            "Thema": topic,
            "Anteil": round(float((dominant == topic).mean()), 3),
            "Top-Wörter": " ".join(terms),
            "Beispiele": " | ".join(_shorten(descriptions.iloc[i]) for i in strongest),
        })
    return pd.DataFrame(rows).sort_values("Anteil", ascending=False)


def _shorten(text: str, length: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= length else text[: length - 1] + "…"


def plot_crosstab(crosstab: pd.DataFrame, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.1 * crosstab.shape[1] + 4, 0.5 * crosstab.shape[0] + 1.5))
    ax.imshow(crosstab.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (row, col), value in np.ndenumerate(crosstab.values):
        if value >= 0.05:
            ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.5 else "#1a1a1a")
    ax.set_xticks(range(crosstab.shape[1]), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(crosstab.shape[0]), crosstab.index, fontsize=8)
    ax.set_title("NMF-Themen je Kategorie (Zeilenanteile)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


class SubtopicAnalysis:
    """Modelliert jede Kategorie einzeln, um Sammelthemen aufzuschlüsseln.

    k wird je Kategorie wie bei den Hauptthemen gewählt: unter den k mit
    Stabilität >= min_stability das mit der höchsten Kohärenz c_v. Erreicht kein
    k die Schwelle, gilt das stabilste k, und die Kategorie wird als nicht stabil
    markiert. Sonst würde die Kohärenz ein zufälliges, großes k auswählen.
    """

    def __init__(self, k_range: range = range(2, 9), min_docs: int = 500, min_stability: float = 0.75,
                 runs: int = 3) -> None:
        self._k_range = k_range
        self._min_docs = min_docs
        self._min_stability = min_stability
        self._runs = runs

    def run(self, corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        subtopics, metrics = [], []
        for category, group in corpus.groupby("service_name"):
            if len(group) < self._min_docs:
                continue
            documents = group["tokens"].tolist()
            spec = nmf_spec(CorpusVectorizer(ngram_range=(1, 1)).tfidf(documents))
            evaluator = TopicEvaluator([d.split() for d in documents], [None] * len(documents))
            selection = KSelection(evaluator, len(documents), self._runs)

            scores = pd.DataFrame([selection.evaluate(spec, k) for k in self._k_range]).drop(columns="nmi")
            scores.insert(0, "Kategorie", category)
            k, is_stable = self._choose(scores)
            scores["gewählt"] = scores["k"] == k
            metrics.append(scores)

            table = topic_table(spec.fit(k, seed=0), group["description"], n_terms=8, n_examples=1)
            table.insert(0, "Kategorie", category)
            table.insert(1, "stabil", is_stable)
            subtopics.append(table)
        return pd.concat(subtopics, ignore_index=True), pd.concat(metrics, ignore_index=True)

    def _choose(self, scores: pd.DataFrame) -> tuple[int, bool]:
        stable = scores[scores["stability"] >= self._min_stability]
        if stable.empty:
            return int(scores.loc[scores["stability"].idxmax(), "k"]), False
        return int(stable.loc[stable["c_v"].idxmax(), "k"]), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", type=Path, default=Path("data/processed/requests_clean.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    writer = ResultWriter(args.out)

    corpus = pd.read_csv(args.clean)
    documents = corpus["tokens"].tolist()
    vectorizer = CorpusVectorizer(ngram_range=(1, 1))
    nmf = nmf_spec(vectorizer.tfidf(documents)).fit(args.k, seed=0)
    lda = lda_spec(vectorizer.bag_of_words(documents)).fit(args.k, seed=0)

    for name, topics in [("nmf", nmf), ("lda", lda)]:
        table = topic_table(topics, corpus["description"])
        writer.write(table, f"topics_{name}")
        print(f"\n{name.upper()} (k={args.k})\n" + table[["Thema", "Anteil", "Top-Wörter"]].to_string(index=False))

    crosstab = pd.crosstab(corpus["service_name"], nmf.dominant_topic(), normalize="index")
    writer.write(crosstab, "crosstab_nmf", index=True)
    labels = [f"{t}: {' '.join(terms)}" for t, terms in enumerate(nmf.top_terms(3))]
    plot_crosstab(crosstab, labels, args.out / "crosstab_nmf.png")

    subtopics, metrics = SubtopicAnalysis().run(corpus)
    writer.write(subtopics, "subtopics")
    writer.write(metrics, "subtopics_k")
    print("\nUnterthemen je Kategorie\n" + subtopics[["Kategorie", "Thema", "Anteil", "Top-Wörter"]].to_string(index=False))


if __name__ == "__main__":
    main()
