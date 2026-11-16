"""Bewertet LDA und NMF für verschiedene Themenzahlen k und speichert Kennzahlen und Grafik.

    python -m maengelmelder.select_k [--k-min 4] [--k-max 30] [--k-step 2] [--runs 3]

Jeder Lauf trainiert auf einer anderen Zufallsstichprobe (80 %) der Meldungen.
Kohärenz, Diversity und NMI werden über die Läufe gemittelt, die Stabilität
vergleicht die Läufe untereinander.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maengelmelder.topics import TopicEvaluator, TopicModelSpec, lda_spec, nmf_spec, subsample
from maengelmelder.vectorize import CorpusVectorizer

UNCATEGORISED = "Sonstiges"
METRICS = {
    "c_v": "Kohärenz c_v",
    "c_npmi": "Kohärenz c_npmi",
    "stability": "Stabilität",
    "diversity": "Topic Diversity",
    "nmi": "NMI zu service_name",
}


class KSelection:
    def __init__(self, evaluator: TopicEvaluator, n_docs: int, runs: int = 3, fraction: float = 0.8) -> None:
        self._evaluator = evaluator
        self._samples = [subsample(n_docs, fraction, seed) for seed in range(runs)]

    def evaluate(self, spec: TopicModelSpec, k: int) -> dict[str, float]:
        fits = [spec.fit(k, seed, rows) for seed, rows in enumerate(self._samples)]
        ev = self._evaluator
        return {
            "model": spec.name,
            "k": k,
            "c_v": np.mean([ev.coherence(f, "c_v") for f in fits]),
            "c_npmi": np.mean([ev.coherence(f, "c_npmi") for f in fits]),
            "stability": ev.stability(fits),
            "diversity": np.mean([ev.diversity(f) for f in fits]),
            "nmi": np.mean([ev.nmi(f) for f in fits]),
        }


def plot(results: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.6), sharex=True)
    for ax, (column, title) in zip(axes, METRICS.items()):
        for model, group in results.groupby("model"):
            ax.plot(group["k"], group[column], marker="o", markersize=3, label=model)
        ax.set_title(title)
        ax.set_xlabel("Anzahl Themen k")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", type=Path, default=Path("data/processed/requests_clean.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--k-step", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    corpus = pd.read_csv(args.clean)
    documents = corpus["tokens"].tolist()
    categories = [None if c == UNCATEGORISED else c for c in corpus["service_name"]]

    # Nur Unigramme: Mit Bigrammen füllen Varianten wie "laterne defekt" /
    # "defekt laterne" die Top-Wörter und treiben die Kohärenz künstlich hoch.
    vectorizer = CorpusVectorizer(ngram_range=(1, 1))
    specs = [lda_spec(vectorizer.bag_of_words(documents)), nmf_spec(vectorizer.tfidf(documents))]

    selection = KSelection(TopicEvaluator([d.split() for d in documents], categories), len(documents), args.runs)
    rows = []
    for k in range(args.k_min, args.k_max + 1, args.k_step):
        for spec in specs:
            started = time.perf_counter()
            rows.append(selection.evaluate(spec, k))
            print(f"{spec.name} k={k:>2} ({time.perf_counter() - started:.0f}s) "
                  + " ".join(f"{m}={rows[-1][m]:.3f}" for m in METRICS), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.out / "k_selection.csv", index=False)
    plot(results, args.out / "k_selection.png")
    print(f"-> {args.out / 'k_selection.csv'}, {args.out / 'k_selection.png'}")


if __name__ == "__main__":
    main()
