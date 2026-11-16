import numpy as np
import pandas as pd

from maengelmelder.analyze import SubtopicAnalysis, topic_table
from maengelmelder.topics import FittedTopics


def scores(stability, c_v):
    return pd.DataFrame({"k": range(2, 2 + len(stability)), "stability": stability, "c_v": c_v})


def test_choose_takes_most_coherent_among_stable_k():
    k, stable = SubtopicAnalysis()._choose(scores([0.9, 0.8, 0.5], [0.3, 0.4, 0.9]))

    assert (k, stable) == (3, True)


def test_choose_falls_back_to_most_stable_k_and_flags_it():
    k, stable = SubtopicAnalysis()._choose(scores([0.6, 0.7, 0.4], [0.3, 0.2, 0.9]))

    assert (k, stable) == (3, False)


def test_topic_table_reports_share_and_strongest_example():
    topics = FittedTopics(
        components=np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]),
        doc_topics=np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]),
        vocabulary=np.array(["müll", "gehweg", "laterne"]),
    )
    descriptions = pd.Series(["Müll auf dem Gehweg", "Laterne defekt", "Sperrmüll"])

    table = topic_table(topics, descriptions, n_terms=2, n_examples=1).set_index("Thema")

    assert table.loc[0, "Anteil"] == round(2 / 3, 3)
    assert table.loc[0, "Top-Wörter"] == "müll gehweg"
    assert table.loc[1, "Beispiele"] == "Laterne defekt"
