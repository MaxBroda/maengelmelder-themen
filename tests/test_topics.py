import numpy as np
import pytest

from maengelmelder.topics import FittedTopics, TopicEvaluator, nmf_spec, subsample
from maengelmelder.vectorize import CorpusVectorizer

VOCABULARY = np.array(["müll", "sperrmüll", "matratze", "laterne", "defekt", "dunkel"])


def topics(*weights):
    components = np.array(weights, dtype=float)
    return FittedTopics(components, np.eye(len(weights)), VOCABULARY)


@pytest.fixture
def evaluator():
    texts = [["müll", "sperrmüll", "matratze"], ["laterne", "defekt", "dunkel"]] * 5
    return TopicEvaluator(texts, ["Müll", "Laterne"] * 5, top_n=3)


def test_top_terms_are_sorted_by_weight():
    fitted = topics([3, 2, 1, 0, 0, 0], [0, 0, 0, 1, 3, 2])

    assert fitted.top_terms(2) == [["müll", "sperrmüll"], ["defekt", "dunkel"]]


def test_identical_runs_in_different_order_are_fully_stable(evaluator):
    a = topics([3, 2, 1, 0, 0, 0], [0, 0, 0, 3, 2, 1])
    b = topics([0, 0, 0, 3, 2, 1], [3, 2, 1, 0, 0, 0])

    assert evaluator.stability([a, b]) == pytest.approx(1.0)


def test_runs_with_different_topics_are_less_stable(evaluator):
    a = topics([3, 2, 1, 0, 0, 0], [0, 0, 0, 3, 2, 1])
    b = topics([3, 0, 0, 2, 1, 0], [0, 3, 2, 0, 0, 1])

    assert evaluator.stability([a, b]) < 1.0


def test_diversity_drops_when_topics_repeat_words():
    distinct = topics([3, 2, 1, 0, 0, 0], [0, 0, 0, 3, 2, 1])
    repeated = topics([3, 2, 1, 0, 0, 0], [3, 2, 1, 0, 0, 0])

    assert TopicEvaluator.diversity(distinct, top_n=3) == 1.0
    assert TopicEvaluator.diversity(repeated, top_n=3) == 0.5


def test_nmi_ignores_uncategorised_documents():
    texts = [["müll"], ["laterne"], ["müll"], ["laterne"]]
    evaluator = TopicEvaluator(texts, ["Müll", "Laterne", None, None])
    fitted = FittedTopics(np.ones((2, 6)), np.array([[1, 0], [0, 1], [0, 1], [1, 0]]), VOCABULARY)

    assert evaluator.nmi(fitted) == pytest.approx(1.0)


def test_nmf_spec_fits_on_subsample_and_scores_all_documents():
    documents = ["müll sperrmüll matratze", "laterne defekt dunkel"] * 10
    spec = nmf_spec(CorpusVectorizer(min_df=1, ngram_range=(1, 1)).tfidf(documents))

    fitted = spec.fit(k=2, seed=0, rows=subsample(len(documents), 0.5, seed=0))

    assert fitted.doc_topics.shape == (20, 2)
    assert sorted(map(set, fitted.top_terms(3)), key=sorted) == sorted(
        [{"müll", "sperrmüll", "matratze"}, {"laterne", "defekt", "dunkel"}], key=sorted
    )
