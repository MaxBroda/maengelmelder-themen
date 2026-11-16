from maengelmelder.vectorize import CorpusVectorizer, rank_shift

DOCUMENTS = [
    "müll gehweg sperrmüll",
    "müll gehweg matratze",
    "müll container glasscherbe",
    "laterne defekt gehweg",
    "laterne defekt dunkel",
]


def test_both_methods_share_vocabulary_including_bigrams():
    vectorizer = CorpusVectorizer(min_df=2)

    bow = vectorizer.bag_of_words(DOCUMENTS)
    tfidf = vectorizer.tfidf(DOCUMENTS)

    assert list(bow.vocabulary) == list(tfidf.vocabulary)
    assert "laterne defekt" in bow.vocabulary
    assert bow.n_bigrams == 2


def test_tfidf_weights_rows_to_unit_length():
    tfidf = CorpusVectorizer(min_df=1).tfidf(DOCUMENTS)

    row_norms = (tfidf.matrix.multiply(tfidf.matrix)).sum(axis=1)

    assert all(abs(norm - 1.0) < 1e-9 for norm in row_norms.A1)


def test_rank_shift_reports_bow_minus_tfidf_rank():
    vectorizer = CorpusVectorizer(min_df=1, ngram_range=(1, 1))
    shift = rank_shift(vectorizer.bag_of_words(DOCUMENTS), vectorizer.tfidf(DOCUMENTS), within=20, n=3)

    assert (shift["Verschiebung"] == shift["Bag-of-Words"] - shift["TF-IDF"]).all()
