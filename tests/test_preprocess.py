import pandas as pd
import pytest

from maengelmelder.preprocess import Preprocessor, TextCleaner, default_lemmatizer


@pytest.fixture(scope="module")
def lemmatizer():
    return default_lemmatizer()


@pytest.mark.parametrize(
    "text, removed",
    [
        ("Siehe https://example.org/foto.jpg bitte", "https://"),
        ("Müll an der Keplerstraße 11 abgelegt", "Keplerstraße"),
        ("Glas auf der Harthaer Straße 56 verteilt", "Harthaer"),
        ("Bank am Rosa-Luxemburg-Platz beschmiert", "Rosa-Luxemburg-Platz"),
        ("Laterne defekt. 01099 Dresden", "01099"),
        ("Bitte beseitigen. Mit freundlichen Grüßen", "Grüßen"),
        ("Container voll, vielen Dank!", "Dank"),
    ],
)
def test_cleaner_removes_noise(text, removed):
    assert removed not in TextCleaner().clean(text)


@pytest.mark.parametrize("text", ["Scherben auf dem Radweg", "Spielplatz voller Müll", "Straße stark verschmutzt"])
def test_cleaner_keeps_content_words(text):
    assert TextCleaner().clean(text) == text


def test_lemmatizer_returns_lowercase_content_lemmas(lemmatizer):
    [tokens] = lemmatizer.lemmas(["Bitte die kaputten Laternen reparieren, siehe Bild"])

    assert "laterne" in tokens
    assert "reparieren" in tokens
    assert not {"bitte", "siehe", "bild", "die"} & set(tokens)


def test_preprocessor_drops_duplicates_and_short_texts(lemmatizer):
    raw = pd.DataFrame({
        "service_request_id": [1, 2, 3],
        "service_name": ["Sonstiges"] * 3,
        "requested_datetime": ["2024-01-01"] * 3,
        "description": [
            "Kaputte Laterne am Spielplatz flackert ständig",
            "Kaputte Laterne am Spielplatz flackert ständig",
            "Bitte entfernen, danke!",
        ],
    })

    result = Preprocessor(TextCleaner(), lemmatizer).run(raw)

    assert result["service_request_id"].tolist() == [1]
