import pytest
from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.utils import (
    NormalizationStrategies
)

POSSIBLE_STRATEGIES = [
    NormalizationStrategies.DOMINANT_SCRIPT,
    NormalizationStrategies.DOMINANT_SCRIPT_AND_BLOCK,
    NormalizationStrategies.CONTEXT_AWARE,
]

TEXTS_TO_TEST = [
    ("hеllо wоrld", "hello world"),  # Mixed scripts
    (
        "heIIo world",
        "hello world",
    ),  # Mixes I and l, which are both Latin but not the same case (Unicode category is different: Ll vs. Lu)
    ("расе", "расе"),  # All Cyrillic
    ("言語是溝通的橋樑", "訁語是溝通的橋樑"),  # Chinese sentence with 言 and 訁
    (
        "言行一致是成功的關鍵",
        "訁行一致是成功的關鍵",
    ),  # Another Chinese sentence with 言 and 訁
    ("", ""),  # Empty string
    ("hello world", "hello world"),  # No homoglyphs
]


@pytest.fixture
def replacer():
    """
    Fixture to set up a HomoglyphReplacer instance for testing.
    """
    return HomoglyphReplacer()


def test_normalize_unsupported_strategy(replacer):
    """
    Test the `normalize` method with unsupported strategies.
    """
    with pytest.raises(NotImplementedError):
        replacer.normalize("test", strategy="unsupported_strategy")


@pytest.mark.parametrize(
    "input_text, expected_output",
    TEXTS_TO_TEST,
)
@pytest.mark.parametrize("strategy", POSSIBLE_STRATEGIES)
def test_normalize(replacer, input_text, expected_output, strategy):
    """
    Test the `normalize` method with various inputs and strategies.
    """
    result = replacer.normalize(input_text, strategy=strategy)
    assert result == expected_output


# Test that succeeds if at least one of the strategies works in all cases
@pytest.mark.parametrize(
    "input_text, expected_output",
    TEXTS_TO_TEST,
)
def test_normalize_with_fallback(replacer, input_text, expected_output):
    """
    Test the `normalize` method with various inputs and strategies. Unlike the previous test,
    this one has a looser assertion: it only checks that at least one of the strategies works.

    That means that *there is at least one strategy that works for all texts* - not necessarily the same one, though.
    """
    # Try each strategy and check if at least one works
    for strategy in POSSIBLE_STRATEGIES:
        result = replacer.normalize(input_text, strategy=strategy)
        if result == expected_output:
            break
    else:
        pytest.fail(f"None of the strategies worked for input: {input_text}")
