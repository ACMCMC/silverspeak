import pytest
from silverspeak.homoglyphs.random_attack import random_attack
from silverspeak.homoglyphs.normalize import normalize_text as normalize
from silverspeak.homoglyphs.utils import NormalizationStrategies, TypesOfHomoglyphs
import math
import logging


POSSIBLE_STRATEGIES = [
    NormalizationStrategies.DOMINANT_SCRIPT,
    NormalizationStrategies.DOMINANT_SCRIPT_AND_BLOCK,
    NormalizationStrategies.LOCAL_CONTEXT,
    NormalizationStrategies.TOKENIZATION,
    NormalizationStrategies.LANGUAGE_MODEL,
    NormalizationStrategies.LLM_PROMPT,
    NormalizationStrategies.SPELL_CHECK,
    NormalizationStrategies.NGRAM,
    NormalizationStrategies.OCR_CONFIDENCE,
    NormalizationStrategies.GRAPH_BASED,
]

MIXED_SCRIPT_PHRASES = [
    'En una tarde tranquila, los niños juegan en el parque mientras los adultos disfrutan de una conversación bajo la sombra de los árboles, y uno de ellos menciona: "这是一个测试句子。"',
    'On a peaceful summer evening, children play in the park while adults chat under the trees, and someone says: "Это тестовое предложение。"',
    "Par une soirée d'été paisible, les enfants jouent dans le parc tandis que les adultes discutent sous les arbres, et l'un d'eux ajoute : \"これはテスト文です。\"",
    'An einem ruhigen Sommerabend spielen Kinder im Park, während Erwachsene unter den Bäumen plaudern, und jemand bemerkt: "هذا جملة اختبارية。"',
    'Numa tarde de verão tranquila, as crianças brincam no parque enquanto os adultos conversam à sombra das árvores, e alguém comenta: "Αυτή είναι μια δοκιμαστική πρόταση。"',
]

SINGLE_SCRIPT_PHRASES = [
    "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains, painting the sky in hues of orange and purple.",  # English
    "Dans un petit village niché au cœur des montagnes, les habitants se réunissent chaque soir pour partager des histoires autour d'un feu de camp.",  # French
    "In einem kleinen Dorf, das tief im Wald versteckt liegt, erzählen sich die Menschen Legenden von längst vergangenen Zeiten.",  # German
    "En una tarde tranquila de verano, los niños juegan en el parque mientras los adultos disfrutan de una conversación bajo la sombra de los árboles.",  # Spanish
    "夕暮れ時、静かな村では子供たちが遊び、大人たちは木陰でお茶を飲みながら談笑しています。",  # Japanese
    "在一个宁静的夏日黄昏，孩子们在公园里玩耍，大人们在树荫下聊天，享受着微风的轻拂。",  # Chinese
    "В тихий летний вечер дети играют на улице, а взрослые сидят на скамейках, обсуждая последние новости.",  # Russian
    "조용한 여름 저녁, 아이들은 공원에서 놀고 어른들은 나무 그늘 아래에서 대화를 나누며 시간을 보냅니다.",  # Korean
    "एक शांत गर्मी की शाम में, बच्चे पार्क में खेलते हैं और वयस्क पेड़ों की छाया में बैठकर बातचीत का आनंद लेते हैं।",  # Hindi
    "في مساء صيفي هادئ، يلعب الأطفال في الحديقة بينما يجلس الكبار تحت ظلال الأشجار يتبادلون الأحاديث.",  # Arabic
    "Numa tarde de verão tranquila, as crianças brincam no parque enquanto os adultos conversam à sombra das árvores.",  # Portuguese
    "Σε ένα ήσυχο καλοκαιρινό απόγευμα, τα παιδιά παίζουν στο πάρκο ενώ οι ενήλικες συζητούν κάτω από τη σκιά των δέντρων.",  # Greek
    "Sıcak bir yaz akşamında, çocuklar parkta oynarken yetişkinler ağaçların gölgesinde sohbet ediyor.",  # Turkish
    "U mirnoj ljetnoj večeri, djeca se igraju u parku dok odrasli razgovaraju u hladu drveća.",  # Serbian
]


@pytest.mark.parametrize(
    "original_text",
    [
        "This is a test sentence with some unique characters: ä, ö, ü.",
        "",
        "!@#$%^&*()_+-=[]{}|;':,./<>?",
    ],
)
@pytest.mark.parametrize("strategy", POSSIBLE_STRATEGIES)
def test_random_attack_and_normalize(original_text, strategy):
    """
    Test the random_attack and normalize functions with various inputs and strategies.
    """
    # Apply random attack
    attacked_text = random_attack(original_text, percentage=0.2, random_seed=42)

    # Normalize the attacked text using the given strategy
    normalized_text = normalize(attacked_text, strategy=strategy)

    # Assert that the normalized text matches the original
    assert normalized_text == original_text


@pytest.mark.parametrize(
    "phrase",
    SINGLE_SCRIPT_PHRASES,
)
@pytest.mark.parametrize("strategy", POSSIBLE_STRATEGIES)
def test_random_attack_and_normalize_single_script(phrase, strategy):
    """
    Test the random_attack and normalize functions with single-script phrases and strategies.
    """
    # Apply random attack
    attacked_text = random_attack(
        phrase,
        percentage=0.2,
        random_seed=42,
    )

    # Normalize the attacked text using the given strategy
    normalized_text = normalize(text=attacked_text, strategy=strategy)

    # Assert that the normalized text matches the original
    assert normalized_text == phrase


@pytest.mark.parametrize(
    "phrase",
    MIXED_SCRIPT_PHRASES,
)
@pytest.mark.parametrize("strategy", POSSIBLE_STRATEGIES)
def test_random_attack_and_normalize_mixed_scripts(phrase, strategy):
    """
    Test the random_attack and normalize functions with mixed-script phrases and strategies.
    """
    # Apply random attack
    attacked_text = random_attack(
        phrase,
        percentage=0.2,
        random_seed=42,
    )

    # Normalize the attacked text using the given strategy
    normalized_text = normalize(text=attacked_text, strategy=strategy)

    # Assert that the normalized text matches the original
    assert normalized_text == phrase


@pytest.mark.parametrize(
    "phrase",
    MIXED_SCRIPT_PHRASES + SINGLE_SCRIPT_PHRASES,
)
def test_random_attack_and_normalize_with_tolerance_and_fallback(
    phrase, tolerance=0.05
):
    """
    Test the random_attack and normalize functions with mixed-script phrases,
    allowing for a tolerance of up to 5% mismatches and ensuring at least one strategy works.
    """
    # Apply random attack
    attacked_text = random_attack(
        phrase,
        percentage=0.2,
        random_seed=42,
    )

    # Calculate the tolerance threshold
    max_mismatches = int(len(phrase) * tolerance)

    # Check if at least one strategy works
    for strategy in POSSIBLE_STRATEGIES:
        normalized_text = normalize(text=attacked_text, strategy=strategy)
        mismatches = len(set(normalized_text) ^ set(phrase))
        if mismatches <= max_mismatches:
            logging.info(
                f"Strategy {strategy} succeeded with {mismatches} mismatches (within tolerance) for input: {phrase}"
            )
            return  # At least one strategy succeeded

    # If no strategy succeeded within tolerance, fail the test
    pytest.fail(
        f"None of the strategies worked within the tolerance for input: \"{phrase}\" and attacked text: \"{attacked_text}\""
    )
