# Usage

## Attacks

```python
from silverspeak import random_attack, greedy_attack, targeted_attack

text = "Hello world"
attacked = random_attack(text=text, percentage=0.1, random_seed=2242)
attacked = greedy_attack(text=text, percentage=0.1, random_seed=2242)
attacked = targeted_attack(text=text, percentage=0.1, random_seed=2242)
```

CLI:

```bash
python -m silverspeak attack --method random --percentage 0.1 --seed 2242
python -m silverspeak attack --method greedy --percentage 0.1 --same-script
python -m silverspeak attack --method targeted --percentage 0.2 --seed 2242
```

Attack options: `--same-script`, `--same-block`, `--homoglyph-types identical,confusables,ocr,ocr_refined`.

## Normalization (fast pipeline)

Default path. Uses the bundled HKB graph; no path argument needed in the CLI.

```python
from silverspeak import normalize_fast
from silverspeak.homoglyphs.hkb.kb import DEFAULT_HKB_PATH

result = normalize_fast(
    text="hеllо wоrld",
    graph_path=DEFAULT_HKB_PATH,
    min_score=0.0,
    score_margin=0.0,
)
print(result.text)
print(result.chars_changed)   # list of (pos, old, new) tuples
print(result.ambiguous)       # unresolved ties, returned as metadata
```

`NormalizeResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Normalized output |
| `chars_changed` | `list` | Positions and substitutions applied |
| `ambiguous` | `list` | Chars where top candidates tied within `score_margin` |

CLI (reads stdin by default):

```bash
echo "hеllо wоrld" | python -m silverspeak normalize
echo "hеllо wоrld" | python -m silverspeak normalize --report
echo "hеllо wоrld" | python -m silverspeak normalize --min-score 0.5 --score-margin 0.1
```

## Normalization (legacy strategies)

Ten heuristic strategies via `HomoglyphReplacer`. Some require optional extras (see [Installation](installation.md)).

```python
from silverspeak import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

normalized = normalize_text(
    text="Hеllo wоrld",
    strategy=NormalizationStrategies.LOCAL_CONTEXT,
)
```

CLI:

```bash
python -m silverspeak normalize --pipeline legacy --strategy local-context
python -m silverspeak normalize --pipeline legacy --strategy dominant-script
```

Available legacy strategies: `dominant-script`, `dominant-script-block`, `local-context`, `tokenization`, `language-model`, plus ngram, OCR, and graph-based (need extras).

See [Normalization Strategies](normalization_strategies.md) for details on each legacy strategy.

## Benchmarking

```python
from silverspeak import run_benchmark, random_attack, normalize_fast
from silverspeak.homoglyphs.hkb.kb import DEFAULT_HKB_PATH

def normalize_fn(text):
    return normalize_fast(
        text=text,
        graph_path=DEFAULT_HKB_PATH,
        min_score=0.0,
        score_margin=0.0,
    ).text

report = run_benchmark(
    clean_samples=["hello", "Привет", "你好世界"],
    round_trip_samples=["Hello world", "café résumé"],
    attack_fn=lambda text: random_attack(text=text, percentage=0.1, random_seed=2242),
    normalize_fn=normalize_fn,
)
print(f"clean FPR: {report.clean_fpr}")          # should be 0.0
print(f"round trips: {len(report.round_trips)}")
for trip in report.round_trips:
    print(trip.original, "->", trip.char_accuracy)
```

See [HKB](hkb.md) for graph details.
