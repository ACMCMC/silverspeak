# Homoglyph Knowledge Base (HKB)

SilverSpeak v3 stores homoglyph relationships in a prebuilt **Homoglyph Knowledge Base** shipped as `silverspeak/homoglyphs/hkb_data/graph.json.gz`.

Each character maps to a ranked list of neighbor edges with `dst`, `score`, and `source` tags. Collisions from the old flat-JSON merge are resolved by ranked edges instead of last-write-wins.

## Sources

| Source | Origin | Coverage |
|--------|--------|----------|
| `identical` | Latin phishing / identical cross-script pairs | Latin-centric |
| `confusables` | Unicode TR39 confusables | IDN-oriented |
| `ocr_refined` | OCR/ViT similarity | CJK-focused |
| `visual` | [confusable-vision](https://github.com/paultendo/confusable-vision) SSIM | Cross-script visual neighbors |

Current graph stats (v2): ~12k edges, ~7k chars, ~1.4k visual pairs, ~85 KB compressed.

## Fast normalization

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
print(result.chars_changed)
print(result.ambiguous)   # never replaced with U+FFFD; returned as metadata
```

Pipeline steps:

1. Strip invisible / format characters
2. NFKC normalization
3. Detect dominant script
4. Replace chars whose script differs from dominant using HKB canonical candidates

CLI:

```bash
echo "hеllо wоrld" | python -m silverspeak normalize
echo "hеllо wоrld" | python -m silverspeak normalize --report
```

## Query the HKB

```python
from silverspeak import HomoglyphKB
from silverspeak.homoglyphs.hkb.kb import DEFAULT_HKB_PATH

kb = HomoglyphKB(graph_path=DEFAULT_HKB_PATH)
kb.homoglyphs_of(char="a", sources=["visual"], min_score=0.7)
kb.canonical_candidates(char="а", script="Latin", min_score=0.0)
kb.is_ambiguous(char="а")
kb.coverage_report(text="hello Привет")
```

## Rebuild the HKB

From a source checkout:

```bash
PYTHONPATH=. python3 scripts/fetch_visual_data.py
PYTHONPATH=. python3 scripts/build_hkb.py
```

`fetch_visual_data.py` downloads confusable-vision discovery JSON into `hkb_data/`. `build_hkb.py` merges all source maps and writes `graph.json.gz`.

## Design notes

- TR39 skeleton data is for detection/clustering only, not direct output
- Visual neighbors are precomputed at build time; no FAISS or torch at runtime
- Ambiguity is returned in `NormalizeResult.ambiguous`, not inlined as replacement chars
- Only characters where `script(char) != dominant_script` are candidates for replacement
