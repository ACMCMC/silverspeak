# Installation

Requires **Python 3.11+**.

## PyPI

```bash
pip install silverspeak
```

### Optional extras

```bash
pip install "silverspeak[spell-check]"
pip install "silverspeak[contextual-spell-check]"
pip install "silverspeak[ngram-analysis]"
pip install "silverspeak[graph-analysis]"
pip install "silverspeak[advanced]"
pip install "silverspeak[all]"
```

| Extra | Packages | Used for |
|-------|----------|----------|
| `spell-check` | symspellpy, pyspellchecker | SymSpell normalization strategy |
| `contextual-spell-check` | neuspell | Contextual spell-check strategy |
| `ngram-analysis` | nltk | N-gram frequency strategy |
| `graph-analysis` | networkx | Graph-based strategy |
| `advanced` | torch, transformers, pillow, doctr | OCR confidence, language model strategies |

The fast HKB pipeline needs only the base install (pandas, unicodedataplus).

## From source

```bash
git clone https://github.com/ACMCMC/silverspeak.git
cd silverspeak
pip install poetry
poetry install --with dev
```

Optional Poetry groups:

```bash
poetry install --with spell-check
poetry install --with advanced
poetry install --with ngram-analysis
poetry install --with graph-analysis
```

## Verify

```bash
python -c "from silverspeak import normalize_fast, HomoglyphKB; print('ok')"
```
