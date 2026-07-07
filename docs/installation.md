# Installation

Requires **Python 3.11+**.

## PyPI

```bash
pip install silverspeak
```

The HKB fast pipeline needs only the base install (pandas, unicodedataplus).

## From source

```bash
git clone https://github.com/ACMCMC/silverspeak.git
cd silverspeak
pip install poetry
poetry install --with dev
```

## Verify

```bash
python -c "from silverspeak import normalize_fast, HomoglyphKB; print('ok')"
```
