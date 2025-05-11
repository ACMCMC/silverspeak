[![Acceptability Test Workflow](https://github.com/ACMCMC/silverspeak/actions/workflows/acceptability-test.yml/badge.svg)](https://github.com/ACMCMC/silverspeak/actions/workflows/acceptability-test.yml) 👈 This means that the library is able to always give correct results with at least one of its strategies on all test cases.

[![Full Test Workflow](https://github.com/ACMCMC/silverspeak/actions/workflows/full-test.yml/badge.svg)](https://github.com/ACMCMC/silverspeak/actions/workflows/full-test.yml) 👈 This should fail. It means that not all strategies work perfectly in every single test case, which is to be expected and means you need to choose your strategies wisely. Check [the docs](http://acmcmc.me/silverspeak/) for more context.

# SilverSpeak
This is a Python library to perform homoglyph-based attacks on text.

We also include the experiments supplementing the paper "SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs".

![SilverSpeak Logo](docs/source/_static/silverspeak_logo_editable.svg)

## Installation

### Basic Installation
You can install this package from PyPI by running:
```
pip install silverspeak
```

### Optional Dependencies

SilverSpeak provides optional dependencies for enhanced normalization strategies:

#### Spell Checking Dependencies
```
pip install "silverspeak[spell-check]"
```

#### Contextual Spell Checking
```
pip install "silverspeak[contextual-spell-check]"
```

#### N-gram Analysis
```
pip install "silverspeak[ngram-analysis]"
```

#### Graph-based Analysis
```
pip install "silverspeak[graph-analysis]"
```

#### OCR-based Analysis
```
pip install pytesseract pillow
```

#### All Optional Dependencies
```
pip install "silverspeak[spell-check,contextual-spell-check,ngram-analysis,graph-analysis]"
pip install pytesseract pillow
```

## Documentation

For full documentation, visit [the docs](http://acmcmc.me/silverspeak/).

### Using the Logo

The project includes an editable SVG logo (`silverspeak_logo_editable.svg`) that is used in the documentation. If you're contributing to the documentation:

1. The logo is automatically included in the documentation build
2. You can refer to it in RST files using: `.. image:: _static/silverspeak_logo_editable.svg`
3. To modify the logo, edit the SVG file directly using a vector graphics editor like Inkscape

## Usage Examples

### Basic Attack Example
```python
from silverspeak.homoglyphs.random_attack import random_attack

text = "Hello, world!"
attacked_text = random_attack(text, 0.1)
print(attacked_text)
```

### Normalization Strategies

SilverSpeak offers multiple strategies for normalizing text that contains homoglyphs:

#### Basic Normalization
```python
from silverspeak.homoglyphs import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Text with homoglyphs
text = "Tһis іs а tеst with ѕome һomoglурhs."

# Default strategy (dominant script)
normalized = normalize_text(text)
print(normalized)
```

#### Advanced Normalization Strategies

##### Spell Check Strategy
```python
from silverspeak.homoglyphs import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Requires: poetry install --with spell-check
normalized = normalize_text(
    "Tһis іs а tеst with ѕome һomoglурhs.",
    strategy=NormalizationStrategies.SPELL_CHECK,
    language="en"  # Optional, default is English
)
print(normalized)
```

##### Language Model Strategy
```python
from silverspeak.homoglyphs import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Uses BERT to predict masked tokens
normalized = normalize_text(
    "Tһis іs а tеst with ѕome һomoglурhs.",
    strategy=NormalizationStrategies.LANGUAGE_MODEL,
    word_level=True  # Use word-level masking (recommended)
)
print(normalized)
```

##### LLM Prompt Strategy
```python
from silverspeak.homoglyphs import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Uses LLMs to fix homoglyphs via prompting
normalized = normalize_text(
    "Tһis іs а tеst with ѕome һomoglурhs.",
    strategy=NormalizationStrategies.LLM_PROMPT,
    model_name="google/gemma-2-1b-it"  # Optional
)
print(normalized)
```

## Installation from source
First, you may want to work in a virtual environment. If you don't have one, you can create it by running:
```
python -m venv .venv
```

Then, activate it with:
```
source .venv/bin/activate
```

You can also use Conda, or any other tool of your preference.

The Python version used in this project is `3.11.0`.

Also, remember to install the requirements by running:
```
pip install -r requirements.txt
```

And finally, install this package by running:
```
pip install -e .
```

## Reproducing the experimental results from the paper
To reproduce the results, you'll need a free Hugging Face account. You can register for an account here: https://huggingface.co/

Then, you'll need to sign into your account using the CLI with a token that has `write` permissions (more information [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli)). To do that, just run:
```
huggingface-cli login
```

[note] When prompted "Add token as git credential?", you should answer "Yes".

Then, set the `MY_HUGGINGFACE_USER` environment variable to the username of the account you just registered on Hugging Face by running:
```
export MY_HUGGINGFACE_USER='your_username'
```

Then, you can run the `run_experiments.sh` script. This script will run the experiments for all the models and datasets.

Finally, run the following command to generate the plots and tables:
```
python experiments/visualization.py
```

You will also find two notebooks (`experiments/divergence_embeddings_attacks.ipynb` and `experiments/perplexity_tests.ipynb`), to reproduce some smaller parts of the paper.

## Datasets
We make our datasets, in versions with and without results, at the following URL: https://huggingface.co/silverspeak
Specifically, the datasets are provided in two versions, one without the results of the experiments and one including them. The datasets are named as follows:
- Datasets without results:
    - `silverspeak/cheat`
    - `silverspeak/essay`
    - `silverspeak/reuter`
    - `silverspeak/writing_prompts`
    - `silverspeak/realnewslike`
- Datasets with results:
    - `silverspeak/cheat_with_results`
    - `silverspeak/essay_with_results`
    - `silverspeak/reuter_with_results`
    - `silverspeak/writing_prompts_with_results`
    - `silverspeak/realnewslike_with_results`

## AI Disclaimer
We used AI code generation assitance from GitHub Copilot for this project. Nonetheless, the coding process has been essentially manual, with the AI code generator exclusively helping us to speed up the process.

## Reproducibility statement
We have tested the code in this repository on a NVIDIA A100 GPU, and have run the experiments twice, independently, to ensure the results are reproducible. We confirm that the results obtained were identical, and thus expect no variation in the results when running the code again. We manually set random seeds where necessary to ensure reproducibility.

## Side note: where does the name "SilverSpeak" come from?
The name SilverSpeak comes from the expression _"Hablar en plata"_ in Spanish. While a literal translation would be _"Speak in silver"_, it means _"Speak clearly"_. Therefore, some people would understand the underlying meaning, while those unfamiliar with the expression would likely misunderstand it.

Homoglyph-based attacks are an effective evasion technique since they change the meaning that detectors perceive, while maintaining the same appearance to a human observer. We think the idea can be a metaphor of the system getting _"lost in translation"_, especially considering that homoglyphs are frequently identical characters in different languages.

Hereby the rationale behind our choice of the name _SilverSpeak_ to refer to the family of homoglyph-based attacks that we use in our paper. The attacks play with the understood meaning of the text, depending on who is the observer, taking advantage of codification differences across alphabets.
