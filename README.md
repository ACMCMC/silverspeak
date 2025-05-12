[![Acceptability Test Workflow](https://github.com/ACMCMC/silverspeak/actions/workflows/acceptability-test.yml/badge.svg)](https://github.com/ACMCMC/silverspeak/actions/workflows/acceptability-test.yml) 👈 This means that the library is able to always give correct results with at least one of its strategies on all test cases.

[![Full Test Workflow](https://github.com/ACMCMC/silverspeak/actions/workflows/full-test.yml/badge.svg)](https://github.com/ACMCMC/silverspeak/actions/workflows/full-test.yml) 👈 This should fail. It means that not all strategies work perfectly in every single test case, which is to be expected and means you need to choose your strategies wisely. Check [the docs](http://acmcmc.me/silverspeak/) for more context.

[![Linting](https://github.com/ACMCMC/silverspeak/actions/workflows/linting.yml/badge.svg)](https://github.com/ACMCMC/silverspeak/actions/workflows/linting.yml) 👈 This means that the code is well formatted and follows the PEP8 style guide. It also means that the code is type-checked using mypy, and that the code is linted using flake8.

# SilverSpeak
This is a Python library to perform homoglyph-based attacks on text.

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

## Contributing
**Contributions are very welcome!** SilverSpeak is still a work in progress, and while we're working hard to finish it, we'd greatly appreciate any help from the community.

Here are some ways you can contribute:
- Implementing new homoglyph attack strategies
- Improving existing normalization techniques
- Enhancing documentation and examples
- Writing tests to ensure reliability
- Reporting bugs or suggesting features via GitHub issues

To contribute, please feel free to fork the repository, make your changes, and submit a pull request. If you're unsure about a contribution or have questions, you can also open an issue for discussion.

## Usage Examples

### Basic Attack Example
```python
from silverspeak.homoglyphs.random_attack import random_attack

text = "Hello, world!"
attacked_text = random_attack(text, 0.1)
print(attacked_text)
```

# Reproducing the experimental results from the research paper
This library was developed as part of the paper ["SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs"](https://aclanthology.org/2025.genaidetect-1.1/). A part of the code in this repository is used to reproduce the results of the paper (check out the `v1.0.0` tag).

# Side note: where does the name "SilverSpeak" come from?

The name SilverSpeak comes from the expression _"Hablar en plata"_ in Spanish. While a literal translation would be _"Speak in silver"_, it means _"Speak clearly"_. Therefore, some people would understand the underlying meaning, while those unfamiliar with the expression would likely misunderstand it.

Homoglyph-based attacks are an effective evasion technique since they change the meaning that detectors perceive, while maintaining the same appearance to a human observer. We think the idea can be a metaphor of the system getting _"lost in translation"_, especially considering that homoglyphs are frequently identical characters in different languages.

Hereby the rationale behind our choice of the name _SilverSpeak_ to refer to the family of homoglyph-based attacks that we use in our paper. The attacks play with the understood meaning of the text, depending on who is the observer, taking advantage of codification differences across alphabets.
