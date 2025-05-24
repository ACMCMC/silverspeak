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

## Why SilverSpeak?

SilverSpeak offers significant advantages over existing homoglyph replacement libraries. Here's how it compares:

### Comprehensive Comparison with Popular Homoglyph Libraries

| Feature | SilverSpeak | confusable_homoglyphs | life4/homoglyphs | codebox/homoglyph | decancer | squatm3 | unisec | homoglyphic | glyphcheck |
|---------|-------------|----------------------|-----------------|-------------------|----------|--------|-------|-------------|------------|
| **Homoglyph Sources** | Unicode standard + OCR-based confusables | Unicode standard only | Unicode standard only | Unicode standard + partial additions | Unicode standard only | Unicode standard only | Unicode standard only | Unicode standard only | Limited set |
| **Context-Awareness** | ✅ Local context matching for natural replacements | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness |
| **Normalization Strategies** | Multiple strategies (dominant script, context-based, language model, etc.) | Basic detection only | Basic ASCII conversion | Simple search capabilities | Text cleanup only | Limited detection | Limited detection | Simple detection | Limited detection |
| **Attack Reversal** | ✅ Advanced attack normalization | ❌ Detection only | ❌ Limited to ASCII conversion | ❌ No reversal | ✅ Basic cleanup | ❌ Limited | ❌ Limited | ⚠️ Partial | ❌ No |
| **Targeted Attack Support** | ✅ Sophisticated targeting | ❌ No | ❌ No | ❌ No | ❌ No | ⚠️ Limited | ❌ No | ❌ No | ❌ No |
| **Active Maintenance** | ✅ Actively maintained | ⚠️ Minimal maintenance | ⚠️ Minimal | ⚠️ Minimal | ⚠️ Minimal | ❌ Abandoned | ⚠️ Minimal | ⚠️ Minimal | ❌ Abandoned |
| **Python Compatibility** | ✅ Modern Python | ⚠️ Legacy Python support | ✅ Python 2 & 3 | ❌ JavaScript/Java focus | ✅ Deno/JS focus | ✅ Python 3 | ✅ Ruby | ✅ C# | ✅ Go |
| **Language Model Integration** | ✅ Advanced | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **Performance Optimization** | ✅ Optimized for both attack and defense | ⚠️ Basic detection | ⚠️ Basic conversion | ⚠️ Search-focused | ⚠️ Basic cleanup | ⚠️ Domain focus | ⚠️ Basic security | ⚠️ Basic detection | ⚠️ Basic detection |
| **Documentation** | ✅ Comprehensive with examples | ⚠️ Basic | ⚠️ Basic | ⚠️ Basic | ⚠️ Basic | ⚠️ Limited | ⚠️ Basic | ⚠️ Basic | ⚠️ Limited |
| **OCR Character Support** | ✅ Extensive | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |
| **Research-Backed** | ✅ Published research | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |

### Feature Details of Major Alternatives

| Library | Primary Focus | Strengths | Limitations |
|---------|--------------|-----------|------------|
| **confusable_homoglyphs** | Character confusion detection | Simple API, basic detection | No attack reversal, limited to Unicode confusables, minimal maintenance |
| **life4/homoglyphs** | ASCII conversion | Simple conversion to ASCII | Limited homoglyph set, no context awareness, basic functionality |
| **codebox/homoglyph** | Search capabilities | Good for basic searches | JavaScript/Java only, no reversal capabilities, limited features |
| **decancer** | Text cleanup | Removes "cancerous" text formatting | Limited to cleanup, no sophisticated attack capabilities |
| **squatm3** | Domain squatting detection | Good for cybersecurity use cases | Narrow focus on domains, abandoned, limited features |
| **unisec** | Unicode security toolkit | General Unicode security | Not focused on homoglyphs specifically, limited features |
| **homoglyphic** | Simple detection | Easy integration with C# | Basic detection only, no advanced features |
| **glyphcheck** | Domain name checking | Simple API | Very limited feature set, abandoned |

### Key Advantages of SilverSpeak

1. **Advanced Context Matching**: SilverSpeak analyzes the surrounding text to select homoglyphs that match the local context in terms of Unicode properties (script, block, category, bidirectional, east_asian_width), ensuring visually coherent replacements. This feature is entirely absent in all other libraries.

2. **OCR-Based Confusables**: Beyond standard Unicode confusables, SilverSpeak includes characters that appear similar in OCR systems, making it substantially more comprehensive than libraries relying solely on Unicode definitions. Our OCR-based approach captures visual similarities that aren't defined in the Unicode standard.

3. **Multiple Normalization Strategies**: Offers various approaches to reverting homoglyph attacks:
   - Dominant script and block detection
   - Local context-based normalization
   - Tokenization-aware normalization
   - Language model-based normalization
   - Spell checking normalization
   - LLM prompt-based normalization
   
   Most other libraries either provide no normalization features or offer only basic detection/conversion without sophisticated reversal capabilities.

4. **Targeted Attack Capabilities**: Unlike competing libraries that focus solely on detection or simple random replacements, SilverSpeak provides sophisticated targeted attack strategies that can evade specific AI text detectors while maintaining human readability.

5. **Language Model Integration**: Built-in support for utilizing language models to improve normalization quality, a feature unavailable in any other homoglyph library.

6. **Comprehensive Testing**: Extensively tested against various attack scenarios, including tests against modern AI-generated text detectors, ensuring reliability in real-world applications.

7. **Active Research**: Developed as part of published academic research ["SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs"](https://aclanthology.org/2025.genaidetect-1.1/), with ongoing improvements based on new findings. None of the other libraries are backed by peer-reviewed research.

### Use Cases Where SilverSpeak Excels

While other libraries focus primarily on detection or simple character substitution, SilverSpeak shines in the following scenarios:

1. **AI Text Detection Evasion**: SilverSpeak was specifically designed to enable text to evade AI-generated text detectors while maintaining human readability, a capability not found in any other homoglyph library.

2. **Security Research**: For researchers examining how homoglyph attacks can affect modern NLP systems and security tools, SilverSpeak provides the most comprehensive toolkit available.

3. **Text Normalization**: When you need to normalize text containing homoglyphs back to standard characters, SilverSpeak offers multiple strategies with varying levels of sophistication to handle even the most complex cases.

4. **Language Processing Pipeline Protection**: SilverSpeak can be integrated into text processing pipelines to sanitize input text before it reaches sensitive language processing components.

5. **Content Moderation**: Unlike simple libraries that only detect basic homoglyphs, SilverSpeak's context-aware approach helps identify and normalize sophisticated homoglyph-based attempts to bypass content filters.

6. **Academic Research**: The research-backed approach and multiple normalization strategies make SilverSpeak ideal for academic studies in linguistic security and text processing.

7. **Cross-Language Text Processing**: With its advanced Unicode property analysis, SilverSpeak handles mixed-script text more effectively than alternatives.

Most other libraries were designed for simple use cases like domain spoofing detection or basic homoglyph identification, making them insufficient for advanced applications like those listed above.

# Reproducing the experimental results from the research paper
This library was developed as part of the paper ["SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs"](https://aclanthology.org/2025.genaidetect-1.1/). A part of the code in this repository is used to reproduce the results of the paper (check out the `v1.0.0` tag).

# Side note: where does the name "SilverSpeak" come from?

The name SilverSpeak comes from the expression _"Hablar en plata"_ in Spanish. While a literal translation would be _"Speak in silver"_, it means _"Speak clearly"_. Therefore, some people would understand the underlying meaning, while those unfamiliar with the expression would likely misunderstand it.

Homoglyph-based attacks are an effective evasion technique since they change the meaning that detectors perceive, while maintaining the same appearance to a human observer. We think the idea can be a metaphor of the system getting _"lost in translation"_, especially considering that homoglyphs are frequently identical characters in different languages.

Hereby the rationale behind our choice of the name _SilverSpeak_ to refer to the family of homoglyph-based attacks that we use in our paper. The attacks play with the understood meaning of the text, depending on who is the observer, taking advantage of codification differences across alphabets.
