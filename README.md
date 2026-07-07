[![Acceptability Test Workflow](https://github.com/ACMCMC/silverspeak/actions/workflows/acceptability-test.yml/badge.svg)](https://github.com/ACMCMC/silverspeak/actions/workflows/acceptability-test.yml)

[![Linting](https://github.com/ACMCMC/silverspeak/actions/workflows/linting.yml/badge.svg)](https://github.com/ACMCMC/silverspeak/actions/workflows/linting.yml)

[![Contributors Welcome](https://img.shields.io/badge/contributors-welcome-brightgreen.svg?style=flat)](https://github.com/ACMCMC/silverspeak/blob/main/README.md#contributing)

# SilverSpeak
This is a Python library to perform homoglyph-based attacks on text.

![SilverSpeak Logo](docs/assets/silverspeak_logo_editable.svg)

## Installation

### Basic Installation
```
pip install silverspeak
```

## Documentation

For full documentation, visit [https://acmcmc.github.io/silverspeak/](https://acmcmc.github.io/silverspeak/).

## Contributing
**Contributions are very welcome!** SilverSpeak is still a work in progress, and while we're working hard to finish it, we'd greatly appreciate any help from the community.

Here are some ways you can contribute:
- Implementing new homoglyph attack strategies
- Improving documentation and examples
- Writing tests to ensure reliability
- Reporting bugs or suggesting features via GitHub issues

To contribute, please feel free to fork the repository, make your changes, and submit a pull request. If you're unsure about a contribution or have questions, you can also open an issue for discussion.

## Usage Examples

### Basic Attack Example
```python
from silverspeak import random_attack

text = "Hello, world!"
attacked_text = random_attack(text=text, percentage=0.1, random_seed=2242)
print(attacked_text)
```

## Why SilverSpeak?

SilverSpeak offers significant advantages over existing homoglyph replacement libraries. Here's how it compares:

### Comprehensive Comparison with Popular Homoglyph Libraries

| Feature | [SilverSpeak](https://github.com/ACMCMC/silverspeak) | [confusable_homoglyphs](https://github.com/vhf/confusable_homoglyphs) | [life4/homoglyphs](https://github.com/life4/homoglyphs) | [codebox/homoglyph](https://github.com/codebox/homoglyph) | [decancer](https://github.com/null8626/decancer) | [squatm3](https://github.com/david3107/squatm3) | [unisec](https://github.com/Acceis/unisec) | [homoglyphic](https://github.com/AlanRVA/Homoglyphic) | [glyphcheck](https://github.com/NebulousLabs/glyphcheck) |
|---------|-------------|----------------------|-----------------|-------------------|----------|--------|-------|-------------|------------|
| **Homoglyph Sources** | Unicode standard + OCR-based confusables | Unicode standard only | Unicode standard only | Unicode standard + partial additions | Unicode standard only | Unicode standard only | Unicode standard only | Unicode standard only | Limited set |
| **Context-Awareness** | ✅ Local context matching for natural replacements | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness | ❌ No context awareness |
| **Normalization** | HKB fast pipeline with ambiguity metadata | Basic detection only | Basic ASCII conversion | Simple search capabilities | Text cleanup only | Limited detection | Limited detection | Simple detection | Limited detection |
| **Attack Reversal** | ✅ Advanced attack normalization | ❌ Detection only | ❌ Limited to ASCII conversion | ❌ No reversal | ✅ Basic cleanup | ❌ Limited | ❌ Limited | ⚠️ Partial | ❌ No |
| **Targeted Attack Support** | ✅ Sophisticated targeting | ❌ No | ❌ No | ❌ No | ❌ No | ⚠️ Limited | ❌ No | ❌ No | ❌ No |
| **Python Compatibility** | ✅ Modern Python | ⚠️ Legacy Python support | ✅ Python 2 & 3 | ❌ JavaScript/Java focus | ✅ Deno/JS focus | ✅ Python 3 | ✅ Ruby | ✅ C# | ✅ Go |
| **Language Model Integration** | Not required at runtime (HKB graph) | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **Performance Optimization** | ✅ Optimized for both attack and defense | ⚠️ Basic detection | ⚠️ Basic conversion | ⚠️ Search-focused | ⚠️ Basic cleanup | ⚠️ Domain focus | ⚠️ Basic security | ⚠️ Basic detection | ⚠️ Basic detection |
| **Documentation** | ✅ Comprehensive with examples | ⚠️ Basic | ⚠️ Basic | ⚠️ Basic | ⚠️ Basic | ⚠️ Limited | ⚠️ Basic | ⚠️ Basic | ⚠️ Limited |
| **OCR Character Support** | ✅ Extensive | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |

### Feature Details of Major Alternatives

| Library | Primary Focus | Strengths | Limitations |
|---------|--------------|-----------|------------|
| **[confusable_homoglyphs](https://github.com/vhf/confusable_homoglyphs)** | Character confusion detection | Simple API, basic detection | No attack reversal, limited to Unicode confusables |
| **[life4/homoglyphs](https://github.com/life4/homoglyphs)** | ASCII conversion | Simple conversion to ASCII | Limited homoglyph set, no context awareness, basic functionality |
| **[codebox/homoglyph](https://github.com/codebox/homoglyph)** | Search capabilities | Good for basic searches | JavaScript/Java only, no reversal capabilities, limited features |
| **[decancer](https://github.com/null8626/decancer)** | Text cleanup | Removes "cancerous" text formatting | Limited to cleanup, no sophisticated attack capabilities |
| **[squatm3](https://github.com/david3107/squatm3)** | Domain squatting detection | Good for cybersecurity use cases | Narrow focus on domains, limited features |
| **[unisec](https://github.com/Acceis/unisec)** | Unicode security toolkit | General Unicode security | Not focused on homoglyphs specifically, limited features |
| **[homoglyphic](https://github.com/AlanRVA/Homoglyphic)** | Simple detection | Easy integration with C# | Basic detection only, no advanced features |
| **[glyphcheck](https://github.com/NebulousLabs/glyphcheck)** | Source code homoglyph detection | Simple API | Very limited feature set |

### Key Advantages of SilverSpeak

1. **Advanced Context Matching**: SilverSpeak analyzes the surrounding text to select homoglyphs that match the local context in terms of Unicode properties (script, block, category, bidirectional, east_asian_width), ensuring visually coherent replacements. This feature is entirely absent in all other libraries.

2. **OCR-Based Confusables**: Beyond standard Unicode confusables, SilverSpeak includes characters that appear similar in OCR systems, making it substantially more comprehensive than libraries relying solely on Unicode definitions. Our OCR-based approach captures visual similarities that aren't defined in the Unicode standard.

3. **HKB normalization**: ranked homoglyph graph for fast, deterministic normalization with an audit trail.

4. **Targeted Attack Capabilities**: Unlike competing libraries that focus solely on detection or simple random replacements, SilverSpeak provides sophisticated targeted attack strategies that can evade specific AI text detectors while maintaining human readability.

5. **Comprehensive Testing**: benchmark harness with clean-text FPR gate and round-trip recovery metrics.

6. **Active Research**: developed as part of published academic research ["SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs"](https://aclanthology.org/2025.genaidetect-1.1/).

### Use Cases Where SilverSpeak Excels

While other libraries focus primarily on detection or simple character substitution, SilverSpeak shines in the following scenarios:

1. **AI Text Detection Evasion**: SilverSpeak was specifically designed to enable text to evade AI-generated text detectors while maintaining human readability, a capability not found in any other homoglyph library.

2. **Security Research**: For researchers examining how homoglyph attacks can affect modern NLP systems and security tools, SilverSpeak provides the most comprehensive toolkit available.

3. **Text Normalization**: HKB-based fast pipeline with structured `NormalizeResult` output.

4. **Language Processing Pipeline Protection**: SilverSpeak can be integrated into text processing pipelines to sanitize input text before it reaches sensitive language processing components.

5. **Content Moderation**: Unlike simple libraries that only detect basic homoglyphs, SilverSpeak's context-aware approach helps identify and normalize sophisticated homoglyph-based attempts to bypass content filters.

6. **Academic Research**: research-backed homoglyph attack and normalization toolkit.

7. **Cross-Language Text Processing**: With its advanced Unicode property analysis, SilverSpeak handles mixed-script text more effectively than alternatives.

Most other libraries were designed for simple use cases like domain spoofing detection or basic homoglyph identification, making them insufficient for advanced applications like those listed above.

# Reproducing the experimental results from the research paper
This library was developed as part of the paper ["SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs"](https://aclanthology.org/2025.genaidetect-1.1/). A part of the code in this repository is used to reproduce the results of the paper (check out the `v1.0.0` tag).

# Side note: where does the name "SilverSpeak" come from?

The name SilverSpeak comes from the expression _"Hablar en plata"_ in Spanish. While a literal translation would be _"Speak in silver"_, it means _"Speak clearly"_. Therefore, some people would understand the underlying meaning, while those unfamiliar with the expression would likely misunderstand it.

Homoglyph-based attacks are an effective evasion technique since they change the meaning that detectors perceive, while maintaining the same appearance to a human observer. We think the idea can be a metaphor of the system getting _"lost in translation"_, especially considering that homoglyphs are frequently identical characters in different languages.

Hereby the rationale behind our choice of the name _SilverSpeak_ to refer to the family of homoglyph-based attacks that we use in our paper. The attacks play with the understood meaning of the text, depending on who is the observer, taking advantage of codification differences across alphabets.
