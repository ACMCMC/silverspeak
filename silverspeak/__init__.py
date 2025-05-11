"""
SilverSpeak: A professional library for text normalization and homoglyph detection/replacement.

This library provides tools for detecting and normalizing homoglyphs (characters
that look similar but have different Unicode code points), which can be used for
text normalization, security applications, and adversarial text generation.

Main components:
- random_attack: Generate text with random homoglyph replacements
- greedy_attack: Generate text with strategically chosen homoglyph replacements
- normalize_text: Normalize text by replacing homoglyphs with standard characters
- HomoglyphReplacer: Core class for homoglyph replacement operations

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

try:
    from importlib.metadata import version as _version
    __version__ = _version("silverspeak")
except (ImportError, ModuleNotFoundError):
    # If package is not installed in a way that metadata is available
    # fallback to reading from pyproject.toml
    import os
    import tomli
    
    _package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _pyproject_path = os.path.join(_package_root, "pyproject.toml")
    
    if os.path.isfile(_pyproject_path):
        with open(_pyproject_path, "rb") as f:
            _pyproject_data = tomli.load(f)
            __version__ = _pyproject_data.get("tool", {}).get("poetry", {}).get("version", "unknown")
    else:
        __version__ = "unknown"

from silverspeak.homoglyphs.greedy_attack import greedy_attack
from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.random_attack import random_attack
from silverspeak.homoglyphs.utils import NormalizationStrategies, TypesOfHomoglyphs


def get_version() -> str:
    """
    Get the current version of the SilverSpeak package.
    
    Returns:
        str: The current version string
    """
    return __version__


__all__ = [
    "random_attack",
    "greedy_attack",
    "normalize_text",
    "HomoglyphReplacer",
    "TypesOfHomoglyphs",
    "NormalizationStrategies",
    "get_version",
    "__version__"
]
