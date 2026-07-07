"""
SilverSpeak: homoglyph attacks and HKB-based normalization.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("silverspeak")
except (ImportError, ModuleNotFoundError):
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

from silverspeak.homoglyphs.attacks.greedy_attack import greedy_attack
from silverspeak.homoglyphs.attacks.random_attack import random_attack
from silverspeak.homoglyphs.attacks.targeted_attack import targeted_attack
from silverspeak.homoglyphs.benchmark import measure_clean_fpr, measure_round_trip, run_benchmark
from silverspeak.homoglyphs.fast_normalize import normalize_fast
from silverspeak.homoglyphs.hkb import HomoglyphKB, build_hkb
from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.normalize_result import NormalizeResult
from silverspeak.homoglyphs.utils import TypesOfHomoglyphs


def get_version() -> str:
    return __version__


__all__ = [
    "random_attack",
    "greedy_attack",
    "targeted_attack",
    "normalize_fast",
    "NormalizeResult",
    "run_benchmark",
    "measure_clean_fpr",
    "measure_round_trip",
    "HomoglyphReplacer",
    "TypesOfHomoglyphs",
    "HomoglyphKB",
    "build_hkb",
    "get_version",
    "__version__",
]
