"""
Homoglyphs attack strategies package.

This package provides various attack strategies for replacing text characters with homoglyphs
(characters that look visually similar but have different Unicode code points).

Available attack strategies:
- Random attack (replaces random characters)
- Greedy attack (maximizes the number of replacements)
- Targeted attack (focuses on specific patterns or characters)

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

from .greedy_attack import greedy_attack
from .random_attack import random_attack
from .targeted_attack import targeted_attack

__all__ = [
    "random_attack",
    "greedy_attack",
    "targeted_attack",
]
