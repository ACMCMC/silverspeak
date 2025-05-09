#!/usr/bin/env python3
"""
SilverSpeak Command Line Interface

This module provides a command-line interface for the SilverSpeak library.
It allows users to apply homoglyph attacks and normalization to text from
the command line.

Example usage:
    # Apply random attack replacing 5% of characters
    python -m silverspeak attack --method random --percentage 0.05 --input input.txt --output output.txt

    # Apply greedy attack with same script constraint
    python -m silverspeak attack --method greedy --percentage 0.1 --same-script --input input.txt --output output.txt

    # Normalize text using dominant script strategy
    python -m silverspeak normalize --strategy dominant-script --input input.txt --output output.txt

Author: Aldan Creo (ACMC) <os@acmc.fyi>
Version: 1.0.0
License: See LICENSE file in the project root
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, TextIO, Union

from silverspeak.homoglyphs.greedy_attack import greedy_attack
from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.random_attack import random_attack
from silverspeak.homoglyphs.utils import NormalizationStrategies, TypesOfHomoglyphs


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the CLI.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="SilverSpeak: A tool for text normalization and homoglyph manipulation"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Attack command
    attack_parser = subparsers.add_parser("attack", help="Apply homoglyph attack to text")
    attack_parser.add_argument(
        "--method", choices=["random", "greedy"], default="random", help="Attack method to use (default: random)"
    )
    attack_parser.add_argument(
        "--percentage", type=float, default=0.05, help="Percentage of characters to replace (default: 0.05)"
    )
    attack_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    attack_parser.add_argument("--same-script", action="store_true", help="Only use homoglyphs from the same script")
    attack_parser.add_argument(
        "--same-block", action="store_true", help="Only use homoglyphs from the same Unicode block"
    )
    attack_parser.add_argument(
        "--homoglyph-types",
        type=str,
        default="identical,confusables",
        help="Comma-separated list of homoglyph types to use (identical, confusables, ocr, ocr_refined)",
    )

    # Normalize command
    normalize_parser = subparsers.add_parser("normalize", help="Normalize text by replacing homoglyphs")
    normalize_parser.add_argument(
        "--strategy",
        type=str,
        default="dominant-script",
        help="Normalization strategy (dominant-script, dominant-script-block, local-context, tokenization, language-model)",
    )

    # Common arguments
    for subparser in [attack_parser, normalize_parser]:
        subparser.add_argument("--input", type=str, default="-", help="Input file path (- for stdin, default: -)")
        subparser.add_argument("--output", type=str, default="-", help="Output file path (- for stdout, default: -)")

    # General arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def read_input(input_path: str) -> str:
    """
    Read input text from a file or stdin.

    Args:
        input_path: Path to the input file or '-' for stdin.

    Returns:
        The input text as a string.
    """
    if input_path == "-":
        return sys.stdin.read()
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()


def write_output(output_path: str, text: str) -> None:
    """
    Write output text to a file or stdout.

    Args:
        output_path: Path to the output file or '-' for stdout.
        text: Text to write.
    """
    if output_path == "-":
        sys.stdout.write(text)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)


def parse_homoglyph_types(types_str: str) -> List[TypesOfHomoglyphs]:
    """
    Parse homoglyph types from a comma-separated string.

    Args:
        types_str: Comma-separated string of homoglyph types.

    Returns:
        List of TypesOfHomoglyphs enum values.
    """
    type_map = {
        "identical": TypesOfHomoglyphs.IDENTICAL,
        "confusables": TypesOfHomoglyphs.CONFUSABLES,
        "ocr": TypesOfHomoglyphs.OCR,
        "ocr_refined": TypesOfHomoglyphs.OCR_REFINED,
    }

    types = []
    for type_name in types_str.split(","):
        type_name = type_name.strip().lower()
        if type_name in type_map:
            types.append(type_map[type_name])
        else:
            logging.warning(f"Unknown homoglyph type: {type_name}, ignoring")

    return types or [TypesOfHomoglyphs.IDENTICAL, TypesOfHomoglyphs.CONFUSABLES]


def parse_normalization_strategy(strategy: str) -> NormalizationStrategies:
    """
    Parse normalization strategy string to the format expected by normalize_text.
    
    Args:
        strategy: Strategy name from command line.
    
    Returns:
        NormalizationStrategies: Normalized strategy enum value.
    """
    strategy_map = {
        "dominant-script": NormalizationStrategies.DOMINANT_SCRIPT,
        "dominant-script-block": NormalizationStrategies.DOMINANT_SCRIPT_AND_BLOCK,
        "local-context": NormalizationStrategies.LOCAL_CONTEXT,
        "tokenization": NormalizationStrategies.TOKENIZATION,
        "language-model": NormalizationStrategies.LANGUAGE_MODEL
    }
    
    return strategy_map.get(strategy, NormalizationStrategies.DOMINANT_SCRIPT)


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Read input text
    try:
        input_text = read_input(args.input)
    except Exception as e:
        logger.error(f"Error reading input: {e}")
        sys.exit(1)

    # Process based on command
    if args.command == "attack":
        # Set random seed if provided
        if args.seed is not None:
            random.seed(args.seed)

        # Parse homoglyph types
        homoglyph_types = parse_homoglyph_types(args.homoglyph_types)

        # Apply attack
        try:
            if args.method == "random":
                result = random_attack(
                    input_text,
                    percentage=args.percentage,
                    random_seed=args.seed,
                    same_script=args.same_script,
                    same_block=args.same_block,
                    types_of_homoglyphs_to_use=homoglyph_types,
                )
            else:  # greedy
                result = greedy_attack(
                    input_text,
                    percentage=args.percentage,
                    same_script=args.same_script,
                    same_block=args.same_block,
                    types_of_homoglyphs_to_use=homoglyph_types,
                )

            logger.info(f"Applied {args.method} attack with {args.percentage:.1%} replacement")
        except Exception as e:
            logger.error(f"Error applying attack: {e}")
            sys.exit(1)

    elif args.command == "normalize":
        # Parse normalization strategy
        strategy = parse_normalization_strategy(args.strategy)

        # Apply normalization
        try:
            result = normalize_text(input_text, strategy=strategy)
            logger.info(f"Applied normalization with strategy: {args.strategy}")
        except Exception as e:
            logger.error(f"Error applying normalization: {e}")
            sys.exit(1)

    else:
        # No command specified
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)

    # Write output
    try:
        write_output(args.output, result)
    except Exception as e:
        logger.error(f"Error writing output: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
