"""
OCR confidence-based normalization strategy for homoglyphs.

This module provides functions to normalize text containing homoglyphs by simulating
OCR processing and analyzing character recognition confidence. Characters with lower
confidence scores are candidates for homoglyph normalization.

Author: GitHub Copilot
"""

import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)

# Lazy loading flags and cached imports
_PIL_AVAILABLE = None
_DOCTR_AVAILABLE = None
_PIL_MODULES = None
_DOCTR_MODULES = None
_DATA_DIR_CREATED = False

def _ensure_data_dir():
    """Ensure data directory exists (lazy creation)."""
    global _DATA_DIR_CREATED
    if not _DATA_DIR_CREATED:
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
            _DATA_DIR_CREATED = True
        except Exception as e:
            logger.warning(f"Failed to create data directory: {e}")

def _check_pil_availability():
    """Check if PIL is available (lazy check)."""
    global _PIL_AVAILABLE, _PIL_MODULES
    if _PIL_AVAILABLE is None:
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            _PIL_MODULES = {
                'Image': Image,
                'ImageDraw': ImageDraw, 
                'ImageFont': ImageFont,
                'np': np
            }
            _PIL_AVAILABLE = True
            logger.debug("PIL/Pillow available for OCR confidence analysis")
        except ImportError as e:
            _PIL_AVAILABLE = False
            logger.debug(f"PIL/Pillow not available: {e}")
    return _PIL_AVAILABLE

def _check_doctr_availability():
    """Check if DocTR is available (lazy check)."""
    global _DOCTR_AVAILABLE, _DOCTR_MODULES
    if _DOCTR_AVAILABLE is None:
        if not _check_pil_availability():
            _DOCTR_AVAILABLE = False
            return False
            
        try:
            from doctr.models import ocr_predictor
            from doctr.models.preprocessor import PreProcessor
            _DOCTR_MODULES = {
                'ocr_predictor': ocr_predictor,
                'PreProcessor': PreProcessor
            }
            _DOCTR_AVAILABLE = True
            logger.debug("DocTR OCR library available")
        except (ImportError, OSError) as e:
            _DOCTR_AVAILABLE = False
            logger.debug(f"DocTR OCR not available: {e}")
    return _DOCTR_AVAILABLE

def _get_ocr_paths():
    """Get OCR file paths (lazy path resolution)."""
    try:
        # Load OCR character mappings from the project resources
        ocr_chars_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_chars_map.json")
        ocr_chars_refined_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_chars_refined_map.json")
        default_confusion_matrix_path = os.path.join(os.path.dirname(__file__), "data", "ocr_confusion_matrix.json")
        return ocr_chars_map_path, ocr_chars_refined_map_path, default_confusion_matrix_path
    except Exception:
        # Fallback paths if there's an error determining the paths
        return "ocr_chars_map.json", "ocr_chars_refined_map.json", "ocr_confusion_matrix.json"

# Default fonts for rendering text
DEFAULT_FONTS = [
    "Arial",
    "Times New Roman",
    "Courier New",
    "Georgia",
    "Verdana",
    "Helvetica",
    "Calibri",
    "Tahoma",
    "Trebuchet MS",
]


def load_confusion_matrix(path: str = None) -> Dict[str, Dict[str, float]]:
    """
    Load OCR character confusion matrix from a file or use built-in defaults.

    Args:
        path: Path to the confusion matrix JSON file

    Returns:
        Dictionary mapping characters to their confusion probabilities
    """
    # Try to load custom confusion matrix if specified
    if path is not None and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load confusion matrix from {path}: {e}")

    # Get OCR paths lazily
    ocr_chars_map_path, ocr_chars_refined_map_path, default_confusion_matrix_path = _get_ocr_paths()
    
    # Try to load SilverSpeak's OCR maps
    for ocr_path in [ocr_chars_refined_map_path, ocr_chars_map_path]:
        if os.path.exists(ocr_path):
            try:
                logger.info(f"Using SilverSpeak's OCR character map from {ocr_path}")
                with open(ocr_path, "r", encoding="utf-8") as f:
                    ocr_map = json.load(f)

                # Convert map to confusion matrix
                confusion_matrix = {}
                for char, confusables in ocr_map.items():
                    if not confusables:
                        continue

                    confusion_matrix[char] = {}
                    base_prob = 1.0 / len(confusables)

                    for confusable in confusables:
                        confusion_matrix[char][confusable] = base_prob

                        if confusable not in confusion_matrix:
                            confusion_matrix[confusable] = {}
                        confusion_matrix[confusable][char] = base_prob * 0.8

                return confusion_matrix
            except Exception as e:
                logger.error(f"Failed to load OCR map from {ocr_path}: {e}")

    # Return a simplified default confusion matrix focusing on common homoglyphs
    logger.warning("Using built-in default OCR confusion matrix")
    return {
        "l": {"I": 0.5, "1": 0.3, "|": 0.2},
        "I": {"l": 0.5, "1": 0.3, "|": 0.2},
        "1": {"l": 0.3, "I": 0.3, "|": 0.1},
        "|": {"l": 0.2, "I": 0.2, "1": 0.1},
        "O": {"0": 0.6, "o": 0.4},
        "0": {"O": 0.6, "o": 0.3},
        "o": {"0": 0.3, "O": 0.4},
        "S": {"5": 0.3},
        "5": {"S": 0.3},
        "B": {"8": 0.2},
        "8": {"B": 0.2},
        "Z": {"2": 0.3},
        "2": {"Z": 0.3},
    }


class OCRConfidenceAnalyzer:
    """Analyzer that uses OCR confidence scores to detect and fix homoglyphs."""

    def __init__(
        self,
        confusion_matrix: Dict[str, Dict[str, float]] = None,
        confidence_threshold: float = 0.7,
        fonts: List[str] = None,
        font_size: int = 24,
    ):
        """
        Initialize OCR confidence analyzer.

        Args:
            confusion_matrix: Character confusion matrix for OCR
            confidence_threshold: Threshold below which characters are considered suspicious
            fonts: List of fonts to render text with
            font_size: Font size for rendering
        """
        self.confusion_matrix = confusion_matrix or load_confusion_matrix()
        self.confidence_threshold = confidence_threshold
        self.fonts = fonts or DEFAULT_FONTS
        self.font_size = font_size
        self.ocr_model = None  # Will be lazily initialized
        
    def _ensure_ocr_model(self):
        """Lazily initialize the DocTR OCR model."""
        if self.ocr_model is None and _check_doctr_availability():
            try:
                logger.info("Loading DocTR OCR model...")
                self.ocr_model = _DOCTR_MODULES['ocr_predictor'](pretrained=True)
                logger.info("DocTR OCR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DocTR OCR model: {e}")
                logger.warning("Will fall back to using confusion matrix only")
        
    def analyze_text(self, text: str) -> List[Tuple[int, float, List[str]]]:
        """
        Analyze text and identify potential homoglyphs based on OCR confidence.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (position, confidence_score, [candidate_replacements])
        """
        suspicious_chars = []

        # Check if DocTR is available and initialize model
        if _check_doctr_availability():
            self._ensure_ocr_model()
            
            if self.ocr_model is not None:
                try:
                    # Create an image with the text
                    img = self._render_text(text)
                    
                    # Convert PIL Image to numpy array for doctr
                    img_np = _PIL_MODULES['np'].array(img)
                    
                    # Use preprocessor directly instead of DocumentFile
                    preprocessor = _DOCTR_MODULES['PreProcessor'](output_size=(img.height, img.width), batch_size=1)
                    processed_img = preprocessor([img_np])
                    
                    # Process the image with doctr
                    result = self.ocr_model(processed_img)
                    
                    # Extract text and confidence scores from doctr result
                    # DocTR results are structured as pages -> blocks -> lines -> words
                    current_pos = 0
                    
                    # Iterate through pages (there should be only one)
                    for page in result.pages:
                        # Iterate through blocks of text
                        for block in page.blocks:
                            # Iterate through lines in each block
                            for line in block.lines:
                                line_text = ""
                                
                                # Iterate through words in each line
                                for word in line.words:
                                    word_text = word.value
                                    word_conf = word.confidence
                                    
                                    # Check if confidence is below threshold
                                    if word_conf < self.confidence_threshold:
                                        # Analyze each character in the word
                                        for char_idx, char in enumerate(word_text):
                                            text_pos = current_pos + char_idx
                                            if text_pos < len(text):
                                                replacements = self._get_potential_replacements(char)
                                                if replacements:
                                                    suspicious_chars.append((text_pos, float(word_conf), replacements))
                                    
                                    # Update line text
                                    line_text += word_text + " "
                                    
                                # Move position counter forward - align with original text
                                current_pos += len(line_text.rstrip())

                    return suspicious_chars

                except Exception as e:
                    logger.error(f"Error analyzing text with DocTR: {e}")
                    logger.exception(e)
                    # Fall back to using confusion matrix directly

        # Fall back to using confusion matrix directly
        for i, char in enumerate(text):
            if char in self.confusion_matrix:
                replacements = self._get_potential_replacements(char)
                if replacements:
                    # Use a fixed medium confidence score
                    suspicious_chars.append((i, 0.6, replacements))

        return suspicious_chars

    def _render_text(self, text: str):
        """
        Render text as an image for OCR processing.

        Args:
            text: Text to render

        Returns:
            PIL Image object with rendered text
        """
        # Ensure PIL is available
        if not _check_pil_availability():
            raise RuntimeError("PIL/Pillow is required for text rendering but not available")
            
        Image = _PIL_MODULES['Image']
        ImageDraw = _PIL_MODULES['ImageDraw'] 
        ImageFont = _PIL_MODULES['ImageFont']
        
        # Try to find an available font
        font = None
        for font_name in self.fonts:
            try:
                font = ImageFont.truetype(font_name, self.font_size)
                break
            except (OSError, IOError):
                continue

        if font is None:
            # Fall back to default font
            font = ImageFont.load_default()

        # Calculate image size based on text
        # This is a rough estimate
        width = max(1, len(text) * self.font_size // 2)
        height = self.font_size * 2

        # Create image with white background
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw text in black
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)

        return img

    def _get_potential_replacements(self, char: str) -> List[str]:
        """
        Get potential character replacements from confusion matrix.

        Args:
            char: Character to find replacements for

        Returns:
            List of potential replacement characters
        """
        if char in self.confusion_matrix:
            # Sort by confusion probability (highest first)
            return sorted(
                self.confusion_matrix[char].keys(), key=lambda x: self.confusion_matrix[char][x], reverse=True
            )
        return []


def normalize_with_ocr(
    text: str, normalization_map: Dict[str, List[str]], char_positions: List[Tuple[int, float, List[str]]]
) -> str:
    """
    Normalize text using identified suspicious character positions.

    Args:
        text: Original text
        normalization_map: Mapping from homoglyphs to standard characters
        char_positions: List of suspicious character positions with confidence scores

    Returns:
        Normalized text
    """
    # Sort positions in reverse order to avoid index shifting when replacing
    char_positions.sort(key=lambda x: x[0], reverse=True)

    result = list(text)  # Convert to list for easier character replacement

    # Build reverse map for faster lookup
    reverse_map = {}
    for homoglyph, standards in normalization_map.items():
        for std in standards:
            if std not in reverse_map:
                reverse_map[std] = set()
            reverse_map[std].add(homoglyph)

    # Process each suspicious character
    for pos, confidence, candidates in char_positions:
        if pos >= len(result):
            continue

        current_char = result[pos]

        # Check if this is a known homoglyph
        for std_char, homoglyphs in reverse_map.items():
            if current_char in homoglyphs:
                # It's a homoglyph, replace with standard character
                result[pos] = std_char
                break

        # If character wasn't replaced but is in candidates list, use most probable replacement
        if result[pos] == current_char:
            for candidate in candidates:
                if candidate in reverse_map and current_char in reverse_map[candidate]:
                    result[pos] = candidate
                    break

    return "".join(result)


def apply_ocr_confidence_strategy(
    text: str, mapping: Dict[str, List[str]], confidence_threshold: float = 0.7, **kwargs
) -> str:
    """
    Apply OCR confidence-based normalization strategy to fix homoglyphs.

    Args:
        text: Text to normalize
        mapping: Homoglyph normalization map
        confidence_threshold: Threshold below which characters are considered suspicious
        **kwargs: Additional arguments

    Returns:
        Normalized text with homoglyphs replaced
    """
    if _check_doctr_availability():
        logger.info("Applying OCR confidence normalization strategy using DocTR")
    else:
        logger.info("Applying OCR-based normalization using confusion matrix (DocTR not fully available)")

    try:
        # Create an OCR analyzer
        analyzer = OCRConfidenceAnalyzer(confidence_threshold=confidence_threshold)

        # Analyze text for suspicious characters
        suspicious_chars = analyzer.analyze_text(text)
        
        # Log how many suspicious characters were found
        logger.info(f"Found {len(suspicious_chars)} suspicious characters")
        
        # Normalize text based on analysis
        normalized_text = normalize_with_ocr(text, mapping, suspicious_chars)
        
        # Count changes made
        changes = sum(1 for a, b in zip(text, normalized_text) if a != b)
        logger.info(f"Normalization completed with {changes} character(s) modified")

        return normalized_text

    except Exception as e:
        logger.error(f"Error in OCR confidence normalization: {e}")
        logger.warning("Returning original text due to error")
        return text
