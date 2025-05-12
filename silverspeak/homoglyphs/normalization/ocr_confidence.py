"""
OCR confidence-based normalization strategy for homoglyphs.

This module provides functions to normalize text containing homoglyphs by simulating
OCR processing and analyzing character recognition confidence. Characters with lower
confidence scores are candidates for homoglyph normalization.

Author: GitHub Copilot
"""

import logging
import os
from typing import Dict, List, Tuple, Set, Optional, Any, Union, TYPE_CHECKING
import numpy as np
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)

# Create data directory for models if it doesn't exist
try:
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
except Exception as e:
    logger.warning(f"Failed to create data directory: {e}")

# Try to import optional OCR dependencies
try:
    from PIL import Image, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    Image = None  # Define Image as None if import fails
    ImageDraw = None
    ImageFont = None
    logger.warning("PIL/Pillow not available, OCR confidence strategy will use simplified approach")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning(
        "Pytesseract not available, OCR confidence strategy will use simplified approach. "
        "To enable full OCR capabilities, install with: pip install pytesseract"
    )


# Try to use existing OCR confusion matrices from SilverSpeak's resources
try:
    # Load OCR character mappings from the project resources
    OCR_CHARS_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_chars_map.json")
    OCR_CHARS_REFINED_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_chars_refined_map.json")
    
    # Default confusion matrix path (in case we create one for storing)
    DEFAULT_CONFUSION_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "data", "ocr_confusion_matrix.json")
except Exception:
    # Fallback paths if there's an error determining the paths
    OCR_CHARS_MAP_PATH = "ocr_chars_map.json"
    OCR_CHARS_REFINED_MAP_PATH = "ocr_chars_refined_map.json"
    DEFAULT_CONFUSION_MATRIX_PATH = "ocr_confusion_matrix.json"

# Default fonts for rendering text
DEFAULT_FONTS = [
    "Arial", "Times New Roman", "Courier New", "Georgia", "Verdana", 
    "Helvetica", "Calibri", "Tahoma", "Trebuchet MS"
]


def load_confusion_matrix(path: str = None) -> Dict[str, Dict[str, float]]:
    """
    Load OCR character confusion matrix from a file or generate from existing OCR maps.
    
    Args:
        path: Path to the confusion matrix JSON file
        
    Returns:
        Dictionary mapping characters to their confusion probabilities
    """
    # Try to load custom confusion matrix if specified
    if path is not None:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load confusion matrix from {path}: {e}")
    
    # Try to load and convert SilverSpeak's OCR maps
    try:
        # First try the refined OCR map
        if os.path.exists(OCR_CHARS_REFINED_MAP_PATH):
            logger.info(f"Using SilverSpeak's refined OCR character map from {OCR_CHARS_REFINED_MAP_PATH}")
            with open(OCR_CHARS_REFINED_MAP_PATH, 'r', encoding='utf-8') as f:
                ocr_map = json.load(f)
            return _convert_ocr_map_to_confusion_matrix(ocr_map)
        
        # Try the regular OCR map
        if os.path.exists(OCR_CHARS_MAP_PATH):
            logger.info(f"Using SilverSpeak's OCR character map from {OCR_CHARS_MAP_PATH}")
            with open(OCR_CHARS_MAP_PATH, 'r', encoding='utf-8') as f:
                ocr_map = json.load(f)
            return _convert_ocr_map_to_confusion_matrix(ocr_map)
    except Exception as e:
        logger.error(f"Failed to load SilverSpeak OCR maps: {e}")
    
    # Return a simplified default confusion matrix focusing on common homoglyphs
    logger.warning("Using built-in default OCR confusion matrix")
    return {
        'l': {'I': 0.5, '1': 0.3, '|': 0.2},
        'I': {'l': 0.5, '1': 0.3, '|': 0.2},
        '1': {'l': 0.3, 'I': 0.3, '|': 0.1},
        '|': {'l': 0.2, 'I': 0.2, '1': 0.1},
        'O': {'0': 0.6, 'o': 0.4},
        '0': {'O': 0.6, 'o': 0.3},
        'o': {'0': 0.3, 'O': 0.4},
        'S': {'5': 0.3},
        '5': {'S': 0.3},
        'B': {'8': 0.2},
        '8': {'B': 0.2},
        'Z': {'2': 0.3},
        '2': {'Z': 0.3},
    }


def _convert_ocr_map_to_confusion_matrix(ocr_map: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Convert SilverSpeak's OCR character map to a confusion matrix.
    
    Args:
        ocr_map: SilverSpeak's OCR character map
        
    Returns:
        Confusion matrix mapping each character to its potential confusions
    """
    confusion_matrix = {}
    
    # For each character in the OCR map
    for char, confusables in ocr_map.items():
        if not confusables:
            continue
            
        # Create entry for this character
        confusion_matrix[char] = {}
        
        # Compute base probability (evenly distributed among confusables)
        base_prob = 1.0 / len(confusables)
        
        # Add each confusable with its probability
        for confusable in confusables:
            confusion_matrix[char][confusable] = base_prob
            
            # Also add the reverse mapping with a slightly lower probability
            if confusable not in confusion_matrix:
                confusion_matrix[confusable] = {}
            confusion_matrix[confusable][char] = base_prob * 0.8
    
    return confusion_matrix


class OCRConfidenceAnalyzer:
    """Analyzer that uses OCR confidence scores to detect and fix homoglyphs."""
    
    def __init__(
        self, 
        confusion_matrix: Dict[str, Dict[str, float]] = None,
        confidence_threshold: float = 0.7,
        use_tesseract: bool = True,
        fonts: List[str] = None,
        font_size: int = 24
    ):
        """
        Initialize OCR confidence analyzer.
        
        Args:
            confusion_matrix: Character confusion matrix for OCR
            confidence_threshold: Threshold below which characters are considered suspicious
            use_tesseract: Whether to use Tesseract OCR for confidence analysis
            fonts: List of fonts to render text with
            font_size: Font size for rendering
        """
        self.confusion_matrix = confusion_matrix or load_confusion_matrix()
        self.confidence_threshold = confidence_threshold
        self.use_tesseract = use_tesseract and TESSERACT_AVAILABLE
        self.fonts = fonts or DEFAULT_FONTS
        self.font_size = font_size
        
        # Check if we have the necessary dependencies
        self.can_render = PILLOW_AVAILABLE
    
    def analyze_text(self, text: str) -> List[Tuple[int, float, List[str]]]:
        """
        Analyze text and identify potential homoglyphs based on OCR confidence.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (position, confidence_score, [candidate_replacements])
        """
        suspicious_chars = []
        
        if self.can_render and self.use_tesseract:
            # Use actual OCR confidence scores
            suspicious_chars = self._analyze_with_tesseract(text)
        else:
            # Fall back to using the confusion matrix directly
            suspicious_chars = self._analyze_with_matrix(text)
            
        return suspicious_chars
    
    def _analyze_with_tesseract(self, text: str) -> List[Tuple[int, float, List[str]]]:
        """
        Analyze text using Tesseract OCR to get confidence scores.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (position, confidence_score, [candidate_replacements])
        """
        suspicious_chars = []
        
        try:
            # Create an image with the text
            img = self._render_text(text)
            
            # Run OCR with confidence data
            ocr_data = pytesseract.image_to_data(
                img, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume a single uniform block of text
            )
            
            # Parse OCR results and find low-confidence characters
            word_confidences = ocr_data['conf']
            words = ocr_data['text']
            
            current_pos = 0
            for i, word in enumerate(words):
                if not word:
                    continue
                    
                confidence = word_confidences[i]
                if confidence < self.confidence_threshold * 100:  # Tesseract uses 0-100 scale
                    # Find potential replacements for each character in this word
                    for char_idx, char in enumerate(word):
                        text_pos = current_pos + char_idx
                        if text_pos < len(text):
                            replacements = self._get_potential_replacements(char)
                            if replacements:
                                suspicious_chars.append((text_pos, confidence / 100, replacements))
                                
                current_pos += len(word) + 1  # +1 for space
                
        except Exception as e:
            logger.error(f"Error analyzing text with Tesseract: {e}")
            
        return suspicious_chars
    
    def _analyze_with_matrix(self, text: str) -> List[Tuple[int, float, List[str]]]:
        """
        Analyze text using only the confusion matrix.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (position, confidence_score, [candidate_replacements])
        """
        suspicious_chars = []
        
        for i, char in enumerate(text):
            replacements = self._get_potential_replacements(char)
            if replacements:
                # Use the highest confusion probability as a proxy for low confidence
                conf_scores = [self.confusion_matrix.get(char, {}).get(r, 0) for r in replacements]
                if conf_scores:
                    max_conf = max(conf_scores)
                    # If there's a high probability of confusion, flag as suspicious
                    if max_conf > (1 - self.confidence_threshold):
                        suspicious_chars.append((i, 1 - max_conf, replacements))
                        
        return suspicious_chars
    
    def _render_text(self, text: str) -> "Optional[Any]":
        """
        Render text as an image for OCR processing.
        
        Args:
            text: Text to render
            
        Returns:
            PIL Image object with rendered text, or None if rendering is not available
        """
        if not self.can_render:
            raise ImportError("PIL/Pillow is required for text rendering")
            
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
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
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
                self.confusion_matrix[char].keys(), 
                key=lambda x: self.confusion_matrix[char][x],
                reverse=True
            )
        return []


def normalize_with_ocr(
    text: str, 
    normalization_map: Dict[str, List[str]],
    char_positions: List[Tuple[int, float, List[str]]]
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
    
    return ''.join(result)


def apply_ocr_confidence_strategy(
    text: str,
    mapping: Dict[str, List[str]],
    confidence_threshold: float = 0.7,
    use_tesseract: bool = True,
    **kwargs
) -> str:
    """
    Apply OCR confidence-based normalization strategy to fix homoglyphs.
    
    Args:
        text: Text to normalize
        mapping: Homoglyph normalization map
        confidence_threshold: Threshold below which characters are considered suspicious
        use_tesseract: Whether to use Tesseract OCR for confidence analysis
        **kwargs: Additional arguments
        
    Returns:
        Normalized text with homoglyphs replaced
    """
    logger.info("Applying OCR confidence normalization strategy")
    
    try:
        # Create an OCR analyzer
        analyzer = OCRConfidenceAnalyzer(
            confidence_threshold=confidence_threshold,
            use_tesseract=use_tesseract
        )
        
        # Analyze text for suspicious characters
        suspicious_chars = analyzer.analyze_text(text)
        
        # Normalize text based on analysis
        normalized_text = normalize_with_ocr(text, mapping, suspicious_chars)
        
        logger.info("OCR confidence normalization completed")
        return normalized_text
        
    except Exception as e:
        logger.error(f"Error in OCR confidence normalization: {e}")
        logger.warning("Returning original text due to error")
        return text
