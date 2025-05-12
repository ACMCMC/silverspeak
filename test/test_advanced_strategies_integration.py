"""
Integration tests for advanced normalization strategies in SilverSpeak.

These tests validate the end-to-end functionality of the new normalization strategies:
1. N-gram frequency-based strategy
2. OCR confidence-based strategy
3. Graph-based network strategy

Tests use the high-level normalize_text API to ensure proper integration
in the SilverSpeak library.
"""

import pytest
import unittest.mock

from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies


class TestAdvancedStrategiesIntegration:
    """Integration tests for advanced normalization strategies."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text with homoglyphs for testing."""
        return "Tһis іs а tеst with ѕome һomoglурhs."
    
    @pytest.fixture
    def expected_result(self):
        """Expected normalized text."""
        return "This is a test with some homoglyphs."
    
    def test_ngram_strategy_integration(self, sample_text, expected_result, monkeypatch):
        """Test n-gram strategy integration with normalize_text API."""
        # Setup a mock function that will be called from within HomoglyphReplacer.normalize
        called_with = {}
        
        def mock_apply_ngram_strategy(*args, **kwargs):
            nonlocal called_with
            called_with = kwargs
            return expected_result
        
        # Apply the monkeypatch at the correct import location 
        monkeypatch.setattr("silverspeak.homoglyphs.normalization.ngram.apply_ngram_strategy", mock_apply_ngram_strategy)
            
        result = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.NGRAM,
            language="english",
            threshold=0.01
        )
            
        # Verify result
        assert result == expected_result
        
        # Verify the strategy was called with correct parameters
        assert called_with.get("text") == sample_text
        assert called_with.get("language") == "english"
        assert called_with.get("threshold") == 0.01
    
    def test_ocr_confidence_strategy_integration(self, sample_text, expected_result):
        """Test OCR confidence strategy integration with normalize_text API."""
        # Mock the underlying strategy implementation
        with unittest.mock.patch("silverspeak.homoglyphs.normalization.ocr_confidence.apply_ocr_confidence_strategy") as mock_apply:
            mock_apply.return_value = expected_result
            
            result = normalize_text(
                sample_text,
                strategy=NormalizationStrategies.OCR_CONFIDENCE,
                confidence_threshold=0.75,
                use_tesseract=False
            )
            
            # Verify result
            assert result == expected_result
            
            # Verify the strategy was called with correct parameters
            mock_apply.assert_called_once()
            args, kwargs = mock_apply.call_args
            assert kwargs["text"] == sample_text
            assert kwargs["confidence_threshold"] == 0.75
            assert kwargs["use_tesseract"] is False
    
    def test_graph_based_strategy_integration(self, sample_text, expected_result):
        """Test graph-based strategy integration with normalize_text API."""
        # Mock the underlying strategy implementation
        with unittest.mock.patch("silverspeak.homoglyphs.normalization.graph_based.apply_graph_strategy") as mock_apply:
            mock_apply.return_value = expected_result
            
            result = normalize_text(
                sample_text,
                strategy=NormalizationStrategies.GRAPH_BASED,
                similarity_threshold=0.8,
                centrality_measure="degree"
            )
            
            # Verify result
            assert result == expected_result
            
            # Verify the strategy was called with correct parameters
            mock_apply.assert_called_once()
            args, kwargs = mock_apply.call_args
            assert kwargs["text"] == sample_text
            assert kwargs["similarity_threshold"] == 0.8
            assert kwargs["centrality_measure"] == "degree"
    
    @pytest.mark.parametrize(
        "strategy,module_path,function_name",
        [
            (NormalizationStrategies.NGRAM, "silverspeak.homoglyphs.normalization.ngram", "apply_ngram_strategy"),
            (NormalizationStrategies.OCR_CONFIDENCE, "silverspeak.homoglyphs.normalization.ocr_confidence", "apply_ocr_confidence_strategy"),
            (NormalizationStrategies.GRAPH_BASED, "silverspeak.homoglyphs.normalization.graph_based", "apply_graph_strategy")
        ]
    )
    def test_strategy_fallback_on_error(self, sample_text, expected_result, strategy, module_path, function_name):
        """Test that strategies fall back to local context on error."""
        # Mock the apply_* function to raise an exception
        with unittest.mock.patch(f"{module_path}.{function_name}") as mock_apply:
            mock_apply.side_effect = Exception("Test error")
            
            # Mock the fallback strategy
            with unittest.mock.patch("silverspeak.homoglyphs.homoglyph_replacer.apply_local_context_strategy") as mock_fallback:
                mock_fallback.return_value = expected_result
                
                # Should not raise exception due to fallback
                result = normalize_text(sample_text, strategy=strategy)
                
                # Verify the original strategy was called
                mock_apply.assert_called_once()
                
                # Verify the fallback was called
                mock_fallback.assert_called_once()
                
                # Verify the result comes from the fallback
                assert result == expected_result
