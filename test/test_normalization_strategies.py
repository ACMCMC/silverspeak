import pytest
import logging
import unittest.mock
from typing import Dict, List
import unicodedata
import unicodedataplus

from silverspeak.homoglyphs.normalization import (
    apply_dominant_script_strategy,
    apply_dominant_script_and_block_strategy,
    apply_local_context_strategy,
    apply_tokenizer_strategy,
    apply_language_model_strategy,
    configure_logging,
    VALID_LOG_LEVELS,
)
from silverspeak.homoglyphs.script_block_category_utils import (
    detect_dominant_script,
    detect_dominant_block,
)


class MockReplacer:
    """Mock replacer class for testing strategies."""
    
    def get_normalization_map_for_script_block_and_category(self, **kwargs):
        """Return a simple normalization map for testing."""
        script = kwargs.get("script", "")
        block = kwargs.get("block", "")
        
        if script == "Unknown" or block == "Unknown":
            return {}
            
        # Simple test mapping
        return {
            "а": ["a"],  # Cyrillic 'а' to Latin 'a'
            "е": ["e"],  # Cyrillic 'е' to Latin 'e'
            "о": ["o"],  # Cyrillic 'о' to Latin 'o'
            "с": ["c"],  # Cyrillic 'с' to Latin 'c'
            "р": ["p"],  # Cyrillic 'р' to Latin 'p'
        }


@pytest.fixture
def mock_replacer():
    return MockReplacer()


@pytest.fixture
def test_normalization_map():
    """Fixture providing a test normalization map."""
    return {
        "а": ["a"],  # Cyrillic 'а' to Latin 'a'
        "е": ["e"],  # Cyrillic 'е' to Latin 'e'
        "о": ["o"],  # Cyrillic 'о' to Latin 'o'
        "с": ["c"],  # Cyrillic 'с' to Latin 'c'
        "р": ["p"],  # Cyrillic 'р' to Latin 'p'
    }


class TestScriptAndBlockDetection:
    """Tests for script and block detection functions."""
    
    @pytest.mark.parametrize(
        "text,expected_script",
        [
            ("hello world", "Latin"),
            ("привет мир", "Cyrillic"),
            ("你好，世界", "Han"),
            ("こんにちは世界", "Han"),  # Mixed Han and Hiragana
            ("", "Unknown"),
            ("1234567890", "Common"),
            ("hello привет", "Latin"),  # Mixed but Latin dominant
        ]
    )
    def test_detect_dominant_script(self, text, expected_script):
        """Test the detect_dominant_script function with various inputs."""
        with unittest.mock.patch("logging.warning") as mock_warning:
            result = detect_dominant_script(text)
            if text == "":
                mock_warning.assert_called_with("Empty text provided to detect_dominant_script")
            elif text == "こんにちは世界":
                # This should log a warning as it has mixed scripts
                mock_warning.assert_called()
            
            assert result == expected_script
    
    @pytest.mark.parametrize(
        "text,expected_block",
        [
            ("hello world", "Basic Latin"),
            ("привет мир", "Cyrillic"),
            ("你好，世界", "CJK Unified Ideographs"),
            ("", "Unknown"),
            ("1234567890", "Basic Latin"),
        ]
    )
    def test_detect_dominant_block(self, text, expected_block):
        """Test the detect_dominant_block function with various inputs."""
        with unittest.mock.patch("logging.warning") as mock_warning:
            result = detect_dominant_block(text)
            if text == "":
                mock_warning.assert_called_with("Empty text provided to detect_dominant_block")
            
            assert result == expected_block


class TestDominantScriptStrategy:
    """Tests for the dominant script strategy."""
    
    def test_apply_dominant_script_strategy_basic(self, mock_replacer):
        """Test the basic functionality of apply_dominant_script_strategy."""
        text = "привет мир"  # Cyrillic
        result = apply_dominant_script_strategy(mock_replacer, text)
        assert result == "пpивет миp"  # Only 'р' is replaced with 'p'
    
    def test_apply_dominant_script_strategy_mixed(self, mock_replacer):
        """Test apply_dominant_script_strategy with mixed scripts."""
        text = "hello мир"  # Mixed Latin and Cyrillic
        result = apply_dominant_script_strategy(mock_replacer, text)
        assert result == "hello миp"  # 'р' is replaced with 'p'
    
    def test_apply_dominant_script_strategy_empty(self, mock_replacer):
        """Test apply_dominant_script_strategy with empty text."""
        with unittest.mock.patch("logging.warning") as mock_warning:
            result = apply_dominant_script_strategy(mock_replacer, "")
            mock_warning.assert_called_with("Empty text provided for normalization")
            assert result == ""
    
    def test_apply_dominant_script_strategy_no_replacer(self):
        """Test apply_dominant_script_strategy with no replacer."""
        with pytest.raises(ValueError):
            apply_dominant_script_strategy(None, "test")


class TestDominantScriptAndBlockStrategy:
    """Tests for the dominant script and block strategy."""
    
    def test_apply_dominant_script_and_block_strategy_basic(self, mock_replacer):
        """Test the basic functionality of apply_dominant_script_and_block_strategy."""
        text = "привет мир"  # Cyrillic
        result = apply_dominant_script_and_block_strategy(mock_replacer, text)
        assert result == "пpивет миp"  # 'р' is replaced with 'p'
    
    def test_apply_dominant_script_and_block_strategy_empty(self, mock_replacer):
        """Test apply_dominant_script_and_block_strategy with empty text."""
        with unittest.mock.patch("logging.warning") as mock_warning:
            result = apply_dominant_script_and_block_strategy(mock_replacer, "")
            mock_warning.assert_called_with("Empty text provided for normalization")
            assert result == ""
    
    def test_apply_dominant_script_and_block_strategy_no_replacer(self):
        """Test apply_dominant_script_and_block_strategy with no replacer."""
        with pytest.raises(ValueError):
            apply_dominant_script_and_block_strategy(None, "test")
    
    def test_apply_dominant_script_and_block_strategy_unknown(self, mock_replacer):
        """Test apply_dominant_script_and_block_strategy with unknown script/block."""
        with unittest.mock.patch("silverspeak.homoglyphs.script_block_category_utils.detect_dominant_script") as mock_script:
            with unittest.mock.patch("silverspeak.homoglyphs.script_block_category_utils.detect_dominant_block") as mock_block:
                mock_script.return_value = "Unknown"
                mock_block.return_value = "Unknown"
                
                with unittest.mock.patch("logging.warning") as mock_warning:
                    result = apply_dominant_script_and_block_strategy(mock_replacer, "test")
                    mock_warning.assert_called()
                    # Since the normalization map is empty for Unknown script/block, the text should be unchanged
                    assert result == "test"


class TestLocalContextStrategy:
    """Tests for the local context strategy."""
    
    def test_apply_local_context_strategy_basic(self, test_normalization_map):
        """Test the basic functionality of apply_local_context_strategy."""
        text = "привет мир"  # Cyrillic with 'р' that can be replaced with 'p'
        result = apply_local_context_strategy(text, test_normalization_map)
        # The 'р' should be replaced with 'p' as it matches more context properties
        assert "p" in result
    
    def test_apply_local_context_strategy_empty_text(self, test_normalization_map):
        """Test apply_local_context_strategy with empty text."""
        with unittest.mock.patch("logging.warning") as mock_warning:
            result = apply_local_context_strategy("", test_normalization_map)
            mock_warning.assert_called_with("Empty text provided for normalization")
            assert result == ""
    
    def test_apply_local_context_strategy_empty_map(self):
        """Test apply_local_context_strategy with empty normalization map."""
        with unittest.mock.patch("logging.warning") as mock_warning:
            result = apply_local_context_strategy("test", {})
            mock_warning.assert_called_with("Empty normalization map provided")
            assert result == "test"
    
    def test_apply_local_context_strategy_window_size(self, test_normalization_map):
        """Test apply_local_context_strategy with different window sizes."""
        text = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"  # Russian alphabet
        
        # Test with different window sizes
        result_small = apply_local_context_strategy(text, test_normalization_map, N=4)
        result_large = apply_local_context_strategy(text, test_normalization_map, N=20)
        
        # Results should be different with different window sizes
        assert result_small != result_large


class TestTokenizerStrategy:
    """Tests for the tokenizer strategy."""
    
    @pytest.mark.parametrize(
        "weights,expected_different",
        [
            # Different weight combinations should produce different results
            ((0.7, 0.1, 0.1, 0.1), True),
            ((0.1, 0.7, 0.1, 0.1), True),
            ((0.1, 0.1, 0.7, 0.1), True),
            ((0.1, 0.1, 0.1, 0.7), True),
            # Same weights should produce the same results
            ((0.25, 0.25, 0.25, 0.25), False),
        ]
    )
    def test_apply_tokenizer_strategy_weights(self, test_normalization_map, weights, expected_different):
        """Test that different weight combinations affect the result."""
        # Skip this test if transformers is not installed
        pytest.importorskip("transformers")
        
        text = "привет мир"  # Cyrillic with several characters that can be replaced
        
        # Mock the tokenizer to avoid actual loading
        with unittest.mock.patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_cls:
            mock_tokenizer = unittest.mock.MagicMock()
            mock_tokenizer.get_vocab.return_value = {"test": 1, "te": 2, "es": 3, "st": 4}
            mock_tokenizer_cls.return_value = mock_tokenizer
            
            # Apply the strategy with different weights
            result1 = apply_tokenizer_strategy(
                text, 
                test_normalization_map,
                LONGEST_START_WEIGHT=weights[0],
                LONGEST_TOKEN_WEIGHT=weights[1],
                NUM_POSSIBLE_TOKENS_WEIGHT=weights[2],
                NUM_TOKENS_CONTAINING_CHAR_WEIGHT=weights[3],
            )
            
            # Apply with default weights
            result2 = apply_tokenizer_strategy(text, test_normalization_map)
            
            if expected_different:
                assert result1 != result2
            else:
                assert result1 == result2


class TestLanguageModelStrategy:
    """Tests for the language model strategy."""
    
    def test_apply_language_model_strategy_basic(self, test_normalization_map):
        """Test the basic functionality of apply_language_model_strategy."""
        # Skip this test if torch or transformers is not installed
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        
        text = "привет мир"  # Cyrillic with 'р' that can be replaced with 'p'
        
        # Create mocks for the language model and tokenizer
        with unittest.mock.patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_cls:
            with unittest.mock.patch("transformers.AutoModelForMaskedLM.from_pretrained") as mock_model_cls:
                import torch
                
                # Mock tokenizer
                mock_tokenizer = unittest.mock.MagicMock()
                mock_tokenizer.mask_token = "[MASK]"
                mock_tokenizer.mask_token_id = 103
                mock_tokenizer.encode.return_value = [103]
                mock_tokenizer.return_value = {"input_ids": torch.tensor([[101, 103, 102]]), "attention_mask": torch.tensor([[1, 1, 1]])}
                mock_tokenizer_cls.return_value = mock_tokenizer
                
                # Mock model output
                mock_output = unittest.mock.MagicMock()
                mock_output.logits = torch.zeros((1, 3, 1000))
                # Make 'p' the highest probability for the masked position
                mock_output.logits[0, 1, ord('p')] = 10.0
                
                # Mock model
                mock_model = unittest.mock.MagicMock()
                mock_model.return_value = mock_output
                mock_model_cls.return_value = mock_model
                
                # Apply the strategy
                result = apply_language_model_strategy(
                    text,
                    test_normalization_map,
                    batch_size=2,
                    max_length=10,
                )
                
                # Verify the function ran without errors
                assert isinstance(result, str)


class TestSpellCheckStrategy:
    """Tests for the spell check normalization strategy."""
    
    @pytest.mark.xfail(reason="Requires symspellpy and pyspellchecker dependencies")
    def test_basic_spell_check(self):
        """Test basic spell check functionality."""
        try:
            from silverspeak.homoglyphs.normalization.spell_check import apply_spell_check_strategy
            
            # Text with homoglyphs (Cyrillic characters)
            text = "Tһis іs а tеst."  # Contains homoglyphs
            
            result = apply_spell_check_strategy(
                text=text,
                normalization_map={
                    "һ": ["h"],  # Cyrillic 'һ' to Latin 'h'
                    "і": ["i"],  # Cyrillic 'і' to Latin 'i'
                    "а": ["a"],  # Cyrillic 'а' to Latin 'a'
                    "е": ["e"],  # Cyrillic 'е' to Latin 'e'
                },
                language="en"
            )
            
            assert result == "This is a test."
        except ImportError:
            pytest.skip("Required dependencies not installed")
    
    @pytest.mark.xfail(reason="Requires symspellpy and pyspellchecker dependencies")
    def test_custom_dictionary(self):
        """Test spell check with custom dictionary."""
        try:
            from silverspeak.homoglyphs.normalization.spell_check import apply_spell_check_strategy
            
            # Text with homoglyphs and domain-specific terms
            text = "SіlvеrSреаk"  # Contains homoglyphs
            
            result = apply_spell_check_strategy(
                text=text,
                normalization_map={
                    "і": ["i"],  # Cyrillic 'і' to Latin 'i'
                    "е": ["e"],  # Cyrillic 'е' to Latin 'e'
                    "р": ["p"],  # Cyrillic 'р' to Latin 'p'
                    "а": ["a"],  # Cyrillic 'а' to Latin 'a'
                },
                language="en",
                custom_words=["SilverSpeak"]
            )
            
            assert result == "SilverSpeak"
        except ImportError:
            pytest.skip("Required dependencies not installed")


class TestLLMPromptStrategy:
    """Tests for the LLM prompt-based normalization strategy."""
    
    @pytest.mark.xfail(reason="Requires transformers dependency")
    def test_llm_prompt_basic(self):
        """Test basic LLM prompt strategy."""
        try:
            from silverspeak.homoglyphs.normalization.llm_prompt import apply_llm_prompt_strategy
            import unittest.mock
            
            # Text with homoglyphs
            text = "Tһis іs а tеst."  # Contains homoglyphs
            
            # Mock the pipeline
            with unittest.mock.patch("silverspeak.homoglyphs.normalization.llm_prompt.pipeline") as mock_pipeline:
                mock_pipe_instance = unittest.mock.MagicMock()
                mock_pipe_instance.return_value = [{"generated_text": "This is a test."}]
                mock_pipeline.return_value = mock_pipe_instance
                
                result = apply_llm_prompt_strategy(
                    text=text,
                    normalization_map={
                        "һ": ["h"],
                        "і": ["i"],
                        "а": ["a"],
                        "е": ["e"]
                    }
                )
                
                # Verify the result matches the mocked return value
                assert result == "This is a test."
                
                # Verify the pipeline was called with the expected model name
                mock_pipeline.assert_called_once()
        except ImportError:
            pytest.skip("Required dependencies not installed")


class TestLanguageModelStrategy:
    """Tests for the language model normalization strategy."""
    
    @pytest.mark.xfail(reason="Requires transformers and torch dependencies")
    def test_language_model_word_level(self):
        """Test language model strategy with word-level masking."""
        try:
            import unittest.mock
            from silverspeak.homoglyphs.normalization.language_model import apply_language_model_strategy
            
            # Mock the language model and tokenizer
            with unittest.mock.patch("silverspeak.homoglyphs.normalization.language_model.AutoModelForMaskedLM") as mock_model_class:
                with unittest.mock.patch("silverspeak.homoglyphs.normalization.language_model.AutoTokenizer") as mock_tokenizer_class:
                    mock_model = unittest.mock.MagicMock()
                    mock_tokenizer = unittest.mock.MagicMock()
                    
                    # Mock the tokenize method
                    mock_tokenizer.tokenize.return_value = ["this", "is", "a", "test"]
                    
                    # Mock the encode method
                    mock_tokenizer.encode.return_value = [101, 1996, 2003, 1037, 3231, 102]
                    
                    # Mock the convert_ids_to_tokens method
                    mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "this", "is", "a", "test", "[SEP]"]
                    
                    # Mock the decode method
                    mock_tokenizer.decode.return_value = "this is a test"
                    
                    # Mock the __call__ method of the model
                    mock_model.return_value.logits = unittest.mock.MagicMock()
                    
                    mock_model_class.from_pretrained.return_value = mock_model
                    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                    
                    # Text with homoglyphs
                    text = "Tһis іs а tеst."  # Contains homoglyphs
                    
                    result = apply_language_model_strategy(
                        text=text,
                        normalization_map={
                            "һ": ["h"],
                            "і": ["i"],
                            "а": ["a"],
                            "е": ["e"]
                        },
                        word_level=True
                    )
                    
                    # With our mocks, we expect the result to be the mocked decode return value
                    assert result == "this is a test"
        except ImportError:
            pytest.skip("Required dependencies not installed")


class TestLoggingConfiguration:
    """Tests for the logging configuration functionality."""
    
    def test_configure_logging_default(self):
        """Test the default logging configuration."""
        with unittest.mock.patch("logging.basicConfig") as mock_basic_config:
            configure_logging()
            mock_basic_config.assert_called_once()
            args, kwargs = mock_basic_config.call_args
            assert kwargs["level"] == logging.INFO
    
    @pytest.mark.parametrize(
        "level,expected",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
            ("invalid", logging.INFO),  # Should default to INFO
        ]
    )
    def test_configure_logging_levels(self, level, expected):
        """Test various logging levels."""
        with unittest.mock.patch("logging.basicConfig") as mock_basic_config:
            configure_logging(level=level)
            mock_basic_config.assert_called_once()
            args, kwargs = mock_basic_config.call_args
            assert kwargs["level"] == expected
    
    def test_configure_logging_custom_format(self):
        """Test custom format string."""
        custom_format = "%(levelname)s: %(message)s"
        with unittest.mock.patch("logging.basicConfig") as mock_basic_config:
            configure_logging(format_string=custom_format)
            mock_basic_config.assert_called_once()
            args, kwargs = mock_basic_config.call_args
            assert kwargs["format"] == custom_format


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])