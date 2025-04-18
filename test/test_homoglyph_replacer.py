import unittest
from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer


class TestHomoglyphReplacer(unittest.TestCase):
    def setUp(self):
        """
        Set up a HomoglyphReplacer instance for testing.
        """
        self.replacer = HomoglyphReplacer()

    def test_normalize_dominant_script_latin_cyrillic(self):
        """
        Test the `normalize` method with the 'dominant_script' strategy.
        """
        # Mock data for testing
        self.replacer.reverse_chars_map = {
            "а": "a",  # Cyrillic 'a'
            "е": "e",  # Cyrillic 'e'
        }
        self.replacer._base_normalization_map = {
            "а": "a",
            "е": "e",
        }
        self.replacer.normalization_translation_tables = {}
        self.replacer._detect_dominant_script = lambda text: "Latin"

        # Input text containing homoglyphs
        input_text = "rаcеs"  # Cyrillic homoglyphs for "race"
        expected_output = "races"

        # Perform normalization
        output = self.replacer.normalize(input_text, strategy="dominant_script")

        # Assert the output matches the expected result
        self.assertEqual(output, expected_output)

    def test_normalize_unsupported_strategy(self):
        """
        Test the `normalize` method with an unsupported strategy.
        """
        with self.assertRaises(NotImplementedError):
            self.replacer.normalize("test", strategy="unsupported_strategy")

    def test_normalize_tokenization_strategy(self):
        """
        Test the `normalize` method with the 'tokenization' strategy.
        """
        with self.assertRaises(NotImplementedError):
            self.replacer.normalize("test", strategy="tokenization")

    def test_normalize_dominant_script_mixed_scripts(self):
        """
        Test the `normalize` method with mixed scripts in the input text.
        """
        # Mock data for testing
        self.replacer.reverse_chars_map = {
            "а": "a",  # Cyrillic 'a'
            "е": "e",  # Cyrillic 'e'
            "о": "o",  # Cyrillic 'o'
        }
        self.replacer._base_normalization_map = {
            "а": "a",
            "е": "e",
            "о": "o",
        }
        self.replacer.normalization_translation_tables = {}
        self.replacer._detect_dominant_script = lambda text: "Latin"

        # Input text containing mixed scripts
        input_text = "hеllо wоrld"  # Cyrillic homoglyphs for "hello world"
        expected_output = "hello world"

        # Perform normalization
        output = self.replacer.normalize(input_text, strategy="dominant_script")

        # Assert the output matches the expected result
        self.assertEqual(output, expected_output)

    def test_normalize_dominant_script_all_cyrillic(self):
        """
        Test the `normalize` method when the dominant script is Cyrillic.
        """
        # Mock data for testing
        self.replacer.reverse_chars_map = {
            "а": "a",  # Cyrillic 'a'
            "е": "e",  # Cyrillic 'e'
        }
        self.replacer._base_normalization_map = {
            "а": "a",
            "е": "e",
        }
        self.replacer.normalization_translation_tables = {}
        self.replacer._detect_dominant_script = lambda text: "Cyrillic"

        # Input text containing homoglyphs
        input_text = "расе"  # Cyrillic homoglyphs for "race"
        expected_output = "расе"  # No change since dominant script is Cyrillic

        # Perform normalization
        output = self.replacer.normalize(input_text, strategy="dominant_script")

        # Assert the output matches the expected result
        self.assertEqual(output, expected_output)

    def test_normalize_empty_string(self):
        """
        Test the `normalize` method with an empty string.
        """
        # Input text is empty
        input_text = ""
        expected_output = ""

        # Perform normalization
        output = self.replacer.normalize(input_text, strategy="dominant_script")

        # Assert the output matches the expected result
        self.assertEqual(output, expected_output)

    def test_normalize_no_homoglyphs(self):
        """
        Test the `normalize` method with text containing no homoglyphs.
        """
        # Input text with no homoglyphs
        input_text = "hello world"
        expected_output = "hello world"

        # Perform normalization
        output = self.replacer.normalize(input_text, strategy="dominant_script_and_block")

        # Assert the output matches the expected result
        self.assertEqual(output, expected_output)

    def test_normalize_invalid_strategy(self):
        """
        Test the `normalize` method with an invalid strategy.
        """
        with self.assertRaises(NotImplementedError):
            self.replacer.normalize("test", strategy="invalid_strategy")


if __name__ == "__main__":
    unittest.main()
