Tokenizer Strategy
==================

The Tokenizer Strategy is an advanced normalization approach that leverages pre-trained tokenizers to optimize homoglyph replacements for downstream tokenized processing. This strategy analyzes how potential character replacements impact tokenization patterns and vocabulary compatibility, making it particularly valuable for applications involving machine learning models that process tokenized text.

Overview
--------

This strategy addresses the critical challenge of maintaining optimal tokenization behavior when normalizing homoglyphs. By analyzing the tokenizer's vocabulary and scoring potential replacements based on their tokenization characteristics, this approach ensures that normalized text remains compatible with tokenized processing workflows such as machine translation, language modeling, text generation, and natural language understanding tasks.

The strategy is especially important because different tokenizers (subword, byte-pair encoding, SentencePiece, etc.) can produce dramatically different tokenization patterns for visually similar characters, potentially affecting model performance downstream.

Implementation Details
-----------------------

The tokenizer strategy implementation in SilverSpeak follows a sophisticated multi-criteria scoring approach:

1. **Tokenizer Loading and Vocabulary Analysis**:
   The strategy begins by loading the specified tokenizer and processing its vocabulary:

   .. code-block:: python

      from silverspeak.homoglyphs.normalization import apply_tokenizer_strategy
      from transformers import AutoTokenizer

      # The strategy loads the tokenizer internally
      tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
      vocab = list(tokenizer.get_vocab().keys())
      vocab = sorted(vocab, key=len, reverse=True)

   The vocabulary is sorted by token length (longest first) to prioritize longer, more specific tokens during analysis. The strategy also handles tokenizer-specific prefixes (like space tokens) that might affect scoring.

2. **Character-Vocabulary Compatibility Analysis**:
   For each possible character replacement, the strategy identifies all vocabulary tokens that contain the character:

   .. code-block:: python

      # For each possible character, find matching vocabulary tokens
      possible_token_starts = {}
      for possible_char in possible_chars:
          tokens_with_char = [
              (token[:token.rindex(possible_char)], len(token), token)
              for token in vocab if possible_char in token
          ]
          possible_token_starts[possible_char] = tokens_with_char

   This analysis captures how each character fits into the tokenizer's subword patterns, considering both the character's position within tokens and the length of those tokens.

3. **Multi-Criteria Scoring System**:
   The strategy employs a weighted scoring system with four key criteria:

   .. code-block:: python

      # Default weights (can be customized)
      LONGEST_START_WEIGHT = 0.4          # Prioritize longer token prefixes
      LONGEST_TOKEN_WEIGHT = 0.3          # Favor longer overall tokens
      NUM_POSSIBLE_TOKENS_WEIGHT = 0.2    # Consider token count diversity
      NUM_TOKENS_CONTAINING_CHAR_WEIGHT = 0.1  # Account for character frequency

      # Score calculation for each character
      final_scores[char] = (
          LONGEST_START_WEIGHT * normalized_start_score +
          LONGEST_TOKEN_WEIGHT * normalized_token_score +
          NUM_POSSIBLE_TOKENS_WEIGHT * normalized_num_tokens_score +
          NUM_TOKENS_CONTAINING_CHAR_WEIGHT * normalized_char_frequency_score
      )

   Each criterion is normalized to a 0-1 scale before weighting to ensure fair comparison across different scales.

4. **Context-Aware Token Matching**:
   The strategy performs sophisticated context matching by filtering tokens based on how well they align with the existing normalized text:

   .. code-block:: python

      # Filter tokens to match existing context
      for char_key in possible_token_starts.keys():
          possible_token_starts[char_key] = [
              token_tuple for token_tuple in possible_token_starts[char_key]
              if normalized_text.endswith(token_tuple[0])  # Prefix matching
          ]

   This ensures that character selections consider how they fit into the broader tokenization context of the surrounding text.

5. **Optimal Character Selection**:
   The strategy selects the character with the highest composite score:

   .. code-block:: python

      best_char = max(final_scores.keys(), key=lambda k: final_scores[k])
      normalized_text += best_char

Advanced Usage Examples
-----------------------

**Basic Tokenizer Strategy Application**:

.. code-block:: python

   from silverspeak.homoglyphs.normalization import apply_tokenizer_strategy

   text = "Тhis іs а samрle with homoglуphs."  # Mixed Cyrillic homoglyphs
   normalization_map = {
       "Т": ["T"],  # Cyrillic 'Т' to Latin 'T'
       "і": ["i"],  # Cyrillic 'і' to Latin 'i'
       "а": ["a"],  # Cyrillic 'а' to Latin 'a'
       "р": ["p"],  # Cyrillic 'р' to Latin 'p'
       "у": ["u"],  # Cyrillic 'у' to Latin 'u'
   }

   normalized_text = apply_tokenizer_strategy(
       text=text,
       mapping=normalization_map,
       tokenizer_name="google/gemma-3-1b-pt"
   )
   print(f"Original:   {text}")
   print(f"Normalized: {normalized_text}")

**Alternative Usage via normalize_text**:

.. code-block:: python

   from silverspeak.homoglyphs import normalize_text
   from silverspeak.homoglyphs.utils import NormalizationStrategies

   text = "Mathеmatical ехprеssion: f(х) = 2х + 1"  # Mixed scripts
   normalized_text = normalize_text(
       text, 
       strategy=NormalizationStrategies.TOKENIZATION,
       tokenizer_name="microsoft/DialoGPT-medium"  # Different tokenizer
   )
   print(normalized_text)

**Custom Scoring Weights**:

.. code-block:: python

   # Prioritize longer tokens more heavily
   normalized_text = apply_tokenizer_strategy(
       text=text,
       mapping=normalization_map,
       tokenizer_name="bert-base-uncased",
       LONGEST_START_WEIGHT=0.5,     # Increase prefix weight
       LONGEST_TOKEN_WEIGHT=0.4,     # Increase token length weight
       NUM_POSSIBLE_TOKENS_WEIGHT=0.1,
       NUM_TOKENS_CONTAINING_CHAR_WEIGHT=0.0
   )

**Comparison with Different Tokenizers**:

.. code-block:: python

   text = "Prосеssing tеxt with spеcial сharacters"
   
   # Compare normalization results with different tokenizers
   tokenizers = [
       "bert-base-uncased",
       "gpt2",
       "microsoft/DialoGPT-medium",
       "google/gemma-3-1b-pt"
   ]
   
   for tokenizer_name in tokenizers:
       result = apply_tokenizer_strategy(
           text=text,
           mapping=normalization_map,
           tokenizer_name=tokenizer_name
       )
       print(f"{tokenizer_name}: {result}")

Performance Characteristics
---------------------------

**Computational Complexity**:
- **Time Complexity**: O(n × m × v) where n is text length, m is average homoglyph candidates per character, and v is vocabulary size
- **Space Complexity**: O(v) for vocabulary storage plus O(m) for candidate analysis
- **Memory Usage**: Moderate to high due to tokenizer and vocabulary loading

**Scalability Considerations**:
- Vocabulary size significantly impacts performance (larger vocabularies = longer processing)
- Character mapping size affects per-character processing time
- Tokenizer loading is a one-time cost that can be amortized across multiple texts

**Speed Benchmarks** (approximate, varies by hardware):
- Short text (< 100 chars): 0.1-0.5 seconds
- Medium text (100-1000 chars): 0.5-2.0 seconds  
- Long text (> 1000 chars): 2.0+ seconds

**Optimization Strategies**:

.. code-block:: python

   # Pre-load tokenizer for multiple normalizations
   from transformers import AutoTokenizer
   
   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   
   # Process multiple texts efficiently
   texts = ["Text 1", "Text 2", "Text 3"]
   for text in texts:
       result = apply_tokenizer_strategy(
           text=text,
           mapping=normalization_map,
           tokenizer_name="bert-base-uncased"  # Reuses loaded tokenizer
       )

Security Considerations
-----------------------

**Model Security**:
- **Dependency Vulnerabilities**: Relies on HuggingFace transformers library - ensure up-to-date versions
- **Model Integrity**: Downloaded tokenizer models should be verified if used in security-critical applications
- **Resource Consumption**: Large vocabularies can consume significant memory - monitor resource usage

**Input Validation**:

.. code-block:: python

   def secure_tokenizer_normalization(text, mapping, max_length=10000):
       """Apply tokenizer strategy with security constraints."""
       if len(text) > max_length:
           raise ValueError(f"Text length {len(text)} exceeds maximum {max_length}")
       
       if not isinstance(mapping, dict):
           raise TypeError("Mapping must be a dictionary")
       
       # Validate mapping content
       for key, values in mapping.items():
           if not isinstance(key, str) or not isinstance(values, list):
               raise TypeError("Invalid mapping format")
       
       return apply_tokenizer_strategy(text, mapping)

**Privacy Considerations**:
- Tokenizer models may have been trained on diverse data - consider data sensitivity
- No text is transmitted externally (local processing only)
- Be cautious with proprietary or sensitive text content

Best Practices
--------------

**Tokenizer Selection**:

.. code-block:: python

   # Choose tokenizers based on target application
   tokenizer_recommendations = {
       "general_purpose": "bert-base-uncased",
       "multilingual": "bert-base-multilingual-cased", 
       "conversation": "microsoft/DialoGPT-medium",
       "code_generation": "microsoft/CodeBERT-base",
       "translation": "marian-mt-models"
   }

**Weight Tuning Guidelines**:

.. code-block:: python

   # For different optimization goals
   optimization_profiles = {
       "accuracy_focused": {
           "LONGEST_START_WEIGHT": 0.5,
           "LONGEST_TOKEN_WEIGHT": 0.3,
           "NUM_POSSIBLE_TOKENS_WEIGHT": 0.15,
           "NUM_TOKENS_CONTAINING_CHAR_WEIGHT": 0.05
       },
       "speed_focused": {
           "LONGEST_START_WEIGHT": 0.6,
           "LONGEST_TOKEN_WEIGHT": 0.4,
           "NUM_POSSIBLE_TOKENS_WEIGHT": 0.0,
           "NUM_TOKENS_CONTAINING_CHAR_WEIGHT": 0.0
       },
       "balanced": {
           "LONGEST_START_WEIGHT": 0.4,
           "LONGEST_TOKEN_WEIGHT": 0.3,
           "NUM_POSSIBLE_TOKENS_WEIGHT": 0.2,
           "NUM_TOKENS_CONTAINING_CHAR_WEIGHT": 0.1
       }
   }

**Error Handling**:

.. code-block:: python

   try:
       result = apply_tokenizer_strategy(text, mapping, tokenizer_name="custom-model")
   except ImportError:
       # Fallback to simpler strategy
       from silverspeak.homoglyphs.normalization import apply_local_context_strategy
       result = apply_local_context_strategy(text, mapping)
   except Exception as e:
       logger.error(f"Tokenizer strategy failed: {e}")
       # Handle gracefully or re-raise

**Integration with Other Strategies**:

.. code-block:: python

   # Sequential application for enhanced accuracy
   from silverspeak.homoglyphs.utils import NormalizationStrategies
   
   # First pass: tokenizer-based normalization
   intermediate = normalize_text(text, strategy=NormalizationStrategies.TOKENIZATION)
   
   # Second pass: local context refinement
   final_result = normalize_text(intermediate, strategy=NormalizationStrategies.LOCAL_CONTEXT)

Limitations and Considerations
------------------------------

**Known Limitations**:
- **Vocabulary Coverage**: Effectiveness limited by tokenizer's vocabulary coverage
- **Language Bias**: Tokenizers trained on specific languages may perform poorly on others
- **Subword Artifacts**: Subword tokenization can create unexpected character preferences
- **Memory Requirements**: Large tokenizer models require significant RAM

**When to Use This Strategy**:
- ✅ Text will undergo tokenized processing (ML models, translation systems)
- ✅ Tokenizer compatibility is critical for downstream tasks
- ✅ Computational resources are available for tokenizer loading
- ✅ Vocabulary-based normalization is preferred over context-based approaches

**When to Consider Alternatives**:
- ❌ Simple, fast normalization is required
- ❌ Target tokenizer is unknown or varies frequently  
- ❌ Memory constraints are significant
- ❌ Real-time processing with minimal latency is essential

**Comparison with Other Strategies**:

.. code-block:: python

   # Performance comparison example
   strategies_comparison = {
       "tokenizer": "High accuracy for tokenized workflows, higher memory usage",
       "local_context": "Fast, context-aware, moderate accuracy", 
       "dominant_script": "Very fast, script-based, lower accuracy for mixed scripts",
       "language_model": "Highest accuracy, very high computational cost"
   }

The Tokenizer Strategy represents a sophisticated approach to homoglyph normalization that prioritizes compatibility with modern NLP workflows. While it requires more computational resources than simpler strategies, its ability to optimize for specific tokenizer vocabularies makes it invaluable for applications where downstream tokenization quality directly impacts performance.