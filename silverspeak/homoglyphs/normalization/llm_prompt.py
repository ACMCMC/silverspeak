"""
LLM prompt-based normalization strategy for homoglyph replacement.

This module provides functionality to normalize text using a large language model
with a prompt-based approach, asking the model to identify and fix homoglyphs in text.
Unlike the masking approach which requires a specific masked language model architecture,
this approach can work with any text generation model that can follow instructions.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
from typing import Dict, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)


def apply_llm_prompt_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    model_name: str = "google/gemma-3-1b-it",
    device: Optional[str] = None,
    max_length: int = 512,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Normalize text by prompting a language model to fix homoglyph replacements.
    
    This strategy uses a prompt-based approach with a language model to identify and 
    correct homoglyphs in text. The model is prompted to normalize the text by replacing
    homoglyphs with their standard equivalents.

    Args:
        text (str): The input text to normalize.
        mapping (Mapping[str, List[str]]): A mapping from original characters to
            their possible homoglyph replacements.
        model_name (str): The HuggingFace model name to load.
            Defaults to "google/gemma-3-1b-it".
        device (Optional[str]): Device to run the model on ('cuda', 'cpu', etc.).
            Defaults to cuda if available, otherwise cpu.
        max_length (int): Maximum length of text segments to process. Longer text will be split.
            Defaults to 512 tokens.
        temperature (float): Sampling temperature for model generation.
            Lower values like 0.0 make output more deterministic. Defaults to 0.0.
        system_prompt (Optional[str]): Custom system prompt to use instead of the default one.
            Default is None, which will use an internally generated prompt.
        **kwargs: Additional keyword arguments passed to the model's generate method.

    Returns:
        str: The normalized text with homoglyphs replaced based on the language model's output.

    Raises:
        ImportError: If required dependencies are not installed.
        RuntimeError: If model loading fails or other runtime errors occur.
        
    Note:
        This strategy works best with instruction-tuned models that can follow detailed prompts.
        The quality of normalization depends on the model's ability to identify and correct homoglyphs.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        logging.error(
            "This normalization strategy requires the transformers and torch libraries. "
            "Install them with: pip install transformers torch"
        )
        raise ImportError("Missing required dependencies: transformers, torch")

    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not mapping:
        logging.warning("Empty mapping provided for normalization")
        return text

    # Set default device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Using device: {device} with model: {model_name}")

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        model.to(device)
        model.eval()
        
        # Create a text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer '{model_name}': {e}")
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")
    
    # Create the default system prompt if not provided
    if system_prompt is None:
        # Create a summary of homoglyph patterns for the prompt
        homoglyph_examples = []
        for original_char, homoglyphs in list(mapping.items())[:5]:  # Use just a few examples
            formatted_homoglyphs = ", ".join([f"'{h}'" for h in homoglyphs])
            homoglyph_examples.append(f"'{original_char}' can be replaced with {formatted_homoglyphs}")
        
        homoglyph_info = "; ".join(homoglyph_examples)
        if len(mapping) > 5:
            homoglyph_info += "; and many more..."
        
        system_prompt = f"""You are a text normalization assistant that specializes in correcting homoglyphs.
Homoglyphs are characters that look similar but have different Unicode code points.
For example: {homoglyph_info}

Your task is to read the provided text which may contain homoglyphs (visually similar characters from different scripts)
and produce a normalized version with the correct characters.

Important instructions:
1. Identify any homoglyphs or suspicious characters that might be replacements
2. Replace them with their correct characters (which are often in the same alphabet/script as the surrounding text)
3. Preserve the exact wording, spacing, and punctuation of the original text
4. If you're uncertain about a character, keep it as is
5. Return ONLY the normalized text without any explanations or additional comments

Process the text as if it has been deliberately modified with homoglyphs as part of an obfuscation attack.
"""

    # Process text in segments if it's too long
    normalized_text_segments = []
    
    for i in range(0, len(text), max_length):
        segment = text[i:i + max_length]
        
        # Create the full prompt
        prompt = f"{system_prompt}\n\nText to normalize: {segment}\n\nNormalized text:"
        
        # Generate the normalized text
        try:
            outputs = generator(
                prompt, 
                max_new_tokens=len(segment) * 2,  # Allow enough tokens for response
                temperature=temperature,
                do_sample=temperature > 0,
                num_return_sequences=1,
                **kwargs
            )
            
            # Extract the generated text
            generated_text = outputs[0]["generated_text"]
            
            # Extract just the normalized part (after the prompt)
            if "Normalized text:" in generated_text:
                normalized_segment = generated_text.split("Normalized text:")[1].strip()
            else:
                # If the model didn't follow the format, try to get text after the original
                prompt_end = prompt.split("\n\n")[-2]  # Use the "Text to normalize:" part
                if prompt_end in generated_text:
                    normalized_segment = generated_text.split(prompt_end)[1].strip()
                    # Further clean up if needed
                    if "Normalized text:" in normalized_segment:
                        normalized_segment = normalized_segment.split("Normalized text:")[1].strip()
                else:
                    # Fallback
                    logging.warning(f"Could not extract normalized text from model output, using original segment")
                    normalized_segment = segment
            
            # Log a sample of the normalization for debugging
            if i == 0:  # Only log the first segment
                logging.debug(f"Original: {segment[:100]}{'...' if len(segment) > 100 else ''}")
                logging.debug(f"Normalized: {normalized_segment[:100]}{'...' if len(normalized_segment) > 100 else ''}")
            
            normalized_text_segments.append(normalized_segment)
            
        except Exception as e:
            logging.error(f"Error during text generation: {e}")
            # Fallback to the original segment
            normalized_text_segments.append(segment)
    
    # Join all processed segments
    return "".join(normalized_text_segments)
