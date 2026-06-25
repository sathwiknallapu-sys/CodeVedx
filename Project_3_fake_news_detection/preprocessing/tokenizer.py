"""
--------------------------------------------------------
AI Based Fake News Detection Tool
Tokenizer Module
--------------------------------------------------------

Author : Your Name

Description:
This module is responsible for converting cleaned news
text into tokens. It also provides utilities for word
frequency analysis and token statistics.

--------------------------------------------------------
"""

from collections import Counter
import nltk

# Download tokenizer if not available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from nltk.tokenize import word_tokenize


def tokenize_text(text: str) -> list:
    """
    Tokenize input text into words.

    Parameters
    ----------
    text : str

    Returns
    -------
    list
        List of word tokens.
    """

    if not isinstance(text, str):
        return []

    return word_tokenize(text)


def token_count(tokens: list) -> int:
    """
    Returns total number of tokens.
    """

    return len(tokens)


def unique_token_count(tokens: list) -> int:
    """
    Returns number of unique words.
    """

    return len(set(tokens))


def word_frequency(tokens: list, top_n: int = 10):
    """
    Returns most common words.

    Parameters
    ----------
    tokens : list
    top_n : int

    Returns
    -------
    list
    """

    frequency = Counter(tokens)

    return frequency.most_common(top_n)


def tokenize_and_analyze(text: str) -> dict:
    """
    Complete tokenizer pipeline.

    Returns:
        Dictionary containing:
            tokens
            total_tokens
            unique_tokens
            top_words
    """

    tokens = tokenize_text(text)

    result = {
        "tokens": tokens,
        "total_tokens": token_count(tokens),
        "unique_tokens": unique_token_count(tokens),
        "top_words": word_frequency(tokens)
    }

    return result


# -------------------------------------------------------
# Manual Testing
# -------------------------------------------------------

if __name__ == "__main__":

    sample_text = """
    Artificial Intelligence is transforming healthcare.
    AI helps doctors detect diseases quickly.
    AI also improves medical diagnosis.
    """

    analysis = tokenize_and_analyze(sample_text)

    print("\nTokens\n")
    print(analysis["tokens"])

    print("\nTotal Tokens")
    print(analysis["total_tokens"])

    print("\nUnique Tokens")
    print(analysis["unique_tokens"])

    print("\nMost Frequent Words")
    print(analysis["top_words"])