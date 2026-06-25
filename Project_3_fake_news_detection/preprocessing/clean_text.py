"""
--------------------------------------------------------
AI Based Fake News Detection Tool
Text Cleaning Module
--------------------------------------------------------

Author : Your Name
Description:
This module performs complete NLP text preprocessing
before converting news into numerical features.

Functions Included:
1. Lowercase conversion
2. HTML removal
3. URL removal
4. Email removal
5. Number removal
6. Punctuation removal
7. Emoji removal
8. Extra whitespace removal
9. Stopword removal
10. Lemmatization

--------------------------------------------------------
"""

import re
import string

from bs4 import BeautifulSoup

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------------------------------
# Download NLTK Resources
# -----------------------------------------------------

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# -----------------------------------------------------
# Initialize
# -----------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()


# -----------------------------------------------------
# Remove HTML
# -----------------------------------------------------

def remove_html(text: str) -> str:
    """
    Removes HTML tags.
    """

    return BeautifulSoup(text, "html.parser").get_text()


# -----------------------------------------------------
# Remove URLs
# -----------------------------------------------------

def remove_urls(text: str) -> str:
    """
    Removes hyperlinks.
    """

    pattern = r"http\S+|www\S+|https\S+"

    return re.sub(pattern, "", text)


# -----------------------------------------------------
# Remove Emails
# -----------------------------------------------------

def remove_email(text: str) -> str:

    pattern = r"\S+@\S+"

    return re.sub(pattern, "", text)


# -----------------------------------------------------
# Remove Numbers
# -----------------------------------------------------

def remove_numbers(text: str) -> str:

    return re.sub(r"\d+", "", text)


# -----------------------------------------------------
# Remove Emojis
# -----------------------------------------------------

def remove_emojis(text: str) -> str:

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.sub("", text)


# -----------------------------------------------------
# Remove Punctuation
# -----------------------------------------------------

def remove_punctuation(text: str) -> str:

    translator = str.maketrans("", "", string.punctuation)

    return text.translate(translator)


# -----------------------------------------------------
# Remove Extra Spaces
# -----------------------------------------------------

def remove_extra_spaces(text: str) -> str:

    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------------------
# Remove Stopwords
# -----------------------------------------------------

def remove_stopwords(text: str) -> str:

    words = text.split()

    filtered_words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_words)


# -----------------------------------------------------
# Lemmatization
# -----------------------------------------------------

def lemmatize_text(text: str) -> str:

    words = text.split()

    lemmas = [
        LEMMATIZER.lemmatize(word)
        for word in words
    ]

    return " ".join(lemmas)


# -----------------------------------------------------
# Complete Cleaning Pipeline
# -----------------------------------------------------

def clean_news_text(text: str) -> str:
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    text : str

    Returns
    -------
    Cleaned text
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = remove_html(text)

    text = remove_urls(text)

    text = remove_email(text)

    text = remove_numbers(text)

    text = remove_emojis(text)

    text = remove_punctuation(text)

    text = remove_extra_spaces(text)

    text = remove_stopwords(text)

    text = lemmatize_text(text)

    return text


# -----------------------------------------------------
# Manual Testing
# -----------------------------------------------------

if __name__ == "__main__":

    sample = """
    BREAKING NEWS!!!

    Visit https://news.com

    Contact admin@gmail.com

    India won the cricket match in 2025 😍🔥🔥
    """

    print("Original Text:\n")
    print(sample)

    print("\n")

    cleaned = clean_news_text(sample)

    print("Cleaned Text:\n")
    print(cleaned)