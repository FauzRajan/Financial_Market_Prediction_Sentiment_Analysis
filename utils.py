# src/utils.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
import os

# --- NLTK Data Downloads ---
# These downloads are necessary for the text preprocessing functions to work.
# We'll put them here to ensure they are available when the script is run.
# They will only download the first time they are executed.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' NLTK data...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading 'wordnet' NLTK data...")
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' NLTK data...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt_tab' NLTK data...")
    nltk.download('punkt_tab')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Cleans and preprocesses text for sentiment analysis.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned and preprocessed text.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs

    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags (optional, but good for general text)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers (optional, depends on if numbers carry sentiment in financial context)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)
