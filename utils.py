# utils.py
import re
import polars as pl

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove common corporate/street stop words to prevent candidate explosion
    stop_words = {'inc', 'llc', 'ltd', 'corp', 'gmbh', 'co', 'the', 'st', 'rd', 'ave', 'blvd', 'court'}
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    return " ".join(tokens)