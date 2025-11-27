import argparse
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sentence_transformers import SentenceTransformer


# ---------- Text preprocessing helpers ----------

HTML_TAG_REGEX = re.compile(r"<[^>]+>")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  
    "\U0001F300-\U0001F5FF"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F1E0-\U0001F1FF" 
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001FA70-\U0001FAFF"  
    "\U00002600-\U000026FF"  
    "\U00002B50"
    "]+",
    flags=re.UNICODE,
)

PATTERN = r"[^a-zA-Z0-9]+"

STOP_WORDS = None
STEMMER = PorterStemmer()

#  Check if NLTK stopwords are downloaded; if not, download them
def ensure_nltk_resources():
   
    global STOP_WORDS
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        STOP_WORDS = set(stopwords.words("english"))

# Clean and preprocess text for embedding generation
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove HTML tags and emojis
    text = HTML_TAG_REGEX.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)

    # Lowercase and remove non-alphanumeric chars
    text = text.lower()
    text = re.sub(PATTERN, " ", text)

    # Tokenize
    words = text.split()

    # Remove stopwords and very short tokens
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]

    # Stem
    stemmed = [STEMMER.stem(w) for w in words]

    return " ".join(stemmed)



