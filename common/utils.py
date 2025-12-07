import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------- Configuration ----------

HTML_TAG_REGEX = re.compile(r"<[^>]+>")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001fa70-\U0001faff"
    "\U00002600-\U000026ff"
    "\U00002b50"
    "]+",
    flags=re.UNICODE,
)

PATTERN = r"[^a-zA-Z0-9]+"

STOP_WORDS = None
STEMMER = PorterStemmer()


# ---------- NLTK Setup ----------


def ensure_nltk_resources():
    """Check if NLTK stopwords are downloaded; if not, download them"""
    global STOP_WORDS
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        STOP_WORDS = set(stopwords.words("english"))


# ---------- Text Preprocessing ----------


def preprocess_text(text: str) -> str:
    """Clean and preprocess text for embedding generation"""
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
    if STOP_WORDS is not None:
        words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    else:
        words = [w for w in words if len(w) > 1]

    # Stem
    stemmed = [STEMMER.stem(w) for w in words]

    return " ".join(stemmed)
