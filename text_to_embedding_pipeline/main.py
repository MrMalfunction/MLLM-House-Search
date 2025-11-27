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

# ---------- Data / embedding pipeline ----------

# This function loads the CSV data and prints basic info
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded data from {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

# Create a text_for_embeddings column by combining relevant fields
def add_text_for_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [
        col
        for col in ["house_id", "bedrooms", "bathrooms", "area", "zipcode", "price", "description"]
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns for text_for_embeddings: {missing_cols}")

    df["text_for_embeddings"] = (
        "House ID: "
        + df["house_id"].astype(str)
        + ". "
        + df["bedrooms"].astype(str)
        + " bedrooms, "
        + df["bathrooms"].astype(str)
        + " bathrooms, "
        + df["area"].astype(str)
        + " sqft, "
        + "zipcode "
        + df["zipcode"].astype(str)
        + ", "
        + "price $"
        + df["price"].astype(str)
        + ". "
        + df["description"].fillna("")
    )
    return df

# Preprocess the text_for_embeddings column
def add_processed_text(df: pd.DataFrame) -> pd.DataFrame:
    if "text_for_embeddings" not in df.columns:
        raise ValueError("text_for_embeddings column not found; run add_text_for_embeddings first.")
    
    df["processed_textembeddings"] = df["text_for_embeddings"].apply(preprocess_text)
    return df

# Load the SentenceTransformer model from HuggingFace
def load_embedding_model(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model

# Compute embeddings
def compute_embeddings(
    texts: pd.Series,
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> np.ndarray:
    # Replace NaNs with empty strings
    texts_list = texts.fillna("").tolist()
    embeddings = model.encode(
        texts_list,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    return embeddings

# Save processed DataFrame and embeddings to disk
def save_outputs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: str,
    csv_name: str = "houses_processed.csv",
    embeddings_name: str = "house_embeddings.npy",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / csv_name
    npy_path = output_dir / embeddings_name

    df.to_csv(csv_path, index=False)
    np.save(npy_path, embeddings)
# ---------- Orchestration / CLI ----------
# Main pipeline function
def run_pipeline(
    input_csv: str,
    output_dir: str,
    model_name: str,
    text_column: str = "text_for_embeddings",
):
  
    ensure_nltk_resources()

    df = load_data(input_csv)
    df = add_text_for_embeddings(df)
    df = add_processed_text(df)

    model = load_embedding_model(model_name)
    texts_to_embed = df[text_column]

    embeddings = compute_embeddings(texts_to_embed, model=model, batch_size=64)
    save_outputs(df, embeddings, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="House description embedding pipeline")
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV file (house_descriptions_2b etc.)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write processed CSV and embeddings",
    )
    parser.add_argument(
        "--model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--text_column",
        default="text_for_embeddings",
        help="Which column to embed (text_for_embeddings or processed_textembeddings)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        text_column=args.text_column,
    )


