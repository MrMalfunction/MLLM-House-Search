import argparse
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import json



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
PINECONE_API_KEY = "PINECONE_API_KEY"  # Replace with your actual Pinecone API key
# Pinecone has a 2MB request size limit; stay slightly under it
MAX_REQUEST_BYTES = 1_900_000  

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

def pinecone_index_get(index_name: str, dimension: int, metric: str):
    api_key_pinecone = PINECONE_API_KEY

    if not api_key_pinecone:
        raise ValueError("Pinecone API key is not set. Please set the PINECONE_API_KEY variablebefore using Pinecone services.")
    pinecone = Pinecone(api_key=api_key_pinecone)
    if index_name not in [idx['name'] for idx in pinecone.list_indexes()]:
           pinecone.create_index(name=index_name, dimension=dimension, metric=metric, spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    index = pinecone.Index(index_name)      
    return index

def update_to_pinecone(index, df: pd.DataFrame, embeddings: np.ndarray, column_id: str="house_id", batch_size: int = 100):

    if len(df) != embeddings.shape[0]:
        raise ValueError("Number of rows in df and embeddings do not match")

    current_batch = []

    def flush_batch():
        nonlocal current_batch
        if not current_batch:
            return
        index.upsert(vectors=current_batch)
        current_batch = []

    for i, (_, row) in enumerate(df.iterrows()):
        vector_id = str(row[column_id])
        values = embeddings[i].tolist()

        metadata = {
            "house_id": str(row["house_id"]),
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": float(row["bathrooms"]),
            "area": float(row["area"]),
            "zipcode": str(row["zipcode"]),
            "price": float(row["price"]),
            "description": row.get("description", "") or "",
        }

        vec = {
            "id": vector_id,
            "values": values,
            "metadata": metadata,
        }

        current_batch.append(vec)

        # 1) limit #vectors per batch
        too_many_vectors = len(current_batch) >= batch_size

        # 2) keep JSON payload under ~1.9MB
        approx_bytes = len(json.dumps({"vectors": current_batch}).encode("utf-8"))
        too_large_bytes = approx_bytes > MAX_REQUEST_BYTES

        if too_many_vectors or too_large_bytes:
            # remove last if size blew up, flush previous batch, then start new with this vec
            if too_large_bytes and len(current_batch) > 1:
                last_vec = current_batch.pop()
                index.upsert(vectors=current_batch)
                current_batch = [last_vec]
            else:
                flush_batch()

    # flush any remaining vectors
    flush_batch()
        
def pinecone_house_search(index, model, query: str, top_k: int = 10):
    # Preprocess and embed the query the same way as training text
    processed_query = preprocess_text(query)
    query_embedding = model.encode([processed_query])[0].tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    print(f"\nTop {top_k} matching houses (Pinecone):")
    for rank, match in enumerate(results["matches"], 1):
        md = match["metadata"]
        score = match["score"]
        print(f"#{rank}: House ID {md['house_id']} | Score: {score:.4f}")
        print(
            f"   {md['bedrooms']} bd, {md['bathrooms']} ba, "
            f"{md['area']} sqft, zipcode {md['zipcode']}, price ${md['price']}"
        )
        print(f"   {md.get('description', '')}")
        print()


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
    parser.add_argument(
        "--pinecone_index",
        default=None,
        help="If provided, update embeddings into this Pinecone index",
    )
    parser.add_argument(
        "--pinecone_metric",
        default="cosine",
        help="Pinecone similarity metric (cosine, dotproduct, or euclidean)",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    ensure_nltk_resources()
    # Run pipeline and get processed data
    df = load_data(args.input_csv)
    df = add_text_for_embeddings(df)
    df = add_processed_text(df)
    model = load_embedding_model(args.model_name)
    texts_to_embed = df[args.text_column]
    embeddings = compute_embeddings(texts_to_embed, model=model, batch_size=64)
    save_outputs(df, embeddings, args.output_dir)
    index = None  # Ensure index is always defined

    if args.pinecone_index:
        print(f"\nUpserting {len(df)} vectors into Pinecone index '{args.pinecone_index}'...")
        index = pinecone_index_get(
            index_name=args.pinecone_index,
            dimension=embeddings.shape[1],
            metric=args.pinecone_metric,
        )
        update_to_pinecone(index, df, embeddings, batch_size=100)
        print("Upsert completed.")

    # --- User query and semantic retrieval ---
    print("\n--- Semantic House Search ---")
    user_query = input("Enter your house search query: ")
    processed_query = preprocess_text(user_query)
    query_embedding = model.encode([processed_query])[0]

    if index is not None:
        pinecone_house_search(index, model, user_query, top_k=10)
    else:
        # Fallback: local cosine similarity search
        processed_query = preprocess_text(user_query)
        query_embedding = model.encode([processed_query])[0]

        def cosine_similarity(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarities = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
        top_indices = np.argsort(similarities)[::-1][:10]

        print("\nTop 10 matching houses (local search):")
        for rank, idx in enumerate(top_indices, 1):
            house = df.iloc[idx]
            print(f"#{rank}: House ID {house['house_id']} | Score: {similarities[idx]:.4f}")
            print(f"   {house['description']}")
            print()


