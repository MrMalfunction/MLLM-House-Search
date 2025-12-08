import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pinecone import Pinecone, ServerlessSpec  # type: ignore
from sentence_transformers import SentenceTransformer

from common import ensure_nltk_resources, preprocess_text

# ---------- Configuration ----------

PINECONE_API_KEY = ""  # Replace with your actual Pinecone API key
MAX_REQUEST_BYTES = 1_900_000  # Pinecone has a 2MB request size limit


# ---------- Data Loading & Processing ----------


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV data and print basic info"""
    df = pd.read_csv(csv_path)
    print(f"Loaded data from {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numerical data and remove invalid values"""
    print("\n--- Cleaning Data ---")
    
    # Check for missing values in critical numerical fields
    critical_numeric_cols = ['price', 'area', 'bedrooms', 'bathrooms']
    missing_counts = df[critical_numeric_cols].isnull().sum()
    if missing_counts.any():
        print("Missing values found:")
        print(missing_counts[missing_counts > 0])
    
    # Remove invalid values
    before_invalid = len(df)
    df = df[
        (df['price'] > 0) & 
        (df['area'] > 0) & 
        (df['bedrooms'] >= 0) &  # 0 bedrooms could be studio
        (df['bathrooms'] > 0)
    ]
    removed_invalid = before_invalid - len(df)
    print(f"Removed {removed_invalid} rows with invalid values (price≤0, area≤0, bathrooms≤0)")
    
    return df

def add_text_for_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Create a text_for_embeddings column by combining relevant fields"""
    missing_cols = [
        col
        for col in [
            "house_id",
            "bedrooms",
            "bathrooms",
            "area",
            "zipcode",
            "price",
            "short_description",
            "frontal_description",
            "kitchen_description",
            "bedroom_description",
            "bathroom_description",
        ]
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
        + df["short_description"].fillna("")
        + ". "
        + df["frontal_description"].fillna("")
        + ". "
        + df["kitchen_description"].fillna("")
        + ". "
        + df["bedroom_description"].fillna("")
        + ". "
        + df["bathroom_description"].fillna("")
    )
    return df


def add_processed_text(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the text_for_embeddings column"""
    if "text_for_embeddings" not in df.columns:
        raise ValueError("text_for_embeddings column not found; run add_text_for_embeddings first.")

    df["processed_textembeddings"] = df["text_for_embeddings"].apply(preprocess_text)
    return df


# ---------- Embedding Generation ----------


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the SentenceTransformer model from HuggingFace"""
    model = SentenceTransformer(model_name)
    return model


def embed_single_text_with_sliding_window(
    text: str,
    model: SentenceTransformer,
    max_tokens: int = 256,
    stride: int = 128,
    pool: str = "mean",
) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        embedding_dim = model.get_sentence_embedding_dimension()
        if embedding_dim is None:
            embedding_dim = 768  # Default embedding dimension
        return np.zeros(embedding_dim, dtype=np.float32)

    tokenizer = model.tokenizer

    try:
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)

        # If text fits within token limit, process directly
        if len(tokens) <= max_tokens:
            with torch.no_grad():
                embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
                return embedding.cpu().numpy().astype(np.float32)

        # Text is too long - use sliding window approach
        slices = []
        for i in range(0, len(tokens), stride):
            slice_tokens = tokens[i : i + max_tokens]
            slice_text = tokenizer.decode(slice_tokens, skip_special_tokens=True)
            slices.append(slice_text)

        # Encode all slices
        with torch.no_grad():
            slice_embeddings = model.encode(
                slices, convert_to_tensor=True, show_progress_bar=False, batch_size=8
            )

        # Pool embeddings from all slices
        if pool == "mean":
            final_embedding = slice_embeddings.mean(dim=0)
        elif pool == "max":
            final_embedding = slice_embeddings.max(dim=0).values
        else:
            raise ValueError(f"Invalid pool method: {pool}. Use 'mean' or 'max'.")

        return final_embedding.cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"Warning: Error embedding text: {str(e)}")
        embedding_dim = model.get_sentence_embedding_dimension()
        if embedding_dim is None:
            embedding_dim = 768  # Default embedding dimension
        return np.zeros(embedding_dim, dtype=np.float32)


def compute_embeddings(
    texts: pd.Series,
    model: SentenceTransformer,
    max_tokens: int = 256,
    stride: int = 128,
    pool: str = "mean",
) -> np.ndarray:
    """Compute embeddings for a series of texts"""
    # Replace NaNs with empty strings
    texts_list = texts.fillna("").tolist()
    tokenizer = model.tokenizer
    token_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts_list]
    n_long = sum(1 for length in token_lengths if length > max_tokens)
    print(f"Number of texts longer than {max_tokens} tokens: {n_long}/{len(texts_list)}")
    embeddings = []
    for _idx, text in enumerate(texts_list):
        embedding = embed_single_text_with_sliding_window(
            text=text,
            model=model,
            max_tokens=max_tokens,
            stride=stride,
            pool=pool,
        )
        embeddings.append(embedding)
    return np.array(embeddings, dtype=np.float32)


def save_outputs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: str,
    csv_name: str = "houses_processed.csv",
    embeddings_name: str = "house_embeddings.npy",
):
    """Save processed DataFrame and embeddings to disk"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / csv_name
    npy_path = output_path / embeddings_name

    df.to_csv(str(csv_path), index=False)
    np.save(str(npy_path), embeddings)

    print(f"Saved processed CSV to: {csv_path}")
    print(f"Saved embeddings to: {npy_path}")


# ---------- Pinecone Integration ----------


def pinecone_index_get(index_name: str, dimension: int, metric: str):
    """Initialize or get existing Pinecone index"""
    api_key_pinecone = PINECONE_API_KEY

    if not api_key_pinecone:
        raise ValueError(
            "Pinecone API key is not set. Please set the PINECONE_API_KEY variable before using Pinecone services."
        )

    pinecone = Pinecone(api_key=api_key_pinecone)

    if index_name not in [idx["name"] for idx in pinecone.list_indexes()]:
        print(f"Creating new Pinecone index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Using existing Pinecone index: {index_name}")

    index = pinecone.Index(index_name)
    return index


def update_to_pinecone(
    index,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    column_id: str = "house_id",
    batch_size: int = 100,
):
    """Upload embeddings with metadata to Pinecone index"""
    if len(df) != embeddings.shape[0]:
        raise ValueError("Number of rows in df and embeddings do not match")

    current_batch = []
    total_upserted = 0

    def flush_batch():
        nonlocal current_batch, total_upserted
        if not current_batch:
            return
        index.upsert(vectors=current_batch)
        total_upserted += len(current_batch)
        print(f"Upserted {total_upserted}/{len(df)} vectors...")
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
            "short_description": row.get("short_description", "") or "",
            "frontal_description": row.get("frontal_description", "") or "",
            "kitchen_description": row.get("kitchen_description", "") or "",
            "bedroom_description": row.get("bedroom_description", "") or "",
            "bathroom_description": row.get("bathroom_description", "") or "",
        }

        vec = {
            "id": vector_id,
            "values": values,
            "metadata": metadata,
        }

        current_batch.append(vec)

        # Limit #vectors per batch
        too_many_vectors = len(current_batch) >= batch_size

        # Keep JSON payload under ~1.9MB
        approx_bytes = len(json.dumps({"vectors": current_batch}).encode("utf-8"))
        too_large_bytes = approx_bytes > MAX_REQUEST_BYTES

        if too_many_vectors or too_large_bytes:
            # Remove last if size blew up, flush previous batch, then start new with this vec
            if too_large_bytes and len(current_batch) > 1:
                last_vec = current_batch.pop()
                index.upsert(vectors=current_batch)
                total_upserted += len(current_batch)
                print(f"Upserted {total_upserted}/{len(df)} vectors...")
                current_batch = [last_vec]
            else:
                flush_batch()

    # Flush any remaining vectors
    flush_batch()
    print(f"Total vectors upserted: {total_upserted}")


# ---------- Main Pipeline ----------


def run_pipeline(
    input_csv: str,
    output_dir: str,
    model_name: str,
    text_column: str = "processed_textembeddings",
    pinecone_index: str | None = None,
    pinecone_metric: str = "cosine",
):
    """Main pipeline: CSV -> Text Processing -> Embeddings -> Storage"""

    print("=" * 60)
    print("TEXT-TO-EMBEDDING-TO-PINECONE PIPELINE")
    print("=" * 60)

    # Step 1: Initialize NLTK resources
    print("\n[1/6] Initializing NLTK resources...")
    ensure_nltk_resources()

    # Step 2: Load data
    print("\n[2/6] Loading data...")
    df = load_data(input_csv)
    df = clean_data(df)

    # Step 3: Create text for embeddings
    print("\n[3/6] Creating text for embeddings...")
    df = add_text_for_embeddings(df)
    df = add_processed_text(df)

    # Step 4: Load embedding model
    print(f"\n[4/6] Loading embedding model: {model_name}...")
    model = load_embedding_model(model_name)

    # Step 5: Compute embeddings
    print(f"\n[5/6] Computing embeddings for column: {text_column}...")
    texts_to_embed: pd.Series = df[text_column]  # type: ignore
    embeddings = compute_embeddings(
        texts_to_embed, model=model, max_tokens=256, stride=128, pool="mean"
    )
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Step 6: Save outputs
    print(f"\n[6/6] Saving outputs to: {output_dir}...")
    save_outputs(df, embeddings, output_dir)

    # Upload to Pinecone
    if pinecone_index:
        print(f"\n[OPTIONAL] Uploading to Pinecone index: {pinecone_index}...")
        index = pinecone_index_get(
            index_name=pinecone_index,
            dimension=embeddings.shape[1],
            metric=pinecone_metric,
        )
        update_to_pinecone(index, df, embeddings, batch_size=100)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# ---------- CLI ----------


def parse_args():
    parser = argparse.ArgumentParser(
        description="House description text-to-embedding-to-Pinecone pipeline"
    )
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
        default="processed_textembeddings",
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

    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        text_column=args.text_column,
        pinecone_index=args.pinecone_index,
        pinecone_metric=args.pinecone_metric,
    )
