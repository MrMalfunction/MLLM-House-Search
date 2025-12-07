# House Search Application

This application provides an interactive semantic search interface for house listings using Pinecone vector database. It allows users to query houses using natural language and retrieves the most relevant results based on semantic similarity.

## Overview

The application performs cloud-based semantic search against a Pinecone vector database, enabling fast and scalable similarity searches across house listings.

## Prerequisites

- Python 3.12+
- Pinecone API key
- Required packages (install via `pip install -r requirements.txt`):
  - sentence-transformers
  - pinecone-client
  - nltk

## Configuration

### Pinecone API Key

Before using the application, set your Pinecone API key in `app/main.py`:

```python
PINECONE_API_KEY = "your-actual-api-key"
```

Alternatively, you can set it as an environment variable and modify the code to read from it.

## Usage

### Interactive Mode

Search using a Pinecone index in interactive mode:

```bash
python app/main.py --pinecone_index <index_name>
```

Example:
```bash
python app/main.py --pinecone_index house-embeddings
```

This will start an interactive session where you can enter multiple queries. Type 'exit' or 'quit' to end the session.

### Single Query Mode

Execute a single search without entering interactive mode:

```bash
python app/main.py --pinecone_index <index_name> --query "3 bedroom house with pool"
```

Example:
```bash
python app/main.py --pinecone_index house-embeddings --query "affordable house near downtown"
```

## Command-Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--pinecone_index` | Yes | Name of Pinecone index to search | None |
| `--model_name` | No | SentenceTransformer model to use | `sentence-transformers/all-MiniLM-L6-v2` |
| `--top_k` | No | Number of results to return | 10 |
| `--query` | No | Single query for non-interactive mode | None |

## Example Queries

- "3 bedroom house with modern kitchen"
- "affordable house near good schools"
- "spacious family home with backyard"
- "luxury condo with city views"
- "house with garage and large living room"
- "pet friendly apartment with balcony"

## Search Process

1. User enters a natural language query
2. Query is preprocessed using the same pipeline as training data:
   - Lowercasing
   - HTML tag and emoji removal
   - Stopword removal
   - Stemming
3. Preprocessed query is converted to an embedding vector using the SentenceTransformer model
4. Cosine similarity search is performed against the Pinecone index
5. Top K most similar houses are returned with metadata

## Output Format

Each search result includes:

- **Rank**: Position in the results (1-N)
- **House ID**: Unique identifier for the property
- **Similarity Score**: Cosine similarity score (0-1, higher indicates better match)
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Area**: Property size in square feet
- **Zipcode**: Property location
- **Price**: Listing price in USD
- **Description**: Excerpt from the property description (first 200 characters)

## Notes

- The application uses the same preprocessing pipeline as the embedding generation pipeline to ensure consistency
- Queries are preprocessed identically to the training data for accurate semantic matching
- The Pinecone index must be created and populated using the `text_to_embedding_pipeline` before running this application
- Results are ranked purely by semantic similarity, not by numerical features like price or area
- The default model (`all-MiniLM-L6-v2`) generates 384-dimensional embeddings

## Troubleshooting

**Index not found error:**
- Ensure the Pinecone index name is correct
- Verify that the index has been created and populated using the embedding pipeline

**API key error:**
- Verify your Pinecone API key is set correctly in the code
- Check that your Pinecone account is active and has access to the specified index

**No results returned:**
- Try different search queries with more common terms
- Ensure the Pinecone index contains data
- Verify the embedding model matches the one used during index creation
