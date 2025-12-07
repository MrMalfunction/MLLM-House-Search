import argparse
import sys
from pathlib import Path

from pinecone import Pinecone  # type: ignore
from sentence_transformers import SentenceTransformer

# Add parent directory to path for common package import
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.settings import settings
from common import ensure_nltk_resources, preprocess_text

# ---------- Pinecone Search ----------


def search_pinecone(index, model: SentenceTransformer, query: str, top_k: int = 10):
    """Search for houses in Pinecone using semantic similarity"""
    # Preprocess and embed the query the same way as training text
    processed_query = preprocess_text(query)
    query_embedding = model.encode([processed_query])[0].tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    print(f"\nTop {top_k} matching houses:")
    print("=" * 80)

    if not results["matches"]:
        print("No results found.")
        return

    for rank, match in enumerate(results["matches"], 1):
        md = match["metadata"]
        score = match["score"]
        print(f"\n#{rank}: House ID {md['house_id']} | Similarity Score: {score:.4f}")
        print(
            f"   Bedrooms: {md['bedrooms']} | Bathrooms: {md['bathrooms']} | Area: {md['area']} sqft"
        )
        print(f"   Zipcode: {md['zipcode']}")
        print(f"   Price: ${md['price']:,.2f}")
        print(f"   Description: {md.get('description', 'No description available')[:200]}...")
    print("=" * 80)


# ---------- Initialization ----------


def load_model(model_name: str) -> SentenceTransformer:
    """Load the SentenceTransformer model"""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    return model


def connect_to_pinecone(index_name: str):
    """Connect to an existing Pinecone index"""
    api_key = settings.pinecone_api_key

    if not api_key:
        raise ValueError(
            "Pinecone API key is not set. Please set the PINECONE_API_KEY environment variable."
        )

    pinecone = Pinecone(api_key=api_key)

    # Check if index exists
    index_names = [idx["name"] for idx in pinecone.list_indexes()]
    if index_name not in index_names:
        raise ValueError(
            f"Pinecone index '{index_name}' does not exist. Available indexes: {index_names}"
        )

    print(f"Connected to Pinecone index: {index_name}")
    return pinecone.Index(index_name)


# ---------- Interactive Search ----------


def interactive_search(model: SentenceTransformer, pinecone_index, top_k: int = 10):
    """Interactive search loop for house queries"""
    print("\n" + "=" * 80)
    print("HOUSE SEMANTIC SEARCH - PINECONE")
    print("=" * 80)
    print("\nEnter your house search queries below.")
    print("Type 'exit' or 'quit' to end the session.\n")
    print(f"Results per query: {top_k}")
    print()

    while True:
        try:
            query = input("Enter your search query: ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using House Semantic Search. Goodbye!")
                break

            if not query:
                print("Please enter a valid query.\n")
                continue

            # Perform search
            search_pinecone(pinecone_index, model, query, top_k=top_k)
            print()

        except KeyboardInterrupt:
            print("\n\nSearch interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError during search: {e}")
            print("Please try again.\n")


# ---------- CLI ----------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive house search using Pinecone semantic similarity"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        default=settings.model_name,
        help="SentenceTransformer model name",
    )

    # Pinecone configuration
    parser.add_argument(
        "--pinecone_index",
        default=settings.pinecone_index_name,
        help="Pinecone index name for search",
    )

    # Search configuration
    parser.add_argument(
        "--top_k",
        type=int,
        default=settings.default_top_k,
        help="Number of top results to return",
    )

    # Single query mode (non-interactive)
    parser.add_argument(
        "--query",
        default=None,
        help="Single query to search (non-interactive mode)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize NLTK resources
    ensure_nltk_resources()

    # Load embedding model
    model = load_model(args.model_name)

    # Connect to Pinecone
    pinecone_index = connect_to_pinecone(args.pinecone_index)

    # Single query mode or interactive mode
    if args.query:
        # Single query mode
        search_pinecone(pinecone_index, model, args.query, top_k=args.top_k)
    else:
        # Interactive mode
        interactive_search(model=model, pinecone_index=pinecone_index, top_k=args.top_k)


if __name__ == "__main__":
    main()
