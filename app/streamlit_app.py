import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from app.settings import settings
from common import ensure_nltk_resources, preprocess_text
from text_to_embedding_pipeline.main import compute_embeddings

# Page configuration
st.set_page_config(
    page_title="House Search - Semantic Image Search",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #7F8C8D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stImage {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .stExpander {
        border: 1px solid #D0D0D0;
        border-radius: 8px;
    }
    .stExpander > div > div > div {
        color: #2C3E50;
    }
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #E8E8E8;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = None
if "house_data" not in st.session_state:
    st.session_state.house_data = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None


@st.cache_resource
def load_model(model_name: str):
    """Load the SentenceTransformer model (cached)"""
    ensure_nltk_resources()
    return SentenceTransformer(model_name)


@st.cache_resource
def connect_to_pinecone(index_name: str):
    """Connect to Pinecone index (cached)"""
    try:
        api_key = settings.pinecone_api_key
        pinecone = Pinecone(api_key=api_key)

        # Check if index exists
        index_names = [idx["name"] for idx in pinecone.list_indexes()]
        if index_name not in index_names:
            st.error(
                f"Pinecone index '{index_name}' does not exist. Available indexes: {index_names}"
            )
            return None

        return pinecone.Index(index_name)
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        return None


@st.cache_data
def load_house_data():
    """Load house metadata and image associations from JSON file"""
    # Try parent data directory first (for local development)
    json_path = Path(__file__).parent.parent / "data" / "house_image_associations.json"

    # If not found, try app/data directory (for deployment)
    if not json_path.exists():
        json_path = Path(__file__).parent / "data" / "house_image_associations.json"

    try:
        with open(json_path) as f:
            data = json.load(f)
        # Create a dictionary keyed by house_id for quick lookup
        house_dict = {item["house_id"]: item for item in data}
        return house_dict
    except Exception as e:
        st.error(f"Error loading house data: {e}")
        return {}


def search_pinecone(index, model, query: str, top_k: int = 10, filters=None):
    """Search for houses in Pinecone using semantic similarity with optional filters"""
    try:
        # Preprocess and embed the query
        processed_query = preprocess_text(query)
        query_embedding = compute_embeddings(pd.Series([processed_query]), model)[0].tolist()
        # Build query parameters
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
        }

        # Add filters if provided
        if filters:
            query_params["filter"] = filters

        # Search in Pinecone
        results = index.query(**query_params)

        # Convert to dictionary for easier handling
        return {"matches": results.matches if hasattr(results, "matches") else []}
    except Exception as e:
        st.error(f"Error during search: {e}")
        return None


def display_house_card(house_info, rank, score, pinecone_metadata=None):
    """Display a single house with its details and images"""
    metadata = house_info["metadata"]
    images = house_info.get("images", {})

    # Use Pinecone metadata for description if available
    if pinecone_metadata:
        metadata = {**metadata, **pinecone_metadata}

    with st.container():
        # Header with better formatting
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.markdown(f"### Rank #{rank} - House ID: {metadata['house_id']}")
        with col_header2:
            st.markdown(f"**Similarity:** {score:.4f}")

        # Display key metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Price", f"${metadata.get('price', 0):,.0f}")
        with col2:
            st.metric("Bedrooms", f"{metadata.get('bedrooms', 0):.0f}")
        with col3:
            st.metric("Bathrooms", f"{metadata.get('bathrooms', 0):.1f}")
        with col4:
            st.metric("Area (sqft)", f"{metadata.get('area', 0):,}")
        with col5:
            st.metric("Zipcode", metadata.get("zipcode", "N/A"))

        # Display description if available
        description = metadata.get("description", "")
        if description:
            with st.expander("View Full Description", expanded=False):
                st.markdown(description)

        # Display images in a grid
        st.markdown("#### Property Images")
        st.markdown("")  # Add spacing
        img_cols = st.columns(4)

        image_types = ["frontal", "bedroom", "bathroom", "kitchen"]
        image_labels = [" Frontal View", " Bedroom", " Bathroom", " Kitchen"]

        for idx, (img_type, label) in enumerate(zip(image_types, image_labels, strict=False)):
            img_path = images.get(img_type, "")
            if img_path:
                # Try parent data directory first (for local development)
                full_path = Path(__file__).parent.parent / "data" / img_path

                # If not found, try app/data directory (for deployment)
                if not full_path.exists():
                    full_path = Path(__file__).parent / "data" / img_path

                if full_path.exists():
                    try:
                        img = Image.open(full_path)
                        with img_cols[idx]:
                            st.image(img, caption=label, use_container_width=True)
                    except Exception:
                        with img_cols[idx]:
                            st.warning(f"{label}\n(Image load error)")
                else:
                    with img_cols[idx]:
                        st.info(f"{label}\n(Image not found)")

        st.markdown("---")


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">House Search - Semantic Image Search</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Search for your dream home using natural language descriptions!</p>',
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Number of results
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=min(settings.default_top_k, 20),
            help="Number of houses to retrieve",
        )

        st.markdown("---")
        st.header("Filters")

        # Price range filter
        price_filter = st.checkbox("Filter by Price Range")
        price_min, price_max = None, None
        if price_filter:
            col1, col2 = st.columns(2)
            with col1:
                price_min = st.number_input("Min Price ($)", min_value=0, value=0, step=50000)
            with col2:
                price_max = st.number_input("Max Price ($)", min_value=0, value=1000000, step=50000)

        # Bedrooms filter
        bedrooms_filter = st.checkbox("Filter by Bedrooms")
        bedrooms_min, bedrooms_max = None, None
        if bedrooms_filter:
            col1, col2 = st.columns(2)
            with col1:
                bedrooms_min = st.number_input("Min Bedrooms", min_value=0, value=1, step=1)
            with col2:
                bedrooms_max = st.number_input("Max Bedrooms", min_value=0, value=10, step=1)

        # Bathrooms filter
        bathrooms_filter = st.checkbox("Filter by Bathrooms")
        bathrooms_min, bathrooms_max = None, None
        if bathrooms_filter:
            col1, col2 = st.columns(2)
            with col1:
                bathrooms_min = st.number_input("Min Bathrooms", min_value=0.0, value=1.0, step=0.5)
            with col2:
                bathrooms_max = st.number_input(
                    "Max Bathrooms", min_value=0.0, value=10.0, step=0.5
                )

        # Area filter
        area_filter = st.checkbox("Filter by Area (sqft)")
        area_min, area_max = None, None
        if area_filter:
            col1, col2 = st.columns(2)
            with col1:
                area_min = st.number_input("Min Area", min_value=0, value=500, step=100)
            with col2:
                area_max = st.number_input("Max Area", min_value=0, value=10000, step=100)

        # Zipcode filter
        zipcode_filter = st.checkbox("Filter by Zipcode")
        zipcode = None
        if zipcode_filter:
            zipcode = st.text_input("Zipcode", placeholder="e.g., 85255")

        st.markdown("---")
        st.markdown("###Search Tips")
        st.markdown(
            """
        - Be specific about features you want
        - Describe architectural styles
        - Mention desired amenities
        - Use natural language
        """
        )

    # Initialize resources
    if st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_model(settings.model_name)

    if st.session_state.pinecone_index is None:
        with st.spinner("Connecting to database..."):
            st.session_state.pinecone_index = connect_to_pinecone(settings.pinecone_index_name)

    if st.session_state.house_data is None:
        with st.spinner("Loading house data..."):
            st.session_state.house_data = load_house_data()

    # Check if all resources loaded successfully
    if st.session_state.pinecone_index is None or st.session_state.house_data is None:
        st.error("Failed to initialize application. Please check your configuration.")
        return

    # Search interface
    st.markdown("###  Search for Houses")
    st.markdown("")  # Add spacing

    # Search input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Enter your search query",
            placeholder="e.g., Modern house with large backyard and updated kitchen",
            label_visibility="collapsed",
        )
    with col2:
        search_button = st.button(" Search", type="primary", use_container_width=True)

    # Build filters
    filters = {}
    if price_filter and price_min is not None and price_max is not None:
        filters["price"] = {"$gte": price_min, "$lte": price_max}
    if bedrooms_filter and bedrooms_min is not None and bedrooms_max is not None:
        filters["bedrooms"] = {"$gte": bedrooms_min, "$lte": bedrooms_max}
    if bathrooms_filter and bathrooms_min is not None and bathrooms_max is not None:
        filters["bathrooms"] = {"$gte": bathrooms_min, "$lte": bathrooms_max}
    if area_filter and area_min is not None and area_max is not None:
        filters["area"] = {"$gte": area_min, "$lte": area_max}
    if zipcode_filter and zipcode:
        try:
            filters["zipcode"] = {"$eq": int(zipcode)}
        except ValueError:
            st.sidebar.error("Invalid zipcode format")

    # Perform search
    if search_button and query:
        with st.spinner("Searching for matching houses..."):
            results = search_pinecone(
                st.session_state.pinecone_index,
                st.session_state.model,
                query,
                top_k=top_k,
                filters=filters,
            )

            if results and len(results.get("matches", [])) > 0:
                st.session_state.search_results = results
                st.success(f"Found {len(results['matches'])} matching houses!")
            else:
                st.warning("No results found. Try a different query or adjust filters.")
                st.session_state.search_results = None
    elif not search_button and query:
        # Auto-search on Enter key
        with st.spinner("Searching for matching houses..."):
            results = search_pinecone(
                st.session_state.pinecone_index,
                st.session_state.model,
                query,
                top_k=top_k,
                filters=filters,
            )

            if results and len(results.get("matches", [])) > 0:
                st.session_state.search_results = results

    # Display search results
    if (
        st.session_state.search_results
        and len(st.session_state.search_results.get("matches", [])) > 0
    ):
        st.markdown("---")
        st.markdown("##  Search Results")
        st.markdown("")  # Add spacing

        matches = st.session_state.search_results["matches"]

        for rank, match in enumerate(matches, 1):
            # Handle both dict and object attribute access
            match_metadata = (
                match.metadata if hasattr(match, "metadata") else match.get("metadata", {})
            )
            match_score = match.score if hasattr(match, "score") else match.get("score", 0.0)

            house_id = int(match_metadata.get("house_id", 0))
            score = match_score

            # Get full house info from loaded data
            house_info = st.session_state.house_data.get(house_id)

            if house_info:
                # Pass Pinecone metadata to display the description
                display_house_card(house_info, rank, score, pinecone_metadata=match_metadata)
            else:
                st.warning(f"House ID {house_id} not found in local data.")

    elif (
        st.session_state.search_results is not None
        and len(st.session_state.search_results.get("matches", [])) == 0
    ):
        st.markdown("---")
        st.info(" No results found. Try a different query or adjust your filters.")
    else:
        # Show welcome message when no search has been performed
        st.markdown("---")
        st.markdown("### Welcome!")
        st.markdown(
            """
        Start searching for your dream home by entering a **natural language description** above.

        Our AI-powered search understands what you're looking for and finds the best matching properties based on semantic similarity.
        """
        )

        # Show some example houses
        st.markdown("### Featured Properties")
        st.markdown("Here are some sample properties from our database:")
        st.markdown("")  # Add spacing
        sample_houses = list(st.session_state.house_data.values())[:3]

        for idx, house_info in enumerate(sample_houses, 1):
            display_house_card(house_info, idx, 1.0)


if __name__ == "__main__":
    main()
