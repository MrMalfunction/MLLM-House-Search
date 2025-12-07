"""
Application settings configuration loaded from environment variables.
Handles configuration for Pinecone, models, and search parameters.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file in project root
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_index_name: str

    # Model Configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Search Configuration
    default_top_k: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables.

        Raises:
            ValueError: If required environment variables are not set.
        """
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment."
            )

        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "house-image-search")
        model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        default_top_k = int(os.getenv("DEFAULT_TOP_K", "10"))

        return cls(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            model_name=model_name,
            default_top_k=default_top_k,
        )


# Global settings instance
settings = Settings.from_env()
