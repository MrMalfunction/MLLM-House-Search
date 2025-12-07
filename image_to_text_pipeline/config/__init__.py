"""
Configuration module for house description generation pipeline.
Centralizes all constants and settings for easy tuning and maintenance.
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_PATH = "./models/qwen3-vl-8b"
ATTN_IMPLEMENTATION = "eager"

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
MAX_NEW_TOKENS = 1200
MIN_NEW_TOKENS = 0
REPETITION_PENALTY = 1.15
TEMPERATURE = 1.0
DO_SAMPLE = False
NUM_BEAMS = 1

# Stopping criteria
STOP_SEQUENCES = ["###"]

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================
DEFAULT_BATCH_SIZE = 10
DEFAULT_FLUSH_INTERVAL = 30
DEFAULT_NUM_WORKERS = 2
DEFAULT_BASE_PATH = "../data"

# Output settings
DEFAULT_OUTPUT_FILE = "house_descriptions.parquet"

# Image types to process
IMAGE_TYPES = ["frontal", "kitchen", "bedroom", "bathroom"]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_system_prompt():
    """
    Load the system prompt from prompt.txt file.

    Returns:
        str: System prompt text, or empty string if file not found
    """
    prompt_path = os.path.join(os.path.dirname(__file__), "../prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path) as f:
            return f.read().strip()
    return ""
