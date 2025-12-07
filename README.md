# MLLM House Search

A semantic search system for real estate properties using vision-language models. This project enables natural language-based house discovery through automated image analysis and vector embeddings.

## Overview

The system processes house images and metadata through a three-stage pipeline:

1. **Image-to-Text Pipeline**: Generates descriptions from house images using Qwen-VL vision-language models
2. **Text-to-Embedding Pipeline**: Converts descriptions to semantic embeddings using sentence-transformers
3. **Search Application**: Provides semantic search capabilities via Pinecone vector database

Users query the system using natural language, retrieving the most semantically similar properties.

## Requirements

- Python 3.12+
- Pinecone API key (https://www.pinecone.io)
- NVIDIA GPU with CUDA (recommended for image processing)
- 50GB+ disk space
- 16GB+ RAM

## Installation

### Clone Repository

```bash
git clone https://github.com/MrMalfunction/MLLM-House-Search.git
cd MLLM-House-Search
```

### Install Dependencies

**Using UV (recommended):**

```bash
uv sync
```

**Using pip:**

```bash
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_api_key
PINECONE_INDEX=house-embeddings
```

Retrieve your API key from https://app.pinecone.io.

## Data Format

The dataset must follow this structure:

```
Houses Dataset/
├── HousesInfo.txt
├── 1_bathroom.jpg
├── 1_bedroom.jpg
├── 1_kitchen.jpg
├── 1_frontal.jpg
├── 2_bathroom.jpg
└── ...
```

### Metadata File

`HousesInfo.txt` contains one house per line with space-separated values:

```
bedrooms bathrooms area zipcode price
4 2 3000 85255 550000
3 1.5 2000 85256 400000
```

### Image Naming

Images are named as: `{house_id}_{room_type}.jpg`

Room types: `bathroom`, `bedroom`, `kitchen`, `frontal`

## Usage

### Step 1: Generate Image Descriptions

```bash
python image_to_text_pipeline/main.py \
    --dataset-path "path/to/Houses Dataset" \
    --output descriptions.json
```

Generates text descriptions from house images using Qwen-VL.

### Step 2: Create Embeddings

```bash
python text_to_embedding_pipeline/main.py \
    --input descriptions.json \
    --output embeddings.json
```

Converts descriptions to embeddings and uploads to Pinecone.

### Step 3: Search Houses

**Interactive mode:**

```bash
python app/main.py --pinecone_index house-embeddings
```

**Single query:**

```bash
python app/main.py --pinecone_index house-embeddings --query "3 bedroom house with modern kitchen"
```

**Web interface (optional):**

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
MLLM-House-Search/
├── image_to_text_pipeline/      # Image analysis pipeline
│   ├── main.py
│   └── core/
├── text_to_embedding_pipeline/  # Embedding generation
│   ├── main.py
│   └── examples/
├── app/                         # Search application
│   ├── main.py
│   ├── streamlit_app.py
│   └── settings/
├── common/                      # Shared utilities
├── batch_jobs/                  # Batch processing
├── house_image_pipeline.py      # Image association
├── pyproject.toml               # Dependencies
├── .pre-commit-config.yaml      # Code quality
└── .env                         # Configuration
```

## Architecture

### Image-to-Text Pipeline

Processes house images using Qwen-VL to generate room-specific descriptions.

**Input**: House images (bathroom, bedroom, kitchen, frontal)
**Output**: JSON file with text descriptions

### Text-to-Embedding Pipeline

Converts descriptions to semantic embeddings using sentence-transformers.

**Input**: Descriptions from image-to-text pipeline
**Output**: Embeddings uploaded to Pinecone index

### Search Application

Retrieves semantically similar houses from the Pinecone index.

**Input**: Natural language query
**Output**: Ranked list of matching properties

## Development

### Install Development Tools

```bash
pip install -e ".[dev]"
pre-commit install
```

### Run Tests

```bash
pytest
```

### Code Quality

Automatic checks on commit:

```bash
pre-commit run --all-files
```

## Dependencies

### Core

- `sentence-transformers`: Embedding generation
- `pinecone`: Vector database
- `python-dotenv`: Environment configuration
- `Pillow`: Image processing

### Pipeline

- `torch`, `torchvision`: Deep learning framework
- `transformers`: Vision-language models
- `qwen-vl-utils`: Model utilities
- `pandas`, `pyarrow`: Data processing

### Development

- `pytest`: Testing
- `ruff`: Linting
- `black`: Code formatting
- `pre-commit`: Git hooks

## Dataset

Designed for the [Houses Dataset](https://github.com/emanhamed/Houses-dataset):
- 535 houses
- 2,140 images (4 per house)
- Metadata: bedrooms, bathrooms, area, zipcode, price

## License

This project is licensed under the MIT License. See LICENSE file for details.

The Houses Dataset used in this project is subject to its own licensing terms. Refer to the [Houses Dataset repository](https://github.com/emanhamed/Houses-dataset) for licensing information.