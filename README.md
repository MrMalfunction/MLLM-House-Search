# 🏠 MLLM House Search

A semantic search system for real estate properties using vision-language models and embeddings.

**[Try the Live Demo →](https://29-house-search.streamlit.app/)**

## What is this?

Search for houses using natural language. Instead of filtering by price ranges and bedroom counts, describe what you want: *"Modern 3-bedroom house with a large kitchen and backyard"* — and get semantically similar properties ranked by relevance.

This project demonstrates how to build a semantic search system by:
1. Converting house images to descriptions using Qwen-VL
2. Creating semantic embeddings with Sentence Transformers
3. Searching via Pinecone vector database

## Features

✨ **Natural language search** — Describe what you want in plain English  
🖼️ **Vision-language model** — Understands house images, not just metadata  
⚡ **Fast semantic search** — Vector embeddings for instant results  
🌐 **Web interface** — Clean Streamlit app for easy exploration  
📊 **535+ properties** — 2,140 images in the dataset  

## Quick Start

### Use the Web App (Easiest)

Visit [29-house-search.streamlit.app](https://29-house-search.streamlit.app/) and start searching immediately.

### Run Locally

**1. Clone & Setup**
```bash
git clone https://github.com/MrMalfunction/MLLM-House-Search.git
cd MLLM-House-Search
pip install -e .
```

**2. Configure**
```bash
# Create .env file
echo "PINECONE_API_KEY=your_key_here" > .env
echo "PINECONE_INDEX=house-embeddings" >> .env
```

Get a free Pinecone API key from [pinecone.io](https://www.pinecone.io)

**3. Search**
```bash
# Web interface
streamlit run app/streamlit_app.py

# Or interactive CLI
python app/main.py --pinecone_index house-embeddings
```

## How It Works

```
House Images
    ↓
[Qwen-VL] → Generate descriptions
    ↓
[Sentence-Transformers] → Create embeddings
    ↓
[Pinecone] → Store vectors
    ↓
User Query → Embed → Search → Results
```

### The Pipeline

1. **Image-to-Text**: Qwen-VL analyzes 4 images per house (bathroom, bedroom, kitchen, frontal) and generates descriptions
2. **Embeddings**: Sentences are converted to 384-dimensional vectors using all-MiniLM-L6-v2
3. **Vector DB**: Embeddings + metadata stored in Pinecone for fast similarity search
4. **Search**: User queries are embedded and matched against stored vectors

## Project Structure

```
├── app/                              # Search application
│   ├── main.py                       # CLI interface
│   ├── streamlit_app.py              # Web app
│   └── settings/                     # Configuration
├── image_to_text_pipeline/           # Convert images → descriptions
├── text_to_embedding_pipeline/       # Convert text → embeddings
├── common/                           # Shared utilities
├── house_image_pipeline.py           # Associate images with metadata
└── README.md
```

## Full Pipeline (Advanced)

For processing raw data from scratch:

```bash
# 1. Associate images with house metadata
python house_image_pipeline.py \
    --dataset-path "path/to/Houses Dataset" \
    --output associations.json

# 2. Generate descriptions from images
python image_to_text_pipeline/main.py \
    --input associations.json \
    --output descriptions.parquet \
    --num-workers 2

# 3. Create embeddings and upload to Pinecone
python text_to_embedding_pipeline/main.py \
    --input_csv descriptions.csv \
    --output_dir embeddings/ \
    --pinecone_index house-embeddings

# 4. Search!
python app/main.py --pinecone_index house-embeddings
```

## Requirements

- Python 3.12+
- Pinecone account (free tier available)
- GPU recommended (but not required)
- 16GB RAM for full pipeline

## Dependencies

**Core:**
- `sentence-transformers` — Semantic embeddings
- `pinecone` — Vector database
- `streamlit` — Web interface
- `python-dotenv` — Configuration

**Pipeline (optional):**
- `torch` — Deep learning
- `transformers` — Vision-language models
- `qwen-vl-utils` — Model utilities
- `pandas` — Data processing

## Dataset

Uses the [Houses Dataset](https://github.com/emanhamed/Houses-dataset) by Eman Hamed:
- 535 houses
- 2,140 images (4 per house)
- Metadata: bedrooms, bathrooms, area, zipcode, price

## Example Search Results

**Query:** "Modern 3-bedroom house with spacious kitchen"

```
#1: House ID 42 | Similarity: 0.8234
    3 bed, 2 bath, 2500 sqft, $450,000
    Modern kitchen with stainless steel appliances...

#2: House ID 127 | Similarity: 0.8012
    3 bed, 2.5 bath, 2800 sqft, $520,000
    Contemporary design with updated kitchen...
```

## Architecture Highlights

- **Multi-GPU Support**: Parallel image processing with worker threads
- **Batch Processing**: Efficient batching for embeddings with size limits
- **Resilient Pipeline**: Handles missing images and malformed data gracefully
- **Cached Loading**: Session-based caching in Streamlit for fast reloads
- **Metadata Filtering**: Optional filtering by price, bedrooms, bathrooms, etc.

## Development

```bash
# Install dev tools
pip install -e ".[dev]"

# Format code
black .

# Lint
ruff check .

# Type check
mypy .

# Run tests
pytest
```

Code quality checks run automatically on commit via pre-commit hooks.

## Troubleshooting

**Pinecone connection error?**  
Check that `.env` file has valid `PINECONE_API_KEY` and the index exists.

**CUDA/GPU not detected?**  
Install CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Out of memory during image processing?**  
Reduce workers: `--num-workers 1` and batch size: `--batch-size 50`

**Model download timeout?**  
Pre-download: `huggingface-cli download Qwen/Qwen2-VL-7B-Instruct`

## Configuration

Edit `.env` to customize:

```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=house-embeddings
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_TOP_K=10
```

## License

MIT License — see [LICENSE](LICENSE) file.

The Houses Dataset is subject to its own terms. See [Houses Dataset repository](https://github.com/emanhamed/Houses-dataset).

## Acknowledgments

- Qwen Team for the VLM model
- Hugging Face for transformers
- Pinecone for vector DB
- Eman Hamed for the Houses Dataset

## Questions?

Open an issue on GitHub or check the [documentation](docs/) for more details.

---

**Ready to search?** [Try it now →](https://29-house-search.streamlit.app/)