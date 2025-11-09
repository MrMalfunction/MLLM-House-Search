# House Image Association Pipeline

A Python pipeline to associate each house with its respective images from the [Houses Dataset](https://github.com/emanhamed/Houses-dataset/tree/master/Houses%20Dataset).

## Overview

This pipeline processes the Houses dataset which contains:
- **535 houses** with metadata (bedrooms, bathrooms, area, zipcode, price)
- **2,140 images** (4 images per house: bathroom, bedroom, kitchen, frontal view)

The pipeline associates each house with its corresponding images and outputs a structured JSON file.

## Dataset Structure

The dataset should be organized as follows:

```
Houses Dataset/
├── HousesInfo.txt          # Metadata file (one line per house)
├── 1_bathroom.jpg          # Images named as: {house_id}_{room_type}.jpg
├── 1_bedroom.jpg
├── 1_kitchen.jpg
├── 1_frontal.jpg
├── 2_bathroom.jpg
└── ...
```

### Metadata Format

Each line in `HousesInfo.txt` contains:
```
bedrooms bathrooms area zipcode price
```

Example:
```
4 4 4053 85255 869500
```

### Image Naming Convention

Images are named as: `{house_id}_{room_type}.jpg`

Where:
- `house_id`: House identifier (1-535)
- `room_type`: One of `bathroom`, `bedroom`, `kitchen`, `frontal`

## Installation

No external dependencies required! The pipeline uses only Python standard library.

```bash
# Python 3.7+ required (for dataclasses support)
python3 --version
```

## Usage

### Basic Usage

```bash
python3 house_image_pipeline.py --dataset-path "path/to/Houses Dataset" --output output.json
```

### With Statistics

```bash
python3 house_image_pipeline.py \
    --dataset-path "temp_dataset/Houses Dataset" \
    --output house_image_associations.json \
    --stats
```

### Command Line Arguments

- `--dataset-path`: Path to the 'Houses Dataset' directory (default: `temp_dataset/Houses Dataset`)
- `--output`: Output JSON file path (default: `house_image_associations.json`)
- `--stats`: Print statistics about the dataset

## Output Format

The pipeline generates a JSON file with the following structure:

```json
[
  {
    "house_id": 1,
    "metadata": {
      "house_id": 1,
      "bedrooms": 4.0,
      "bathrooms": 4.0,
      "area": 4053,
      "zipcode": 85255,
      "price": 869500
    },
    "images": {
      "bathroom": "path/to/1_bathroom.jpg",
      "bedroom": "path/to/1_bedroom.jpg",
      "kitchen": "path/to/1_kitchen.jpg",
      "frontal": "path/to/1_frontal.jpg"
    }
  },
  ...
]
```

## Programmatic Usage

You can also use the pipeline programmatically:

```python
from house_image_pipeline import HouseImagePipeline

# Initialize pipeline
pipeline = HouseImagePipeline("path/to/Houses Dataset")

# Associate houses with images
houses = pipeline.associate_houses_with_images()

# Save to JSON
pipeline.save_to_json(houses, "output.json")

# Get statistics
stats = pipeline.get_statistics(houses)
print(stats)
```

## Example Output

When run with `--stats`, the pipeline prints:

```
Reading metadata...
Found metadata for 535 houses
Scanning images...
Found images for 535 houses
Associating houses with images...
Successfully associated 535 houses with their images
Saved associations to house_image_associations.json

=== Dataset Statistics ===
total_houses: 535
houses_with_all_images: 535
houses_with_some_images: 535
houses_with_no_images: 0
image_counts_by_type: {'bathroom': 535, 'bedroom': 535, 'kitchen': 535, 'frontal': 535}
total_images: 2140
```

## Features

- ✅ Automatically parses image filenames to extract house ID and room type
- ✅ Reads and parses metadata from HousesInfo.txt
- ✅ Associates each house with its 4 images
- ✅ Handles missing images gracefully
- ✅ Generates structured JSON output
- ✅ Provides statistics about the dataset
- ✅ No external dependencies

## Dataset Source

The dataset is available at: https://github.com/emanhamed/Houses-dataset/tree/master/Houses%20Dataset

## License

This pipeline is provided as-is. Please refer to the original dataset repository for dataset licensing information.

