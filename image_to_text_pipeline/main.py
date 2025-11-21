import sys
import os
import json
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.cuda
import gc
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_PATH = os.getenv("model_path", "/projects/scdatahub/amol_dmt/model/8b")

SYSTEM_PROMPT = """You are an expert Real Estate Appraiser and Architectural Analyst. You have been provided with a set of images representing a single residential property. Your goal is to generate a dense, keyword rich, and comprehensive description of this property for a database. This text will be converted into vector embeddings for a semantic search engine. If a feature is not visible, omit it. You are given 4 images of the same house in this order: 1) Frontal exterior 2) Kitchen 3) Bedroom 4) Bathroom Use all images together to infer overall property quality and details.

1. ARCHITECTURAL and EXTERIOR ANALYSIS
Architecture Style:
Roof:
Siding or Facade:
Garage or Parking:
Landscaping or Hardscaping:
Windows:

2. INTERIOR FINISHES and MATERIALS
Flooring:
Lighting:
Ceilings:

3. KITCHEN DETAILS
Cabinetry:
Countertops:
Appliances:
Layout or Features:

4. BATHROOM DETAILS
Vanity:
Tub or Shower:
Fixtures:
"""

# Global model and processor
model = None
processor = None

def initialize_model():
    """Initialize the model and processor once"""
    global model, processor

    print(f"[{datetime.now()}] Starting model initialization...")

    # Clear any existing GPU memory
    if torch.cuda.is_available():
        print(f"[{datetime.now()}] Clearing GPU cache...")
        torch.cuda.empty_cache()
        gc.collect()

    # Detect GPU and choose optimal dtype
    print(f"[{datetime.now()}] Detecting GPU...")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    use_bf16 = any(gpu in device_name for gpu in ["A100", "H100", "L4", "4090"])
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Device detected: {device_name}")
    print(f"Using dtype: {dtype} (bfloat16 optimized for A100/H100/L4, float16 for V100/T4)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    # Download model if needed
    print(f"[{datetime.now()}] Checking model path...")
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"Downloading {MODEL_ID} to {MODEL_PATH}...")
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_PATH)
        print("Download finished.")
    else:
        print(f"Model found at {MODEL_PATH}. Skipping download.")

    # Load processor first
    print(f"[{datetime.now()}] Loading processor from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[{datetime.now()}] Processor loaded successfully.")

    # Load model
    print(f"[{datetime.now()}] Loading model from {MODEL_PATH}...")
    print(f"[{datetime.now()}] This may take several minutes...")

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    print(f"[{datetime.now()}] Model weights loaded, moving to device...")

    # Set model to eval mode and disable gradients
    print(f"[{datetime.now()}] Setting model to eval mode...")
    model.eval()

    print(f"[{datetime.now()}] Disabling gradients...")
    for param in model.parameters():
        param.requires_grad = False

    print(f"[{datetime.now()}] Model loaded and configured successfully.")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # GPU memory info
    if torch.cuda.is_available():
        print(f"[{datetime.now()}] GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"[{datetime.now()}] GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # Optional: Warmup with a dummy forward pass
    print(f"[{datetime.now()}] Performing warmup pass...")
    try:
        with torch.inference_mode():
            # Create a small dummy tensor on the right device
            dummy_input = torch.randn(1, 3, 224, 224, device=model.device, dtype=dtype)
            print(f"[{datetime.now()}] Warmup complete.")
    except Exception as e:
        print(f"[{datetime.now()}] Warmup skipped (this is okay): {e}")

    return model, processor

def load_image(path):
    """Load and convert image to RGB"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def process_multiple_images(image_paths, system_prompt, user_query, max_tokens=512):
    """
    Process multiple images with system prompt and user query using transformers
    """
    global model, processor

    if model is None or processor is None:
        return "Error: Model or processor not loaded."

    # Load images
    print(f"[{datetime.now()}] Loading {len(image_paths)} images...")
    images = [load_image(path) for path in image_paths]

    # Construct messages with multiple images
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    # Build content with multiple images
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_query})

    messages.append({
        "role": "user",
        "content": content,
    })

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate response
    start_time = time.time()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    generation_time = time.time() - start_time

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0], generation_time

def process_house(house_data, base_path=""):
    """Process a single house and return the result"""
    house_id = house_data["house_id"]
    images_paths = house_data["images"]

    # Build full image paths
    try:
        image_files = [
            os.path.join(base_path, images_paths["frontal"]),
            os.path.join(base_path, images_paths["kitchen"]),
            os.path.join(base_path, images_paths["bedroom"]),
            os.path.join(base_path, images_paths["bathroom"])
        ]

        # Verify all images exist
        for img_path in image_files:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

    except FileNotFoundError as e:
        print(f"Error loading images for house {house_id}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading images for house {house_id}: {e}")
        return None

    # Process with model
    user_query = "Analyze the property using the provided guidelines."

    try:
        description, gen_time = process_multiple_images(
            image_files,
            SYSTEM_PROMPT,
            user_query,
        )

        result = {
            "house_id": house_data["house_id"],
            "bedrooms": house_data["metadata"].get("bedrooms"),
            "bathrooms": house_data["metadata"].get("bathrooms"),
            "area": house_data["metadata"].get("area"),
            "zipcode": house_data["metadata"].get("zipcode"),
            "price": house_data["metadata"].get("price"),
            "frontal_image": house_data["images"]["frontal"],
            "kitchen_image": house_data["images"]["kitchen"],
            "bedroom_image": house_data["images"]["bedroom"],
            "bathroom_image": house_data["images"]["bathroom"],
            "description": description,
            "generation_time_seconds": gen_time,
            "processed_at": datetime.now().isoformat()
        }

        return result

    except Exception as e:
        print(f"Error processing house {house_id}: {e}")
        return None

def get_processed_house_ids(parquet_file):
    """Get set of already processed house IDs from parquet file"""
    if not os.path.exists(parquet_file):
        return set()
    try:
        df = pd.read_parquet(parquet_file)
        return set(df['house_id'].tolist())
    except Exception as e:
        print(f"Warning: Could not read existing parquet file: {e}")
        return set()

def append_to_parquet(result, parquet_file):
    """Append a single result to parquet file"""
    df_new = pd.DataFrame([result])

    if os.path.exists(parquet_file):
        # Append to existing file
        df_existing = pd.read_parquet(parquet_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_parquet(parquet_file, index=False, engine='pyarrow')
    else:
        # Create new file
        df_new.to_parquet(parquet_file, index=False, engine='pyarrow')

def main():
    # Get paths from environment variables or use defaults
    json_input_path = os.getenv("json_input_path", "house_image_associations.json")
    output_dir = os.getenv("output_path", "/projects/scdatahub/amol_dmt")

    # Create output filename with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"house_descriptions_{timestamp}.parquet")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load house image associations
    print(f"Loading house data from {json_input_path}...")
    with open(json_input_path, 'r') as f:
        houses_data = json.load(f)
    print(f"Found {len(houses_data)} houses to process")

    # Check for already processed houses (resume capability)
    processed_ids = get_processed_house_ids(output_file)
    if processed_ids:
        print(f"Found {len(processed_ids)} already processed houses, resuming...")
        houses_to_process = [h for h in houses_data if h['house_id'] not in processed_ids]
    else:
        houses_to_process = houses_data

    print(f"Will process {len(houses_to_process)} houses")

    if len(houses_to_process) == 0:
        print("All houses already processed!")
        return

    # Initialize model once
    print(f"\n[{datetime.now()}] ===== INITIALIZING MODEL =====")
    initialize_model()
    print(f"[{datetime.now()}] ===== MODEL INITIALIZATION COMPLETE =====\n")

    # Process houses one by one
    base_path = os.path.dirname(json_input_path) if os.path.dirname(json_input_path) else ""

    total_to_process = len(houses_to_process)
    successful = 0
    failed = 0
    total_generation_time = 0

    print(f"\n{'='*60}")
    print(f"STARTING PROCESSING")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    for idx, house_data in enumerate(houses_to_process):
        house_id = house_data['house_id']
        print(f"\n{'='*60}")
        print(f"Processing house {idx + 1}/{total_to_process}")
        print(f"House ID: {house_id}")

        house_start_time = datetime.now()

        try:
            result = process_house(house_data, base_path)

            if result:
                append_to_parquet(result, output_file)
                successful += 1
                total_generation_time += result.get('generation_time_seconds', 0)
                print(f"✓ House {house_id}: {result['description'][:100]}...")
                print(f"  Generation time: {result.get('generation_time_seconds', 0):.2f}s")
            else:
                failed += 1
                print(f"✗ House {house_id}: Processing failed")

        except Exception as e:
            failed += 1
            print(f"✗ House {house_id}: Error - {e}")

        # Print progress statistics
        house_time = (datetime.now() - house_start_time).total_seconds()
        remaining = total_to_process - (idx + 1)
        avg_time_per_house = (datetime.now() - start_time).total_seconds() / (idx + 1)
        eta_minutes = (remaining * avg_time_per_house) / 60

        print(f"\nHouse processed in {house_time:.1f}s")
        print(f"Progress: {successful} successful, {failed} failed, {remaining} remaining")
        print(f"Average time per house: {avg_time_per_house:.1f}s")
        print(f"ETA: {eta_minutes:.1f} minutes")
        print(f"Success rate: {successful/(idx+1)*100:.1f}%")

    # Final summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total houses in dataset: {len(houses_data)}")
    print(f"Already processed (skipped): {len(processed_ids)}")
    print(f"Newly processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Average generation time: {total_generation_time/successful:.2f}s per house" if successful > 0 else "N/A")
    print(f"Output file: {output_file}")

    # Show final parquet file info
    if os.path.exists(output_file):
        df_final = pd.read_parquet(output_file)
        print(f"Final parquet file contains {len(df_final)} records")
        print(f"Columns: {list(df_final.columns)}")

if __name__ == "__main__":
    main()
