import sys
import os
import json
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------------------------------------------------
# Disable torch.compile completely - this is the issue
# ------------------------------------------------------------------
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # Use flashinfer instead

import torch
from PIL import Image
from vllm import LLM, SamplingParams

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

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def process_house(llm, sampling_params, house_data, base_path=""):
    """Process a single house and return the result"""
    house_id = house_data["house_id"]
    images_paths = house_data["images"]

    # Load images
    try:
        frontal_img = load_image(os.path.join(base_path, images_paths["frontal"]))
        kitchen_img = load_image(os.path.join(base_path, images_paths["kitchen"]))
        bedroom_img = load_image(os.path.join(base_path, images_paths["bedroom"]))
        bathroom_img = load_image(os.path.join(base_path, images_paths["bathroom"]))
    except FileNotFoundError as e:
        print(f"Error loading images for house {house_id}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading images for house {house_id}: {e}")
        return None

    # Prepare prompt
    single = "<|vision_start|><|image_pad|><|vision_end|>"
    vision_placeholders = single + single + single + single
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{vision_placeholders}\n"
        f"Analyze the property using the provided guidelines.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    request = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": [frontal_img, kitchen_img, bedroom_img, bathroom_img],
        },
    }

    return request, house_data

def process_batch(llm, sampling_params, batch_data, base_path=""):
    """Process a batch of houses for better GPU utilization"""
    requests = []
    house_metadata = []

    for house_data in batch_data:
        result = process_house(llm, sampling_params, house_data, base_path)
        if result:
            request, metadata = result
            requests.append(request)
            house_metadata.append(metadata)

    if not requests:
        return []

    # Batch inference for better GPU utilization
    try:
        outputs = llm.generate(requests, sampling_params=sampling_params)
    except Exception as e:
        print(f"Error in batch generation: {e}")
        return []

    # Collect results
    results = []
    for idx, (output, metadata) in enumerate(zip(outputs, house_metadata)):
        try:
            description = output.outputs[0].text
            result = {
                "house_id": metadata["house_id"],
                "bedrooms": metadata["metadata"].get("bedrooms"),
                "bathrooms": metadata["metadata"].get("bathrooms"),
                "area": metadata["metadata"].get("area"),
                "zipcode": metadata["metadata"].get("zipcode"),
                "price": metadata["metadata"].get("price"),
                "frontal_image": metadata["images"]["frontal"],
                "kitchen_image": metadata["images"]["kitchen"],
                "bedroom_image": metadata["images"]["bedroom"],
                "bathroom_image": metadata["images"]["bathroom"],
                "description": description,
                "processed_at": datetime.now().isoformat()
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing output for house {metadata['house_id']}: {e}")
            continue

    return results

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

    # CUDA and PyTorch performance settings - optimized for V100
    torch.set_default_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    # Initialize vLLM - disable compilation entirely
    print("Initializing model for V100 PCIe 32GB (compilation disabled for stability)...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=10240,  # Slightly reduced
        gpu_memory_utilization=0.92,  # Slightly conservative
        dtype="float16",
        max_num_seqs=4,
        enforce_eager=True,  # CRITICAL: No compilation
        disable_custom_all_reduce=False,
    )

    sampling = SamplingParams(
        max_tokens=500,
        temperature=0.1,
        top_p=0.9,
        skip_special_tokens=True,
        use_beam_search=False,
    )

    # Process houses in batches
    base_path = os.path.dirname(json_input_path) if os.path.dirname(json_input_path) else ""
    BATCH_SIZE = 4  # Process 4 houses at once

    total_to_process = len(houses_to_process)
    successful = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"STARTING BATCH PROCESSING")
    print(f"Estimated time: ~{total_to_process * 0.75:.1f} minutes")
    print(f"{'='*60}\n")

    # Process in batches
    for batch_idx in range(0, total_to_process, BATCH_SIZE):
        batch = houses_to_process[batch_idx:batch_idx + BATCH_SIZE]
        batch_end = min(batch_idx + BATCH_SIZE, total_to_process)

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx//BATCH_SIZE + 1}/{(total_to_process + BATCH_SIZE - 1)//BATCH_SIZE}")
        print(f"Houses {batch_idx + 1}-{batch_end} of {total_to_process}")
        print(f"House IDs: {[h['house_id'] for h in batch]}")

        batch_start_time = datetime.now()

        try:
            # Process entire batch
            results = process_batch(llm, sampling, batch, base_path)

            # Save each result immediately
            for result in results:
                try:
                    append_to_parquet(result, output_file)
                    successful += 1
                    print(f"✓ House {result['house_id']}: {result['description'][:80]}...")
                except Exception as e:
                    print(f"✗ Error saving house {result['house_id']}: {e}")
                    failed += 1

            # Count failures in batch
            failed_in_batch = len(batch) - len(results)
            if failed_in_batch > 0:
                failed += failed_in_batch
                print(f"✗ {failed_in_batch} houses failed in this batch")

        except Exception as e:
            failed += len(batch)
            print(f"✗ Batch processing error: {e}")

        # Print batch statistics
        batch_time = (datetime.now() - batch_start_time).total_seconds()
        remaining = total_to_process - batch_end
        time_per_house = batch_time / len(batch) if len(batch) > 0 else 0
        eta_minutes = (remaining * time_per_house) / 60

        print(f"\nBatch completed in {batch_time:.1f}s ({time_per_house:.1f}s per house)")
        print(f"Progress: {successful} successful, {failed} failed, {remaining} remaining")
        print(f"ETA: {eta_minutes:.1f} minutes")
        print(f"Throughput: {successful/batch_end*100:.1f}% success rate")

    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total houses in dataset: {len(houses_data)}")
    print(f"Already processed (skipped): {len(processed_ids)}")
    print(f"Newly processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Output file: {output_file}")

    # Show final parquet file info
    if os.path.exists(output_file):
        df_final = pd.read_parquet(output_file)
        print(f"Total records in output file: {len(df_final)}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
