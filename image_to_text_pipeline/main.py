import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import gc
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

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

class HouseDescriptionGenerator:
    def __init__(self, model_path=None, use_cache=True):
        self.model = None
        self.processor = None
        self.model_path = model_path or "./models/qwen3-vl-8b"
        self.use_cache = use_cache

    def initialize_model(self):
        """Initialize the model and processor"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing model...", flush=True)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {device_name}", flush=True)
        else:
            print("No GPU detected, using CPU", flush=True)

        # Determine dtype
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            use_bf16 = any(gpu in device_name for gpu in ["A100", "H100", "L4", "4090"])
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            print(f"Using dtype: {dtype}", flush=True)
        else:
            dtype = torch.float32

        # Load processor and model
        print(f"Loading from: {self.model_path}", flush=True)

        if not os.path.exists(self.model_path):
            print(f"Model not found locally. Downloading {MODEL_ID}...", flush=True)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=MODEL_ID, local_dir=self.model_path)

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Model loaded successfully on {next(self.model.parameters()).device}", flush=True)

        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB", flush=True)

    def load_image(self, path):
        """Load and convert image to RGB"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def generate_description(self, image_paths, max_tokens=512):
        """Generate description from multiple images"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        # Load images
        images = [self.load_image(path) for path in image_paths]

        # Construct messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]

        # Add images and query
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": "Analyze the property using the provided guidelines."})

        messages.append({"role": "user", "content": content})

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=self.use_cache,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        generation_time = time.time() - start_time

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text, generation_time

    def process_house(self, house_data, base_path=""):
        """Process a single house"""
        house_id = house_data["house_id"]

        try:
            # Build image paths
            image_paths = [
                os.path.join(base_path, house_data["images"]["frontal"]),
                os.path.join(base_path, house_data["images"]["kitchen"]),
                os.path.join(base_path, house_data["images"]["bedroom"]),
                os.path.join(base_path, house_data["images"]["bathroom"])
            ]

            # Generate description
            description, gen_time = self.generate_description(image_paths)

            # Build result
            result = {
                "house_id": house_id,
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
            print(f"Error processing house {house_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate descriptions for house images using Qwen3-VL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        default="house_image_associations.json",
        help="Path to input JSON file with house data"
    )
    parser.add_argument(
        "--output", "-o",
        default="house_descriptions.parquet",
        help="Path to output parquet file"
    )
    parser.add_argument(
        "--model-path", "-m",
        default="./models/qwen3-vl-8b",
        help="Path to store/load the model"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume processing (skip already processed houses)"
    )
    parser.add_argument(
        "--save-every", "-s",
        type=int,
        default=5,
        help="Save progress after processing this many houses"
    )

    args = parser.parse_args()

    print(f"Starting job at {datetime.now()}", flush=True)
    print(f"Arguments: {args}", flush=True)

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", flush=True)
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}", flush=True)

    # Load house data
    print(f"Loading house data from {args.input}...", flush=True)
    with open(args.input, 'r') as f:
        houses_data = json.load(f)
    print(f"Found {len(houses_data)} houses", flush=True)

    # Check for resume
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        try:
            print(f"Resume mode: Reading existing output file {args.output}...", flush=True)
            df = pd.read_parquet(args.output)
            processed_ids = set(df['house_id'].tolist())
            print(f"Resuming: {len(processed_ids)} houses already processed", flush=True)
            print(f"Processed house IDs: {sorted(list(processed_ids))[:10]}..." if len(processed_ids) > 10 else f"Processed house IDs: {sorted(list(processed_ids))}", flush=True)
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}", flush=True)
    elif args.resume:
        print(f"Resume mode enabled but output file does not exist yet", flush=True)

    houses_to_process = [
        h for h in houses_data
        if h['house_id'] not in processed_ids
    ]

    if not houses_to_process:
        print("All houses already processed!", flush=True)
        return

    print(f"Will process {len(houses_to_process)} houses", flush=True)
    print(f"First 10 houses to process: {[h['house_id'] for h in houses_to_process[:10]]}", flush=True)

    # Initialize generator
    generator = HouseDescriptionGenerator(model_path=args.model_path)
    generator.initialize_model()

    # Process houses
    base_path = os.path.dirname(args.input) if os.path.dirname(args.input) else "."
    results = []
    successful = 0
    failed = 0
    start_time = datetime.now()

    print(f"\n{'='*60}", flush=True)
    print("STARTING PROCESSING", flush=True)
    print(f"{'='*60}\n", flush=True)

    for idx, house_data in enumerate(houses_to_process):
        house_id = house_data['house_id']
        print(f"[{idx+1}/{len(houses_to_process)}] Processing house {house_id}...", flush=True)

        try:
            result = generator.process_house(house_data, base_path)

            if result:
                results.append(result)
                successful += 1
                print(f"  ✓ Success ({result['generation_time_seconds']:.2f}s)", flush=True)
                print(f"  Description: {result['description'][:100]}...", flush=True)

                # Save after every single house
                try:
                    df_new = pd.DataFrame(results)
                    if os.path.exists(args.output):
                        df_existing = pd.read_parquet(args.output)
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        df_combined.to_parquet(args.output, index=False)
                        print(f"  💾 Saved to {args.output} (Total: {len(df_combined)} houses)", flush=True)
                    else:
                        df_new.to_parquet(args.output, index=False)
                        print(f"  💾 Saved to {args.output} (Total: {len(df_new)} houses)", flush=True)
                    results = []
                except Exception as save_error:
                    print(f"  ⚠️  Warning: Failed to save: {save_error}", flush=True)
                    traceback.print_exc()
                    sys.stdout.flush()
            else:
                failed += 1
                print(f"  ✗ Failed", flush=True)

        except Exception as e:
            failed += 1
            print(f"  ✗ Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

        # Progress stats
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_time = elapsed / (idx + 1)
        remaining = len(houses_to_process) - (idx + 1)
        eta_minutes = (remaining * avg_time) / 60

        print(f"  Progress: {successful} ok, {failed} failed, {remaining} left (ETA: {eta_minutes:.1f}m)\n", flush=True)

    # Save any remaining results (should be empty now since we save after each house)
    if results:
        df_new = pd.DataFrame(results)
        if os.path.exists(args.output):
            df_existing = pd.read_parquet(args.output)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(args.output, index=False)
        else:
            df_new.to_parquet(args.output, index=False)
        print(f"  💾 Saved final batch to {args.output}", flush=True)

    # Final summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}", flush=True)
    print("PROCESSING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total processed: {successful}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Total time: {total_time/60:.1f} minutes", flush=True)
    print(f"Average time per house: {total_time/successful:.1f} seconds" if successful > 0 else "", flush=True)
    print(f"Output: {args.output}", flush=True)

    if os.path.exists(args.output):
        df_final = pd.read_parquet(args.output)
        print(f"Final file contains: {len(df_final)} records", flush=True)
        print(f"House IDs in final file: {sorted(df_final['house_id'].tolist())[:10]}..." if len(df_final) > 10 else f"House IDs in final file: {sorted(df_final['house_id'].tolist())}", flush=True)


if __name__ == "__main__":
    main()
