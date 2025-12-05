"""
House Image Description Generator using Qwen3-VL Model

This module processes house images and generates detailed descriptions using a vision-language model.
Supports parallel processing with multiple GPUs and batched writing to parquet files.
"""

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
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, StoppingCriteria, StoppingCriteriaList
from qwen_vl_utils import process_vision_info
import multiprocessing as mp
from queue import Empty
import threading
import traceback
import re

# Import our custom parser
from parser import parse_delimited_output

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_prompt():
    """Load the prompt template from prompt.txt file"""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    with open(prompt_path, 'r') as f:
        return f.read().strip()


SYSTEM_PROMPT = load_prompt()


class TextStoppingCriteria(StoppingCriteria):
    """Stop generation when specific text sequences appear in the output"""

    def __init__(self, tokenizer, stop_sequences, initial_length):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.initial_length = initial_length

    def __call__(self, input_ids, scores, **kwargs):  # type: ignore
        # Decode only newly generated tokens
        generated_ids = input_ids[0][self.initial_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Check if any stop sequence appears
        for stop_seq in self.stop_sequences:
            if stop_seq in generated_text:
                return True
        return False


class HouseDescriptionGenerator:
    """Main class for generating house descriptions from images"""

    def __init__(self, model_path=None, worker_id=0, gpu_id=0):
        self.model = None
        self.processor = None
        self.model_path = model_path or "./models/qwen3-vl-8b"
        self.worker_id = worker_id
        self.gpu_id = gpu_id

    def initialize_model(self):
        """Initialize the VLM model and processor"""
        print(f"[Worker {self.worker_id}] Initializing model on GPU {self.gpu_id}...", flush=True)

        # Set GPU device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            torch.cuda.empty_cache()
            gc.collect()
            device_name = torch.cuda.get_device_name(self.gpu_id)
            print(f"[Worker {self.worker_id}] Using GPU: {device_name}", flush=True)
        else:
            print(f"[Worker {self.worker_id}] No GPU available, using CPU", flush=True)

        # Determine dtype based on GPU capability
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(self.gpu_id)
            use_bf16 = any(gpu in device_name for gpu in ["A100", "H100", "L4", "4090"])
            dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            dtype = torch.float32

        # Download model if needed
        if not os.path.exists(self.model_path):
            print(f"[Worker {self.worker_id}] Downloading model {MODEL_ID}...", flush=True)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=MODEL_ID, local_dir=self.model_path)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load model to assigned GPU
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map={"": self.gpu_id},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[Worker {self.worker_id}] Model loaded successfully", flush=True)

    def load_image(self, path):
        """Load and convert image to RGB"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def generate_description(self, image_paths, metadata, max_tokens=1500):
        """Generate description from multiple images"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized")

        # Load images
        images = [self.load_image(path) for path in image_paths]

        # Build messages for the model
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]

        # Add images to user message
        content = []
        for img in images:
            content.append({"type": "image", "image": img})

        # Add metadata text
        metadata_str = f"""Property Metadata:
- Bedrooms: {metadata.get('bedrooms', 'N/A')}
- Bathrooms: {metadata.get('bathrooms', 'N/A')}
- Area: {metadata.get('area', 'N/A')}
- Zipcode: {metadata.get('zipcode', 'N/A')}
- Price: {metadata.get('price', 'N/A')}

Analyze the property images and provide the description."""

        content.append({"type": "text", "text": metadata_str})
        messages.append({"role": "user", "content": content})

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_info = process_vision_info(messages)

        # Handle different return formats from process_vision_info
        if len(vision_info) == 3:
            image_inputs, video_inputs, _ = vision_info
        else:
            image_inputs, video_inputs = vision_info

        # Move to correct device
        device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Setup stopping criteria - only use ### as stop word
        stop_sequences = ["###"]
        initial_length = inputs.input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList([
            TextStoppingCriteria(self.processor.tokenizer, stop_sequences, initial_length)
        ])

        # Generate with greedy decoding and strict limits to prevent rambling
        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2000,
                min_new_tokens=100,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.2,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        generation_time = time.time() - start_time

        # Decode output
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
        """
        Process a single house and return raw output.
        Parsing is done by the writer thread to free GPU worker faster.
        """
        house_id = house_data["house_id"]

        try:
            # Build image paths
            image_paths = []
            for img_type in ["frontal", "kitchen", "bedroom", "bathroom"]:
                img_path = house_data["images"][img_type]
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                else:
                    full_path = os.path.join(base_path, img_path)
                    image_paths.append(full_path)

            # Validate images exist
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")

            # Prepare metadata
            metadata = {
                "bedrooms": house_data["metadata"].get("bedrooms"),
                "bathrooms": house_data["metadata"].get("bathrooms"),
                "area": house_data["metadata"].get("area"),
                "zipcode": house_data["metadata"].get("zipcode"),
                "price": house_data["metadata"].get("price")
            }

            # Generate description
            raw_output, gen_time = self.generate_description(image_paths, metadata)

            # Return result with raw output only (no parsing here - done by writer)
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
                "raw_output": raw_output,
                "generation_time_seconds": gen_time,
                "processed_at": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error processing house {house_id}: {e}", flush=True)
            traceback.print_exc()
            return None


def worker_process(worker_id, work_queue, result_queue, model_path, base_path, stop_event, gpu_id=0):
    """Worker process that processes houses from the queue"""
    try:
        print(f"[Worker {worker_id}] Starting on GPU {gpu_id}", flush=True)

        # Initialize generator
        generator = HouseDescriptionGenerator(model_path=model_path, worker_id=worker_id, gpu_id=gpu_id)
        generator.initialize_model()

        print(f"[Worker {worker_id}] Ready", flush=True)

        processed_count = 0
        while not stop_event.is_set():
            try:
                house_data = work_queue.get(timeout=1)

                if house_data is None:  # Stop signal
                    break

                house_id = house_data['house_id']
                print(f"[Worker {worker_id}] Processing house {house_id}...", flush=True)

                result = generator.process_house(house_data, base_path)

                if result:
                    result_queue.put(('success', result))
                    processed_count += 1
                    print(f"[Worker {worker_id}] ✓ Completed {house_id} ({result['generation_time_seconds']:.2f}s)", flush=True)
                else:
                    result_queue.put(('failed', house_id))

            except Empty:
                continue
            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}", flush=True)
                traceback.print_exc()

        print(f"[Worker {worker_id}] Shutting down. Processed {processed_count} houses.", flush=True)

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}", flush=True)
        traceback.print_exc()


def result_writer_thread(result_queue, output_file, stop_event, batch_size=10, flush_interval=30):
    """
    Writer thread that parses results and writes to parquet in batches.
    Parsing is done here to free up GPU workers faster.
    """
    successful = 0
    failed = 0
    buffer = []
    last_flush_time = time.time()

    def flush_buffer():
        """Write buffered results to disk"""
        nonlocal successful
        if not buffer:
            return

        try:
            df_new = pd.DataFrame(buffer)
            if os.path.exists(output_file):
                df_existing = pd.read_parquet(output_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_parquet(output_file, index=False)
                total = len(df_combined)
            else:
                df_new.to_parquet(output_file, index=False)
                total = len(df_new)

            print(f"[Writer] 💾 Saved {len(buffer)} houses (Total: {total})", flush=True)
            buffer.clear()
        except Exception as e:
            print(f"[Writer] ⚠️ Failed to flush: {e}", flush=True)
            traceback.print_exc()

    while not stop_event.is_set():
        try:
            result_type, result_data = result_queue.get(timeout=1)

            if result_type == 'stop':
                flush_buffer()
                break
            elif result_type == 'success':
                # Parse raw output into structured fields (done here, not in GPU worker)
                raw_output = result_data.get('raw_output', '')
                parsed = parse_delimited_output(raw_output)

                # Add parsed fields to result
                result_data['short_description'] = parsed['short_description']
                result_data['frontal_description'] = parsed['frontal_description']
                result_data['kitchen_description'] = parsed['kitchen_description']
                result_data['bedroom_description'] = parsed['bedroom_description']
                result_data['bathroom_description'] = parsed['bathroom_description']

                buffer.append(result_data)
                successful += 1
                print(f"[Writer] 📥 Parsed & buffered house {result_data['house_id']} ({len(buffer)} buffered)", flush=True)

                # Flush if buffer full or time elapsed
                if len(buffer) >= batch_size or (time.time() - last_flush_time) >= flush_interval:
                    flush_buffer()
                    last_flush_time = time.time()

            elif result_type == 'failed':
                failed += 1
                print(f"[Writer] ❌ Failed house {result_data}", flush=True)

        except Empty:
            # Periodic flush even if not full
            if buffer and (time.time() - last_flush_time) >= flush_interval:
                flush_buffer()
                last_flush_time = time.time()
            continue
        except Exception as e:
            print(f"[Writer] Error: {e}", flush=True)
            traceback.print_exc()

    # Final flush
    if buffer:
        print(f"[Writer] Final flush of {len(buffer)} houses", flush=True)
        flush_buffer()

    print(f"[Writer] Done. Saved {successful}, failed {failed}", flush=True)
    return successful, failed


def test_single_house(input_file, model_path, base_path, house_id=None):
    """Test mode: process one house and display output"""
    print(f"\n{'='*60}")
    print("TEST MODE")
    print(f"{'='*60}\n")

    # Load data
    with open(input_file, 'r') as f:
        houses_data = json.load(f)

    # Select house
    if house_id:
        house_data = next((h for h in houses_data if h['house_id'] == house_id), None)
        if not house_data:
            print(f"Error: House ID {house_id} not found")
            return
    else:
        house_data = random.choice(houses_data)

    print(f"Testing house: {house_data['house_id']}")
    print(f"Bedrooms: {house_data['metadata'].get('bedrooms')}")
    print(f"Bathrooms: {house_data['metadata'].get('bathrooms')}")
    print(f"Area: {house_data['metadata'].get('area')}")
    print(f"Price: {house_data['metadata'].get('price')}\n")

    # Initialize and process
    generator = HouseDescriptionGenerator(model_path=model_path, worker_id=0, gpu_id=0)
    generator.initialize_model()

    print("Processing...\n")
    start_time = time.time()
    result = generator.process_house(house_data, base_path)
    processing_time = time.time() - start_time

    if result:
        print(f"\n{'='*60}")
        print("RAW OUTPUT")
        print(f"{'='*60}\n")
        print(result['raw_output'])

        print(f"\n{'='*60}")
        print("PARSED OUTPUT")
        print(f"{'='*60}\n")

        parsed = parse_delimited_output(result['raw_output'])
        print(f"Short Description:\n{parsed['short_description']}\n")
        print(f"Frontal:\n{parsed['frontal_description'][:200]}...\n")
        print(f"Kitchen:\n{parsed['kitchen_description'][:200]}...\n")
        print(f"Bedroom:\n{parsed['bedroom_description'][:200]}...\n")
        print(f"Bathroom:\n{parsed['bathroom_description'][:200]}...\n")

        print(f"\nProcessing time: {processing_time:.2f}s")
        print(f"{'='*60}\n")
    else:
        print("Error: Processing failed")


def main():
    parser = argparse.ArgumentParser(description="Generate house descriptions from images")
    parser.add_argument("--input", "-i", default="house_image_associations.json", help="Input JSON file")
    parser.add_argument("--output", "-o", default="house_descriptions.parquet", help="Output parquet file")
    parser.add_argument("--model-path", "-m", default="./models/qwen3-vl-8b", help="Model path")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from existing output")
    parser.add_argument("--num-workers", "-w", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--base-path", "-b", default="../data", help="Base path for images")
    parser.add_argument("--test", "-t", action="store_true", help="Test mode (single house)")
    parser.add_argument("--test-house-id", type=str, default=None, help="Specific house ID for test")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for writing")
    parser.add_argument("--flush-interval", type=int, default=30, help="Flush interval in seconds")

    args = parser.parse_args()
    base_path = os.path.abspath(args.base_path)

    # Test mode
    if args.test:
        test_single_house(args.input, args.model_path, base_path, args.test_house_id)
        return

    print(f"Starting job at {datetime.now()}")
    print(f"Workers: {args.num_workers}")

    # Detect GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPUs detected: {num_gpus}")
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load houses
    with open(args.input, 'r') as f:
        houses_data = json.load(f)
    print(f"Total houses: {len(houses_data)}")

    # Handle resume
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        df_temp = pd.read_parquet(args.output)
        processed_ids = set(df_temp['house_id'].tolist())
        print(f"Already processed: {len(processed_ids)}")

    houses_to_process = [h for h in houses_data if h['house_id'] not in processed_ids]

    if not houses_to_process:
        print("All houses already processed!")
        return

    print(f"To process: {len(houses_to_process)}")

    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    work_queue = mp.Queue(maxsize=args.num_workers * 2)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    # Start workers
    workers = []
    for worker_id in range(args.num_workers):
        assigned_gpu = worker_id % num_gpus if num_gpus > 0 else 0
        p = mp.Process(
            target=worker_process,
            args=(worker_id, work_queue, result_queue, args.model_path, base_path, stop_event, assigned_gpu)
        )
        p.start()
        workers.append(p)
        print(f"Started worker {worker_id} on GPU {assigned_gpu}")

    # Start writer
    writer_thread = threading.Thread(
        target=result_writer_thread,
        args=(result_queue, args.output, stop_event, args.batch_size, args.flush_interval)
    )
    writer_thread.start()

    print(f"\n{'='*60}")
    print("PROCESSING")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    # Queue work
    try:
        for house_data in houses_to_process:
            work_queue.put(house_data)

        # Send stop signals
        for _ in range(args.num_workers):
            work_queue.put(None)

        # Wait for completion
        for i, worker in enumerate(workers):
            worker.join()
            print(f"Worker {i} finished")

        result_queue.put(('stop', None))
        writer_thread.join()

    except KeyboardInterrupt:
        print("\nInterrupted! Stopping...")
        stop_event.set()
        for worker in workers:
            worker.terminate()
            worker.join()
        writer_thread.join()

    total_time = (datetime.now() - start_time).total_seconds()

    # Summary
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"Output: {args.output}")

    if os.path.exists(args.output):
        df_final = pd.read_parquet(args.output)
        print(f"Total records: {len(df_final)}")
        print(f"Newly processed: {len(df_final) - len(processed_ids)}")


if __name__ == "__main__":
    main()
