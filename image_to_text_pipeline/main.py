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
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import multiprocessing as mp
from queue import Empty
import threading
import traceback

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Load prompt from file
def load_prompt():
    """Load the prompt from prompt.txt file"""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    with open(prompt_path, 'r') as f:
        return f.read().strip()

SYSTEM_PROMPT = load_prompt()

class HouseDescriptionGenerator:
    def __init__(self, model_path=None, use_cache=True, worker_id=0, gpu_id=0):
        self.model = None
        self.processor = None
        self.model_path = model_path or "./models/qwen3-vl-8b"
        self.use_cache = use_cache
        self.worker_id = worker_id
        self.gpu_id = gpu_id

    def initialize_model(self):
        """Initialize the model and processor"""
        print(f"[Worker {self.worker_id}][{datetime.now().strftime('%H:%M:%S')}] Initializing model...", flush=True)

        # Set the correct GPU device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            print(f"[Worker {self.worker_id}] Using GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}", flush=True)
            torch.cuda.empty_cache()
        else:
            print(f"[Worker {self.worker_id}] CUDA not available, using CPU", flush=True)

        try:
            # Initialize processor
            print(f"[Worker {self.worker_id}][{datetime.now().strftime('%H:%M:%S')}] Loading processor from {self.model_path}...", flush=True)
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            print(f"[Worker {self.worker_id}][{datetime.now().strftime('%H:%M:%S')}] Processor loaded", flush=True)

            # Initialize model
            print(f"[Worker {self.worker_id}][{datetime.now().strftime('%H:%M:%S')}] Loading model from {self.model_path}...", flush=True)
            device_map = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="flash_attention_2"
            )
            print(f"[Worker {self.worker_id}][{datetime.now().strftime('%H:%M:%S')}] Model loaded on {device_map}", flush=True)

            # Print memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
                reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024**3
                print(f"[Worker {self.worker_id}] GPU {self.gpu_id} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", flush=True)

            print(f"[Worker {self.worker_id}][{datetime.now().strftime('%H:%M:%S')}] Model initialization complete", flush=True)

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error initializing model: {e}", flush=True)
            traceback.print_exc()
            raise

    def load_image(self, image_path):
        """Load an image from file"""
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error loading image {image_path}: {e}", flush=True)
            raise

    def generate_description(self, image_path):
        """Generate a description for a single image"""
        try:
            # Load image
            image = self.load_image(image_path)

            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT
                        }
                    ]
                }
            ]

            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            # Move inputs to correct device
            device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            inputs = inputs.to(device)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            # Trim generated tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return output_text

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error generating description for {image_path}: {e}", flush=True)
            traceback.print_exc()
            raise

    def process_house(self, house_id, image_paths, base_path):
        """Process all images for a house and generate structured output"""
        try:
            start_time = time.time()

            descriptions = {}
            for image_name, relative_path in image_paths.items():
                # Construct full path
                full_path = os.path.join(base_path, relative_path)

                if not os.path.exists(full_path):
                    print(f"[Worker {self.worker_id}] Warning: Image not found: {full_path}", flush=True)
                    descriptions[image_name] = "ERROR: Image file not found"
                    continue

                # Generate description
                try:
                    description = self.generate_description(full_path)
                    descriptions[image_name] = description
                except Exception as e:
                    descriptions[image_name] = f"ERROR: {str(e)}"

            elapsed = time.time() - start_time

            # Structure output
            result = {
                'house_id': house_id,
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(elapsed, 2),
                'num_images': len(image_paths),
                **descriptions  # Unpack all descriptions as individual columns
            }

            print(f"[Worker {self.worker_id}] ✅ Completed house {house_id} in {elapsed:.1f}s", flush=True)

            return result

        except Exception as e:
            print(f"[Worker {self.worker_id}] ❌ Failed to process house {house_id}: {e}", flush=True)
            traceback.print_exc()
            raise


def worker_process(worker_id, work_queue, result_queue, model_path, base_path, stop_event, gpu_id):
    """Worker process that processes houses"""
    try:
        print(f"[Worker {worker_id}] Starting on GPU {gpu_id}", flush=True)

        # Initialize generator
        generator = HouseDescriptionGenerator(
            model_path=model_path,
            worker_id=worker_id,
            gpu_id=gpu_id
        )
        generator.initialize_model()

        processed = 0
        while not stop_event.is_set():
            try:
                # Get work from queue
                house_data = work_queue.get(timeout=1)

                if house_data is None:  # Poison pill
                    print(f"[Worker {worker_id}] Received stop signal", flush=True)
                    break

                house_id = house_data['house_id']
                image_paths = house_data['image_paths']

                print(f"[Worker {worker_id}] Processing house {house_id}...", flush=True)

                # Process the house
                result = generator.process_house(house_id, image_paths, base_path)
                result_queue.put(('success', result))
                processed += 1

            except Empty:
                continue
            except Exception as e:
                print(f"[Worker {worker_id}] Error processing house: {e}", flush=True)
                traceback.print_exc()
                if 'house_id' in locals():
                    result_queue.put(('failed', house_id))

        print(f"[Worker {worker_id}] Shutting down. Processed {processed} houses.", flush=True)

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}", flush=True)
        traceback.print_exc()


def result_writer_thread(result_queue, output_file, stop_event, batch_size=10, flush_interval=30):
    """Thread that writes results to file in batches for better performance"""
    successful = 0
    failed = 0
    buffer = []
    last_flush_time = time.time()

    def flush_buffer():
        """Flush the buffer to disk"""
        nonlocal buffer, successful
        if not buffer:
            return 0

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

            flushed_count = len(buffer)
            house_ids = [r['house_id'] for r in buffer]
            buffer.clear()
            print(f"[Writer] 💾 Flushed {flushed_count} houses to disk (Total: {total} houses in file)", flush=True)
            print(f"[Writer] 💾 Saved: {', '.join(house_ids)}", flush=True)
            return flushed_count
        except Exception as e:
            print(f"[Writer] ⚠️ Failed to flush buffer: {e}", flush=True)
            traceback.print_exc()
            return 0

    while not stop_event.is_set():
        try:
            result_type, result_data = result_queue.get(timeout=1)

            if result_type == 'stop':
                # Flush any remaining data before stopping
                flush_buffer()
                break
            elif result_type == 'success':
                buffer.append(result_data)
                successful += 1
                print(f"[Writer] 📥 Buffered house {result_data['house_id']} ({len(buffer)} in buffer, {successful} total)", flush=True)

                # Flush if buffer is full or enough time has passed
                current_time = time.time()
                if len(buffer) >= batch_size or (current_time - last_flush_time) >= flush_interval:
                    flush_buffer()
                    last_flush_time = current_time

            elif result_type == 'failed':
                failed += 1
                print(f"[Writer] ❌ Failed house {result_data}", flush=True)

        except Empty:
            # Flush buffer periodically even if not full
            current_time = time.time()
            if buffer and (current_time - last_flush_time) >= flush_interval:
                flush_buffer()
                last_flush_time = current_time
            continue
        except Exception as e:
            print(f"[Writer] Error in writer thread: {e}", flush=True)
            traceback.print_exc()

    # Final flush on shutdown
    if buffer:
        print(f"[Writer] Final flush of {len(buffer)} remaining houses...", flush=True)
        flush_buffer()

    print(f"[Writer] Writer thread shutting down. Saved {successful}, failed {failed}", flush=True)
    return successful, failed


def test_single_house(input_file, model_path, base_path, test_house_id=None):
    """Test mode: process a single random house and print output"""
    print("="*60)
    print("TEST MODE")
    print("="*60)

    # Load house data
    with open(input_file, 'r') as f:
        houses_data = json.load(f)

    # Select house to test
    if test_house_id:
        house_data = next((h for h in houses_data if h['house_id'] == test_house_id), None)
        if not house_data:
            print(f"Error: House ID '{test_house_id}' not found in input file")
            sys.exit(1)
        print(f"Testing specific house: {test_house_id}")
    else:
        house_data = random.choice(houses_data)
        print(f"Testing random house: {house_data['house_id']}")

    print(f"House ID: {house_data['house_id']}")
    print(f"Number of images: {len(house_data['image_paths'])}")
    print(f"Image paths:")
    for img_name, img_path in house_data['image_paths'].items():
        full_path = os.path.join(base_path, img_path)
        exists = "✓" if os.path.exists(full_path) else "✗"
        print(f"  {exists} {img_name}: {img_path}")

    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)

    # Initialize generator
    generator = HouseDescriptionGenerator(model_path=model_path, worker_id=0, gpu_id=0)
    generator.initialize_model()

    print("\n" + "="*60)
    print("PROCESSING HOUSE")
    print("="*60)

    # Process house
    start_time = time.time()
    result = generator.process_house(
        house_data['house_id'],
        house_data['image_paths'],
        base_path
    )
    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"\nStructured output:")
    print(json.dumps(result, indent=2))

    # Save to test output file
    test_output = "test_output.parquet"
    df = pd.DataFrame([result])
    df.to_parquet(test_output, index=False)
    print(f"\n✓ Saved test output to: {test_output}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate descriptions for house images using Qwen3-VL with parallel processing",
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
        "--num-workers", "-w",
        type=int,
        default=2,
        help="Number of parallel workers (models to run)"
    )
    parser.add_argument(
        "--base-path", "-b",
        default="../data",
        help="Base path where house images are stored (relative to script or absolute)"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test mode: process a single random house and print output"
    )
    parser.add_argument(
        "--test-house-id",
        type=str,
        default=None,
        help="Specific house ID to test (only used with --test flag)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of results to buffer before writing to disk"
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=30,
        help="Seconds between automatic buffer flushes"
    )

    args = parser.parse_args()

    # Convert base_path to absolute path
    base_path = os.path.abspath(args.base_path)

    # Handle test mode
    if args.test:
        test_single_house(args.input, args.model_path, base_path, args.test_house_id)
        return

    print(f"Starting parallel processing job at {datetime.now()}", flush=True)
    print(f"Arguments: {args}", flush=True)

    # Clear CUDA_VISIBLE_DEVICES to detect all available GPUs
    # This ensures we can see and use all GPUs in the system
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"Clearing existing CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        del os.environ['CUDA_VISIBLE_DEVICES']

    # Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected {num_gpus} GPU(s)", flush=True)

    if num_gpus > 0:
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)", flush=True)

    # Adjust number of workers based on available GPUs
    if num_gpus == 0:
        print("Warning: No GPUs detected. Processing will be slow on CPU.", flush=True)
        args.num_workers = min(args.num_workers, 1)
    elif num_gpus >= 2:
        print(f"Using {num_gpus} GPUs for parallel processing", flush=True)

    print(f"Number of workers: {args.num_workers}", flush=True)

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
            df_temp = pd.read_parquet(args.output)
            processed_ids = set(df_temp['house_id'].tolist())
            print(f"Resuming: {len(processed_ids)} houses already processed", flush=True)
            print(f"Processed house IDs: {sorted(list(processed_ids))[:10]}..." if len(processed_ids) > 10 else f"Processed house IDs: {sorted(list(processed_ids))}", flush=True)
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}", flush=True)
    elif args.resume:
        print(f"Resume mode enabled but output file does not exist yet", flush=True)

    # Filter out already processed houses
    houses_to_process = [
        h for h in houses_data
        if h['house_id'] not in processed_ids
    ]

    if not houses_to_process:
        print("All houses already processed!", flush=True)
        if os.path.exists(args.output):
            df_temp = pd.read_parquet(args.output)
            print(f"Total records in output file: {len(df_temp)}", flush=True)
        return

    print(f"Will process {len(houses_to_process)} remaining houses", flush=True)
    print(f"First 10 houses to process: {[h['house_id'] for h in houses_to_process[:10]]}", flush=True)

    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    work_queue = mp.Queue(maxsize=args.num_workers * 2)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    print(f"Base path for images: {base_path}", flush=True)
    print(f"Batch size: {args.batch_size}, Flush interval: {args.flush_interval}s", flush=True)

    # Start worker processes
    workers = []
    for worker_id in range(args.num_workers):
        # Assign GPU in round-robin fashion if multiple GPUs available
        assigned_gpu = worker_id % num_gpus if num_gpus > 0 else 0

        p = mp.Process(
            target=worker_process,
            args=(worker_id, work_queue, result_queue, args.model_path, base_path, stop_event, assigned_gpu)
        )
        p.start()
        workers.append(p)
        print(f"Started worker {worker_id} on GPU {assigned_gpu} (PID: {p.pid})", flush=True)

    # Start result writer thread with batch parameters
    writer_thread = threading.Thread(
        target=result_writer_thread,
        args=(result_queue, args.output, stop_event, args.batch_size, args.flush_interval)
    )
    writer_thread.start()

    print(f"\n{'='*60}", flush=True)
    print("STARTING PARALLEL PROCESSING", flush=True)
    print(f"{'='*60}\n", flush=True)

    start_time = datetime.now()

    # Feed work to queue
    try:
        for house_data in houses_to_process:
            work_queue.put(house_data)

        # Send poison pills to stop workers
        for _ in range(args.num_workers):
            work_queue.put(None)

        print(f"[Main] All {len(houses_to_process)} houses queued for processing", flush=True)

        # Wait for workers to finish
        for i, worker in enumerate(workers):
            worker.join()
            print(f"[Main] Worker {i} finished", flush=True)

        # Stop writer thread
        result_queue.put(('stop', None))
        writer_thread.join()

    except KeyboardInterrupt:
        print("\n[Main] Interrupted! Stopping workers...", flush=True)
        stop_event.set()
        for worker in workers:
            worker.terminate()
            worker.join()
        writer_thread.join()

    total_time = (datetime.now() - start_time).total_seconds()

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print("PROCESSING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total time for this run: {total_time/60:.1f} minutes", flush=True)
    print(f"Output: {args.output}", flush=True)

    if os.path.exists(args.output):
        df_final = pd.read_parquet(args.output)
        total_in_file = len(df_final)
        newly_processed = total_in_file - len(processed_ids)
        print(f"\nFinal file contains: {total_in_file} total records", flush=True)
        print(f"Newly processed in this run: {newly_processed}", flush=True)
        print(f"Previously processed: {len(processed_ids)}", flush=True)
        if total_in_file > 0:
            print(f"Average time per house: {total_time/newly_processed:.1f} seconds" if newly_processed > 0 else "", flush=True)
        print(f"House IDs in final file: {sorted(df_final['house_id'].tolist())[:10]}..." if len(df_final) > 10 else f"House IDs in final file: {sorted(df_final['house_id'].tolist())}", flush=True)


if __name__ == "__main__":
    main()
