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
import multiprocessing as mp
from queue import Empty
import threading
import traceback

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SYSTEM_PROMPT = """Analyze this residential property using 4 images (frontal, kitchen, bedroom, bathroom). Generate a dense, keyword-rich description for semantic search. Include: architectural style, exterior features, interior finishes, flooring, kitchen (cabinets, countertops, appliances), bathroom (vanity, tub/shower, fixtures), lighting, and overall quality. Omit features not visible."""

class HouseDescriptionGenerator:
    def __init__(self, model_path=None, use_cache=True, worker_id=0):
        self.model = None
        self.processor = None
        self.model_path = model_path or "./models/qwen3-vl-8b"
        self.use_cache = use_cache
        self.worker_id = worker_id
        self.gpu_id = 0  # All workers share GPU 0

    def initialize_model(self):
        """Initialize the model and processor"""
        print(f"[Worker {self.worker_id}][{datetime.now().strftime('%H:%M:%S')}] Initializing model...", flush=True)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            torch.cuda.empty_cache()
            gc.collect()
            device_name = torch.cuda.get_device_name(self.gpu_id)
            print(f"[Worker {self.worker_id}] GPU detected: {device_name} (GPU {self.gpu_id})", flush=True)
        else:
            print(f"[Worker {self.worker_id}] No GPU detected, using CPU", flush=True)

        # Determine dtype
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(self.gpu_id)
            use_bf16 = any(gpu in device_name for gpu in ["A100", "H100", "L4", "4090"])
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            print(f"[Worker {self.worker_id}] Using dtype: {dtype}", flush=True)
        else:
            dtype = torch.float32

        # Load processor and model
        print(f"[Worker {self.worker_id}] Loading from: {self.model_path}", flush=True)

        if not os.path.exists(self.model_path):
            print(f"[Worker {self.worker_id}] Model not found locally. Downloading {MODEL_ID}...", flush=True)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=MODEL_ID, local_dir=self.model_path)

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map={"": self.gpu_id},  # All workers share GPU 0
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[Worker {self.worker_id}] Model loaded successfully on {next(self.model.parameters()).device}", flush=True)

        if torch.cuda.is_available():
            print(f"[Worker {self.worker_id}] GPU Memory: {torch.cuda.memory_allocated(self.gpu_id) / 1024**3:.2f} GB", flush=True)

    def load_image(self, path):
        """Load and convert image to RGB"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def generate_description(self, image_paths, max_tokens=200):
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
                min_new_tokens=50,
                do_sample=False,
                num_beams=1,
                use_cache=self.use_cache,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
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

            # Validate all images exist
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")

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
            print(f"[Worker {self.worker_id}] Error processing house {house_id}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            return None


def worker_process(worker_id, work_queue, result_queue, model_path, base_path, stop_event):
    """Worker process that processes houses from the queue"""
    try:
        print(f"[Worker {worker_id}] Starting worker process", flush=True)

        # Initialize generator for this worker (all share GPU 0)
        generator = HouseDescriptionGenerator(model_path=model_path, worker_id=worker_id)
        generator.initialize_model()

        print(f"[Worker {worker_id}] Ready to process houses", flush=True)

        processed_count = 0
        while not stop_event.is_set():
            try:
                # Get work from queue with timeout
                house_data = work_queue.get(timeout=1)

                if house_data is None:  # Poison pill
                    print(f"[Worker {worker_id}] Received stop signal", flush=True)
                    break

                house_id = house_data['house_id']
                print(f"[Worker {worker_id}] Processing house {house_id}...", flush=True)

                result = generator.process_house(house_data, base_path)

                if result:
                    result_queue.put(('success', result))
                    processed_count += 1
                    print(f"[Worker {worker_id}] ✓ Completed house {house_id} ({result['generation_time_seconds']:.2f}s) [Total: {processed_count}]", flush=True)
                else:
                    result_queue.put(('failed', house_id))
                    print(f"[Worker {worker_id}] ✗ Failed house {house_id}", flush=True)

            except Empty:
                continue
            except Exception as e:
                print(f"[Worker {worker_id}] Error in worker loop: {e}", flush=True)
                traceback.print_exc()

        print(f"[Worker {worker_id}] Worker shutting down. Processed {processed_count} houses.", flush=True)

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error in worker: {e}", flush=True)
        traceback.print_exc()


def result_writer_thread(result_queue, output_file, stop_event):
    """Thread that writes results to file as they come in"""
    successful = 0
    failed = 0

    while not stop_event.is_set():
        try:
            result_type, result_data = result_queue.get(timeout=1)

            if result_type == 'stop':
                break
            elif result_type == 'success':
                # Save result immediately
                try:
                    df_new = pd.DataFrame([result_data])
                    if os.path.exists(output_file):
                        df_existing = pd.read_parquet(output_file)
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        df_combined.to_parquet(output_file, index=False)
                        total = len(df_combined)
                    else:
                        df_new.to_parquet(output_file, index=False)
                        total = 1
                    successful += 1
                    print(f"[Writer] 💾 Saved house {result_data['house_id']} (Total: {total} houses, {successful} this run)", flush=True)
                except Exception as e:
                    print(f"[Writer] ⚠️ Failed to save: {e}", flush=True)
                    traceback.print_exc()
            elif result_type == 'failed':
                failed += 1
                print(f"[Writer] Failed house {result_data}", flush=True)

        except Empty:
            continue
        except Exception as e:
            print(f"[Writer] Error in writer thread: {e}", flush=True)
            traceback.print_exc()

    print(f"[Writer] Writer thread shutting down. Saved {successful}, failed {failed}", flush=True)
    return successful, failed


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

    args = parser.parse_args()

    print(f"Starting parallel processing job at {datetime.now()}", flush=True)
    print(f"Arguments: {args}", flush=True)
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

    base_path = os.path.dirname(args.input) if os.path.dirname(args.input) else "."

    # Start worker processes
    workers = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, work_queue, result_queue, args.model_path, base_path, stop_event)
        )
        p.start()
        workers.append(p)
        print(f"Started worker {worker_id} (PID: {p.pid})", flush=True)

    # Start result writer thread
    writer_thread = threading.Thread(
        target=result_writer_thread,
        args=(result_queue, args.output, stop_event)
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
