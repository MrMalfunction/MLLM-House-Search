"""
House Image Description Generator using Qwen3-VL Model

Production pipeline for processing house images and generating detailed descriptions.
Supports parallel processing with multiple GPUs and batched writing to parquet files.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import threading
from datetime import datetime

import pandas as pd
import torch
from config import (
    DEFAULT_BASE_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLUSH_INTERVAL,
    DEFAULT_MODEL_PATH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_FILE,
    load_system_prompt,
)
from core.generator import HouseDescriptionGenerator
from parquet_to_csv import convert_parquet_to_csv
from parser import parse_delimited_output
from pipeline.workers import worker_process
from pipeline.writer import result_writer_thread

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


def test_single_house(input_file, model_path, base_path, house_id=None):
    """
    Test mode: process one house and display output for verification.

    Args:
        input_file: Path to input JSON with house data
        model_path: Path to VLM model
        base_path: Base directory for image paths
        house_id: Specific house ID to test (random if None)
    """
    print(f"\n{'=' * 60}")
    print("TEST MODE")
    print(f"{'=' * 60}\n")

    # Load data
    with open(input_file) as f:
        houses_data = json.load(f)

    # Select house
    if house_id:
        house_data = next((h for h in houses_data if h["house_id"] == house_id), None)
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
    system_prompt = load_system_prompt()
    generator.initialize_model(system_prompt)

    print("Processing...\n")
    import time

    start_time = time.time()
    result = generator.process_house(house_data, base_path)
    processing_time = time.time() - start_time

    if result:
        print(f"\n{'=' * 60}")
        print("RAW OUTPUT")
        print(f"{'=' * 60}\n")
        print(result["raw_output"])

        print(f"\n{'=' * 60}")
        print("PARSED OUTPUT")
        print(f"{'=' * 60}\n")

        parsed = parse_delimited_output(result["raw_output"])
        print(f"Short Description:\n{parsed['short_description']}\n")
        print(f"Frontal:\n{parsed['frontal_description'][:200]}...\n")
        print(f"Kitchen:\n{parsed['kitchen_description'][:200]}...\n")
        print(f"Bedroom:\n{parsed['bedroom_description'][:200]}...\n")
        print(f"Bathroom:\n{parsed['bathroom_description'][:200]}...\n")

        print(f"\nProcessing time: {processing_time:.2f}s")
        print(f"{'=' * 60}\n")
    else:
        print("Error: Processing failed")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate house descriptions from images using vision-language model"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="house_image_associations.json",
        help="Input JSON file with house data",
    )
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_FILE, help="Output parquet file")
    parser.add_argument(
        "--model-path", "-m", default=DEFAULT_MODEL_PATH, help="Path to pretrained VLM model"
    )
    parser.add_argument(
        "--resume", "-r", action="store_true", help="Resume from existing output file"
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--base-path", "-b", default=DEFAULT_BASE_PATH, help="Base directory for image paths"
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Test mode (process single house)"
    )
    parser.add_argument(
        "--test-house-id", type=str, default=None, help="Specific house ID for test mode"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for writing results"
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=DEFAULT_FLUSH_INTERVAL,
        help="Flush interval in seconds",
    )

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
    with open(args.input) as f:
        houses_data = json.load(f)
    print(f"Total houses: {len(houses_data)}")

    # Handle resume
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        df_temp = pd.read_parquet(args.output)
        processed_ids = set(df_temp["house_id"].tolist())
        print(f"Already processed: {len(processed_ids)}")

    houses_to_process = [h for h in houses_data if h["house_id"] not in processed_ids]

    if not houses_to_process:
        print("All houses already processed!")
        return

    print(f"To process: {len(houses_to_process)}")

    # Setup multiprocessing
    mp.set_start_method("spawn", force=True)
    work_queue = mp.Queue(maxsize=args.num_workers * 2)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    # Start workers
    workers = []
    for worker_id in range(args.num_workers):
        assigned_gpu = worker_id % num_gpus if num_gpus > 0 else 0
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                work_queue,
                result_queue,
                args.model_path,
                base_path,
                stop_event,
                assigned_gpu,
            ),
        )
        p.start()
        workers.append(p)
        print(f"Started worker {worker_id} on GPU {assigned_gpu}")

    # Start writer
    writer_thread = threading.Thread(
        target=result_writer_thread,
        args=(result_queue, args.output, stop_event, args.batch_size, args.flush_interval),
    )
    writer_thread.start()

    print(f"\n{'=' * 60}")
    print("PROCESSING")
    print(f"{'=' * 60}\n")

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

        result_queue.put(("stop", None))
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
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"Time: {total_time / 60:.1f} minutes")
    print(f"Output: {args.output}")

    if os.path.exists(args.output):
        df_final = pd.read_parquet(args.output)
        print(f"Total records: {len(df_final)}")
        print(f"Newly processed: {len(df_final) - len(processed_ids)}")

        # Convert parquet to CSV
        print(f"\n{'=' * 60}")
        print("CONVERTING TO CSV")
        print(f"{'=' * 60}\n")
        try:
            csv_output = os.path.splitext(args.output)[0] + ".csv"
            csv_path = convert_parquet_to_csv(args.output, csv_output)
            print(f"CSV file created: {csv_path}")
        except Exception as e:
            print(f"Error converting to CSV: {e}")


if __name__ == "__main__":
    main()
