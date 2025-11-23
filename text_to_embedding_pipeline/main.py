import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd
import torch
import gc
import time
from sentence_transformers import SentenceTransformer
import multiprocessing as mp
from queue import Empty
import threading
import traceback
import numpy as np

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@dataclass
class TextInput:
    """Input data structure for text to be embedded"""
    house_id: str
    description: str

    @classmethod
    def from_dict(cls, data: dict):
        """Create TextInput from dictionary"""
        return cls(
            house_id=str(data['house_id']),
            description=str(data.get('description', ''))
        )


@dataclass
class EmbeddingOutput:
    """Output data structure for embeddings"""
    house_id: str
    description: str
    embedding: List[float]
    embedding_dimension: int
    processing_time_seconds: float
    timestamp: str
    model_name: str

    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            'house_id': self.house_id,
            'description': self.description,
            'embedding': self.embedding,
            'embedding_dimension': self.embedding_dimension,
            'processing_time_seconds': self.processing_time_seconds,
            'timestamp': self.timestamp,
            'model_name': self.model_name
        }


class TextEmbeddingGenerator:
    def __init__(self, model_path=None, worker_id=0):
        self.model = None
        self.model_path = model_path or "./local_mpnet_model"
        self.worker_id = worker_id
        self.device = None

    def initialize_model(self):
        """Initialize the sentence transformer model"""
        print(f"[Worker {self.worker_id}][{datetime.now().strftime('%H:%M:%S')}] Initializing model...", flush=True)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            print(f"[Worker {self.worker_id}] GPU detected: {device_name}", flush=True)
        else:
            self.device = "cpu"
            print(f"[Worker {self.worker_id}] No GPU detected, using CPU", flush=True)

        # Download and save the model if not already saved
        if not os.path.exists(self.model_path):
            print(f"[Worker {self.worker_id}] Model not found locally. Downloading {MODEL_NAME}...", flush=True)
            model = SentenceTransformer(MODEL_NAME)
            model.save(self.model_path)
            print(f"[Worker {self.worker_id}] Model saved to {self.model_path}", flush=True)

        # Load the saved model
        self.model = SentenceTransformer(self.model_path, device=self.device)
        self.model.eval()

        print(f"[Worker {self.worker_id}] Model loaded successfully on {self.device}", flush=True)

        if torch.cuda.is_available():
            print(f"[Worker {self.worker_id}] GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB", flush=True)

    def generate_embedding(self, text_input: TextInput, batch_size=32) -> Optional[EmbeddingOutput]:
        """Generate embedding for a single text input"""
        if not text_input.description or text_input.description.strip() == '':
            print(f"[Worker {self.worker_id}] Warning: Empty description for house {text_input.house_id}", flush=True)
            return None

        try:
            start_time = time.time()

            # Generate embedding
            embedding = self.model.encode(
                [text_input.description],
                convert_to_numpy=True,
                device=self.device,
                batch_size=batch_size,
                show_progress_bar=False
            )[0]  # Get the first (and only) embedding

            processing_time = time.time() - start_time

            # Create output
            output = EmbeddingOutput(
                house_id=text_input.house_id,
                description=text_input.description,
                embedding=embedding.tolist(),
                embedding_dimension=len(embedding),
                processing_time_seconds=round(processing_time, 4),
                timestamp=datetime.now().isoformat(),
                model_name=MODEL_NAME
            )

            return output

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error generating embedding for house {text_input.house_id}: {e}", flush=True)
            traceback.print_exc()
            return None

    def process_text(self, text_data: dict) -> Optional[dict]:
        """Process a single text entry and return result"""
        try:
            # Convert to TextInput
            text_input = TextInput.from_dict(text_data)

            # Generate embedding
            result = self.generate_embedding(text_input)

            if result:
                return result.to_dict()
            else:
                return None

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error processing house {text_data.get('house_id', 'unknown')}: {e}", flush=True)
            traceback.print_exc()
            return None


def worker_process(worker_id, work_queue, result_queue, model_path, stop_event):
    """Worker process that generates embeddings from the queue"""
    try:
        print(f"[Worker {worker_id}] Starting worker process", flush=True)

        # Initialize generator for this worker
        generator = TextEmbeddingGenerator(model_path=model_path, worker_id=worker_id)
        generator.initialize_model()

        print(f"[Worker {worker_id}] Ready to process texts", flush=True)

        processed_count = 0
        while not stop_event.is_set():
            try:
                # Get work from queue with timeout
                text_data = work_queue.get(timeout=1)

                if text_data is None:  # Poison pill
                    print(f"[Worker {worker_id}] Received stop signal", flush=True)
                    break

                house_id = text_data.get('house_id', 'unknown')
                print(f"[Worker {worker_id}] Processing house {house_id}...", flush=True)

                result = generator.process_text(text_data)

                if result:
                    result_queue.put(('success', result))
                    processed_count += 1
                    print(f"[Worker {worker_id}] ✓ Completed house {house_id} ({result['processing_time_seconds']:.4f}s) [Total: {processed_count}]", flush=True)
                else:
                    result_queue.put(('failed', house_id))
                    print(f"[Worker {worker_id}] ✗ Failed house {house_id}", flush=True)

            except Empty:
                continue
            except Exception as e:
                print(f"[Worker {worker_id}] Error in worker loop: {e}", flush=True)
                traceback.print_exc()

        print(f"[Worker {worker_id}] Worker shutting down. Processed {processed_count} texts.", flush=True)

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error in worker: {e}", flush=True)
        traceback.print_exc()


def result_writer_thread(result_queue, output_file, output_format, stop_event):
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
                        if output_format == 'parquet':
                            df_existing = pd.read_parquet(output_file)
                        else:  # csv
                            df_existing = pd.read_csv(output_file)
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

                        if output_format == 'parquet':
                            df_combined.to_parquet(output_file, index=False)
                        else:  # csv
                            df_combined.to_csv(output_file, index=False)
                        total = len(df_combined)
                    else:
                        if output_format == 'parquet':
                            df_new.to_parquet(output_file, index=False)
                        else:  # csv
                            df_new.to_csv(output_file, index=False)
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


def load_input_data(input_file: str) -> List[dict]:
    """Load input data from parquet, csv, or json file"""
    file_ext = Path(input_file).suffix.lower()

    if file_ext == '.parquet':
        df = pd.read_parquet(input_file)
        # Ensure required columns exist
        if 'house_id' not in df.columns or 'description' not in df.columns:
            raise ValueError(f"Input parquet file must contain 'house_id' and 'description' columns")
        return df[['house_id', 'description']].to_dict('records')

    elif file_ext == '.csv':
        df = pd.read_csv(input_file)
        # Ensure required columns exist
        if 'house_id' not in df.columns or 'description' not in df.columns:
            raise ValueError(f"Input CSV file must contain 'house_id' and 'description' columns")
        return df[['house_id', 'description']].to_dict('records')

    elif file_ext == '.json':
        with open(input_file, 'r') as f:
            data = json.load(f)
        # Validate structure
        if isinstance(data, list):
            for item in data:
                if 'house_id' not in item or 'description' not in item:
                    raise ValueError(f"Each item in JSON must contain 'house_id' and 'description'")
            return data
        else:
            raise ValueError(f"JSON file must contain a list of objects")

    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .parquet, .csv, or .json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for house descriptions using SentenceTransformers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input file (parquet, csv, or json) with house_id and description columns"
    )
    parser.add_argument(
        "--output", "-o",
        default="house_embeddings.parquet",
        help="Path to output file (parquet or csv)"
    )
    parser.add_argument(
        "--model-path", "-m",
        default="./local_mpnet_model",
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
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--output-format", "-f",
        choices=['parquet', 'csv'],
        default='parquet',
        help="Output file format"
    )

    args = parser.parse_args()

    print(f"Starting text-to-embedding pipeline at {datetime.now()}", flush=True)
    print(f"Arguments: {args}", flush=True)
    print(f"Number of workers: {args.num_workers}", flush=True)

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", flush=True)
        sys.exit(1)

    # Set output format based on file extension if not specified
    if args.output.endswith('.csv'):
        args.output_format = 'csv'
    elif args.output.endswith('.parquet'):
        args.output_format = 'parquet'
    else:
        # Add extension based on format
        args.output = f"{args.output}.{args.output_format}"

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}", flush=True)

    # Load input data
    print(f"Loading data from {args.input}...", flush=True)
    try:
        texts_data = load_input_data(args.input)
    except Exception as e:
        print(f"Error loading input file: {e}", flush=True)
        sys.exit(1)

    print(f"Found {len(texts_data)} texts to process", flush=True)

    # Check for resume
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        try:
            print(f"Resume mode: Reading existing output file {args.output}...", flush=True)
            if args.output_format == 'parquet':
                df_temp = pd.read_parquet(args.output)
            else:
                df_temp = pd.read_csv(args.output)
            processed_ids = set(df_temp['house_id'].astype(str).tolist())
            print(f"Resuming: {len(processed_ids)} houses already processed", flush=True)
            sample_ids = sorted(list(processed_ids))[:10]
            print(f"Sample processed IDs: {sample_ids}..." if len(processed_ids) > 10 else f"Processed IDs: {sample_ids}", flush=True)
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}", flush=True)
    elif args.resume:
        print(f"Resume mode enabled but output file does not exist yet", flush=True)

    # Filter out already processed texts
    texts_to_process = [
        t for t in texts_data
        if str(t['house_id']) not in processed_ids
    ]

    if not texts_to_process:
        print("All texts already processed!", flush=True)
        if os.path.exists(args.output):
            if args.output_format == 'parquet':
                df_temp = pd.read_parquet(args.output)
            else:
                df_temp = pd.read_csv(args.output)
            print(f"Total records in output file: {len(df_temp)}", flush=True)
        return

    print(f"Will process {len(texts_to_process)} remaining texts", flush=True)
    sample_to_process = [t['house_id'] for t in texts_to_process[:10]]
    print(f"First texts to process: {sample_to_process}", flush=True)

    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    work_queue = mp.Queue(maxsize=args.num_workers * 2)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    # Start worker processes
    workers = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, work_queue, result_queue, args.model_path, stop_event)
        )
        p.start()
        workers.append(p)
        print(f"Started worker {worker_id} (PID: {p.pid})", flush=True)

    # Start result writer thread
    writer_thread = threading.Thread(
        target=result_writer_thread,
        args=(result_queue, args.output, args.output_format, stop_event)
    )
    writer_thread.start()

    print(f"\n{'='*60}", flush=True)
    print("STARTING EMBEDDING GENERATION", flush=True)
    print(f"{'='*60}\n", flush=True)

    start_time = datetime.now()

    # Feed work to queue
    try:
        for text_data in texts_to_process:
            work_queue.put(text_data)

        # Send poison pills to stop workers
        for _ in range(args.num_workers):
            work_queue.put(None)

        print(f"[Main] All {len(texts_to_process)} texts queued for processing", flush=True)

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
        if args.output_format == 'parquet':
            df_final = pd.read_parquet(args.output)
        else:
            df_final = pd.read_csv(args.output)
        total_in_file = len(df_final)
        newly_processed = total_in_file - len(processed_ids)
        print(f"\nFinal file contains: {total_in_file} total records", flush=True)
        print(f"Newly processed in this run: {newly_processed}", flush=True)
        print(f"Previously processed: {len(processed_ids)}", flush=True)
        if newly_processed > 0:
            print(f"Average time per text: {total_time/newly_processed:.2f} seconds", flush=True)
        sample_final_ids = sorted(df_final['house_id'].astype(str).tolist())[:10]
        print(f"Sample IDs in final file: {sample_final_ids}..." if len(df_final) > 10 else f"IDs in final file: {sample_final_ids}", flush=True)


if __name__ == "__main__":
    main()
