"""
Worker processes for parallel house processing.
Each worker handles model initialization and processes houses from the queue.
"""

import traceback
from queue import Empty

from config import load_system_prompt
from core.generator import HouseDescriptionGenerator


def worker_process(
    worker_id, work_queue, result_queue, model_path, base_path, stop_event, gpu_id=0
):
    """
    Worker process that processes houses from the queue.

    Args:
        worker_id: Unique identifier for this worker
        work_queue: Multiprocessing queue with house data to process
        result_queue: Multiprocessing queue for results
        model_path: Path to the VLM model
        base_path: Base directory for image paths
        stop_event: Event to signal shutdown
        gpu_id: GPU device ID to use
    """
    try:
        print(f"[Worker {worker_id}] Starting on GPU {gpu_id}", flush=True)

        # Initialize generator with system prompt
        system_prompt = load_system_prompt()
        generator = HouseDescriptionGenerator(
            model_path=model_path, worker_id=worker_id, gpu_id=gpu_id
        )
        generator.initialize_model(system_prompt)

        print(f"[Worker {worker_id}] Ready", flush=True)

        processed_count = 0
        while not stop_event.is_set():
            try:
                house_data = work_queue.get(timeout=1)

                if house_data is None:  # Stop signal
                    break

                house_id = house_data["house_id"]
                print(f"[Worker {worker_id}] Processing house {house_id}...", flush=True)

                result = generator.process_house(house_data, base_path)

                if result:
                    result_queue.put(("success", result))
                    processed_count += 1
                    print(
                        f"[Worker {worker_id}] ✓ Completed {house_id} ({result['generation_time_seconds']:.2f}s)",
                        flush=True,
                    )
                else:
                    result_queue.put(("failed", house_id))

            except Empty:
                continue
            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}", flush=True)
                traceback.print_exc()

        print(
            f"[Worker {worker_id}] Shutting down. Processed {processed_count} houses.", flush=True
        )

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}", flush=True)
        traceback.print_exc()
