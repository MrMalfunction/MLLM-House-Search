"""
Result writer thread for batched output processing.
Parses results and writes to parquet in batches for efficiency.
"""

import os
import time
import traceback
from queue import Empty

import pandas as pd
from config import DEFAULT_BATCH_SIZE, DEFAULT_FLUSH_INTERVAL
from parser import parse_delimited_output


def result_writer_thread(
    result_queue,
    output_file,
    stop_event,
    batch_size=DEFAULT_BATCH_SIZE,
    flush_interval=DEFAULT_FLUSH_INTERVAL,
):
    """
    Writer thread that parses results and writes to parquet in batches.
    Parsing done here frees up GPU workers faster.

    Args:
        result_queue: Multiprocessing queue with results
        output_file: Output parquet file path
        stop_event: Event to signal shutdown
        batch_size: Number of records to buffer before writing
        flush_interval: Seconds between flushes

    Returns:
        Tuple of (successful_count, failed_count)
    """
    successful = 0
    failed = 0
    buffer = []
    last_flush_time = time.time()

    def flush_buffer():
        """Write buffered results to disk."""
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

            if result_type == "stop":
                flush_buffer()
                break
            elif result_type == "success":
                # Parse raw output into structured fields
                raw_output = result_data.get("raw_output", "")
                parsed = parse_delimited_output(raw_output)

                # Add parsed fields to result
                result_data["short_description"] = parsed["short_description"]
                result_data["frontal_description"] = parsed["frontal_description"]
                result_data["kitchen_description"] = parsed["kitchen_description"]
                result_data["bedroom_description"] = parsed["bedroom_description"]
                result_data["bathroom_description"] = parsed["bathroom_description"]

                buffer.append(result_data)
                successful += 1
                print(
                    f"[Writer] 📥 Parsed & buffered house {result_data['house_id']} ({len(buffer)} buffered)",
                    flush=True,
                )

                # Flush if buffer full or time elapsed
                if len(buffer) >= batch_size or (time.time() - last_flush_time) >= flush_interval:
                    flush_buffer()
                    last_flush_time = time.time()

            elif result_type == "failed":
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
