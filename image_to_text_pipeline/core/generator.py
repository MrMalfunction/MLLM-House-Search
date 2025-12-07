"""
Generator module for creating house descriptions from images using VLM.
Handles model initialization, image loading, and description generation.
"""

import gc
import os
import time
import traceback
from datetime import datetime

import torch
from config import (
    ATTN_IMPLEMENTATION,
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    MODEL_ID,
    NUM_BEAMS,
    REPETITION_PENALTY,
    STOP_SEQUENCES,
)
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor, StoppingCriteriaList

from core.stopping import TextStoppingCriteria


class HouseDescriptionGenerator:
    """Generates detailed house descriptions from images using a vision-language model."""

    def __init__(self, model_path=None, worker_id=0, gpu_id=0):
        """
        Initialize the generator.

        Args:
            model_path: Path to the VLM model (default: ./models/qwen3-vl-8b)
            worker_id: Identifier for this worker process
            gpu_id: GPU device ID to use
        """
        self.model = None
        self.processor = None
        self.model_path = model_path or "./models/qwen3-vl-8b"
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.system_prompt = ""

    def initialize_model(self, system_prompt):
        """
        Initialize the VLM model and processor.

        Args:
            system_prompt: System prompt for the model
        """
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
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(repo_id=MODEL_ID, local_dir=self.model_path)
            except Exception as e:
                print(f"[Worker {self.worker_id}] Error downloading model: {e}", flush=True)
                raise

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        # Load model to assigned GPU
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map={"": self.gpu_id},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=ATTN_IMPLEMENTATION,
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.system_prompt = system_prompt
        print(f"[Worker {self.worker_id}] Model loaded successfully", flush=True)

    def load_image(self, path):
        """
        Load and convert image to RGB.

        Args:
            path: Path to the image file

        Returns:
            PIL Image in RGB mode

        Raises:
            FileNotFoundError: If image doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def generate_description(self, image_paths, metadata, max_tokens=1500):
        """
        Generate description from multiple images.

        Args:
            image_paths: List of image file paths
            metadata: Dictionary with property metadata (bedrooms, bathrooms, area, zipcode, price)
            max_tokens: Maximum tokens for display (used in prompt)

        Returns:
            Tuple of (output_text, generation_time_seconds)
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized")

        # Load images
        images = [self.load_image(path) for path in image_paths]

        # Build messages for the model
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

        # Add images to user message
        content = []
        for img in images:
            content.append({"type": "image", "image": img})

        # Add metadata text
        metadata_str = f"""Property Metadata:
- Bedrooms: {metadata.get("bedrooms", "N/A")}
- Bathrooms: {metadata.get("bathrooms", "N/A")}
- Area: {metadata.get("area", "N/A")}
- Zipcode: {metadata.get("zipcode", "N/A")}
- Price: {metadata.get("price", "N/A")}

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
            image_inputs, video_inputs, _ = vision_info  # type: ignore
        else:
            image_inputs, video_inputs = vision_info[0], vision_info[1]  # type: ignore

        # Move to correct device
        device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Setup stopping criteria
        initial_length = inputs.input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList(
            [TextStoppingCriteria(self.processor.tokenizer, STOP_SEQUENCES, initial_length)]
        )

        # Generate with strict settings to prevent rambling
        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                num_beams=NUM_BEAMS,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )
        generation_time = time.time() - start_time

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text, generation_time

    def process_house(self, house_data, base_path="", image_types=None):
        """
        Process a single house and return raw output with metadata.

        Args:
            house_data: Dictionary containing house information
            base_path: Base directory for relative image paths
            image_types: List of image types to process (default: frontal, kitchen, bedroom, bathroom)

        Returns:
            Dictionary with processing results or None on failure
        """
        if image_types is None:
            image_types = ["frontal", "kitchen", "bedroom", "bathroom"]

        house_id = house_data["house_id"]

        try:
            # Build image paths
            image_paths = []
            for img_type in image_types:
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
                "price": house_data["metadata"].get("price"),
            }

            # Generate description
            raw_output, gen_time = self.generate_description(image_paths, metadata)

            # Return result with raw output
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
                "processed_at": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error processing house {house_id}: {e}", flush=True)
            traceback.print_exc()
            return None
