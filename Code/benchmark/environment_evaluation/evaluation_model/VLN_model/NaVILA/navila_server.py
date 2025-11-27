#!/usr/bin/env python3
"""
NaVILA VLN Model Server for SAGE-3D Benchmark.

This script creates a socket-based server that loads the NaVILA model and
processes VLN (Vision-and-Language Navigation) inference requests.

IMPORTANT: 
    - This script should be run within the NaVILA-Bench project environment.
    - NaVILA has two repositories:
      * NaVILA (model): https://github.com/AnjieCheng/NaVILA
      * NaVILA-Bench: https://github.com/yang-zj1026/NaVILA-Bench
    - Copy this file to NaVILA-Bench/scripts/ and run from there.

Usage:
    1. Clone NaVILA-Bench repository:
       git clone https://github.com/yang-zj1026/NaVILA-Bench.git
       cd NaVILA-Bench

    2. Setup environment following NaVILA-Bench installation guide

    3. Install NaVILA model dependencies (from NaVILA repo):
       git clone https://github.com/AnjieCheng/NaVILA.git
       cd NaVILA && pip install -e . && cd ..

    4. Copy this server script to NaVILA-Bench:
       cp navila_server.py /path/to/NaVILA-Bench/scripts/

    5. Download model weights from: https://huggingface.co/a8cheng/navila-llama3-8b-8f

    6. Run the server:
       cd /path/to/NaVILA-Bench
       conda activate navila-eval
       python scripts/navila_server.py \\
           --model_path /path/to/navila-llama3-8b-8f \\
           --port 54321 \\
           --device cuda

    7. The server will listen on the specified port for inference requests.
       Use the SAGE-3D benchmark client to send requests.

Server Protocol:
    - Uses TCP socket communication
    - Request format: JSON with 'images' (base64 list) and 'query' (string)
    - Response format: JSON with model output text
    - Data is prefixed with 8-byte big-endian size header

Example Request:
    {
        "images": ["base64_encoded_image_1", "base64_encoded_image_2", ...],
        "query": "Navigate to the red chair in the living room."
    }

Related Repositories:
    - NaVILA Model: https://github.com/AnjieCheng/NaVILA
    - NaVILA-Bench: https://github.com/yang-zj1026/NaVILA-Bench
    - Model Weights: https://huggingface.co/a8cheng/navila-llama3-8b-8f
"""

import socket
import torch
import json
import argparse
import os
import time
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
import re

# NaVILA/LLaVA specific imports - these require the NaVILA environment
from transformers import AutoTokenizer, AutoConfig
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    process_image,
    tokenizer_image_token,
    get_model_name_from_path
)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.builder import load_pretrained_model


class NaVILAServer:
    """NaVILA model server with multi-GPU support."""

    def __init__(self, args):
        """Initialize the NaVILA server.

        Args:
            args: Command line arguments containing:
                - model_path: Path to NaVILA model checkpoint
                - device: Device to run on ('cuda' or 'cpu')
                - precision: Model precision ('W16A16')
                - conv_mode: Conversation template mode
                - num_video_frames: Number of video frames to process
        """
        self.args = args
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.vision_tower = None
        self.setup()

    def setup(self):
        """Initialize model and tokenizer."""
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("[SERVER] CUDA is not available. Falling back to CPU.")
            self.args.device = "cpu"
        else:
            print(f"[SERVER] CUDA available with {torch.cuda.device_count()} GPU(s)")

        self._disable_initializers()
        self._initialize_tokenizer_and_model()

        if self.args.precision == "W16A16":
            self._load_checkpoint_w16a16()
        else:
            raise ValueError(f"Precision {self.args.precision} not supported")

    def _disable_initializers(self):
        """Disable weight initializers for faster loading."""
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None

    def _initialize_tokenizer_and_model(self):
        """Initialize tokenizer from model path."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.args.model_path, "llm"), use_fast=False
        )
        config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)

    def _load_checkpoint_w16a16(self):
        """Load model checkpoint with W16A16 precision."""
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint shards")
        for _ in pbar:
            model_name = get_model_name_from_path(self.args.model_path)

            # Load model with device_map for multi-GPU
            if torch.cuda.device_count() > 1 and self.args.device != "cpu":
                print(f"[SERVER] Using multi-GPU setup with {torch.cuda.device_count()} GPUs")
                # Use device_map="auto" for automatic multi-GPU distribution
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    self.args.model_path,
                    model_name,
                    None,
                    device_map="auto",
                    load_8bit=False,
                    load_4bit=False
                )
            else:
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    self.args.model_path,
                    model_name,
                    None
                )

            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor

        # Only move to device if not using device_map
        if not (torch.cuda.device_count() > 1 and self.args.device != "cpu"):
            self.model = self.model.to(self.args.device)
            if self.args.device != "cpu":
                self.model = self.model.half()  # Use FP16 to save memory

    def start_server(self, host='localhost', port=54321):
        """Start the socket server.

        Args:
            host: Host address to bind to
            port: Port number to listen on
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"[SERVER] NaVILA Server listening on {host}:{port}")
        print(f"[SERVER] Model: {self.args.model_path}")
        print(f"[SERVER] Device: {self.args.device}")
        print(f"[SERVER] Ready to receive requests...")

        while True:
            conn, addr = server_socket.accept()
            try:
                # Receive data size first (8 bytes, big endian)
                size_data = conn.recv(8)
                if len(size_data) != 8:
                    print(f"[SERVER] Invalid size data from {addr}")
                    continue

                size = int.from_bytes(size_data, 'big')
                print(f"[SERVER] Expecting {size} bytes from {addr}")

                # Receive the actual data
                data = b''
                while len(data) < size:
                    remaining = size - len(data)
                    packet = conn.recv(min(4096, remaining))
                    if not packet:
                        break
                    data += packet

                if len(data) != size:
                    print(f"[SERVER] Incomplete data received from {addr}: {len(data)}/{size}")
                    continue

                # Parse the received data
                request = json.loads(data.decode())
                images = request['images']
                query = request['query']

                print(f"[SERVER] Processing request from {addr}: {len(images)} images")
                print(f"[SERVER] Query: {query[:80]}...")

                # Process images and generate response
                response = self.process_request(images, query)

                # Send response back (8 bytes size header, big endian)
                response_bytes = json.dumps(response).encode()
                try:
                    conn.sendall(len(response_bytes).to_bytes(8, 'big'))
                    conn.sendall(response_bytes)
                    print(f"[SERVER] Response sent to {addr}: {len(response_bytes)} bytes")
                except BrokenPipeError:
                    print(f"[SERVER] Client {addr} disconnected while sending response")
                except Exception as e:
                    print(f"[SERVER] Error sending response to {addr}: {str(e)}")

            except Exception as e:
                print(f"[SERVER] Error processing request from {addr}: {str(e)}")
            finally:
                conn.close()

    def process_request(self, images, query):
        """Process a VLN inference request.

        Args:
            images: List of base64-encoded images
            query: Navigation instruction text

        Returns:
            Model output text (action description)
        """
        try:
            # Process images
            image_tensor = process_images(images, self.image_processor, self.model.config)

            # Handle device placement for multi-GPU
            if torch.cuda.device_count() > 1 and self.args.device != "cpu":
                # Model is already distributed, put tensors on first GPU
                image_tensor = image_tensor.to("cuda:0", dtype=torch.float16)
            else:
                image_tensor = image_tensor.to(self.args.device, dtype=torch.float16)

            # Prepare prompt using conversation template
            conv = conv_templates[self.args.conv_mode].copy()
            instruction = query
            image_token = "<image>\n"

            # Build the prompt with image tokens
            qs = (
                f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                f'of historical observations {image_token * (self.args.num_video_frames-1)}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                f"degree, moving forward a certain distance, or stop if the task is completed."
            )
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Tokenize and prepare input
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0)

            # Handle device placement for input_ids
            if torch.cuda.device_count() > 1 and self.args.device != "cpu":
                input_ids = input_ids.to("cuda:0")
            else:
                input_ids = input_ids.to(self.args.device)

            # Setup stopping criteria
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            # Generate response
            with torch.inference_mode():
                start_time = time.time()
                output_ids = self.model.generate(
                    input_ids,
                    images=[image_tensor],
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
                generation_time = time.time() - start_time
                print(f"[SERVER] Model generation took {generation_time:.2f} seconds")

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            return outputs.strip()

        except Exception as e:
            print(f"[SERVER] Error in process_request: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing request: {str(e)}"


def process_images(images, image_processor, model_cfg):
    """Process a list of images (either PIL Images or base64 strings).

    Args:
        images: List of PIL Images or base64-encoded strings
        image_processor: Image processor from model
        model_cfg: Model configuration

    Returns:
        Stacked tensor of processed images
    """
    model_cfg.image_processor = image_processor
    processed_images = []

    for image in images:
        if isinstance(image, str):
            # Handle base64 encoded image
            try:
                # Decode base64 string to PIL Image
                image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
            except Exception as e:
                print(f"[SERVER] Error decoding base64 image: {e}")
                # Create a blank image if decoding fails
                image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Process the PIL Image
        processed_image = process_image(image, model_cfg, None)
        processed_images.append(processed_image)

    if all(x.shape == processed_images[0].shape for x in processed_images):
        processed_images = torch.stack(processed_images, dim=0)
    return processed_images


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NaVILA VLN Model Server for SAGE-3D Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server with default settings
    python navila_server.py --model_path /path/to/navila-llama3-8b-8f

    # Start server on specific port and host
    python navila_server.py --model_path /path/to/model --host 0.0.0.0 --port 8888

    # Run on CPU (not recommended, very slow)
    python navila_server.py --model_path /path/to/model --device cpu
        """
    )
    parser.add_argument(
        "--host", type=str, default='localhost',
        help="Host address to bind the server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=54321,
        help="Port number to bind the server (default: 54321)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the NaVILA model checkpoint directory"
    )
    parser.add_argument(
        "--precision", type=str, default="W16A16",
        help="Compute precision (default: W16A16)"
    )
    parser.add_argument(
        "--conv_mode", type=str, default="llama_3",
        help="Conversation template mode (default: llama_3)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on: 'cuda' or 'cpu' (default: cuda)"
    )
    parser.add_argument(
        "--num_video_frames", type=int, default=8,
        help="Number of video frames to process (default: 8)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("NaVILA VLN Model Server for SAGE-3D Benchmark")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Server: {args.host}:{args.port}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.precision}")
    print(f"Conv mode: {args.conv_mode}")
    print(f"Num video frames: {args.num_video_frames}")
    print("=" * 60)

    # Add error handling for CUDA initialization
    try:
        server = NaVILAServer(args)
        server.start_server(host=args.host, port=args.port)
    except RuntimeError as e:
        if "forward compatibility" in str(e) or "CUDA" in str(e):
            print(f"[SERVER] CUDA error detected: {e}")
            print("[SERVER] Trying to fall back to CPU...")
            args.device = "cpu"
            server = NaVILAServer(args)
            server.start_server(host=args.host, port=args.port)
        else:
            raise e


if __name__ == "__main__":
    main()
