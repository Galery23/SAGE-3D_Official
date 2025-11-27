#!/usr/bin/env python3
"""
MLLM VLN Server for SAGE-3D Benchmark.

This script creates a socket-based server that loads various MLLM models 
(Qwen2.5-VL, LLaVA, InternVL, etc.) and processes VLN inference requests.

Supported Models:
    - Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    - LLaVA: https://huggingface.co/llava-hf/llava-1.5-7b-hf
    - InternVL: https://huggingface.co/OpenGVLab/InternVL2-8B

Usage:
    python mllm_server.py --model_type qwen-vl --model_path /path/to/model --port 7777

Server Protocol:
    - Uses TCP socket communication
    - Request format: JSON with 'images' (base64 list) and 'query' (string)
    - Response format: JSON with 'result' (action string) and 'status'
    - Data is prefixed with 8-byte big-endian size header
"""

import os
import sys
import json
import socket
import argparse
import traceback
import base64
import io
import re
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image


class VLNPromptTemplate:
    """VLN task prompt templates."""
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get system prompt for VLN task."""
        return """You are a navigation agent. Given an image and instruction, predict the next action.

RESPOND WITH ONLY ONE OF THESE ACTIONS:
- MOVE_FORWARD
- TURN_LEFT  
- TURN_RIGHT
- STOP

Rules:
1. NO explanations or descriptions
2. ONLY output the action word
3. Use STOP when task is complete
4. Choose the action that best follows the instruction"""

    @staticmethod
    def get_user_prompt(instruction: str) -> str:
        """Get user prompt with instruction."""
        return f"""Instruction: {instruction}

Next action:"""


class MLLMAdapter(ABC):
    """Base class for MLLM adapters."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self):
        """Load model from HuggingFace."""
        pass
    
    @abstractmethod
    def generate_response(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """Generate response from model."""
        pass
    
    def extract_action(self, response: str) -> str:
        """Extract VLN action from model response."""
        response_upper = response.upper().strip()
        
        valid_actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
        
        # Direct match
        for action in valid_actions:
            if action in response_upper:
                return action
        
        # Partial match
        if any(word in response_upper for word in ["FORWARD", "AHEAD", "MOVE", "GO"]):
            return "MOVE_FORWARD"
        elif any(word in response_upper for word in ["LEFT"]):
            return "TURN_LEFT"  
        elif any(word in response_upper for word in ["RIGHT"]):
            return "TURN_RIGHT"
        elif any(word in response_upper for word in ["STOP", "HALT", "DONE", "FINISH"]):
            return "STOP"
        
        # Default action
        print(f"[WARN] Cannot parse action from: {response[:100]}...")
        return "MOVE_FORWARD"
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        if self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache()


class QwenVLAdapter(MLLMAdapter):
    """Qwen2.5-VL adapter using HuggingFace Transformers."""
    
    def load_model(self):
        """Load Qwen2.5-VL model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            print("[QWEN] Loading Qwen2.5-VL model...")
            print(f"[QWEN] Model path: {self.model_path}")
            
            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            print("[QWEN] Model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load Qwen2.5-VL: {e}")
            raise
    
    def generate_response(self, image: Image.Image, instruction: str, **kwargs) -> str:
        """Generate response using Qwen2.5-VL."""
        try:
            system_prompt = VLNPromptTemplate.get_system_prompt()
            user_prompt = VLNPromptTemplate.get_user_prompt(instruction)
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', 64),
                    temperature=kwargs.get('temperature', 0.1),
                    do_sample=True
                )
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True
            )[0]
            
            return self.extract_action(output_text)
            
        except Exception as e:
            print(f"[ERROR] Qwen2.5-VL inference failed: {e}")
            return "MOVE_FORWARD"


class LLaVAAdapter(MLLMAdapter):
    """LLaVA adapter using HuggingFace Transformers."""
    
    def load_model(self):
        """Load LLaVA model."""
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            
            print("[LLAVA] Loading LLaVA model...")
            print(f"[LLAVA] Model path: {self.model_path}")
            
            # Load model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            print("[LLAVA] Model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load LLaVA: {e}")
            raise
    
    def generate_response(self, image: Image.Image, instruction: str, **kwargs) -> str:
        """Generate response using LLaVA."""
        try:
            system_prompt = VLNPromptTemplate.get_system_prompt()
            user_prompt = VLNPromptTemplate.get_user_prompt(instruction)
            
            # LLaVA prompt format
            prompt = f"USER: <image>\n{system_prompt}\n\n{user_prompt}\nASSISTANT:"
            
            # Process input
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', 64),
                    temperature=kwargs.get('temperature', 0.1),
                    do_sample=True
                )
            
            # Decode
            output_text = self.processor.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return self.extract_action(output_text)
            
        except Exception as e:
            print(f"[ERROR] LLaVA inference failed: {e}")
            return "MOVE_FORWARD"


class InternVLAdapter(MLLMAdapter):
    """InternVL adapter using HuggingFace Transformers."""
    
    def load_model(self):
        """Load InternVL model."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print("[INTERNVL] Loading InternVL model...")
            print(f"[INTERNVL] Model path: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print("[INTERNVL] Model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load InternVL: {e}")
            raise
    
    def generate_response(self, image: Image.Image, instruction: str, **kwargs) -> str:
        """Generate response using InternVL."""
        try:
            system_prompt = VLNPromptTemplate.get_system_prompt()
            user_prompt = VLNPromptTemplate.get_user_prompt(instruction)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if needed
            max_size = 448
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Prepare pixel values
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            pixel_values = transform(image).unsqueeze(0)
            pixel_values = pixel_values.to(dtype=torch.bfloat16, device=self.model.device)
            
            # Generate using chat method
            generation_config = {
                'max_new_tokens': kwargs.get('max_new_tokens', 32),
                'temperature': kwargs.get('temperature', 0.1),
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                generation_config
            )
            
            return self.extract_action(response)
            
        except Exception as e:
            print(f"[ERROR] InternVL inference failed: {e}")
            return "MOVE_FORWARD"


class MLLMServer:
    """MLLM VLN Server."""
    
    def __init__(self, adapter: MLLMAdapter, host: str = "localhost", port: int = 7777):
        self.adapter = adapter
        self.host = host
        self.port = port
        self.socket = None
        self.is_running = False
        
        # Statistics
        self.total_requests = 0
        self.total_inference_time = 0.0
        
    def start_server(self):
        """Start the server."""
        try:
            # Load model
            print("[SERVER] Loading model...")
            self.adapter.load_model()
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.is_running = True
            
            print(f"[SERVER] Listening on {self.host}:{self.port}")
            print("[SERVER] Ready to receive requests...")
            
            while self.is_running:
                try:
                    client_socket, addr = self.socket.accept()
                    print(f"[SERVER] Client connected: {addr}")
                    self._handle_client(client_socket)
                except Exception as e:
                    if self.is_running:
                        print(f"[ERROR] Client handling failed: {e}")
                        
        except Exception as e:
            print(f"[ERROR] Server startup failed: {e}")
            raise
        finally:
            self.cleanup()
    
    def _handle_client(self, client_socket):
        """Handle client connection."""
        try:
            while True:
                # Receive size header (8 bytes)
                size_data = client_socket.recv(8)
                if not size_data or len(size_data) != 8:
                    break
                
                data_size = int.from_bytes(size_data, 'big')
                
                # Receive JSON data
                json_data = b''
                while len(json_data) < data_size:
                    chunk = client_socket.recv(min(4096, data_size - len(json_data)))
                    if not chunk:
                        break
                    json_data += chunk
                
                if len(json_data) != data_size:
                    print(f"[ERROR] Data size mismatch: expected {data_size}, got {len(json_data)}")
                    continue
                
                # Parse JSON
                try:
                    request = json.loads(json_data.decode('utf-8'))
                    
                    # Handle VLN request
                    if 'images' in request and 'query' in request:
                        response = self._handle_vln_query(request)
                    else:
                        response = {"status": "error", "message": "Missing images or query"}
                        
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    print(f"[ERROR] Data parse failed: {e}")
                    response = {"status": "error", "message": "Invalid data format"}
                
                # Send response
                try:
                    response_json = json.dumps(response)
                    response_data = response_json.encode('utf-8')
                    
                    client_socket.sendall(len(response_data).to_bytes(8, 'big'))
                    client_socket.sendall(response_data)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to send response: {e}")
                    break
                
        except Exception as e:
            print(f"[ERROR] Client handling error: {e}")
            traceback.print_exc()
        finally:
            client_socket.close()
    
    def _handle_vln_query(self, request):
        """Handle VLN query request."""
        try:
            images_b64 = request['images']
            instruction = request['query']
            
            # Decode images
            images = []
            for img_b64 in images_b64:
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                images.append(img)
            
            print(f"[SERVER] Processing {len(images)} images, instruction: {instruction[:50]}...")
            
            # Run inference (use first image)
            start_time = time.time()
            result = self.adapter.generate_response(images[0], instruction)
            inference_time = time.time() - start_time
            
            # Update statistics
            self.total_requests += 1
            self.total_inference_time += inference_time
            
            print(f"[SERVER] Result: {result} ({inference_time:.2f}s)")
            
            return {
                "status": "success",
                "result": result,
                "inference_time": inference_time
            }
            
        except Exception as e:
            print(f"[ERROR] VLN query failed: {e}")
            traceback.print_exc()
            return {
                "status": "error", 
                "message": f"VLN query failed: {str(e)}",
                "result": "MOVE_FORWARD"
            }
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if self.socket:
            self.socket.close()
        if self.adapter:
            self.adapter.cleanup()
        print("[SERVER] Resources cleaned up")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MLLM VLN Server for SAGE-3D Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Models:
    qwen-vl   : Qwen2.5-VL (https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
    llava     : LLaVA (https://huggingface.co/llava-hf/llava-1.5-7b-hf)
    internvl  : InternVL (https://huggingface.co/OpenGVLab/InternVL2-8B)

Examples:
    # Start Qwen2.5-VL server
    python mllm_server.py --model_type qwen-vl \\
        --model_path /path/to/Qwen2.5-VL-7B-Instruct --port 7777

    # Start LLaVA server
    python mllm_server.py --model_type llava \\
        --model_path /path/to/llava-1.5-7b-hf --port 7778

    # Start InternVL server
    python mllm_server.py --model_type internvl \\
        --model_path /path/to/InternVL2-8B --port 7779
        """
    )
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=['qwen-vl', 'llava', 'internvl'],
        help="Model type to load"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to model checkpoint or HuggingFace model ID"
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Host address to bind (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7777,
        help="Port number to bind (default: 7777)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on: 'cuda' or 'cpu' (default: cuda)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("MLLM VLN Server for SAGE-3D Benchmark")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Server: {args.host}:{args.port}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create adapter based on model type
    if args.model_type == 'qwen-vl':
        adapter = QwenVLAdapter(args.model_path, args.device)
    elif args.model_type == 'llava':
        adapter = LLaVAAdapter(args.model_path, args.device)
    elif args.model_type == 'internvl':
        adapter = InternVLAdapter(args.model_path, args.device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Create and start server
    server = MLLMServer(adapter, args.host, args.port)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
    finally:
        server.cleanup()


if __name__ == "__main__":
    main()

