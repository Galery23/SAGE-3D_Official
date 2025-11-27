#!/usr/bin/env python3
"""
NaVid VLM Server for SAGE-3D Benchmark.

This script creates a socket-based server that loads the NaVid model and
processes VLN (Vision-and-Language Navigation) inference requests.

IMPORTANT:
    - This script should be run within the NaVid-VLN-CE project environment.
    - NaVid repository: https://github.com/jzhzhang/NaVid-VLN-CE
    - Copy this file to NaVid-VLN-CE project root and run from there.

Usage:
    1. Clone NaVid-VLN-CE repository:
       git clone https://github.com/jzhzhang/NaVid-VLN-CE.git
       cd NaVid-VLN-CE

    2. Setup environment following NaVid installation guide

    3. Download model weights:
       - NaVid model: https://huggingface.co/jzhzhang/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split
       - EVA-ViT-G: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth

    4. Copy this server script to NaVid-VLN-CE:
       cp navid_server.py /path/to/NaVid-VLN-CE/

    5. Run the server:
       cd /path/to/NaVid-VLN-CE
       conda activate NaVid
       python navid_server.py \\
           --model_path /path/to/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split \\
           --eva_vit_path /path/to/eva_vit_g.pth \\
           --port 4321

Server Protocol:
    - Uses TCP socket communication
    - Request format: JSON with 'images' (base64 list) and 'instruction' (string)
    - Response format: JSON with velocity commands (vx, vy, yaw_rate, duration_s, stop)
    - Data is prefixed with 8-byte big-endian size header

Related Repositories:
    - NaVid: https://github.com/jzhzhang/NaVid-VLN-CE
    - Model Weights: https://huggingface.co/jzhzhang/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split
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
import random
import math
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image

# NaVid specific imports - these require the NaVid environment
try:
    from navid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from navid.conversation import conv_templates, SeparatorStyle
    from navid.model.builder import load_pretrained_model
    from navid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
except ImportError as e:
    print(f"[ERROR] NaVid module import failed: {e}")
    print("Please ensure PYTHONPATH includes NaVid-VLN-CE project root directory")
    print("Example: export PYTHONPATH=$PYTHONPATH:/path/to/NaVid-VLN-CE")
    raise


class NaVidVLMCore:
    """NaVid VLM Core based on original NaVid Agent implementation."""
    
    def __init__(self, model_path: str, eva_vit_path: str, device: str = "cuda"):
        """Initialize NaVid model.
        
        Args:
            model_path: Path to NaVid model checkpoint
            eva_vit_path: Path to EVA-ViT-G weights file
            device: Device to run on ('cuda' or 'cpu')
        """
        print("[NAVID] Initializing NaVid model...")
        
        self.model_path = model_path
        self.eva_vit_path = eva_vit_path
        self.device = device
        self.conv_mode = "vicuna_v1"
        
        # Load model using NaVid's builder
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, self.model_name
        )
        
        print("[NAVID] Model initialization complete")
        
        # NaVid prompt template
        self.prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. "
            "Analyze this series of images to decide your next move, which could involve "
            "turning left or right by a specific degree or moving forward a certain distance."
        )
        
        # State variables
        self.history_rgb_tensor = None
        self.rgb_list = []
        self.pending_action_list = []
        self.last_action = None
        self.count_stop = 0
        self.first_forward = False
        
        self.reset()
        
    def reset(self):
        """Reset state for new episode."""
        self.history_rgb_tensor = None
        self.rgb_list = []
        self.pending_action_list = []
        self.last_action = None
        self.count_stop = 0
        self.first_forward = False
        print("[NAVID] State reset")
        
    def process_images(self, rgb_list):
        """Process images using NaVid's image processor."""
        start_img_index = 0
        
        if self.history_rgb_tensor is not None:
            start_img_index = self.history_rgb_tensor.shape[0]
        
        batch_image = np.asarray(rgb_list[start_img_index:])
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        if self.history_rgb_tensor is None:
            self.history_rgb_tensor = video
        else:
            self.history_rgb_tensor = torch.cat((self.history_rgb_tensor, video), dim=0)
        
        return [self.history_rgb_tensor]
    
    def predict_inference(self, prompt):
        """Run NaVid inference."""
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        # NaVid special tokens
        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IMAGE_SEPARATOR = "<image_sep>"
        
        # Tokenize special tokens
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_separator = self.tokenizer(IMAGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_separator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[WARN] {n_diff_input_output} output_ids differ from input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs

    def extract_result(self, output):
        """Extract action from model output.
        
        Returns:
            Tuple of (action_id, value) where action_id is:
            - 0: STOP
            - 1: MOVE_FORWARD
            - 2: TURN_LEFT
            - 3: TURN_RIGHT
        """
        output_lower = output.lower()
        print(f"[NAVID] Parsing output: {output}")
        
        if "stop" in output_lower:
            print(f"[NAVID] Action: STOP")
            return 0, None
        elif "forward" in output_lower:
            match = re.search(r'-?\d+', output)
            if match is None:
                print(f"[NAVID] Action: FORWARD (default distance 50)")
                return 1, 50.0
            match = match.group()
            print(f"[NAVID] Action: FORWARD, distance: {match}")
            return 1, float(match)
        elif "left" in output_lower:
            match = re.search(r'-?\d+', output)
            if match is None:
                print(f"[NAVID] Action: LEFT (default angle 30)")
                return 2, 30.0
            match = match.group()
            print(f"[NAVID] Action: LEFT, angle: {match}")
            return 2, float(match)
        elif "right" in output_lower:
            match = re.search(r'-?\d+', output)
            if match is None:
                print(f"[NAVID] Action: RIGHT (default angle 30)")
                return 3, 30.0
            match = match.group()
            print(f"[NAVID] Action: RIGHT, angle: {match}")
            return 3, float(match)

        print(f"[NAVID] Action: UNKNOWN, returning None")
        return None, None

    def process_request(self, instruction: str, images: List[Image.Image], current_yaw: float = 0.0) -> Dict[str, Any]:
        """Process VLN request.
        
        Args:
            instruction: Navigation instruction text
            images: List of PIL Images
            current_yaw: Current robot yaw angle
            
        Returns:
            Action response dictionary with velocity commands
        """
        try:
            print(f"[NAVID] Processing request: {instruction[:50]}...")
            
            # Convert images to numpy arrays
            rgb_arrays = []
            for img in images:
                if isinstance(img, Image.Image):
                    rgb_array = np.array(img)
                    rgb_arrays.append(rgb_array)
                else:
                    rgb_arrays.append(img)
            
            # Add latest image to history
            if rgb_arrays:
                self.rgb_list.append(rgb_arrays[-1])
            
            # Return pending action if available
            if len(self.pending_action_list) != 0:
                temp_action = self.pending_action_list.pop(0)
                print(f"[NAVID] Returning pending action: {temp_action}")
                return self._convert_action_to_response(temp_action, "Pending action")
            
            # Run NaVid inference
            navigation_qs = self.prompt_template.format(instruction)
            navigation = self.predict_inference(navigation_qs)
            print(f"[NAVID] Model output: {navigation}")
            
            # Extract action and generate pending actions
            action_index, num = self.extract_result(navigation)
            
            # Generate pending action list based on action type
            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, int(num/25))):
                    self.pending_action_list.append(1)
            elif action_index == 2:
                for _ in range(min(3, int(num/30))):
                    self.pending_action_list.append(2)
            elif action_index == 3:
                for _ in range(min(3, int(num/30))):
                    self.pending_action_list.append(3)
            
            if action_index is None or len(self.pending_action_list) == 0:
                self.pending_action_list.append(random.randint(1, 3))
            
            # Return first action
            if self.pending_action_list:
                next_action = self.pending_action_list.pop(0)
                return self._convert_action_to_response(next_action, navigation)
            else:
                return self._convert_action_to_response(0, navigation)
                
        except Exception as e:
            print(f"[NAVID] Error processing request: {e}")
            traceback.print_exc()
            return {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration_s": 0.0, "stop": True, 
                   "raw_response": f"Error: {e}", "parsed_from": "error"}
    
    def _convert_action_to_response(self, action_id: int, raw_text: str) -> Dict[str, Any]:
        """Convert action ID to VLN response format.
        
        Args:
            action_id: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
            raw_text: Raw model output text
            
        Returns:
            Response dictionary with velocity commands
        """
        if action_id == 0:  # STOP
            return {
                "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration_s": 0.0, "stop": True,
                "raw_response": raw_text, "parsed_from": "navid"
            }
        elif action_id == 1:  # MOVE_FORWARD
            return {
                "vx": 0.25, "vy": 0.0, "yaw_rate": 0.0, "duration_s": 1.0, "stop": False,
                "raw_response": raw_text, "parsed_from": "navid"
            }
        elif action_id == 2:  # TURN_LEFT
            return {
                "vx": 0.0, "vy": 0.0, "yaw_rate": math.radians(30)/1.0, "duration_s": 1.0, "stop": False,
                "raw_response": raw_text, "parsed_from": "navid"
            }
        elif action_id == 3:  # TURN_RIGHT
            return {
                "vx": 0.0, "vy": 0.0, "yaw_rate": -math.radians(30)/1.0, "duration_s": 1.0, "stop": False,
                "raw_response": raw_text, "parsed_from": "navid"
            }
        else:
            return {
                "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration_s": 0.0, "stop": True,
                "raw_response": raw_text, "parsed_from": "navid"
            }


class NaVidSocketServer:
    """Socket server for NaVid VLM."""
    
    def __init__(self, host: str, port: int, vlm_core: NaVidVLMCore):
        """Initialize socket server.
        
        Args:
            host: Host address to bind to
            port: Port number to listen on
            vlm_core: NaVid VLM core instance
        """
        self.host = host
        self.port = port
        self.vlm_core = vlm_core
        self.running = False
        
    def start_server(self):
        """Start the socket server."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        self.running = True
        
        print(f"[SERVER] NaVid Socket Server listening on {self.host}:{self.port}")
        print(f"[SERVER] Ready to receive requests...")
        
        try:
            while self.running:
                conn, addr = server_socket.accept()
                print(f"[SERVER] Client connected: {addr}")
                
                try:
                    self._handle_client(conn, addr)
                except Exception as e:
                    print(f"[SERVER] Error handling client {addr}: {e}")
                finally:
                    conn.close()
                    print(f"[SERVER] Client disconnected: {addr}")
                    
        except Exception as e:
            print(f"[SERVER] Server error: {e}")
        finally:
            server_socket.close()
    
    def _handle_client(self, conn: socket.socket, addr):
        """Handle client connection."""
        # Receive data size (8 bytes, big endian)
        size_data = conn.recv(8)
        if len(size_data) != 8:
            return
            
        data_size = int.from_bytes(size_data, 'big')
        
        # Receive request data
        request_data = b""
        while len(request_data) < data_size:
            chunk = conn.recv(min(4096, data_size - len(request_data)))
            if not chunk:
                break
            request_data += chunk
        
        if len(request_data) != data_size:
            return
        
        # Parse request
        try:
            request = json.loads(request_data.decode('utf-8'))
        except json.JSONDecodeError as e:
            print(f"[SERVER] JSON parse error: {e}")
            return
        
        # Handle request
        if request.get('action') == 'reset':
            self.vlm_core.reset()
            response = {"status": "reset_complete"}
        else:
            # VLN step request
            instruction = request.get('instruction', '') or request.get('query', '')
            current_yaw = request.get('current_yaw', 0.0)
            
            # Decode images
            images = []
            if 'images' in request:
                for img_b64 in request['images']:
                    img_data = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    images.append(img)
            
            # Get NaVid response
            navid_response = self.vlm_core.process_request(instruction, images, current_yaw)
            
            # Build response
            response = navid_response.copy()
            response["status"] = "success"
            if "text_response" not in response:
                response["text_response"] = response.get('raw_response', 'No response')
        
        # Send response
        response_data = json.dumps(response).encode('utf-8')
        response_size = len(response_data)
        
        conn.sendall(response_size.to_bytes(8, 'big'))
        conn.sendall(response_data)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NaVid VLM Server for SAGE-3D Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server with default settings
    python navid_server.py \\
        --model_path /path/to/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split \\
        --eva_vit_path /path/to/eva_vit_g.pth

    # Start server on specific port
    python navid_server.py \\
        --model_path /path/to/model \\
        --eva_vit_path /path/to/eva_vit_g.pth \\
        --port 8888
        """
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to NaVid model checkpoint directory"
    )
    parser.add_argument(
        "--eva_vit_path", type=str, required=True,
        help="Path to EVA-ViT-G weights file (eva_vit_g.pth)"
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Host address to bind the server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=4321,
        help="Port number to bind the server (default: 4321)"
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
    print("NaVid VLM Server for SAGE-3D Benchmark")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"EVA-ViT path: {args.eva_vit_path}")
    print(f"Server: {args.host}:{args.port}")
    print(f"Device: {args.device}")
    print("=" * 60)

    try:
        vlm_core = NaVidVLMCore(
            model_path=args.model_path,
            eva_vit_path=args.eva_vit_path,
            device=args.device
        )
        
        server = NaVidSocketServer(args.host, args.port, vlm_core)
        server.start_server()
            
    except Exception as e:
        print(f"[SERVER] Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

