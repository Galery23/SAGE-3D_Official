#!/usr/bin/env python3
"""
Modular VLM Client Architecture for SAGE-3D Benchmark.

Supports different input types (RGB vs RGB-D) and output types (trajectory vs text).
"""

import io
import json
import math
import socket
import base64
import logging
import numpy as np
import requests
from typing import List, Dict, Any, Optional, Union, Callable
from PIL import Image
from abc import ABC, abstractmethod

try:
    import cv2
except ImportError:
    cv2 = None

# Global log function
_global_log_function = None


def set_log_function(log_func: Callable[[str], None]) -> None:
    """Set global log function."""
    global _global_log_function
    _global_log_function = log_func


def _log_and_print(msg: str) -> None:
    """Internal log function."""
    print(msg, flush=True)
    if _global_log_function:
        try:
            _global_log_function(msg)
        except:
            pass


class InputProcessor(ABC):
    """Base class for input processors."""

    @abstractmethod
    def process_input(self, rgb_images: List[Image.Image], depth_images: List[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Process input data, return standardized format."""
        pass


class RGBInputProcessor(InputProcessor):
    """RGB input processor."""

    def __init__(self, history_frames: int = 8):
        self.history_frames = history_frames
        self.image_history = []

    def process_input(self, rgb_images: List[Image.Image], depth_images: List[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Process RGB input."""
        if not rgb_images or len(rgb_images) == 0:
            raise ValueError("RGB image list is empty")

        current_image = rgb_images[0]

        # Update image history
        self.image_history.append(current_image)
        if len(self.image_history) > self.history_frames:
            self.image_history = self.image_history[-self.history_frames:]

        # Prepare image sequence
        image_sequence = self.image_history.copy()
        while len(image_sequence) < self.history_frames:
            image_sequence.insert(0, image_sequence[0] if image_sequence else current_image)

        return {
            "input_type": "rgb",
            "images": image_sequence,
            "current_image": current_image
        }


class RGBDInputProcessor(InputProcessor):
    """RGB-D input processor."""

    def process_input(self, rgb_images: List[Image.Image], depth_images: List[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Process RGB-D input."""
        if not rgb_images or len(rgb_images) == 0:
            raise ValueError("RGB image list is empty")

        rgb_image = rgb_images[0]

        # Process depth image
        if depth_images is not None and len(depth_images) > 0:
            depth_image = depth_images[0].astype(np.float32)
            _log_and_print(f"[RGBD_INPUT] Using real depth: shape={depth_image.shape}, range=[{depth_image.min():.3f}, {depth_image.max():.3f}]m")
        else:
            # Create default depth image
            if hasattr(rgb_image, 'size'):
                w, h = rgb_image.size
                depth_image = np.full((h, w), 5.0, dtype=np.float32)
            else:
                depth_image = np.full((480, 640), 5.0, dtype=np.float32)
            _log_and_print(f"[RGBD_INPUT] Using default depth: shape={depth_image.shape}")

        return {
            "input_type": "rgbd",
            "rgb_image": rgb_image,
            "depth_image": depth_image
        }


class OutputParser(ABC):
    """Base class for output parsers."""

    @abstractmethod
    def parse_output(self, raw_response: Any, current_yaw: float = 0.0, **kwargs) -> Dict[str, Any]:
        """Parse model output to standard action format."""
        pass


class TrajectoryOutputParser(OutputParser):
    """Trajectory output parser."""

    def parse_output(self, raw_response: Any, current_yaw: float = 0.0, **kwargs) -> Dict[str, Any]:
        """Parse trajectory output to action commands."""
        if isinstance(raw_response, dict) and "trajectory" in raw_response:
            trajectory = raw_response["trajectory"]
        elif isinstance(raw_response, np.ndarray):
            trajectory = raw_response
        else:
            trajectory = np.array(raw_response)

        if len(trajectory.shape) == 3 and trajectory.shape[0] == 1:
            trajectory = trajectory[0]  # Remove batch dimension

        # Parse trajectory to velocity and angular velocity
        vx, vy, yaw_rate = self._parse_trajectory_to_velocity(trajectory, current_yaw)

        return {
            "vx": vx,
            "vy": vy,
            "yaw_rate": yaw_rate,
            "duration_s": 1.0,
            "stop": False,
            "raw_response": f"Trajectory: {trajectory[0] if len(trajectory) > 0 else 'empty'}",
            "parsed_from": "trajectory"
        }

    def _parse_trajectory_to_velocity(self, trajectory: np.ndarray, current_yaw: float = 0.0) -> tuple:
        """Parse trajectory to velocity and angular velocity."""
        try:
            if len(trajectory.shape) == 2 and trajectory.shape[1] >= 3:
                # Calculate angular velocity
                yaw_rate = 0.0
                if len(trajectory) >= 3:
                    directions = []
                    for i in range(min(5, len(trajectory) - 1)):
                        dx = trajectory[i+1][0] - trajectory[i][0]
                        dy = trajectory[i+1][1] - trajectory[i][1]
                        distance = np.sqrt(dx*dx + dy*dy)

                        if distance > 0.005:
                            direction = math.atan2(dy, dx)
                            directions.append(direction)

                    if len(directions) >= 2:
                        angle_changes = []
                        for i in range(len(directions) - 1):
                            angle_diff = directions[i+1] - directions[i]
                            while angle_diff > math.pi:
                                angle_diff -= 2*math.pi
                            while angle_diff < -math.pi:
                                angle_diff += 2*math.pi
                            angle_changes.append(angle_diff)

                        if angle_changes:
                            avg_angle_change = np.mean(angle_changes)
                            yaw_rate = avg_angle_change * 2.0
                            max_yaw_rate = math.radians(60)
                            yaw_rate = np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate)

                # Find first meaningful movement target
                for i in range(len(trajectory)):
                    x, y, z = trajectory[i][:3]
                    distance_2d = np.sqrt(x*x + y*y)

                    if distance_2d > 0.01:  # 1cm threshold
                        scale_factor = 3.0
                        robot_vx = float(-x * scale_factor)
                        robot_vy = float(y * scale_factor)

                        # Coordinate transform
                        cos_yaw = math.cos(current_yaw)
                        sin_yaw = math.sin(current_yaw)

                        world_vx = robot_vx * cos_yaw - robot_vy * sin_yaw
                        world_vy = robot_vx * sin_yaw + robot_vy * cos_yaw

                        # Limit velocity range
                        max_speed = 0.5
                        current_speed = np.sqrt(world_vx*world_vx + world_vy*world_vy)

                        if current_speed > max_speed:
                            world_vx = world_vx * max_speed / current_speed
                            world_vy = world_vy * max_speed / current_speed

                        return world_vx, world_vy, yaw_rate

                return 0.0, 0.0, yaw_rate
            else:
                return 0.0, 0.0, 0.0

        except Exception as e:
            _log_and_print(f"[TRAJECTORY_PARSER] Error parsing trajectory: {e}")
            return 0.0, 0.0, 0.0


class TextOutputParser(OutputParser):
    """Text output parser."""

    def parse_output(self, raw_response: Any, current_yaw: float = 0.0, **kwargs) -> Dict[str, Any]:
        """Parse text output to action commands."""
        if isinstance(raw_response, dict):
            text = raw_response.get("text_response", str(raw_response))
        else:
            text = str(raw_response)

        # Parse text action
        action = self._parse_text_to_action(text)

        return {
            "vx": action["vx"],
            "vy": action["vy"],
            "yaw_rate": action["yaw_rate"],
            "duration_s": action["duration_s"],
            "stop": action["stop"],
            "raw_response": text,
            "parsed_from": "text"
        }

    def _parse_text_to_action(self, text: str) -> Dict[str, Any]:
        """Parse text to action commands."""
        import re
        text_lower = text.lower()

        # Default action
        action = {
            "vx": 0.0,
            "vy": 0.0,
            "yaw_rate": 0.0,
            "duration_s": 1.0,
            "stop": False
        }

        try:
            # Stop commands
            if any(word in text_lower for word in ["stop", "halt", "complete", "finish", "done"]):
                action["stop"] = True
                return action

            # Forward commands
            if any(word in text_lower for word in ["forward", "ahead", "straight", "move"]):
                distance_match = re.search(r'(\d+\.?\d*)\s*(?:meter|metre|m|step)', text_lower)
                if distance_match:
                    distance = float(distance_match.group(1))
                    action["vx"] = min(distance / action["duration_s"], 0.5)
                else:
                    action["vx"] = 0.3  # Default forward speed

            # Turn commands
            turn_left = any(word in text_lower for word in ["left", "turn left"])
            turn_right = any(word in text_lower for word in ["right", "turn right"])

            if turn_left or turn_right:
                angle_match = re.search(r'(\d+\.?\d*)\s*(?:degree|deg|°)', text_lower)
                if angle_match:
                    angle = float(angle_match.group(1))
                    angle_rad = math.radians(angle)
                else:
                    angle_rad = math.radians(30)  # Default 30 degrees

                if turn_left:
                    action["yaw_rate"] = angle_rad / action["duration_s"]
                else:
                    action["yaw_rate"] = -angle_rad / action["duration_s"]

            # Backward commands
            if any(word in text_lower for word in ["back", "backward", "reverse"]):
                action["vx"] = -0.2

            _log_and_print(f"[TEXT_PARSER] Parsed action: {text[:50]}... -> vx={action['vx']:.3f}, yaw_rate={math.degrees(action['yaw_rate']):.1f}°/s")

        except Exception as e:
            _log_and_print(f"[TEXT_PARSER] Text parsing failed: {e}")

        return action


class CommunicationProtocol(ABC):
    """Base class for communication protocols."""

    @abstractmethod
    def send_request(self, processed_input: Dict[str, Any], instruction: str, host: str, port: int, **kwargs) -> Any:
        """Send request to VLM server."""
        pass


class HTTPProtocol(CommunicationProtocol):
    """HTTP communication protocol."""

    def send_request(self, processed_input: Dict[str, Any], instruction: str, host: str, port: int, **kwargs) -> Any:
        """Send request via HTTP."""
        _log_and_print(f"[HTTP_PROTOCOL] Processing HTTP request - input type: {processed_input['input_type']}, target: {host}:{port}")
        if processed_input["input_type"] == "rgbd":
            return self._send_rgbd_request(processed_input, instruction, host, port)
        elif processed_input["input_type"] == "rgb":
            return self._send_rgb_request(processed_input, instruction, host, port)
        else:
            raise ValueError(f"HTTP protocol does not support input type: {processed_input['input_type']}")

    def _send_rgbd_request(self, processed_input: Dict[str, Any], instruction: str, host: str, port: int) -> Dict[str, Any]:
        """Send RGB-D request."""
        rgb_image = processed_input["rgb_image"]
        depth_image = processed_input["depth_image"]

        # Convert RGB image
        if hasattr(rgb_image, 'convert'):
            rgb_array = np.array(rgb_image.convert('RGB'))
        else:
            rgb_array = np.array(rgb_image)

        # Convert to BGR format
        if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
            if cv2 is not None:
                bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = rgb_array[:, :, ::-1]
        else:
            bgr_image = rgb_array

        # Encode RGB image
        if cv2 is not None:
            _, rgb_buffer = cv2.imencode('.jpg', bgr_image)
            rgb_bytes = rgb_buffer.tobytes()
        else:
            rgb_pil = Image.fromarray(bgr_image[:, :, ::-1])
            rgb_bytes_io = io.BytesIO()
            rgb_pil.save(rgb_bytes_io, format='JPEG')
            rgb_bytes = rgb_bytes_io.getvalue()

        # Encode depth image
        depth_array_clamped = np.clip(depth_image, 0.0, 6.5)
        depth_array_encoded = (depth_array_clamped * 10000.0).astype(np.uint16)

        if cv2 is not None:
            _, depth_buffer = cv2.imencode('.png', depth_array_encoded)
            depth_bytes = depth_buffer.tobytes()
        else:
            depth_pil = Image.fromarray(depth_array_encoded)
            depth_bytes_io = io.BytesIO()
            depth_pil.save(depth_bytes_io, format='PNG')
            depth_bytes = depth_bytes_io.getvalue()

        # Send HTTP request
        url = f"http://{host}:{port}/nogoal_step"
        files = {
            'image': ('image.jpg', rgb_bytes, 'image/jpeg'),
            'depth': ('depth.png', depth_bytes, 'image/png')
        }

        response = requests.post(url, files=files, timeout=30)

        if response.status_code == 200:
            result = response.json()
            trajectory = np.array(result['trajectory'])
            _log_and_print(f"[HTTP_PROTOCOL] Received trajectory response, shape: {trajectory.shape}")
            return {"trajectory": trajectory}
        else:
            raise Exception(f"HTTP request failed, status code: {response.status_code}, response: {response.text}")

    def _send_rgb_request(self, processed_input: Dict[str, Any], instruction: str, host: str, port: int) -> Dict[str, Any]:
        """Send RGB request to HTTP API server."""
        from io import BytesIO

        # Get RGB images
        rgb_images = processed_input.get("images", [processed_input.get("current_image")])

        # Use first image
        rgb_image = rgb_images[0] if isinstance(rgb_images, list) and rgb_images else rgb_images

        # Convert to PIL image format
        if hasattr(rgb_image, 'convert'):
            pil_image = rgb_image.convert('RGB')
        else:
            pil_image = Image.fromarray(np.array(rgb_image).astype(np.uint8))

        # Encode to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_data = buffer.getvalue()
        img_b64 = base64.b64encode(img_data).decode('utf-8')

        # Build request data (match NaVid server format)
        request_data = {
            'images': [img_b64],
            'instruction': instruction,
            'current_yaw': 0.0
        }

        # Send HTTP POST request
        url = f"http://{host}:{port}/vln_step"
        headers = {'Content-Type': 'application/json'}

        _log_and_print(f"[HTTP_PROTOCOL] Sending RGB request to: {url}")

        response = requests.post(url, json=request_data, headers=headers, timeout=60)

        if response.status_code == 200:
            result = response.json()
            action_text = result.get('result', 'MOVE_FORWARD')
            _log_and_print(f"[HTTP_PROTOCOL] Received action response: {action_text}")
            return {"text": action_text}
        else:
            raise Exception(f"HTTP request failed, status code: {response.status_code}, response: {response.text}")


class SocketProtocol(CommunicationProtocol):
    """Socket communication protocol."""

    def send_request(self, processed_input: Dict[str, Any], instruction: str, host: str, port: int, **kwargs) -> Any:
        """Send request via Socket."""
        if processed_input["input_type"] == "rgb":
            return self._send_rgb_request(processed_input, instruction, host, port)
        else:
            raise ValueError(f"Socket protocol does not support input type: {processed_input['input_type']}")

    def _send_rgb_request(self, processed_input: Dict[str, Any], instruction: str, host: str, port: int) -> str:
        """Send RGB sequence request."""
        images = processed_input["images"]

        # Encode images to base64
        encoded_images = []
        for img in images:
            if hasattr(img, 'convert'):
                img = img.convert('RGB')
            else:
                img = Image.fromarray(img).convert('RGB')

            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            encoded_images.append(img_str)

        # Prepare request data
        request_data = {
            'images': encoded_images,
            'query': instruction
        }

        # Establish Socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(60)  # 60 second timeout
        sock.connect((host, port))

        try:
            # Send data
            data = json.dumps(request_data).encode('utf-8')
            _log_and_print(f"[SOCKET_CLIENT] Sending data length: {len(data)} bytes")
            sock.sendall(len(data).to_bytes(8, 'big'))
            sock.sendall(data)

            # Receive response
            size_data = sock.recv(8)
            size = int.from_bytes(size_data, 'big')

            response_data = b''
            while len(response_data) < size:
                packet = sock.recv(4096)
                if not packet:
                    break
                response_data += packet

            response = json.loads(response_data.decode('utf-8'))
            _log_and_print(f"[SOCKET_PROTOCOL] Received text response length: {len(response)} chars")
            return response

        finally:
            sock.close()


class ModularVLMClient:
    """Modular VLM client."""

    def __init__(self, input_type: str, output_type: str, protocol: str,
                 host: str = "localhost", port: int = 8888, **kwargs):
        """Initialize modular VLM client.

        Args:
            input_type: Input type ("rgb" or "rgbd")
            output_type: Output type ("trajectory" or "text")
            protocol: Communication protocol ("http" or "socket")
            host: Server address
            port: Server port
        """
        self.input_type = input_type
        self.output_type = output_type
        self.protocol_type = protocol
        self.host = host
        self.port = port
        self.kwargs = kwargs

        # Initialize components
        self.input_processor = self._create_input_processor()
        self.output_parser = self._create_output_parser()
        self.protocol = self._create_protocol()

        _log_and_print(f"[MODULAR_CLIENT] Initialized: {input_type.upper()} + {output_type.upper()} + {protocol.upper()}")

        # If using HTTP protocol with NavDP config, need to initialize navigator
        if protocol == "http" and self._needs_navigator_init():
            self._initialize_navigator()

    def _create_input_processor(self) -> InputProcessor:
        """Create input processor."""
        if self.input_type == "rgb":
            return RGBInputProcessor(**self.kwargs)
        elif self.input_type == "rgbd":
            return RGBDInputProcessor(**self.kwargs)
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")

    def _create_output_parser(self) -> OutputParser:
        """Create output parser."""
        if self.output_type == "trajectory":
            return TrajectoryOutputParser()
        elif self.output_type == "text":
            return TextOutputParser()
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}")

    def _create_protocol(self) -> CommunicationProtocol:
        """Create communication protocol."""
        if self.protocol_type == "http":
            return HTTPProtocol()
        elif self.protocol_type == "socket":
            return SocketProtocol()
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol_type}")

    def _needs_navigator_init(self) -> bool:
        """Determine if navigator needs initialization."""
        return (self.output_type == "trajectory" and
                (self.port == 8888 or self.kwargs.get('model_type') == 'navdp'))

    def _initialize_navigator(self):
        """Initialize NavDP navigator."""
        try:
            _log_and_print(f"[MODULAR_CLIENT] Initializing NavDP navigator...")

            # NavDP standard initialization parameters
            intrinsic = np.array([
                [525.0, 0.0, 320.0],
                [0.0, 525.0, 240.0],
                [0.0, 0.0, 1.0]
            ])

            url = f"http://{self.host}:{self.port}/navigator_reset"
            data = {
                'intrinsic': intrinsic.tolist(),
                'stop_threshold': -0.5,
                'batch_size': 1
            }

            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                _log_and_print(f"[MODULAR_CLIENT] ✓ NavDP navigator initialized successfully")
            else:
                _log_and_print(f"[MODULAR_CLIENT] ✗ NavDP navigator initialization failed, status code: {response.status_code}")

        except Exception as e:
            _log_and_print(f"[MODULAR_CLIENT] ✗ NavDP navigator initialization error: {e}")

    def query(self, rgb_images: List[Image.Image], instruction: str,
              current_yaw: float = 0.0, depth_images: List[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Query VLM model."""
        try:
            # 1. Process input
            processed_input = self.input_processor.process_input(rgb_images, depth_images, **kwargs)

            # 2. Send request
            raw_response = self.protocol.send_request(processed_input, instruction, self.host, self.port, **kwargs)

            # 3. Parse output
            parsed_action = self.output_parser.parse_output(raw_response, current_yaw, **kwargs)

            return parsed_action

        except Exception as e:
            _log_and_print(f"[MODULAR_CLIENT] VLM query failed: {e}")
            return {
                "vx": 0.0,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "duration_s": 1.0,
                "stop": True,
                "raw_response": f"Error: {str(e)}",
                "parsed_from": "error"
            }


# Predefined configurations
PREDEFINED_CONFIGS = {
    "navdp": {
        "input_type": "rgbd",
        "output_type": "trajectory",
        "protocol": "http",
        "port": 8888
    },
    "navila": {
        "input_type": "rgb",
        "output_type": "text",
        "protocol": "socket",
        "port": 54321,
        "history_frames": 8
    },
    "navid": {
        "input_type": "rgb",
        "output_type": "trajectory",
        "protocol": "socket",
        "port": 54321,
        "history_frames": 8
    },
    "example_rgb_trajectory": {
        "input_type": "rgb",
        "output_type": "trajectory",
        "protocol": "http",
        "port": 9999
    },
    "example_rgbd_text": {
        "input_type": "rgbd",
        "output_type": "text",
        "protocol": "socket",
        "port": 10000
    }
}


def create_vlm_client(model_name: str = None, input_type: str = None, output_type: str = None,
                     protocol: str = None, **kwargs) -> ModularVLMClient:
    """Factory function to create VLM client.

    Args:
        model_name: Predefined model name (e.g., "navdp", "navila")
        input_type: Input type ("rgb" or "rgbd")
        output_type: Output type ("trajectory" or "text")
        protocol: Communication protocol ("http" or "socket")
    """
    if model_name and model_name in PREDEFINED_CONFIGS:
        config = PREDEFINED_CONFIGS[model_name].copy()
        config.update(kwargs)  # Allow overriding predefined config
        return ModularVLMClient(**config)
    elif input_type and output_type and protocol:
        return ModularVLMClient(input_type, output_type, protocol, **kwargs)
    else:
        raise ValueError("Must provide model_name or (input_type, output_type, protocol)")


def query_vlm(images: List[Image.Image], instruction: str, host: str = "localhost", port: int = 8888,
              current_yaw: float = 0.0, depth_images: List[np.ndarray] = None,
              model_type: str = None, input_type: str = None, output_type: str = None,
              protocol: str = None, **kwargs) -> Dict[str, Any]:
    """Unified VLM query function supporting multiple configurations.

    Method 1: Use predefined model
        query_vlm(..., model_type="navdp")

    Method 2: Use modular configuration
        query_vlm(..., input_type="rgb", output_type="trajectory", protocol="http")
    """
    # Prefer modular configuration
    if input_type and output_type and protocol:
        _log_and_print(f"[QUERY_VLM] Using modular config: {input_type} + {output_type} + {protocol}")
        client = create_vlm_client(input_type=input_type, output_type=output_type,
                                 protocol=protocol, host=host, port=port, **kwargs)
    elif model_type and model_type in PREDEFINED_CONFIGS:
        _log_and_print(f"[QUERY_VLM] Using predefined model: {model_type}")
        client = create_vlm_client(model_name=model_type, host=host, port=port, **kwargs)
    else:
        # Default fallback to NavDP
        _log_and_print(f"[QUERY_VLM] Using default config: navdp")
        client = create_vlm_client(model_name="navdp", host=host, port=port, **kwargs)

    return client.query(images, instruction, current_yaw=current_yaw, depth_images=depth_images, **kwargs)








