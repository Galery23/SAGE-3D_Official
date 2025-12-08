#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLN Trajectory Generator: Convert InteriorGS 2D semantic maps into VLN training data.

This script generates navigation trajectories and natural language instructions from:
1. InteriorGS scene folders (with labels.json)
2. Textual scene maps (from scene_text_generator.py)
3. 2D semantic maps (from semantic_map_builder.py)

All paths and API configurations are exposed via CLI arguments for maximum flexibility.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import heapq
import numpy as np
import requests
from scipy.ndimage import distance_transform_edt

# Visualization (optional)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not available, visualization disabled")


# ==================== Configuration Defaults ====================

DEFAULT_LABEL_ROOT = Path("/path/to/data/InteriorGS")
DEFAULT_SCENE_TEXT_ROOT = Path("/path/to/data/Scene_Map")
DEFAULT_SEM_MAP_ROOT = Path("/path/to/data/2D_Semantic_Map")
DEFAULT_ENDPOINT_OUT_ROOT = Path("Data/endpoints")
DEFAULT_TRAJ_OUT_ROOT = Path("Data/trajectories")
DEFAULT_PROMPT_PAIRWISE = Path(
    "prompts/trajectory_generation/prompt_pairwise_judgement.json"
)
DEFAULT_PROMPT_PAIRWISE_BATCH = Path(
    "prompts/trajectory_generation/prompt_pairwise_judgement_batch.json"
)
DEFAULT_PROMPT_TRAJ_TO_INSTR = Path(
    "prompts/trajectory_generation/prompt_traj_to_instruction.json"
)

# Navigation parameters
ROBOT_RADIUS_M = 0.2
SCALE_M_PER_PX = 0.05
FIXED_Z = 0.5
SAMPLE_STEP = 1

# Processing parameters
JUDGE_WORKERS = 32
INSTR_WORKERS = 32
MIN_TRAJS_PER_SCENE = 100
MAX_PAIRS_PER_BATCH = 50
BATCH_PAIRS_PER_LLM_CALL = 10
MAX_TOTAL_PAIRS_CHECK = 5000
MIN_DISTANCE_THRESHOLD = 2.0
MAX_DISTANCE_THRESHOLD = 20.0
MAX_INSTR_RETRY = 5
INCREMENTAL_SAVE_THRESHOLD = 5  # Save endpoints every N reachable paths
INSTR_WAIT_TIMEOUT = 180  # Timeout for instruction generation (seconds)
PRINT_EVERY_JUDGE = 25  # Print progress every N judge operations
PRINT_EVERY_REACH = 10  # Print progress every N reachable paths

# Bad scenes to skip (optional)
BAD_SCENE_IDS = set()

# ==================== OpenAI API Helper ====================


class OpenAIClient:
    """Standard OpenAI Chat Completions API client."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model = model
        self.timeout = timeout
        self.url = f"{self.base_url}/chat/completions"

    def chat_completion(self, messages: Sequence[Dict[str, str]]) -> str:
        """Send chat completion request and return content."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }

        response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI API returned no choices")

        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("OpenAI API returned empty content")

        return content


# ==================== Company API Client (for testing) ====================

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("[WARN] langchain not available, Company API client disabled")


class CompanyAPIClient:
    """Company API client using langchain_openai (for testing purposes only)."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://oneapi.qunhequnhe.com/v1",
        model: str = "gpt-5",
        timeout: int = 60,
    ):
        if not HAS_LANGCHAIN:
            raise ImportError("langchain_openai is required for CompanyAPIClient. Install it with: pip install langchain-openai langchain-core")
        
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        
        # Initialize langchain client
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            streaming=False,
            timeout=self.timeout,
        )
    
    def chat_completion(self, messages: Sequence[Dict[str, str]]) -> str:
        """Send chat completion request using langchain and return content."""
        # Convert messages format from OpenAI format to langchain format
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                from langchain_core.messages import AIMessage
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                from langchain_core.messages import SystemMessage
                langchain_messages.append(SystemMessage(content=content))
        
        # Call API
        response = self.llm.invoke(langchain_messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        if not content or not content.strip():
            raise RuntimeError("Company API returned empty content")
        
        return content.strip()


# ==================== Utility Functions ====================


def normalize_label(label: str) -> str:
    """Normalize label to lowercase snake_case."""
    return label.strip().lower().replace(" ", "_")


def robust_json_parse(text: str) -> Any:
    """Parse JSON with fallback strategies."""
    try:
        return json.loads(text)
    except Exception:
        # Try extracting JSON from markdown code blocks
        for pattern in [r"```json\s*(\{.*?\}|\[.*?\])\s*```", r"```\s*(\{.*?\}|\[.*?\])\s*```"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    continue
        # Try finding first { or [
        for char in ["{", "["]:
            start = text.find(char)
            if start != -1:
                end = text.rfind("}" if char == "{" else "]")
                if end > start:
                    try:
                        return json.loads(text[start : end + 1])
                    except Exception:
                        continue
    return None


def load_prompt_template(path: Path) -> List[Dict[str, str]]:
    """Load prompt template from JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_scene_text(scene_key: str, scene_text_root: Path) -> str:
    """Load textual scene map."""
    path = scene_text_root / f"semantic_map_{scene_key}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ==================== Path Planning ====================


def astar_pixel(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]] | None:
    """A* path planning on pixel grid."""
    H, W = grid.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]

        for d in dirs:
            nx, ny = cur[0] + d[0], cur[1] + d[1]
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 1:
                continue

            nb = (nx, ny)
            step = math.hypot(nx - cur[0], ny - cur[1])
            tg = g_score[cur] + step
            if nb not in g_score or tg < g_score[nb]:
                came_from[nb] = cur
                g_score[nb] = tg
                f = tg + math.hypot(nx - goal[0], ny - goal[1])
                heapq.heappush(open_set, (f, nb))

    return None


def instance_centroid_px(mask_coords: List[Tuple[int, int]]) -> Tuple[int, int] | None:
    """Compute instance centroid in pixel coordinates."""
    if not mask_coords:
        return None
    m = np.array(mask_coords)
    c = m.mean(axis=0)
    return (int(round(c[1])), int(round(c[0])))


def boundary_pixels(mask_coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Extract boundary pixels from mask."""
    s = set((int(y), int(x)) for (y, x) in mask_coords)
    b = []
    for (y, x) in s:
        neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        if any(n not in s for n in neighbors):
            b.append((y, x))
    return b


def get_nearest_free_pixel_on_side(
    instance_mask: List[Tuple[int, int]], base_map: np.ndarray, towards_px: Tuple[int, int] | None = None, max_search_dist: int = 50
) -> Tuple[int, int] | None:
    """Find nearest free pixel near instance boundary."""
    H, W = base_map.shape
    b_pixels = boundary_pixels(instance_mask)
    if not b_pixels:
        return None

    visited = set()
    q = deque()
    for (by, bx) in b_pixels:
        if 0 <= bx < W and 0 <= by < H:
            visited.add((bx, by))
            q.append((bx, by, 0))

    while q:
        x, y, d = q.popleft()
        if d > max_search_dist:
            break
        if 0 <= x < W and 0 <= y < H and base_map[y, x] == 0:
            if towards_px is None:
                return (x, y)
            else:
                bx, by = np.mean([(px, py) for (py, px) in instance_mask], axis=0)
                vec_item_to_point = np.array([x - bx, y - by])
                vec_item_to_towards = np.array([towards_px[0] - bx, towards_px[1] - by])
                if np.dot(vec_item_to_point, vec_item_to_towards) >= 0:
                    return (x, y)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny, d + 1))

    return None


# ==================== LLM Interaction ====================


def build_pairwise_messages(template: List[Dict[str, str]], scene_text: str, start_item: str, end_item: str) -> List[Dict[str, str]]:
    """Build messages for pairwise judgement."""
    messages = []
    for msg in template:
        content = msg["content"]
        content = content.replace("{scene_map}", scene_text)
        content = content.replace("{start_item}", start_item)
        content = content.replace("{end_item}", end_item)
        messages.append({"role": msg["role"], "content": content})
    return messages


def build_batch_pairwise_messages(
    template: List[Dict[str, str]], scene_text: str, pairs: List[Tuple[str, str]]
) -> List[Dict[str, str]]:
    """Build messages for batch pairwise judgement (v2)."""
    pairs_list_str = ""
    for i, (start, end) in enumerate(pairs):
        pairs_list_str += f'Pair {i}: Start="{start}", End="{end}"\n'

    messages = []
    for msg in template:
        content = msg["content"]
        content = content.replace("{scene_map}", scene_text)
        content = content.replace("{pairs_list}", pairs_list_str.strip())
        messages.append({"role": msg["role"], "content": content})
    return messages


def build_instruction_messages(template: List[Dict[str, str]], scene_text: str, start_item: str, end_item: str) -> List[Dict[str, str]]:
    """Build messages for instruction generation."""
    text_block = f'"text_map": {json.dumps(scene_text)}'
    json_block = f'"start": "{start_item}",\n"end": "{end_item}"'

    messages = []
    for msg in template:
        content = msg["content"]
        content = content.replace("{text}", text_block)
        content = content.replace("{json}", json_block)
        messages.append({"role": msg["role"], "content": content})
    return messages


def llm_judge_pairs_batch_v2(
    client: OpenAIClient | CompanyAPIClient, template: List[Dict[str, str]], scene_text: str, pairs: List[Tuple[str, str]]
) -> List[Tuple[str, str, bool, bool]]:
    """Batch judge pairs using v2 prompt."""
    if not pairs:
        return []

    try:
        messages = build_batch_pairwise_messages(template, scene_text, pairs)
        response_text = client.chat_completion(messages)

        # Parse JSON response
        parsed = robust_json_parse(response_text)
        if not isinstance(parsed, list):
            return [(start, end, False, False) for start, end in pairs]

        results = []
        for i, (start, end) in enumerate(pairs):
            if i < len(parsed):
                result = parsed[i]
                meaningful = result.get("meaningful", False)
                if isinstance(meaningful, bool):
                    results.append((start, end, meaningful, True))
                elif str(meaningful).lower() in ["true", "yes", "1"]:
                    results.append((start, end, True, True))
                else:
                    results.append((start, end, False, True))
            else:
                results.append((start, end, False, True))

        return results

    except Exception as e:
        print(f"[ERROR] Batch judge API call failed: {e}")
        return [(start, end, False, False) for start, end in pairs]


def llm_generate_instructions(
    client: OpenAIClient | CompanyAPIClient, template: List[Dict[str, str]], scene_text: str, start_item: str, end_item: str, scene_key: str
) -> Tuple[List[Dict], bool]:
    """Generate instructions for a trajectory."""
    try:
        messages = build_instruction_messages(template, scene_text, start_item, end_item)
        response_text = client.chat_completion(messages)

        parsed = robust_json_parse(response_text)
        instructions_list: List[Dict] = []

        if isinstance(parsed, list):
            for inst in parsed:
                if isinstance(inst, dict):
                    inst["scene_id"] = scene_key
                    instructions_list.append(inst)
        elif isinstance(parsed, dict):
            parsed["scene_id"] = scene_key
            instructions_list.append(parsed)

        if not instructions_list:
            instructions_list.append(
                {
                    "instruction_type": "Default",
                    "start": start_item,
                    "end": end_item,
                    "generated_instruction": f"Navigate from {start_item} to {end_item}.",
                    "scene_id": scene_key,
                }
            )

        # Check if we have valid (non-default) instructions
        has_valid = any(
            isinstance(inst, dict)
            and str(inst.get("instruction_type", "")).lower() != "default"
            and inst.get("generated_instruction")
            for inst in instructions_list
        )

        return instructions_list, has_valid

    except Exception as e:
        print(f"[ERROR] Instruction generation failed: {e}")
        return [
            {
                "instruction_type": "Default",
                "start": start_item,
                "end": end_item,
                "generated_instruction": f"Navigate from {start_item} to {end_item}.",
                "scene_id": scene_key,
            }
        ], False


# ==================== Map Building ====================


def build_2d_map(
    sem_data: List[Dict], scale: float = SCALE_M_PER_PX, robot_radius_m: float = ROBOT_RADIUS_M
) -> Tuple[np.ndarray | None, float | None, float | None, float | None]:
    """Build 2D navigation map from semantic data."""
    try:
        # Extract bounds
        all_y: List[float] = []
        all_x: List[float] = []
        for inst in sem_data:
            for y, x in inst.get("mask_coords_m", []):
                try:
                    y_val = float(y)
                    x_val = float(x)
                except (ValueError, TypeError):
                    continue
                all_y.append(y_val)
                all_x.append(x_val)

        if not all_y or not all_x:
            return None, None, None, None

        min_y, max_y = min(all_y), max(all_y)
        min_x, max_x = min(all_x), max(all_x)
        h = int(np.ceil((max_y - min_y) / scale)) + 1
        w = int(np.ceil((max_x - min_x) / scale)) + 1

        # World to pixel conversion
        def world2pix(x_m: float | str | int, y_m: float | str | int) -> Tuple[int, int]:
            try:
                x_val = float(x_m)
                y_val = float(y_m)
            except (ValueError, TypeError):
                return -1, -1
            px = int(round((x_val - min_x) / scale))
            py = int(round((y_val - min_y) / scale))
            return py, px

        # Store pixel coordinates
        for inst in sem_data:
            pixel_coords = []
            for y_m, x_m in inst.get("mask_coords_m", []):
                py, px = world2pix(x_m, y_m)
                if py == -1 or px == -1:
                    continue
                if 0 <= py < h and 0 <= px < w:
                    pixel_coords.append((py, px))
            inst["mask_coords"] = pixel_coords

        # Build obstacle map (Unable Area + Wall)
        def lab(inst: Dict) -> str:
            return str(inst.get("category_label", "")).lower()

        grid_map = np.zeros((h, w), dtype=np.uint8)
        for inst in sem_data:
            l = lab(inst)
            if l in ["unable area", "wall"]:
                for py, px in inst.get("mask_coords", []):
                    if 0 <= py < h and 0 <= px < w:
                        grid_map[py, px] = 1

        # Inflate for robot radius
        if robot_radius_m > 0:
            dist_m = distance_transform_edt(grid_map == 0, sampling=scale)
            grid_map = (dist_m <= robot_radius_m).astype(np.uint8)

        return grid_map, scale, min_x, min_y

    except Exception as e:
        print(f"[ERROR] Failed to build 2D map: {e}")
        return None, None, None, None


# ==================== Distance and Filtering ====================


def calculate_distance(inst1: Dict, inst2: Dict) -> float:
    """Calculate Euclidean distance between two instances (meters)."""
    coords1 = inst1.get("mask_coords_m", [])
    coords2 = inst2.get("mask_coords_m", [])

    if not coords1 or not coords2:
        return float("inf")

    try:
        coords1_arr = np.array([[float(y), float(x)] for y, x in coords1])
        coords2_arr = np.array([[float(y), float(x)] for y, x in coords2])

        if coords1_arr.size == 0 or coords2_arr.size == 0:
            return float("inf")

        centroid1 = coords1_arr.mean(axis=0)
        centroid2 = coords2_arr.mean(axis=0)

        return np.linalg.norm(centroid1 - centroid2)
    except (ValueError, TypeError) as e:
        print(f"[WARN] Distance calculation failed: {e}")
        return float("inf")


def should_skip_same_category(item1_id: str, item2_id: str, itemid2inst: Dict[str, Dict]) -> bool:
    """Check if pairs should be skipped due to same category."""
    if item1_id not in itemid2inst or item2_id not in itemid2inst:
        return False

    inst1 = itemid2inst[item1_id]
    inst2 = itemid2inst[item2_id]

    label1 = inst1.get("category_label", "").lower()
    label2 = inst2.get("category_label", "").lower()

    # Skip same category
    if label1 == label2:
        return True

    # Skip similar functional groups
    similar_groups = [
        {"chair", "stool", "armchair"},
        {"table", "desk", "dining_table"},
        {"bed", "sofa", "couch"},
        {"cabinet", "shelf", "bookshelf", "wardrobe"},
        {"lamp", "light", "ceiling_light"},
    ]

    for group in similar_groups:
        if label1 in group and label2 in group:
            return True

    return False


def filter_pairs_by_distance_and_category(
    pairs: List[Tuple[str, str]], itemid2inst: Dict[str, Dict], min_dist: float = MIN_DISTANCE_THRESHOLD, max_dist: float = MAX_DISTANCE_THRESHOLD
) -> List[Tuple[str, str]]:
    """Filter pairs by distance and category."""
    filtered = []

    for s, e in pairs:
        # Skip same category
        if should_skip_same_category(s, e, itemid2inst):
            continue

        # Skip invalid distances
        if s in itemid2inst and e in itemid2inst:
            distance = calculate_distance(itemid2inst[s], itemid2inst[e])
            if distance < min_dist or distance > max_dist:
                continue

        filtered.append((s, e))

    return filtered


def build_connectivity_map(grid_map: np.ndarray, itemid2inst: Dict[str, Dict]) -> Dict[Tuple[int, int], set]:
    """Precompute connectivity map using union-find algorithm to avoid repeated A* calculations."""
    H, W = grid_map.shape
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(x: Tuple[int, int]) -> Tuple[int, int]:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: Tuple[int, int], y: Tuple[int, int]) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build union-find for all free pixels
    free_pixels = []
    for y in range(H):
        for x in range(W):
            if grid_map[y, x] == 0:  # Free space
                pixel = (x, y)
                free_pixels.append(pixel)
                parent[pixel] = pixel

    # Connect adjacent free pixels
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for x, y in free_pixels:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and grid_map[ny, nx] == 0:
                union((x, y), (nx, ny))

    # Find connectivity component for each item
    item_connectivity: Dict[str, Tuple[int, int]] = {}

    for item_id, inst in itemid2inst.items():
        if "mask_coords" not in inst or not inst["mask_coords"]:
            continue

        nearest_free = get_nearest_free_pixel_on_side(inst["mask_coords"], grid_map)
        if nearest_free:
            item_connectivity[item_id] = find(nearest_free)

    # Group items by connectivity component
    connectivity_groups: Dict[Tuple[int, int], set] = defaultdict(set)
    for item_id, component in item_connectivity.items():
        connectivity_groups[component].add(item_id)

    return dict(connectivity_groups)


def are_items_connected(item1: str, item2: str, connectivity_map: Dict[Tuple[int, int], set]) -> bool:
    """Quickly check if two items are connected."""
    for component_items in connectivity_map.values():
        if item1 in component_items and item2 in component_items:
            return True
    return False


# ==================== Trajectory Generation ====================


def generate_trajectory_points(
    path: List[Tuple[int, int]], scale: float, min_x: float, min_y: float, fixed_z: float = FIXED_Z, sample_step: int = SAMPLE_STEP
) -> List[Dict]:
    """Convert path pixels to trajectory points with poses."""
    xs, ys = zip(*path)
    xs_w = [min_x + (x + 0.5) * scale for x in xs]
    ys_w = [min_y + (y + 0.5) * scale for y in ys]
    world_path = list(zip(xs_w, ys_w))
    sampled_points = world_path[::sample_step]

    points_list = []
    for j, (wx, wy) in enumerate(sampled_points):
        if j < len(sampled_points) - 1:
            wx_next, wy_next = sampled_points[j + 1]
        else:
            wx_next, wy_next = sampled_points[j]

        dx = wx_next - wx
        dy = wy_next - wy
        yaw = math.atan2(dy, dx)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        points_list.append(
            {
                "point": str(j),
                "position": [wx, wy, fixed_z],
                "rotation": [0.0, 0.0, qz, qw],
                "action": [],
                "camera_images": [],
                "focal_length": 7.0,
                "horizontal_aperture": 20.954999923706055,
                "vertical_aperture": 20.954999923706055,
                "focus_distance": 0.0,
                "clipping_range": [1.0, 1000000.0],
            }
        )

    return points_list


def validate_and_generate_path(
    start_item: str,
    end_item: str,
    itemid2inst: Dict[str, Dict],
    grid_map: np.ndarray,
    scale: float,
    min_x: float,
    min_y: float,
) -> Dict | None:
    """Validate and generate path between two items."""
    start_inst = itemid2inst.get(start_item)
    goal_inst = itemid2inst.get(end_item)

    if not start_inst or not goal_inst:
        return None

    start_cent_px = instance_centroid_px(start_inst.get("mask_coords", []))
    goal_cent_px = instance_centroid_px(goal_inst.get("mask_coords", []))

    if not start_cent_px or not goal_cent_px:
        return None

    start_px = get_nearest_free_pixel_on_side(start_inst.get("mask_coords", []), grid_map, towards_px=goal_cent_px)
    goal_px = get_nearest_free_pixel_on_side(goal_inst.get("mask_coords", []), grid_map, towards_px=start_cent_px)

    if not start_px or not goal_px:
        return None

    # A* path planning
    path = astar_pixel(grid_map, start_px, goal_px)
    if not path:
        return None

    points_list = generate_trajectory_points(path, scale, min_x, min_y)

    return {"start": start_item, "end": end_item, "points": points_list}


# ==================== File I/O Helpers ====================


def item_id_from_label_counts(label: str, counter: Dict[str, int]) -> str:
    """Generate item ID from label and counter."""
    lbl_norm = normalize_label(label)
    counter[lbl_norm] += 1
    return f"{lbl_norm}_{counter[lbl_norm]}"


def extract_key_from_scene_dir_name(name: str) -> str:
    """Extract scene key from directory name."""
    return name.strip()


def count_existing_trajectories(scene_key: str, traj_root: Path) -> int:
    """Count existing trajectories from trajectory files."""
    total = 0
    tail_id = scene_key.split("_")[-1] if "_" in scene_key else scene_key
    traj_dir = traj_root / tail_id

    if not traj_dir.exists():
        return 0

    # Check main trajectory file
    main_traj_path = traj_dir / f"trajectories_{scene_key}.json"
    if main_traj_path.exists():
        try:
            with main_traj_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "scenes" in data and len(data["scenes"]) > 0 and "samples" in data["scenes"][0]:
                total += len(data["scenes"][0]["samples"])
        except Exception as e:
            print(f"[WARN] Failed to read trajectory file {main_traj_path}: {e}")

    # Check part files
    part_num = 2
    while True:
        part_path = traj_dir / f"trajectories_{scene_key}_part{part_num}.json"
        if not part_path.exists():
            break
        try:
            with part_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "scenes" in data and len(data["scenes"]) > 0 and "samples" in data["scenes"][0]:
                total += len(data["scenes"][0]["samples"])
        except Exception as e:
            print(f"[WARN] Failed to read part trajectory file {part_path}: {e}")
        part_num += 1

    return total


def get_existing_endpoint_pairs_from_trajectories(scene_key: str, traj_root: Path) -> set:
    """Extract existing endpoint pairs from trajectory files."""
    existing = set()
    tail_id = scene_key.split("_")[-1] if "_" in scene_key else scene_key
    traj_dir = traj_root / tail_id

    if not traj_dir.exists():
        return existing

    # Check all trajectory files
    for traj_file in traj_dir.glob(f"trajectories_{scene_key}*.json"):
        try:
            with traj_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "scenes" in data and len(data["scenes"]) > 0:
                samples = data["scenes"][0].get("samples", [])
                for sample in samples:
                    instructions = sample.get("instructions", [])
                    if instructions:
                        first = instructions[0]
                        start = first.get("start")
                        end = first.get("end")
                        if start and end:
                            existing.add((start, end))
        except Exception as e:
            print(f"[WARN] Failed to read trajectory file {traj_file}: {e}")

    return existing


def save_trajectory_file(
    samples: List[Dict], scene_key: str, output_path: Path, part_num: int | None = None
) -> bool:
    """Save trajectory samples to JSON file."""
    try:
        tail_id = scene_key.split("_")[-1] if "_" in scene_key else scene_key

        traj_data = {
            "dataset_metadata": {"name": "GVLN"},
            "scenes": [{"scene_id": 0, "scene_name": tail_id, "samples": samples}],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(traj_data, f, indent=2)

        return True
    except Exception as e:
        print(f"[ERROR] Failed to save trajectory file {output_path}: {e}")
        return False


def get_next_part_number(scene_key: str, endpoint_root: Path) -> int:
    """Get next part number for split files."""
    part_num = 2
    while True:
        endpoints_path = endpoint_root / f"{scene_key}_endpoints_part{part_num}.json"
        if not endpoints_path.exists():
            return part_num
        part_num += 1


def append_to_json_file(file_path: Path, new_data: List[Dict], data_type: str = "endpoints") -> bool:
    """
    Append data to JSON file.
    
    Args:
        file_path: Path to JSON file
        new_data: New data to append
        data_type: "endpoints" or "trajectories"
    
    Returns:
        bool: Whether append was successful
    """
    try:
        if file_path.exists():
            # Read existing data
            with file_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)

            if data_type == "endpoints":
                # For endpoints, append to list
                if isinstance(existing_data, list):
                    existing_data.extend(new_data)
                else:
                    existing_data = new_data
            else:
                # For trajectories, append to scenes[0].samples
                if "scenes" in existing_data and len(existing_data["scenes"]) > 0:
                    existing_data["scenes"][0]["samples"].extend(new_data)
                else:
                    # Infer scene_name from file path
                    scene_name = "unknown"
                    filename = file_path.name
                    if "trajectories_" in filename:
                        parts = filename.replace("trajectories_", "").replace(".json", "").split("_part")[0]
                        if "_" in parts:
                            scene_name = parts.split("_")[-1]

                    existing_data = {
                        "dataset_metadata": {"name": "GVLN"},
                        "scenes": [{"scene_id": 0, "scene_name": scene_name, "samples": new_data}],
                    }
        else:
            # File doesn't exist, create new data
            if data_type == "endpoints":
                existing_data = new_data
            else:
                # Infer scene_name from file path
                scene_name = "unknown"
                filename = file_path.name
                if "trajectories_" in filename:
                    parts = filename.replace("trajectories_", "").replace(".json", "").split("_part")[0]
                    if "_" in parts:
                        scene_name = parts.split("_")[-1]

                existing_data = {
                    "dataset_metadata": {"name": "GVLN"},
                    "scenes": [{"scene_id": 0, "scene_name": scene_name, "samples": new_data}],
                }

        # Write file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

        return True
    except Exception as e:
        print(f"[ERROR] Failed to append to {file_path}: {e}")
        return False


# ==================== Endpoint and Trajectory File Management ====================


def check_endpoint_trajectory_pairs(
    scene_key: str, endpoint_root: Path, traj_root: Path
) -> Dict[str, Any]:
    """
    Check correspondence between endpoint pair files and trajectory files.
    
    Returns:
        {
            "complete_pairs": [(endpoints_file, trajectory_file), ...],
            "missing_trajectories": [(endpoints_file, expected_traj_file), ...],
            "total_endpoints": int,
            "complete_trajectories": int
        }
    """
    result = {
        "complete_pairs": [],
        "missing_trajectories": [],
        "total_endpoints": 0,
        "complete_trajectories": 0,
    }

    tail_id = scene_key.split("_")[-1] if "_" in scene_key else scene_key
    traj_dir = traj_root / tail_id

    # Check main files
    main_endpoints_path = endpoint_root / f"{scene_key}_endpoints.json"
    main_traj_path = traj_dir / f"trajectories_{scene_key}.json"

    if main_endpoints_path.exists():
        try:
            with main_endpoints_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                endpoint_count = len(data)
                result["total_endpoints"] += endpoint_count

                if main_traj_path.exists():
                    result["complete_pairs"].append((str(main_endpoints_path), str(main_traj_path)))
                    try:
                        with main_traj_path.open("r", encoding="utf-8") as tf:
                            traj_data = json.load(tf)
                        if "scenes" in traj_data and len(traj_data["scenes"]) > 0 and "samples" in traj_data["scenes"][0]:
                            actual_traj_count = len(traj_data["scenes"][0]["samples"])
                            result["complete_trajectories"] += actual_traj_count
                            print(f"[{scene_key}] Main file complete: {endpoint_count} endpoint pairs + {actual_traj_count} trajectories")
                    except Exception as te:
                        result["complete_trajectories"] += endpoint_count
                        print(f"[{scene_key}] Main file complete: {endpoint_count} endpoint pairs + trajectory file (parse failed: {te})")
                else:
                    result["missing_trajectories"].append((str(main_endpoints_path), str(main_traj_path)))
                    print(f"[{scene_key}] Main file missing trajectory: {endpoint_count} endpoint pairs, need to generate {main_traj_path.name}")
        except Exception as e:
            print(f"[WARN] Failed to read main endpoint file: {e}")

    # Check part files
    part_num = 2
    while True:
        part_endpoints_path = endpoint_root / f"{scene_key}_endpoints_part{part_num}.json"
        part_traj_path = traj_dir / f"trajectories_{scene_key}_part{part_num}.json"

        if not part_endpoints_path.exists():
            break

        try:
            with part_endpoints_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                endpoint_count = len(data)
                result["total_endpoints"] += endpoint_count

                if part_traj_path.exists():
                    result["complete_pairs"].append((str(part_endpoints_path), str(part_traj_path)))
                    try:
                        with part_traj_path.open("r", encoding="utf-8") as tf:
                            traj_data = json.load(tf)
                        if "scenes" in traj_data and len(traj_data["scenes"]) > 0 and "samples" in traj_data["scenes"][0]:
                            actual_traj_count = len(traj_data["scenes"][0]["samples"])
                            result["complete_trajectories"] += actual_traj_count
                            print(f"[{scene_key}] Part{part_num} file complete: {endpoint_count} endpoint pairs + {actual_traj_count} trajectories")
                    except Exception as te:
                        result["complete_trajectories"] += endpoint_count
                        print(f"[{scene_key}] Part{part_num} file complete: {endpoint_count} endpoint pairs + trajectory file (parse failed: {te})")
                else:
                    result["missing_trajectories"].append((str(part_endpoints_path), str(part_traj_path)))
                    print(f"[{scene_key}] Part{part_num} file missing trajectory: {endpoint_count} endpoint pairs, need to generate {part_traj_path.name}")
        except Exception as e:
            print(f"[WARN] Failed to read Part{part_num} endpoint file: {e}")

        part_num += 1

    return result


def generate_trajectories_from_endpoints(
    scene_key: str,
    endpoints_file: Path,
    output_traj_file: Path,
    itemid2inst: Dict[str, Dict],
    grid_map: np.ndarray,
    scale: float,
    min_x: float,
    min_y: float,
    scene_text: str,
    clients: List[OpenAIClient],
    instr_template: List[Dict[str, str]],
    sem_data: List[Dict] | None = None,
) -> bool:
    """
    Generate trajectory file from existing endpoint pair file.
    
    Returns:
        bool: Whether generation was successful
    """
    try:
        # Read endpoint pairs
        if not endpoints_file.exists():
            print(f"[ERROR] Endpoint file does not exist: {endpoints_file}")
            return False

        with endpoints_file.open("r", encoding="utf-8") as f:
            endpoint_pairs = json.load(f)

        if not isinstance(endpoint_pairs, list):
            print(f"[ERROR] Endpoint file format error: {endpoints_file}")
            return False

        print(f"[{scene_key}] Starting trajectory generation from {len(endpoint_pairs)} endpoint pairs (including LLM instruction generation)")

        # Step 1: Generate all reachable trajectory entries (path validation)
        reachable_entries = []

        for i, pair in enumerate(endpoint_pairs):
            start_item = pair.get("start")
            end_item = pair.get("end")

            if not start_item or not end_item:
                continue

            # Path validation and trajectory generation (reuse existing logic)
            result = validate_and_generate_path(start_item, end_item, itemid2inst, grid_map, scale, min_x, min_y)
            if result:
                reachable_entries.append(result)

            if (i + 1) % 10 == 0:
                print(f"[{scene_key}] Path generation progress: {i + 1}/{len(endpoint_pairs)}, reachable: {len(reachable_entries)}")

        if not reachable_entries:
            print(f"[{scene_key}] No reachable paths generated")
            return False

        print(f"[{scene_key}] Path generation complete: {len(reachable_entries)}/{len(endpoint_pairs)} reachable paths")

        # Step 2: Concurrently generate LLM instructions
        print(f"[{scene_key}] Starting concurrent instruction generation: {len(reachable_entries)} trajectories")

        instructions_map: Dict[Tuple[str, str], List[Dict]] = {}
        instr_api_fail = 0

        instr_start_time = time.time()
        print(f"[{scene_key}] Submitted {len(reachable_entries)} instruction generation tasks to thread pool")

        with ThreadPoolExecutor(max_workers=min(INSTR_WORKERS, len(reachable_entries))) as ex:
            fut_map = {}
            for idx, entry in enumerate(reachable_entries):
                s, e = entry["start"], entry["end"]
                client = clients[idx % len(clients)]
                fut = ex.submit(
                    generate_instructions_with_retry,
                    client,
                    instr_template,
                    scene_text,
                    s,
                    e,
                    scene_key,
                    MAX_INSTR_RETRY,
                )
                fut_map[fut] = (s, e)

            done_cnt = 0
            for fut in as_completed(fut_map.keys()):
                s, e = fut_map[fut]
                try:
                    instr_list, api_ok = fut.result()
                    if not api_ok:
                        instr_api_fail += 1
                    instructions_map[(s, e)] = instr_list
                except Exception as ex_err:
                    print(f"[LLM FAIL] Instruction thread exception: {ex_err}")
                    instructions_map[(s, e)] = [
                        {
                            "instruction_type": "Default",
                            "start": s,
                            "end": e,
                            "generated_instruction": f"Navigate from {s} to {e}.",
                            "scene_id": scene_key,
                        }
                    ]
                    instr_api_fail += 1
                done_cnt += 1
                if done_cnt % 10 == 0 or done_cnt == len(reachable_entries):
                    elapsed = time.time() - instr_start_time
                    avg_time = elapsed / done_cnt if done_cnt > 0 else 0
                    success_cnt = done_cnt - instr_api_fail
                    print(
                        f"[{scene_key}] Instruction generation progress {done_cnt}/{len(reachable_entries)} "
                        f"(success: {success_cnt}, failed: {instr_api_fail}) | avg time: {avg_time:.2f}s/item"
                    )

        instr_end_time = time.time()
        instr_duration = instr_end_time - instr_start_time
        avg_per_instruction = instr_duration / len(reachable_entries) if len(reachable_entries) > 0 else 0
        print(
            f"[{scene_key}] Instruction generation complete: total time {instr_duration:.2f}s | "
            f"avg per item: {avg_per_instruction:.3f}s | API failures: {instr_api_fail}/{len(reachable_entries)}"
        )

        # Step 3: Assemble final trajectory samples
        trajectory_samples = []
        for i, entry in enumerate(reachable_entries):
            start_item = entry["start"]
            end_item = entry["end"]
            points_list = entry["points"]
            instr_list = instructions_map.get((start_item, end_item), [
                {
                    "instruction_type": "Default",
                    "start": start_item,
                    "end": end_item,
                    "generated_instruction": f"Navigate from {start_item} to {end_item}.",
                    "scene_id": scene_key,
                }
            ])

            sample = {
                "trajectory_id": str(i),
                "instructions": instr_list,
                "points": points_list,
            }
            trajectory_samples.append(sample)

        # Save trajectory file
        if trajectory_samples:
            tail_id = scene_key.split("_")[-1] if "_" in scene_key else scene_key

            traj_data = {
                "dataset_metadata": {"name": "GVLN"},
                "scenes": [{"scene_id": 0, "scene_name": tail_id, "samples": trajectory_samples}],
            }

            output_traj_file.parent.mkdir(parents=True, exist_ok=True)
            with output_traj_file.open("w", encoding="utf-8") as f:
                json.dump(traj_data, f, indent=2)

            print(
                f"[{scene_key}] Successfully generated {len(trajectory_samples)}/{len(endpoint_pairs)} trajectories, "
                f"saved to {output_traj_file.name}"
            )

            # Generate visualizations for the trajectories
            if HAS_MATPLOTLIB:
                try:
                    # Determine visualization directory name based on output filename
                    output_basename = output_traj_file.name
                    if "_part" in output_basename:
                        part_match = re.search(r"_part(\d+)", output_basename)
                        if part_match:
                            part_num = int(part_match.group(1))
                            vis_dir_name = f"recollected_nav_vis_part{part_num}"
                        else:
                            vis_dir_name = "recollected_nav_vis"
                    else:
                        vis_dir_name = "recollected_nav_vis"

                    # Create visualization directory
                    vis_dir = output_traj_file.parent / vis_dir_name
                    vis_dir.mkdir(parents=True, exist_ok=True)

                    # Generate visualization for each trajectory
                    vis_count = 0
                    for sample in trajectory_samples:
                        trajectory_id = sample.get("trajectory_id", "unknown")
                        points = sample.get("points", [])

                        if points:
                            path_points = [(pt["position"][0], pt["position"][1]) for pt in points]
                            start_item = sample.get("instructions", [{}])[0].get("start") if sample.get("instructions") else None
                            end_item = sample.get("instructions", [{}])[0].get("end") if sample.get("instructions") else None

                            vis_path = vis_dir / f"{scene_key}_traj_{trajectory_id}.png"
                            if sem_data and visualize_trajectory_on_map(sem_data, path_points, vis_path, scale, min_x, min_y, start_item, end_item):
                                vis_count += 1

                            if vis_count % 20 == 0:
                                print(f"[{scene_key}] Visualization progress: {vis_count}/{len(trajectory_samples)}")

                    print(f"[{scene_key}] Generated {vis_count}/{len(trajectory_samples)} visualizations")
                except Exception as e:
                    print(f"[{scene_key}] Visualization generation failed: {e}")

            return True
        else:
            print(f"[{scene_key}] Failed to generate any valid trajectories")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to generate trajectories from endpoints: {e}")
        import traceback

        traceback.print_exc()
        return False


# ==================== Visualization Functions ====================


def visualize_trajectory_on_map(
    sem_data: List[Dict],
    path_points: List[Tuple[float, float]],
    vis_path: Path,
    scale: float,
    min_x: float,
    min_y: float,
    start_item_id: str | None = None,
    end_item_id: str | None = None,
) -> bool:
    """
    Visualize trajectory path on 2D map.
    
    Args:
        sem_data: Semantic data
        path_points: Trajectory path points list [(x, y), ...]
        vis_path: Path to save the image
        scale: Pixel to world coordinate scale ratio
        min_x, min_y: Minimum world coordinates
        start_item_id: Start item_id (for text annotation)
        end_item_id: End item_id (for text annotation)
    
    Returns:
        bool: Whether visualization was successful
    """
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not available, skipping visualization")
        return False

    try:
        if not path_points or len(path_points) < 2:
            return False

        # Calculate map dimensions
        all_y = []
        all_x = []

        for inst in sem_data:
            if not isinstance(inst, dict):
                continue

            mask_coords_m = inst.get("mask_coords_m", [])
            for y, x in mask_coords_m:
                try:
                    all_y.append(float(y))
                    all_x.append(float(x))
                except (ValueError, TypeError):
                    continue

        if not all_y or not all_x:
            return False

        min_y_calc, max_y = min(all_y), max(all_y)
        min_x_calc, max_x = min(all_x), max(all_x)
        h = int(np.ceil((max_y - min_y_calc) / scale)) + 1
        w = int(np.ceil((max_x - min_x_calc) / scale)) + 1

        # Use provided min_x, min_y (for consistency)
        min_x = min_x_calc
        min_y = min_y_calc

        # World to pixel coordinate conversion
        def world2pix(x_m: float, y_m: float) -> Tuple[int, int]:
            px = int(round((x_m - min_x) / scale))
            py = int(round((y_m - min_y) / scale))
            return py, px

        # Create pixel coordinates for each instance
        for inst in sem_data:
            if not isinstance(inst, dict):
                continue

            pixel_coords = []
            mask_coords_m = inst.get("mask_coords_m", [])
            for y_m, x_m in mask_coords_m:
                try:
                    py, px = world2pix(float(x_m), float(y_m))
                    if 0 <= py < h and 0 <= px < w:
                        pixel_coords.append((py, px))
                except (ValueError, TypeError):
                    continue
            inst["mask_coords"] = pixel_coords

        # Build colored map
        bg_color = (31 / 255, 119 / 255, 180 / 255, 1.0)  # Dark blue
        wall_color = (158 / 255, 218 / 255, 229 / 255, 0.95)  # Light blue
        unable_color = (1.0, 128 / 255, 128 / 255, 1.0)  # Pink

        color_map_img = np.zeros((h, w, 4), dtype=float)
        color_map_img[:, :] = bg_color

        def lab(inst: Dict) -> str:
            return inst.get("category_label", "").lower()

        # Color by category label
        for inst in sem_data:
            if not isinstance(inst, dict):
                continue

            l = lab(inst)
            mask_coords = inst.get("mask_coords", [])
            if l == "wall":
                for py, px in mask_coords:
                    if 0 <= py < h and 0 <= px < w:
                        color_map_img[py, px] = wall_color
            elif l == "unable area":
                for py, px in mask_coords:
                    if 0 <= py < h and 0 <= px < w:
                        color_map_img[py, px] = unable_color

        # Image world coordinate extent
        img_extent = [min_x, min_x + w * scale, min_y, min_y + h * scale]

        # Create figure
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
        ax.set_facecolor((bg_color[0], bg_color[1], bg_color[2]))
        ax.imshow(color_map_img, extent=img_extent, origin="lower", interpolation="nearest")

        # Add start/end text annotations
        if len(path_points) >= 2:
            start_pos = path_points[0]
            goal_pos = path_points[-1]

            start_text = f"START: {start_item_id}" if start_item_id else "START"
            ax.text(
                start_pos[0],
                start_pos[1],
                start_text,
                color="yellow",
                fontsize=12,
                ha="center",
                va="center",
                fontweight="bold",
            )

            goal_text = f"GOAL: {end_item_id}" if end_item_id else "GOAL"
            ax.text(
                goal_pos[0],
                goal_pos[1],
                goal_text,
                color="yellow",
                fontsize=12,
                ha="center",
                va="center",
                fontweight="bold",
            )

        # Draw trajectory path
        xs_w = [pt[0] for pt in path_points]
        ys_w = [pt[1] for pt in path_points]
        ax.plot(xs_w, ys_w, "-", color="red", linewidth=3, alpha=0.9)
        ax.scatter([xs_w[0], xs_w[-1]], [ys_w[0], ys_w[-1]], color="red", s=80)

        ax.set_title("2D Navigation Map - Trajectory Visualization")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")

        # Save image
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return True

    except Exception as e:
        print(f"[ERROR] Visualization generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def generate_missing_visualizations(
    scene_key: str, traj_root: Path, sem_map_root: Path, scale: float = SCALE_M_PER_PX
) -> None:
    """
    Check existing trajectory data and generate visualizations for missing ones.
    """
    if not HAS_MATPLOTLIB:
        print(f"[SKIP] {scene_key} matplotlib not available, skipping visualization generation")
        return

    print(f"[{scene_key}] Checking if missing visualizations need to be generated...")

    tail_id = scene_key.split("_")[-1] if "_" in scene_key else scene_key
    traj_dir = traj_root / tail_id

    if not traj_dir.exists():
        return

    # Find all trajectory files
    trajectory_files = list(traj_dir.glob(f"trajectories_{scene_key}*.json"))

    if not trajectory_files:
        return

    # Check which trajectory files are missing visualizations
    missing_vis_files = []

    for traj_file in trajectory_files:
        filename = traj_file.name

        # Determine corresponding visualization directory name
        if "_part" in filename:
            part_match = re.search(r"_part(\d+)", filename)
            if part_match:
                part_num = int(part_match.group(1))
                vis_dir_name = f"recollected_nav_vis_part{part_num}"
            else:
                vis_dir_name = "recollected_nav_vis"
        else:
            vis_dir_name = "recollected_nav_vis"

        vis_dir = traj_dir / vis_dir_name

        # Check if visualization directory exists and has content
        needs_generation = False
        if not vis_dir.exists():
            needs_generation = True
        else:
            vis_files = [f for f in vis_dir.glob("*.png")]
            if len(vis_files) == 0:
                needs_generation = True

        if needs_generation:
            missing_vis_files.append((traj_file, vis_dir, vis_dir_name))

    if not missing_vis_files:
        print(f"[{scene_key}] All trajectory data has corresponding visualizations")
        return

    print(f"[{scene_key}] Found {len(missing_vis_files)} trajectory files missing visualizations, generating...")

    # Load scene data for visualization
    try:
        sem_map_path = sem_map_root / f"2D_Semantic_Map_{scene_key}_Complete.json"
        if not sem_map_path.exists():
            print(f"[{scene_key}] Semantic map file does not exist, skipping visualization generation: {sem_map_path}")
            return

        with sem_map_path.open("r", encoding="utf-8") as f:
            sem_data = json.load(f)

        # Build 2D map to get scale and bounds
        all_y = []
        all_x = []
        for inst in sem_data:
            for y, x in inst.get("mask_coords_m", []):
                try:
                    all_y.append(float(y))
                    all_x.append(float(x))
                except (ValueError, TypeError):
                    continue

        if not all_y or not all_x:
            print(f"[{scene_key}] Cannot calculate map bounds, skipping visualization")
            return

        min_y, max_y = min(all_y), max(all_y)
        min_x, max_x = min(all_x), max(all_x)

        # Generate visualizations for each missing file
        for traj_file, vis_dir, vis_dir_name in missing_vis_files:
            try:
                print(f"[{scene_key}] Generating visualizations: {traj_file.name} -> {vis_dir_name}")

                # Read trajectory data
                with traj_file.open("r", encoding="utf-8") as f:
                    traj_data = json.load(f)

                if "scenes" not in traj_data or len(traj_data["scenes"]) == 0:
                    continue

                samples = traj_data["scenes"][0].get("samples", [])
                if not samples:
                    continue

                # Create visualization directory
                vis_dir.mkdir(parents=True, exist_ok=True)

                # Generate visualization for each sample
                vis_count = 0
                for sample in samples:
                    trajectory_id = sample.get("trajectory_id", "unknown")
                    points = sample.get("points", [])
                    instructions = sample.get("instructions", [])

                    if points:
                        # Extract trajectory path points
                        path_points = [(pt["position"][0], pt["position"][1]) for pt in points]

                        # Extract start/end item IDs from instructions
                        start_item_id = None
                        end_item_id = None
                        if instructions and len(instructions) > 0:
                            first_instr = instructions[0]
                            start_item_id = first_instr.get("start")
                            end_item_id = first_instr.get("end")

                        # Generate visualization image
                        vis_path = vis_dir / f"{scene_key}_traj_{trajectory_id}.png"
                        success = visualize_trajectory_on_map(
                            sem_data, path_points, vis_path, scale, min_x, min_y, start_item_id, end_item_id
                        )

                        if success:
                            vis_count += 1

                print(f"[{scene_key}] {traj_file.name}: Generated {vis_count}/{len(samples)} visualizations")

            except Exception as e:
                print(f"[{scene_key}] Failed to generate visualizations for {traj_file.name}: {e}")

        print(f"[{scene_key}] Missing visualization generation complete")

    except Exception as e:
        print(f"[{scene_key}] Failed to load scene data, cannot generate visualizations: {e}")


# ==================== Main Processing Loop ====================


def generate_instructions_with_retry(
    client: OpenAIClient | CompanyAPIClient,
    template: List[Dict[str, str]],
    scene_text: str,
    start_item: str,
    end_item: str,
    scene_key: str,
    max_retry: int = MAX_INSTR_RETRY,
) -> Tuple[List[Dict], bool]:
    """Generate instructions with retry logic."""
    last_list, last_ok = [], False

    for attempt in range(max_retry + 1):
        try:
            instr_list, has_valid = llm_generate_instructions(client, template, scene_text, start_item, end_item, scene_key)
            if has_valid:
                if attempt > 0:
                    print(f"[{scene_key}] Instruction generation retry succeeded: {start_item}->{end_item} (attempt {attempt+1})")
                return instr_list, True
            last_list, last_ok = instr_list, has_valid
        except Exception as e:
            print(f"[{scene_key}] Instruction generation exception (attempt {attempt+1}/{max_retry}): {start_item}->{end_item}, error: {e}")

        # Exponential backoff
        if attempt < max_retry:
            sleep_s = min(2 ** attempt, 16)
            time.sleep(sleep_s)

    return last_list if last_list else [
        {
            "instruction_type": "Default",
            "start": start_item,
            "end": end_item,
            "generated_instruction": f"Navigate from {start_item} to {end_item}.",
            "scene_id": scene_key,
        }
    ], False


def process_scene(
    scene_dir: Path,
    clients: List[OpenAIClient | CompanyAPIClient],
    label_root: Path,
    scene_text_root: Path,
    sem_map_root: Path,
    endpoint_root: Path,
    traj_root: Path,
    pairwise_template: List[Dict[str, str]],
    pairwise_batch_template: List[Dict[str, str]],
    instr_template: List[Dict[str, str]],
    min_trajs: int = MIN_TRAJS_PER_SCENE,
    max_pairs_check: int = MAX_TOTAL_PAIRS_CHECK,
    max_pairs_per_batch: int = MAX_PAIRS_PER_BATCH,
    batch_pairs_per_llm: int = BATCH_PAIRS_PER_LLM_CALL,
    judge_workers: int = JUDGE_WORKERS,
    instr_workers: int = INSTR_WORKERS,
) -> None:
    """Process a single scene to generate trajectories."""
    scene_key = extract_key_from_scene_dir_name(scene_dir.name)

    # First, check and generate missing visualizations for existing trajectories
    if HAS_MATPLOTLIB:
        generate_missing_visualizations(scene_key, traj_root, sem_map_root)

    # Check existing trajectories
    existing_trajectories = count_existing_trajectories(scene_key, traj_root)
    print(f"[{scene_key}] Existing trajectories: {existing_trajectories}/{min_trajs}")

    # Check endpoint-trajectory file correspondence and generate missing trajectories
    file_status = check_endpoint_trajectory_pairs(scene_key, endpoint_root, traj_root)
    missing_trajectories = file_status["missing_trajectories"]
    total_endpoints = file_status["total_endpoints"]
    complete_trajectories = file_status["complete_trajectories"]

    print(
        f"[{scene_key}] File status: actual trajectories={existing_trajectories}, "
        f"total endpoints={total_endpoints}, missing trajectories={len(missing_trajectories)}, target={min_trajs}"
    )

    # If there are missing trajectory files, prioritize generating those
    if missing_trajectories:
        print(f"[{scene_key}] Found {len(missing_trajectories)} endpoint files missing trajectory files, generating them first...")

        # Load required files for trajectory generation
        scene_text = load_scene_text(scene_key, scene_text_root)
        if not scene_text:
            print(f"[SKIP] {scene_key} missing scene text, cannot generate missing trajectories")
        else:
            sem_map_path = sem_map_root / f"2D_Semantic_Map_{scene_key}_Complete.json"
            if sem_map_path.exists():
                with sem_map_path.open("r", encoding="utf-8") as f:
                    sem_data = json.load(f)

                # Build 2D map
                grid_map, scale, min_x, min_y = build_2d_map(sem_data)
                if grid_map is not None:
                    # Create item mapping
                    itemid2inst: Dict[str, Dict] = {}
                    for inst in sem_data:
                        item_id = inst.get("item_id", "")
                        if item_id:
                            itemid2inst[item_id] = inst

                    # Generate missing trajectories
                    generated_count = 0
                    for endpoints_file_str, traj_file_str in missing_trajectories:
                        endpoints_file = Path(endpoints_file_str)
                        traj_file = Path(traj_file_str)
                        print(f"[{scene_key}] Generating trajectory from {endpoints_file.name} -> {traj_file.name}")
                        success = generate_trajectories_from_endpoints(
                            scene_key,
                            endpoints_file,
                            traj_file,
                            itemid2inst,
                            grid_map,
                            scale,
                            min_x,
                            min_y,
                            scene_text,
                            clients,
                            instr_template,
                            sem_data,
                        )
                        if success:
                            generated_count += 1

                    print(f"[{scene_key}] Successfully generated {generated_count}/{len(missing_trajectories)} missing trajectory files")

                    # Re-check trajectory count after generating missing ones
                    existing_trajectories = count_existing_trajectories(scene_key, traj_root)
                    print(f"[{scene_key}] Updated trajectory count: {existing_trajectories}/{min_trajs}")

    if existing_trajectories >= min_trajs:
        print(f"[SKIP] {scene_key} already has enough trajectories ({existing_trajectories}/{min_trajs})")
        return

    # Load required files for new trajectory generation
    labels_path = scene_dir / "labels.json"
    if not labels_path.exists():
        print(f"[SKIP] {scene_key} missing labels.json")
        return

    scene_text = load_scene_text(scene_key, scene_text_root)
    if not scene_text:
        print(f"[SKIP] {scene_key} missing scene text")
        return

    sem_map_path = sem_map_root / f"2D_Semantic_Map_{scene_key}_Complete.json"
    if not sem_map_path.exists():
        print(f"[SKIP] {scene_key} missing semantic map: {sem_map_path}")
        return

    # Load semantic map
    with sem_map_path.open("r", encoding="utf-8") as f:
        sem_data = json.load(f)

    # Build 2D map
    grid_map, scale, min_x, min_y = build_2d_map(sem_data)
    if grid_map is None:
        print(f"[ERROR] {scene_key} failed to build 2D map")
        return

    # Create item mapping
    itemid2inst: Dict[str, Dict] = {}
    map_item_ids = set()
    for inst in sem_data:
        item_id = str(inst.get("item_id", "")).strip()
        if item_id:
            itemid2inst[item_id] = inst
            map_item_ids.add(item_id)

    # Precompute connectivity map for optimization
    print(f"[{scene_key}] Precomputing connectivity map...")
    connectivity_map = build_connectivity_map(grid_map, itemid2inst)
    print(f"[{scene_key}] Connectivity map complete: {len(connectivity_map)} connected components")

    # Prepare visualization base map (Unable Area + Wall only)
    if HAS_MATPLOTLIB:
        h, w = grid_map.shape
        bg_color = (31 / 255, 119 / 255, 180 / 255, 1.0)  # Deep blue
        wall_color = (158 / 255, 218 / 255, 229 / 255, 0.95)  # Light blue
        unable_color = (1.0, 128 / 255, 128 / 255, 1.0)  # Pink

        def lab(inst: Dict) -> str:
            return str(inst.get("category_label", "")).lower()

        color_map_img = np.zeros((h, w, 4), dtype=float)
        color_map_img[:, :] = bg_color

        wall_cids = {inst["category_id"] for inst in sem_data if lab(inst) == "wall"}
        unable_cids = {inst["category_id"] for inst in sem_data if lab(inst) == "unable area"}

        for inst in sem_data:
            cid = inst["category_id"]
            l = lab(inst)
            if (cid in wall_cids) or (l == "wall"):
                for py, px in inst.get("mask_coords", []):
                    if 0 <= py < h and 0 <= px < w:
                        color_map_img[py, px] = wall_color
            elif (cid in unable_cids) or (l == "unable area"):
                for py, px in inst.get("mask_coords", []):
                    if 0 <= py < h and 0 <= px < w:
                        color_map_img[py, px] = unable_color

        img_extent = [min_x, min_x + w * scale, min_y, min_y + h * scale]
    else:
        color_map_img = None
        img_extent = None

    # Load labels and create candidates
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)

    if isinstance(labels_data, dict) and "labels" in labels_data:
        labels = labels_data["labels"]
    elif isinstance(labels_data, list):
        labels = labels_data
    else:
        print(f"[ERROR] {scene_key} invalid labels.json format")
        return

    label_counter = defaultdict(int)
    label_item_ids = []
    for obj in labels:
        lbl = obj.get("label", "")
        if lbl:
            item_id = item_id_from_label_counts(lbl, label_counter)
            label_item_ids.append(item_id)

    candidates = sorted([iid for iid in label_item_ids if iid in map_item_ids])
    print(f"[{scene_key}] Candidate objects: {len(candidates)}")
    max_possible_pairs = len(candidates) * (len(candidates) - 1)
    print(f"[{scene_key}] Maximum possible endpoint pairs: {max_possible_pairs:,}")

    # Get existing pairs (from trajectory files, more reliable)
    existing_pairs = get_existing_endpoint_pairs_from_trajectories(scene_key, traj_root)
    print(f"[{scene_key}] Existing endpoint pairs (from trajectories): {len(existing_pairs)}")

    # Data consistency check
    if len(existing_pairs) > 0 and existing_trajectories > 0:
        dedup_ratio = len(existing_pairs) / existing_trajectories
        print(f"[{scene_key}] Deduplication ratio: {dedup_ratio:.3f} ({len(existing_pairs)} unique pairs / {existing_trajectories} trajectories)")
        if dedup_ratio < 0.9:
            print(f"[{scene_key}] WARNING: Possible duplicate trajectories detected (low deduplication ratio)")

    # Generator function: batch generate endpoint pairs incrementally
    pairs_per_batch = 5000  # Generate 5000 pairs per batch
    candidate_start_idx = 0

    def generate_candidate_pairs_batch():
        """Generate endpoint pairs batch by batch."""
        nonlocal candidate_start_idx
        batch = []
        while len(batch) < pairs_per_batch and candidate_start_idx < len(candidates):
            s = candidates[candidate_start_idx]
            for e in candidates:
                if s != e and (s, e) not in existing_pairs:
                    batch.append((s, e))
                    if len(batch) >= pairs_per_batch:
                        break
            candidate_start_idx += 1
        return batch

    # Process in batches with sub-batch processing
    current_traj_count = existing_trajectories
    batch_num = 0
    total_pairs_processed = 0
    current_part_num: int | None = None
    current_endpoints_path: Path | None = None

    while current_traj_count < min_trajs and candidate_start_idx < len(candidates) and total_pairs_processed < max_pairs_check:
        batch_num += 1

        # Generate current batch pairs
        current_batch_pairs = generate_candidate_pairs_batch()
        if not current_batch_pairs:
            print(f"[{scene_key}] No more endpoint pairs to generate, stopping")
            break

        print(f"\n[{scene_key}] === Batch {batch_num} ===")
        print(f"[{scene_key}] Generated endpoint pairs: {len(current_batch_pairs)}")

        # Filter pairs
        filtered_pairs = filter_pairs_by_distance_and_category(current_batch_pairs, itemid2inst)
        print(f"[{scene_key}] After filtering: {len(filtered_pairs)} pairs")

        if not filtered_pairs:
            print(f"[{scene_key}] No valid pairs in this batch, continuing...")
            total_pairs_processed += len(current_batch_pairs)
            continue

        # Determine part number and file paths (only once per batch)
        if current_part_num is None:
            if existing_pairs or batch_num > 1:
                current_part_num = get_next_part_number(scene_key, endpoint_root)
                endpoints_suffix = f"_part{current_part_num}"
                traj_suffix = f"_part{current_part_num}"
                current_endpoints_path = endpoint_root / f"{scene_key}_endpoints{endpoints_suffix}.json"
            else:
                endpoints_suffix = ""
                traj_suffix = ""
                current_endpoints_path = endpoint_root / f"{scene_key}_endpoints.json"
                print(f"[{scene_key}] Starting with main file")
        else:
            endpoints_suffix = f"_part{current_part_num}" if current_part_num else ""
            traj_suffix = f"_part{current_part_num}" if current_part_num else ""

        # Calculate needed trajectories
        needed_trajs = min_trajs - current_traj_count
        max_pairs_this_batch = min(len(filtered_pairs), max_pairs_check - total_pairs_processed)
        print(f"[{scene_key}] Batch {batch_num}: Need {needed_trajs} trajectories, processing {max_pairs_this_batch} pairs")

        # Shuffle pairs
        random.shuffle(filtered_pairs)
        round_pairs = filtered_pairs[:max_pairs_this_batch]

        # Sub-batch processing
        pairs_llm_round = []  # All meaningful pairs in this batch
        reachable_entries_round = []  # All reachable paths in this batch
        total_checked_round = 0
        total_ok_round = 0
        total_api_fail_round = 0
        sub_batch_id = 0
        cursor = 0

        print(f"[{scene_key}] Sub-batch size: {max_pairs_per_batch}")

        # Process in sub-batches
        while len(reachable_entries_round) < needed_trajs and cursor < len(round_pairs):
            sub_batch_id += 1
            batch_pairs = round_pairs[cursor : cursor + max_pairs_per_batch]
            cursor += len(batch_pairs)
            total_pairs_processed += len(batch_pairs)

            print(f"[{scene_key}] Sub-batch {sub_batch_id}: Judging {len(batch_pairs)} pairs (progress: {cursor}/{len(round_pairs)})")

            # Group pairs for batch LLM calls
            batch_groups = []
            for i in range(0, len(batch_pairs), batch_pairs_per_llm):
                batch_groups.append(batch_pairs[i : i + batch_pairs_per_llm])

            # Batch judge pairs
            checked = 0
            ok_count = 0
            api_fail = 0
            pairs_llm_batch = []

            judge_start_time = time.time()
            with ThreadPoolExecutor(max_workers=min(judge_workers, len(batch_groups))) as ex:
                futures = []
                for idx, pairs_group in enumerate(batch_groups):
                    client = clients[idx % len(clients)]
                    futures.append(ex.submit(llm_judge_pairs_batch_v2, client, pairwise_batch_template, scene_text, pairs_group))

                for fut in as_completed(futures):
                    results = fut.result()
                    for s, e, ok, api_success in results:
                        checked += 1
                        total_checked_round += 1
                        if ok:
                            ok_count += 1
                            total_ok_round += 1
                            pairs_llm_batch.append({"start": s, "end": e})
                        if not api_success:
                            api_fail += 1
                            total_api_fail_round += 1

                    if checked % PRINT_EVERY_JUDGE == 0 or checked >= len(batch_pairs):
                        print(f"[{scene_key}] Sub-batch {sub_batch_id}: Judged {checked}/{len(batch_pairs)} | ok:{ok_count} | api_fail:{api_fail}")

            judge_duration = time.time() - judge_start_time
            print(f"[{scene_key}] Sub-batch {sub_batch_id} judgment time: {judge_duration:.2f}s | avg: {judge_duration/len(batch_pairs):.3f}s per pair")

            pairs_llm_round.extend(pairs_llm_batch)

            # Validate paths and generate trajectories
            reach_ok_batch = 0
            if pairs_llm_batch:
                path_start_time = time.time()
                with ThreadPoolExecutor(max_workers=min(32, len(pairs_llm_batch))) as path_ex:
                    path_futures = [
                        path_ex.submit(
                            validate_and_generate_path,
                            pair["start"],
                            pair["end"],
                            itemid2inst,
                            grid_map,
                            scale,
                            min_x,
                            min_y,
                        )
                        for pair in pairs_llm_batch
                    ]

                    for path_fut in as_completed(path_futures):
                        result = path_fut.result()
                        if result:
                            reachable_entries_round.append(result)
                            reach_ok_batch += 1
                            if reach_ok_batch % PRINT_EVERY_REACH == 0:
                                print(f"[{scene_key}] Sub-batch {sub_batch_id}: Reachable paths {reach_ok_batch}/{len(pairs_llm_batch)} | Total reachable: {len(reachable_entries_round)}")

                path_duration = time.time() - path_start_time
                print(f"[{scene_key}] Sub-batch {sub_batch_id} path validation time: {path_duration:.2f}s | avg: {path_duration/len(pairs_llm_batch):.3f}s per pair")
            else:
                print(f"[{scene_key}] Sub-batch {sub_batch_id}: No meaningful pairs, skipping path validation")

            print(f"[{scene_key}] Sub-batch {sub_batch_id} complete: OK {ok_count}/{len(batch_pairs)} | Reachable {reach_ok_batch} | Total reachable: {len(reachable_entries_round)}")

            # Incremental save: save endpoints every N reachable paths
            if reach_ok_batch > 0 and len(reachable_entries_round) % INCREMENTAL_SAVE_THRESHOLD == 0:
                last_saved_index = len(reachable_entries_round) - (len(reachable_entries_round) % INCREMENTAL_SAVE_THRESHOLD)
                current_batch_entries = reachable_entries_round[last_saved_index:]
                current_pairs_to_save = [{"start": entry["start"], "end": entry["end"]} for entry in current_batch_entries]

                if current_endpoints_path and append_to_json_file(current_endpoints_path, current_pairs_to_save, "endpoints"):
                    print(f"[{scene_key}] Incremental save: {len(current_pairs_to_save)} endpoints saved to {current_endpoints_path.name}")

            # Break if we have enough trajectories
            if len(reachable_entries_round) >= needed_trajs:
                break

        print(f"[{scene_key}] Batch {batch_num} summary: checked={total_checked_round}, ok={total_ok_round}, api_fail={total_api_fail_round} | Reachable={len(reachable_entries_round)}")

        if not reachable_entries_round:
            print(f"[{scene_key}] Batch {batch_num}: No reachable paths, continuing...")
            # Save meaningful but unreachable pairs for debugging
            if pairs_llm_round and current_endpoints_path:
                with current_endpoints_path.open("w", encoding="utf-8") as f:
                    json.dump(pairs_llm_round, f, indent=2)
                print(f"[{scene_key}] Saved {len(pairs_llm_round)} meaningful but unreachable pairs to {current_endpoints_path.name}")
            continue

        # Save remaining endpoints (if any)
        last_saved_count = (len(reachable_entries_round) // INCREMENTAL_SAVE_THRESHOLD) * INCREMENTAL_SAVE_THRESHOLD
        if len(reachable_entries_round) > last_saved_count and current_endpoints_path:
            unsaved_entries = reachable_entries_round[last_saved_count:]
            unsaved_pairs = [{"start": entry["start"], "end": entry["end"]} for entry in unsaved_entries]
            if append_to_json_file(current_endpoints_path, unsaved_pairs, "endpoints"):
                print(f"[{scene_key}] Saved remaining {len(unsaved_pairs)} endpoints to {current_endpoints_path.name}")

        # Generate instructions with timeout control
        print(f"[{scene_key}] Generating instructions for {len(reachable_entries_round)} trajectories...")
        # Limit to what we need
        needed_trajs = min_trajs - current_traj_count
        entries_to_process = reachable_entries_round[:needed_trajs]

        # Generate instructions with timeout control
        instructions = [None] * len(entries_to_process)
        instr_api_fail = 0

        def default_instr(s: str, e: str) -> List[Dict]:
            return [
                {
                    "instruction_type": "Default",
                    "start": s,
                    "end": e,
                    "generated_instruction": f"Navigate from {s} to {e}.",
                    "scene_id": scene_key,
                }
            ]

        instr_start_time = time.time()
        with ThreadPoolExecutor(max_workers=min(instr_workers, len(entries_to_process))) as ex:
            fut_map: Dict[Any, Tuple[int, str, str]] = {}
            for idx, entry in enumerate(entries_to_process):
                s, e = entry["start"], entry["end"]
                client = clients[idx % len(clients)]
                fut = ex.submit(generate_instructions_with_retry, client, instr_template, scene_text, s, e, scene_key, MAX_INSTR_RETRY)
                fut_map[fut] = (idx, s, e)

            pending = set(fut_map.keys())
            done_cnt = 0
            while pending:
                done, not_done = wait(pending, timeout=INSTR_WAIT_TIMEOUT, return_when=FIRST_COMPLETED)
                if not done:
                    # Timeout: cancel remaining tasks and use default instructions
                    to_cancel = list(not_done)
                    for fut in to_cancel:
                        idx, s, e = fut_map[fut]
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        instructions[idx] = default_instr(s, e)
                        instr_api_fail += 1
                        pending.remove(fut)
                    print(f"[{scene_key}] Instruction generation timeout, using defaults for {len(to_cancel)} remaining items")
                    break

                for fut in done:
                    idx, s, e = fut_map[fut]
                    try:
                        instr_list, api_ok = fut.result()
                        if not api_ok:
                            instr_api_fail += 1
                    except Exception as ex_err:
                        print(f"[{scene_key}] Instruction generation exception: {ex_err}")
                        instr_list = default_instr(s, e)
                        instr_api_fail += 1
                    instructions[idx] = instr_list
                    pending.remove(fut)
                    done_cnt += 1
                    if done_cnt % 10 == 0 or not pending:
                        print(f"[{scene_key}] Instruction progress {done_cnt}/{len(entries_to_process)} | API failures: {instr_api_fail}")

        instr_duration = time.time() - instr_start_time
        print(f"[{scene_key}] Batch {batch_num} instruction generation time: {instr_duration:.2f}s | avg: {instr_duration/len(entries_to_process):.3f}s | API failures: {instr_api_fail}")

        # Create samples and generate visualizations
        samples_out = []
        tail_id = scene_key.split("_")[-1] if "_" in scene_key else scene_key
        traj_dir = traj_root / tail_id
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Prepare visualization directory
        if current_part_num:
            vis_dir_name = f"recollected_nav_vis_part{current_part_num}"
        else:
            vis_dir_name = "recollected_nav_vis"
        vis_dir = traj_dir / vis_dir_name
        vis_dir.mkdir(parents=True, exist_ok=True)

        def inst_centroid_world(inst: Dict) -> Tuple[float, float] | None:
            """Calculate instance centroid in world coordinates."""
            mask_coords = inst.get("mask_coords", [])
            if not mask_coords:
                return None
            m = np.array(mask_coords)
            c = m.mean(axis=0)  # (y, x)
            cx = min_x + (c[1] + 0.5) * scale
            cy = min_y + (c[0] + 0.5) * scale
            return cx, cy

        for tid, (entry, instr_list) in enumerate(zip(entries_to_process, instructions)):
            s, e = entry["start"], entry["end"]
            if instr_list is None:
                instr_list = default_instr(s, e)

            # Create sample
            samples_out.append(
                {
                    "trajectory_id": str(current_traj_count + tid),
                    "instructions": instr_list,
                    "points": entry["points"],
                }
            )

            # Generate visualization
            if HAS_MATPLOTLIB and color_map_img is not None and img_extent is not None:
                world_path = [(p["position"][0], p["position"][1]) for p in entry["points"]]
                start_inst = itemid2inst.get(s)
                goal_inst = itemid2inst.get(e)

                fig = plt.figure(figsize=(12, 12))
                ax = plt.gca()
                ax.set_facecolor((bg_color[0], bg_color[1], bg_color[2]))
                ax.imshow(color_map_img, extent=img_extent, origin="lower", interpolation="nearest")

                # Annotate start/goal
                start_cw = inst_centroid_world(start_inst) if start_inst else None
                goal_cw = inst_centroid_world(goal_inst) if goal_inst else None

                if start_cw:
                    ax.text(start_cw[0], start_cw[1], f"START: {s}", color="yellow", fontsize=12, ha="center", va="center", fontweight="bold")
                if goal_cw:
                    ax.text(goal_cw[0], goal_cw[1], f"GOAL: {e}", color="yellow", fontsize=12, ha="center", va="center", fontweight="bold")

                # Draw trajectory path
                if len(world_path) >= 2:
                    xs_w = [wp[0] for wp in world_path]
                    ys_w = [wp[1] for wp in world_path]
                    ax.plot(xs_w, ys_w, "-", color="red", linewidth=3, alpha=0.9)
                    ax.scatter([xs_w[0], xs_w[-1]], [ys_w[0], ys_w[-1]], color="red", s=80)

                ax.set_title(f"2D Navigation Map (obstacles: Unable + Wall) - {tail_id} | traj {current_traj_count + tid}")
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")

                vis_path = vis_dir / f"{tail_id}_traj_{current_traj_count + tid}_vis.png"
                plt.savefig(vis_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

        # Save trajectory file
        traj_file = traj_dir / f"trajectories_{scene_key}{traj_suffix}.json"
        save_trajectory_file(samples_out, scene_key, traj_file, current_part_num)
        print(f"[{scene_key}] Saved trajectory file: {traj_file.name} ({len(samples_out)} samples)")
        if HAS_MATPLOTLIB:
            print(f"[{scene_key}] Visualization directory: {vis_dir_name}")

        # Update counts
        current_traj_count += len(samples_out)
        for pair in pairs_llm_round:
            existing_pairs.add((pair["start"], pair["end"]))

        print(f"[{scene_key}] Batch {batch_num} complete: Current trajectory count {current_traj_count}/{min_trajs}")

        # Check if we've reached the target
        if current_traj_count >= min_trajs:
            print(f"[{scene_key}] Reached target of {min_trajs} trajectories")
            break

    print(f"[{scene_key}] Processing complete: Final trajectory count {current_traj_count}/{min_trajs}, Total pairs processed: {total_pairs_processed}")


# ==================== CLI Interface ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate VLN trajectories from InteriorGS 2D semantic maps.")

    # Input/output paths
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT, help="Root directory with InteriorGS scene folders")
    parser.add_argument("--scene-text-root", type=Path, default=DEFAULT_SCENE_TEXT_ROOT, help="Directory with textual scene maps")
    parser.add_argument("--sem-map-root", type=Path, default=DEFAULT_SEM_MAP_ROOT, help="Directory with 2D semantic maps")
    parser.add_argument("--endpoint-root", type=Path, default=DEFAULT_ENDPOINT_OUT_ROOT, help="Output directory for endpoint pairs")
    parser.add_argument("--traj-root", type=Path, default=DEFAULT_TRAJ_OUT_ROOT, help="Output directory for trajectories")

    # Prompt files
    parser.add_argument("--prompt-pairwise", type=Path, default=DEFAULT_PROMPT_PAIRWISE, help="Path to pairwise judgement prompt")
    parser.add_argument("--prompt-pairwise-batch", type=Path, default=DEFAULT_PROMPT_PAIRWISE_BATCH, help="Path to batch pairwise judgement prompt")
    parser.add_argument("--prompt-traj-to-instr", type=Path, default=DEFAULT_PROMPT_TRAJ_TO_INSTR, help="Path to trajectory-to-instruction prompt")

    # API configuration
    parser.add_argument("--api-type", type=str, choices=["openai", "company"], default="openai", 
                       help="API type to use: 'openai' for OpenAI API, 'company' for company internal API (testing)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), 
                       help="API key (or set OPENAI_API_KEY env var for OpenAI, or COMPANY_API_KEY for company API)")
    parser.add_argument("--api-base", type=str, default=None, 
                       help="API base URL (default: OpenAI=https://api.openai.com/v1, Company=https://oneapi.qunhequnhe.com/v1)")
    parser.add_argument("--model", type=str, default=None, 
                       help="Model name to use (default: OpenAI=gpt-4o-mini, Company=gpt-5)")
    parser.add_argument("--api-timeout", type=int, default=60, help="API request timeout in seconds")

    # Processing parameters
    parser.add_argument("--min-trajs", type=int, default=MIN_TRAJS_PER_SCENE, help="Minimum trajectories per scene")
    parser.add_argument("--max-pairs-check", type=int, default=MAX_TOTAL_PAIRS_CHECK, help="Maximum pairs to check per scene")
    parser.add_argument("--judge-workers", type=int, default=JUDGE_WORKERS, help="Number of workers for pair judgement")
    parser.add_argument("--instr-workers", type=int, default=INSTR_WORKERS, help="Number of workers for instruction generation")
    parser.add_argument("--max-pairs-per-batch", type=int, default=MAX_PAIRS_PER_BATCH, help="Max pairs per batch")
    parser.add_argument("--batch-pairs-per-llm-call", type=int, default=BATCH_PAIRS_PER_LLM_CALL, help="Pairs per LLM call in batch")

    # Scene filtering
    parser.add_argument("--only", nargs="+", help="Only process specified scene folders")
    parser.add_argument("--skip-bad-scenes", action="store_true", help="Skip known bad scenes")
    parser.add_argument("--bad-scene-ids", nargs="+", default=[], help="Additional bad scene IDs to skip")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set default API parameters based on API type
    if args.api_type == "company":
        # Company API defaults
        if not args.api_key:
            args.api_key = os.environ.get("COMPANY_API_KEY", "")
        if not args.api_base:
            args.api_base = "https://oneapi.qunhequnhe.com/v1"
        if not args.model:
            args.model = "gpt-5"
        
        # Validate API key
        if not args.api_key:
            print("[ERROR] Company API key required. Provide --api-key or set COMPANY_API_KEY environment variable.")
            return
        
        # Validate langchain is available
        if not HAS_LANGCHAIN:
            print("[ERROR] Company API requires langchain-openai. Install it with: pip install langchain-openai langchain-core")
            return
        
        # Create company API clients
        clients = [CompanyAPIClient(args.api_key, args.api_base, args.model, args.api_timeout)]
        print(f"[INFO] Using Company API: {args.api_base} with model {args.model}")
    else:
        # OpenAI API defaults
        if not args.api_key:
            args.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not args.api_base:
            args.api_base = "https://api.openai.com/v1"
        if not args.model:
            args.model = "gpt-4o-mini"
        
        # Validate API key
        if not args.api_key:
            print("[ERROR] OpenAI API key required. Provide --api-key or set OPENAI_API_KEY environment variable.")
            return
        
        # Create OpenAI API clients
        clients = [OpenAIClient(args.api_key, args.api_base, args.model, args.api_timeout)]
        print(f"[INFO] Using OpenAI API: {args.api_base} with model {args.model}")
    
    print(f"[INFO] Using {len(clients)} API client(s)")

    # Load prompt templates
    try:
        pairwise_template = load_prompt_template(args.prompt_pairwise)
        pairwise_batch_template = load_prompt_template(args.prompt_pairwise_batch)
        instr_template = load_prompt_template(args.prompt_traj_to_instr)
    except Exception as e:
        print(f"[ERROR] Failed to load prompt templates: {e}")
        return

    # Collect bad scene IDs
    bad_scenes = set(BAD_SCENE_IDS)
    if args.skip_bad_scenes:
        bad_scenes.update(BAD_SCENE_IDS)
    bad_scenes.update(args.bad_scene_ids)

    # Find scene directories
    if not args.label_root.exists():
        print(f"[ERROR] Label root does not exist: {args.label_root}")
        return

    scene_dirs = sorted([d for d in args.label_root.iterdir() if d.is_dir()])
    if args.only:
        scene_dirs = [d for d in scene_dirs if d.name in args.only]

    print(f"[INFO] Found {len(scene_dirs)} scene directories")

    # Create output directories
    args.endpoint_root.mkdir(parents=True, exist_ok=True)
    args.traj_root.mkdir(parents=True, exist_ok=True)

    # Process each scene
    processed = 0
    skipped = 0

    for idx, scene_dir in enumerate(scene_dirs, 1):
        scene_name = scene_dir.name

        # Check if should skip
        if args.skip_bad_scenes:
            scene_id = scene_name.split("_")[-1] if "_" in scene_name else scene_name
            if scene_id in bad_scenes:
                print(f"\n[SKIP] Scene {scene_name} ({idx}/{len(scene_dirs)}) - Bad scene")
                skipped += 1
                continue

        try:
            print(f"\n[PROCESS] Scene {scene_name} ({idx}/{len(scene_dirs)})")
            process_scene(
                scene_dir,
                clients,
                args.label_root,
                args.scene_text_root,
                args.sem_map_root,
                args.endpoint_root,
                args.traj_root,
                pairwise_template,
                pairwise_batch_template,
                instr_template,
                args.min_trajs,
                args.max_pairs_check,
                args.max_pairs_per_batch,
                args.batch_pairs_per_llm_call,
                args.judge_workers,
                args.instr_workers,
            )
            processed += 1
        except Exception as e:
            print(f"[ERROR] Scene {scene_name} processing failed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n[STATS] Total scenes: {len(scene_dirs)}, Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()

