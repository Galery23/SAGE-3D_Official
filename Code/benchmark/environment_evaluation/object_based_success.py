#!/usr/bin/env python3
"""
Object-Based Success Evaluation for SAGE-3D Benchmark.

Evaluates navigation success based on target object bounding boxes
from 2D semantic maps.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


def reverse_position_mapping(
    px_3d: float,
    py_3d: float,
    map_data: List[Dict],
    flip_x: bool = True,
    flip_y: bool = True,
    negate_xy: bool = True
) -> Tuple[float, float]:
    """Reverse mapping: convert 3D trajectory coordinates back to 2D for visualization.

    This is the inverse of the forward mapping process.

    Args:
        px_3d, py_3d: Coordinates from 3D trajectory
        map_data: Map data for boundary calculation
        flip_x, flip_y, negate_xy: Mapping parameters (should match original mapping)

    Returns:
        (px_2d, py_2d): Converted 2D coordinates
    """
    # Get map boundaries
    all_y = [float(y) for inst in map_data for y, x in inst.get('mask_coords_m', [])]
    all_x = [float(x) for inst in map_data for y, x in inst.get('mask_coords_m', [])]

    if not all_x or not all_y:
        return px_3d, py_3d

    min_y, max_y = min(all_y), max(all_y)
    min_x, max_x = min(all_x), max(all_x)

    # Reverse mapping process (opposite order of forward mapping)
    px, py = px_3d, py_3d

    # 1. Reverse negation if applied
    if negate_xy:
        px = -px
        py = -py

    # 2. Reverse mirroring if applied
    if flip_x:
        px = (min_x + max_x) - px
    if flip_y:
        py = (min_y + max_y) - py

    return px, py


class ObjectBasedSuccessEvaluator:
    """Evaluator for object-based success determination using target object bounding boxes."""

    def __init__(self, semantic_map_path: str, collision_detector=None, verbose: bool = True):
        """Initialize the evaluator.

        Args:
            semantic_map_path: Path to 2D semantic map JSON file
            collision_detector: Optional collision detector instance
            verbose: Whether to print debug information
        """
        self.semantic_map_path = semantic_map_path
        self.collision_detector = collision_detector
        self.verbose = verbose
        self.semantic_map_data = []
        self.object_bbox_cache = {}

        self._load_semantic_map()

    def _log(self, msg: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def _load_semantic_map(self) -> None:
        """Load 2D semantic map data."""
        try:
            with open(self.semantic_map_path, 'r') as f:
                self.semantic_map_data = json.load(f)
            self._log(f"[OBJECT_SUCCESS] ✓ Loaded semantic map: {len(self.semantic_map_data)} objects")

            # Build item_id to object info mapping
            for obj in self.semantic_map_data:
                if 'item_id' in obj:
                    self.object_bbox_cache[obj['item_id']] = obj

        except Exception as e:
            self._log(f"[OBJECT_SUCCESS] ✗ Failed to load semantic map: {e}")
            self.semantic_map_data = []

    def extract_end_object_id(self, episode: Dict[str, Any]) -> Optional[str]:
        """Extract end object ID from episode data.

        Args:
            episode: Episode data dictionary

        Returns:
            End object ID, or None if extraction fails
        """
        try:
            instructions = episode.get("instructions", [])
            if not instructions:
                return None

            first_instruction = instructions[0]
            if isinstance(first_instruction, dict) and "end" in first_instruction:
                end_object_id = first_instruction["end"]
                self._log(f"[OBJECT_SUCCESS] Extracted end object ID: {end_object_id}")
                return end_object_id

            return None

        except Exception as e:
            self._log(f"[OBJECT_SUCCESS] ✗ Failed to extract end object ID: {e}")
            return None

    def get_object_bbox(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get object bounding box information.

        Args:
            object_id: Object ID

        Returns:
            Object info dict containing bbox, or None
        """
        if object_id in self.object_bbox_cache:
            return self.object_bbox_cache[object_id]

        for obj in self.semantic_map_data:
            if obj.get('item_id') == object_id:
                self.object_bbox_cache[object_id] = obj
                return obj

        self._log(f"[OBJECT_SUCCESS] ⚠ Object not found: {object_id}")
        return None

    def get_object_center(self, object_id: str) -> Optional[np.ndarray]:
        """Get object center coordinates.

        Args:
            object_id: Object ID

        Returns:
            Object center [x, y] or None
        """
        obj_info = self.get_object_bbox(object_id)
        if not obj_info:
            return None

        try:
            bbox_m = obj_info.get('bbox_m', [])
            if len(bbox_m) != 4:
                return None

            x_center = (float(bbox_m[0]) + float(bbox_m[2])) / 2.0
            y_center = (float(bbox_m[1]) + float(bbox_m[3])) / 2.0

            return np.array([x_center, y_center])

        except Exception as e:
            self._log(f"[OBJECT_SUCCESS] ✗ Failed to compute object center: {e}")
            return None

    def is_position_in_object_area(
        self,
        position: np.ndarray,
        object_id: str,
        expansion_radius: float = 1.0
    ) -> bool:
        """Check if position is within object area (expanded bbox).

        Args:
            position: Agent 3D position [x, y, z]
            object_id: Target object ID
            expansion_radius: Bbox expansion radius in meters

        Returns:
            Whether position is within object area
        """
        obj_info = self.get_object_bbox(object_id)
        if not obj_info:
            return False

        try:
            # Convert 3D position to 2D coordinate system
            pos_2d_x, pos_2d_y = reverse_position_mapping(
                position[0], position[1], self.semantic_map_data
            )

            self._log(f"[OBJECT_SUCCESS] Coord transform: 3D=({position[0]:.2f}, {position[1]:.2f}) -> 2D=({pos_2d_x:.2f}, {pos_2d_y:.2f})")

            # Parse bbox info
            bbox_m = obj_info.get('bbox_m', [])
            if len(bbox_m) != 4:
                self._log(f"[OBJECT_SUCCESS] ⚠ Object {object_id} bbox format error: {bbox_m}")
                return False

            # bbox format: [x_min, y_min, x_max, y_max]
            x_min = float(bbox_m[0]) - expansion_radius
            y_min = float(bbox_m[1]) - expansion_radius
            x_max = float(bbox_m[2]) + expansion_radius
            y_max = float(bbox_m[3]) + expansion_radius

            # Check if converted 2D position is within expanded bbox
            in_bbox = (x_min <= pos_2d_x <= x_max) and (y_min <= pos_2d_y <= y_max)

            self._log(f"[OBJECT_SUCCESS] Position check: 2D_pos=({pos_2d_x:.2f}, {pos_2d_y:.2f}), "
                     f"bbox=[{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}], result={in_bbox}")

            return in_bbox

        except Exception as e:
            self._log(f"[OBJECT_SUCCESS] ✗ Position check failed: {e}")
            return False

    def is_collision_free_area(self, position: np.ndarray) -> bool:
        """Check if position is collision-free.

        Args:
            position: Position [x, y, z]

        Returns:
            Whether position is collision-free
        """
        if self.collision_detector is None:
            self._log(f"[OBJECT_SUCCESS] ⚠ No collision detector, assuming collision-free")
            return True

        try:
            if hasattr(self.collision_detector, 'check_collision_at_position'):
                is_collision = self.collision_detector.check_collision_at_position(position[0], position[1])
                return not is_collision
            else:
                self._log(f"[OBJECT_SUCCESS] ⚠ Collision detector missing check_collision_at_position method")
                return True

        except Exception as e:
            self._log(f"[OBJECT_SUCCESS] ✗ Collision detection failed: {e}")
            return True

    def evaluate_success(
        self,
        current_position: np.ndarray,
        episode: Dict[str, Any],
        expansion_radius: float = 1.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate if agent successfully reached target.

        Args:
            current_position: Current position [x, y, z]
            episode: Episode data
            expansion_radius: Bbox expansion radius

        Returns:
            (success, info_dict) tuple
        """
        result_info = {
            "method": "object_based",
            "end_object_id": None,
            "object_found": False,
            "in_object_area": False,
            "collision_free": False,
            "fallback_to_point": False
        }

        # 1. Extract end object ID
        end_object_id = self.extract_end_object_id(episode)
        result_info["end_object_id"] = end_object_id

        if not end_object_id:
            self._log(f"[OBJECT_SUCCESS] Cannot extract end object ID, trying smart position evaluation")
            result_info["fallback_to_smart_position"] = True
            smart_success, smart_info = self._smart_position_success(current_position, episode, expansion_radius)
            result_info.update(smart_info)
            return smart_success, result_info

        # 2. Get object bbox info
        obj_info = self.get_object_bbox(end_object_id)
        if not obj_info:
            self._log(f"[OBJECT_SUCCESS] Object {end_object_id} not found, trying smart position evaluation")
            result_info["fallback_to_smart_position"] = True
            smart_success, smart_info = self._smart_position_success(current_position, episode, expansion_radius)
            result_info.update(smart_info)
            return smart_success, result_info

        result_info["object_found"] = True

        # 3. Check if within object area
        in_area = self.is_position_in_object_area(current_position, end_object_id, expansion_radius)
        result_info["in_object_area"] = in_area

        # If not in labeled object area and object is far, try smart evaluation
        if not in_area:
            obj_center = self.get_object_center(end_object_id)
            if obj_center is not None:
                agent_2d_x, agent_2d_y = reverse_position_mapping(
                    current_position[0], current_position[1], self.semantic_map_data
                )
                agent_2d_pos = np.array([agent_2d_x, agent_2d_y])
                distance_to_labeled_object = np.linalg.norm(agent_2d_pos - obj_center)

                self._log(f"[OBJECT_SUCCESS] Distance to labeled object {end_object_id}: {distance_to_labeled_object:.3f}m")

                # If labeled object is too far (>5m), might be mislabeled
                if distance_to_labeled_object > 5.0:
                    self._log(f"[OBJECT_SUCCESS] ⚠ Labeled object {end_object_id} too far ({distance_to_labeled_object:.1f}m), "
                             "might be mislabeled, trying smart evaluation")
                    result_info["labeled_object_too_far"] = True
                    result_info["distance_to_labeled_object"] = distance_to_labeled_object

                    smart_success, smart_info = self._smart_position_success(current_position, episode, expansion_radius)
                    result_info.update(smart_info)
                    result_info["fallback_to_smart_position"] = True
                    return smart_success, result_info

            return False, result_info

        # 4. Check collision-free
        collision_free = self.is_collision_free_area(current_position)
        result_info["collision_free"] = collision_free

        # 5. Final determination
        success = in_area and collision_free

        self._log(f"[OBJECT_SUCCESS] Final result: object={end_object_id}, in_area={in_area}, "
                 f"collision_free={collision_free}, success={success}")

        return success, result_info

    def _fallback_point_success(self, current_position: np.ndarray, episode: Dict[str, Any]) -> bool:
        """Fallback to traditional point-based success evaluation.

        Args:
            current_position: Current position
            episode: Episode data

        Returns:
            Whether successful
        """
        try:
            goals = episode.get("goals", [])
            if not goals:
                return False

            goal_position = np.array(goals[0]["position"])
            goal_radius = goals[0].get("radius", 0.5)

            distance = np.linalg.norm(current_position - goal_position)
            success = distance < goal_radius

            self._log(f"[OBJECT_SUCCESS] Point fallback: distance={distance:.3f}, radius={goal_radius}, success={success}")

            return success

        except Exception as e:
            self._log(f"[OBJECT_SUCCESS] ✗ Point fallback failed: {e}")
            return False

    def _smart_position_success(
        self,
        current_position: np.ndarray,
        episode: Dict[str, Any],
        expansion_radius: float = 1.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """Smart position-based success evaluation using trajectory endpoint.

        Args:
            current_position: Current position
            episode: Episode data
            expansion_radius: Expansion radius

        Returns:
            (is_success, info_dict)
        """
        info = {
            "method": "smart_position",
            "found_candidates": 0,
            "best_target": None,
            "final_success": False
        }

        try:
            self._log(f"[OBJECT_SUCCESS] Using smart position evaluation")

            # Get trajectory endpoint
            gt_locations = episode.get("gt_locations", [])
            if not gt_locations:
                info["error"] = "Cannot get trajectory endpoint"
                return self._fallback_point_success(current_position, episode), info

            target_3d_pos = np.array(gt_locations[-1])

            # Convert to 2D coordinates
            target_2d_x, target_2d_y = reverse_position_mapping(
                target_3d_pos[0], target_3d_pos[1], self.semantic_map_data
            )
            target_2d_pos = np.array([target_2d_x, target_2d_y])

            info["target_3d"] = target_3d_pos[:2].tolist()
            info["target_2d"] = [target_2d_x, target_2d_y]

            self._log(f"[OBJECT_SUCCESS] Trajectory endpoint 3D: {target_3d_pos[:2]}")
            self._log(f"[OBJECT_SUCCESS] Trajectory endpoint 2D: ({target_2d_x:.2f}, {target_2d_y:.2f})")

            # Search for reasonable target objects near endpoint
            search_radius = 2.0
            candidate_objects = []

            for obj in self.semantic_map_data:
                item_id = obj.get('item_id', '')
                bbox_m = obj.get('bbox_m', [])
                category = obj.get('category_label', '')

                if len(bbox_m) == 4:
                    try:
                        center_x = (float(bbox_m[0]) + float(bbox_m[2])) / 2.0
                        center_y = (float(bbox_m[1]) + float(bbox_m[3])) / 2.0
                        center_pos = np.array([center_x, center_y])

                        distance_to_target = np.linalg.norm(center_pos - target_2d_pos)

                        if distance_to_target <= search_radius:
                            priority = self._get_object_priority(item_id, category)
                            candidate_objects.append({
                                'item_id': item_id,
                                'category': category,
                                'distance': distance_to_target,
                                'priority': priority,
                                'bbox_m': bbox_m
                            })
                    except (ValueError, TypeError):
                        continue

            info["found_candidates"] = len(candidate_objects)

            if not candidate_objects:
                self._log(f"[OBJECT_SUCCESS] ⚠ No suitable target objects found within {search_radius}m of endpoint")
                info["error"] = f"No target objects found within {search_radius}m of endpoint"

                # Fallback to distance-based evaluation
                agent_2d_x, agent_2d_y = reverse_position_mapping(
                    current_position[0], current_position[1], self.semantic_map_data
                )
                distance_2d = np.linalg.norm(np.array([agent_2d_x, agent_2d_y]) - target_2d_pos)

                success = distance_2d <= expansion_radius
                info["fallback_distance"] = distance_2d
                info["final_success"] = success

                if success:
                    self._log(f"[OBJECT_SUCCESS] Distance-based success (distance: {distance_2d:.3f}m)")
                else:
                    self._log(f"[OBJECT_SUCCESS] Distance-based failure (distance: {distance_2d:.3f}m)")

                return success, info

            # Sort by priority and distance to select best target
            candidate_objects.sort(key=lambda x: (x['priority'], x['distance']))
            best_target = candidate_objects[0]

            info["best_target"] = {
                "item_id": best_target['item_id'],
                "category": best_target['category'],
                "distance": best_target['distance'],
                "priority": best_target['priority']
            }

            self._log(f"[OBJECT_SUCCESS] Selected best target: {best_target['item_id']} ({best_target['category']})")
            self._log(f"[OBJECT_SUCCESS] Distance to endpoint: {best_target['distance']:.3f}m, priority: {best_target['priority']}")

            # Check if agent is within this object's area
            in_object_area = self.is_position_in_object_area(current_position, best_target['item_id'], expansion_radius)
            info["in_object_area"] = in_object_area

            if not in_object_area:
                info["final_success"] = False
                self._log(f"[OBJECT_SUCCESS] Not within inferred target object {best_target['item_id']} area")
                return False, info

            # Check collision-free
            is_collision_free = self.is_collision_free_area(current_position)
            info["collision_free"] = is_collision_free

            if not is_collision_free:
                info["final_success"] = False
                self._log(f"[OBJECT_SUCCESS] Within inferred target {best_target['item_id']} area but collision detected")
                return False, info

            info["final_success"] = True
            self._log(f"[OBJECT_SUCCESS] ✓ Successfully reached inferred target {best_target['item_id']} ({best_target['category']}) collision-free area")
            return True, info

        except Exception as e:
            info["error"] = str(e)
            self._log(f"[OBJECT_SUCCESS] ✗ Smart position evaluation failed: {e}")
            return self._fallback_point_success(current_position, episode), info

    def _get_object_priority(self, item_id: str, category: str) -> int:
        """Get object priority (lower value = higher priority).

        Args:
            item_id: Object item ID
            category: Object category label

        Returns:
            Priority value (1-10)
        """
        item_lower = item_id.lower()
        category_lower = category.lower()

        # Screen and projector - highest priority
        if any(kw in item_lower or kw in category_lower for kw in ['screen', 'projector']):
            return 1

        # Table related - high priority
        if any(kw in item_lower or kw in category_lower for kw in ['table', 'desk']):
            return 2

        # Chair related - medium priority
        if 'chair' in item_lower or 'chair' in category_lower:
            return 3

        # Other furniture - lower priority
        if any(furniture in category_lower for furniture in ['furniture', 'cabinet', 'shelf', 'bookcase']):
            return 4

        # Unusable areas - lowest priority
        if 'unable' in item_lower or 'unable' in category_lower:
            return 10

        # Default priority
        return 5








