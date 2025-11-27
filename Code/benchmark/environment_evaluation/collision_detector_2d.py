#!/usr/bin/env python3
"""
2D Semantic Map Collision Detector for SAGE-3D Benchmark.

Performs collision detection based on 2D semantic maps,
avoiding complex 3D physics system issues.
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import distance_transform_edt


def _should_print_debug() -> bool:
    """Check if debug messages should be printed."""
    return not os.environ.get('SILENT_LOGGING_MODE', False)


def _debug_print(msg: str) -> None:
    """Conditionally print debug messages."""
    if _should_print_debug():
        print(msg)


class SemanticMap2DCollisionDetector:
    """2D semantic map-based collision detector."""

    def __init__(self, map_json_path: str, robot_radius_m: float = 0.1, scale: float = 0.05):
        """Initialize collision detector.

        Args:
            map_json_path: Path to 2D semantic map JSON file
            robot_radius_m: Robot radius in meters
            scale: Grid resolution (meters/pixel)
        """
        self.map_json_path = map_json_path
        self.robot_radius_m = robot_radius_m
        self.scale = scale

        # Map boundaries
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

        # Obstacle map
        self.obstacle_map = None
        self.map_height = 0
        self.map_width = 0

        # Load map data
        self._load_map_data()

    def _load_map_data(self) -> None:
        """Load 2D semantic map data and build obstacle map."""
        if not os.path.exists(self.map_json_path):
            _debug_print(f"[COLLISION_2D] Warning: Map file does not exist: {self.map_json_path}")
            return

        try:
            with open(self.map_json_path, 'r') as f:
                map_data = json.load(f)

            _debug_print(f"[COLLISION_2D] Loaded 2D semantic map: {self.map_json_path}")
            _debug_print(f"[COLLISION_2D] Map contains {len(map_data)} instances")

            # Calculate map boundaries
            all_y = [float(y) for inst in map_data for y, x in inst.get('mask_coords_m', [])]
            all_x = [float(x) for inst in map_data for y, x in inst.get('mask_coords_m', [])]

            if not all_x or not all_y:
                _debug_print(f"[COLLISION_2D] Error: No valid coordinates in map data")
                return

            self.min_y, self.max_y = min(all_y), max(all_y)
            self.min_x, self.max_x = min(all_x), max(all_x)

            _debug_print(f"[COLLISION_2D] Map bounds: X=[{self.min_x:.2f}, {self.max_x:.2f}], Y=[{self.min_y:.2f}, {self.max_y:.2f}]")

            # Calculate grid map dimensions
            self.map_height = int(np.ceil((self.max_y - self.min_y) / self.scale)) + 1
            self.map_width = int(np.ceil((self.max_x - self.min_x) / self.scale)) + 1

            _debug_print(f"[COLLISION_2D] Grid map size: {self.map_height} x {self.map_width} (scale={self.scale}m/pixel)")

            # Build obstacle map
            self._build_obstacle_map(map_data)

        except Exception as e:
            _debug_print(f"[COLLISION_2D] Error: Failed to load map data: {e}")
            import traceback
            traceback.print_exc()

    def _build_obstacle_map(self, map_data: List[Dict]) -> None:
        """Build obstacle map from semantic data."""
        # Initialize obstacle map
        obstacle_map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        # Track obstacle categories
        obstacle_categories = set()

        for inst in map_data:
            category_label = str(inst.get('category_label', '')).lower()

            # Only treat 'unable area' and 'wall' as obstacles (consistent with A* algorithm)
            if category_label in ['unable area', 'wall']:
                obstacle_categories.add(category_label)

                # Convert obstacle coordinates to pixel coordinates and mark
                for y, x in inst.get('mask_coords_m', []):
                    py, px = self._world_to_pixel(x, y)
                    if 0 <= py < self.map_height and 0 <= px < self.map_width:
                        obstacle_map[py, px] = 1

        _debug_print(f"[COLLISION_2D] Obstacle categories: {obstacle_categories}")
        _debug_print(f"[COLLISION_2D] Original obstacle pixels: {np.sum(obstacle_map)}")

        # Use Euclidean distance transform for robot radius inflation
        if self.robot_radius_m > 0:
            # Calculate distance transform (in meters)
            dist_m = distance_transform_edt(obstacle_map == 0, sampling=self.scale)

            # Inflate: areas within robot radius are also obstacles
            inflated_obstacle = (dist_m <= self.robot_radius_m).astype(np.uint8)

            obstacle_map = inflated_obstacle

            _debug_print(f"[COLLISION_2D] Robot radius inflation: {self.robot_radius_m}m")
            _debug_print(f"[COLLISION_2D] Inflated obstacle pixels: {np.sum(obstacle_map)}")

        self.obstacle_map = obstacle_map

    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        px = int(round((float(x) - self.min_x) / self.scale))
        py = int(round((float(y) - self.min_y) / self.scale))
        return py, px

    def _pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        x = self.min_x + (px + 0.5) * self.scale
        y = self.min_y + (py + 0.5) * self.scale
        return x, y

    def forward_position_mapping(
        self,
        px_3d: float,
        py_3d: float,
        flip_x: bool = True,
        flip_y: bool = True,
        negate_xy: bool = True
    ) -> Tuple[float, float]:
        """Forward position mapping: convert 3D trajectory coordinates to 2D map coordinates.

        This is the inverse of reverse_position_mapping.

        Original trajectory transformation process (from Trajectory_trans.py):
        1. First mirror flip: flip_x, flip_y
        2. Then negate: negate_xy

        Forward mapping (3D->2D) should be the reverse:
        1. First negate (if originally negated)
        2. Then inverse mirror flip (flip again)

        Args:
            px_3d, py_3d: Coordinates from 3D trajectory
            flip_x, flip_y, negate_xy: Mapping parameters (should match trajectory transformation)

        Returns:
            (px_2d, py_2d): Converted 2D map coordinates
        """
        if self.min_x is None or self.max_x is None or self.min_y is None or self.max_y is None:
            return px_3d, py_3d

        px, py = px_3d, py_3d

        # Step 1: If originally negated, first negate back
        if negate_xy:
            px = -px
            py = -py

        # Step 2: If originally flipped, do inverse operation (flip again)
        if flip_x:
            px = (self.min_x + self.max_x) - px
        if flip_y:
            py = (self.min_y + self.max_y) - py

        return px, py

    def check_collision_3d(self, pos_3d: np.ndarray) -> bool:
        """Check if 3D position has collision.

        Args:
            pos_3d: 3D position [x, y, z]

        Returns:
            True if collision, False if no collision
        """
        if self.obstacle_map is None:
            _debug_print(f"[COLLISION_2D] Warning: Obstacle map not initialized, skipping collision detection")
            return False

        try:
            # Map 3D position to 2D map coordinates
            px_2d, py_2d = self.forward_position_mapping(pos_3d[0], pos_3d[1])

            # Convert to pixel coordinates
            py, px = self._world_to_pixel(px_2d, py_2d)

            # Check if within map bounds
            if not (0 <= py < self.map_height and 0 <= px < self.map_width):
                # Out of bounds handling strategy:
                # 1. If slightly out of bounds (possibly numerical error), allow tolerance
                margin = 2  # 2 pixel tolerance
                if (-margin <= py < self.map_height + margin and
                    -margin <= px < self.map_width + margin):
                    # Within tolerance, constrain to boundary
                    py = max(0, min(self.map_height - 1, py))
                    px = max(0, min(self.map_width - 1, px))
                    _debug_print(f"[COLLISION_2D] Position slightly out of bounds, constrained: pixel({px}, {py})")
                else:
                    # Severely out of bounds, treat as collision
                    _debug_print(f"[COLLISION_2D] Position severely out of map bounds: 3D{pos_3d[:2]} -> 2D({px_2d:.3f}, {py_2d:.3f}) -> pixel({px}, {py})")
                    return True

            # Check obstacle map
            is_collision = self.obstacle_map[py, px] == 1

            if is_collision:
                _debug_print(f"[COLLISION_2D] Collision detected: 3D pos {pos_3d[:2]} -> 2D pos ({px_2d:.3f}, {py_2d:.3f}) -> pixel ({px}, {py}) = obstacle")

            return is_collision

        except Exception as e:
            _debug_print(f"[COLLISION_2D] Collision detection error: {e}")
            return False

    def check_path_collision_3d(self, start_pos_3d: np.ndarray, end_pos_3d: np.ndarray, num_samples: int = 10) -> bool:
        """Check if 3D path has collision.

        Args:
            start_pos_3d: Start 3D position
            end_pos_3d: End 3D position
            num_samples: Number of sample points along path

        Returns:
            True if path has collision, False if collision-free
        """
        if num_samples <= 1:
            return self.check_collision_3d(end_pos_3d)

        # Sample multiple points along path
        for i in range(1, num_samples + 1):
            t = i / float(num_samples)
            sample_pos = start_pos_3d * (1 - t) + end_pos_3d * t

            if self.check_collision_3d(sample_pos):
                return True

        return False

    def check_collision_at_position(self, x: float, y: float) -> bool:
        """Check collision at 2D position (for object_based_success compatibility).

        Args:
            x, y: 2D position coordinates

        Returns:
            True if collision, False if no collision
        """
        return self.check_collision_3d(np.array([x, y, 0.0]))

    def get_collision_info(self) -> Dict:
        """Get collision detector information."""
        return {
            "map_path": self.map_json_path,
            "robot_radius_m": self.robot_radius_m,
            "scale": self.scale,
            "map_bounds": {
                "x": [self.min_x, self.max_x] if self.min_x is not None else None,
                "y": [self.min_y, self.max_y] if self.min_y is not None else None
            },
            "map_size": [self.map_height, self.map_width],
            "obstacle_pixels": int(np.sum(self.obstacle_map)) if self.obstacle_map is not None else 0,
            "total_pixels": self.map_height * self.map_width,
            "obstacle_ratio": float(np.sum(self.obstacle_map)) / (self.map_height * self.map_width) if self.obstacle_map is not None else 0.0
        }


def test_collision_detector():
    """Test collision detector."""
    test_map_path = "/path/to/your/semantic_map.json"  # Replace with actual path

    if not os.path.exists(test_map_path):
        _debug_print(f"[TEST] Skipping test: Map file does not exist {test_map_path}")
        return

    detector = SemanticMap2DCollisionDetector(test_map_path, robot_radius_m=0.1)

    # Test some positions
    test_positions = [
        np.array([0.0, 0.0, 0.5]),
        np.array([1.0, 1.0, 0.5]),
        np.array([-1.0, -1.0, 0.5]),
    ]

    for pos in test_positions:
        collision = detector.check_collision_3d(pos)
        _debug_print(f"[TEST] Position {pos[:2]} collision detection: {'collision' if collision else 'no collision'}")

    # Print detector info
    info = detector.get_collision_info()
    _debug_print(f"[TEST] Collision detector info: {info}")


if __name__ == "__main__":
    test_collision_detector()








