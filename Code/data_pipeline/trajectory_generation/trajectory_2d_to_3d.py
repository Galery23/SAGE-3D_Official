#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory 2D to 3D Transformation: Transform 2D navigation trajectories to 3D space coordinates.

This script processes trajectory files generated from 2D semantic maps and transforms them
to 3D space coordinates using coordinate flipping and rotation adjustments based on 2D map boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple

# ==================== Default Configuration ====================

DEFAULT_TRAJ_ROOT = Path("Data/trajectories")
DEFAULT_MAP_ROOT = Path("/path/to/data/2D_Semantic_Map")

# Transformation flags
DEFAULT_FLIP_X = True
DEFAULT_FLIP_Y = True
DEFAULT_NEGATE_XY = True


# ==================== Transformation Functions ====================


def flip_position(
    px: float,
    py: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    flip_x: bool = True,
    flip_y: bool = True,
    negate: bool = False,
) -> Tuple[float, float]:
    """Flip position coordinates using mirroring and optional negation.
    
    Args:
        px, py: Original x, y coordinates
        min_x, max_x, min_y, max_y: Map boundaries
        flip_x: Whether to mirror horizontally
        flip_y: Whether to mirror vertically
        negate: Whether to negate both coordinates after mirroring
    
    Returns:
        Transformed (x, y) coordinates
    """
    if flip_x:
        px = (min_x + max_x) - px
    if flip_y:
        py = (min_y + max_y) - py
    if negate:
        px = -px
        py = -py
    return px, py


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw angle from quaternion rotation around Z-axis.
    
    Args:
        qx, qy, qz, qw: Quaternion components
    
    Returns:
        Yaw angle in radians
    """
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Generate quaternion from yaw angle.
    
    Args:
        yaw: Yaw angle in radians
    
    Returns:
        Quaternion components (qx, qy, qz, qw)
    """
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return 0.0, 0.0, qz, qw


def extract_map_bounds(map_data: list) -> Tuple[float, float, float, float] | None:
    """Extract map boundaries from semantic map data.
    
    Args:
        map_data: List of instance dictionaries from semantic map
    
    Returns:
        (min_x, max_x, min_y, max_y) or None if extraction fails
    """
    try:
        all_y = []
        all_x = []
        for inst in map_data:
            for y, x in inst.get("mask_coords_m", []):
                try:
                    all_y.append(float(y))
                    all_x.append(float(x))
                except (ValueError, TypeError):
                    continue
        
        if not all_y or not all_x:
            return None
        
        min_y, max_y = min(all_y), max(all_y)
        min_x, max_x = min(all_x), max(all_x)
        return min_x, max_x, min_y, max_y
    except Exception:
        return None


def transform_trajectory_points(
    points: list,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    flip_x: bool = True,
    flip_y: bool = True,
    negate_xy: bool = True,
) -> None:
    """Transform trajectory points in-place.
    
    Args:
        points: List of point dictionaries with position and rotation
        min_x, max_x, min_y, max_y: Map boundaries
        flip_x: Whether to mirror horizontally
        flip_y: Whether to mirror vertically
        negate_xy: Whether to negate coordinates after mirroring
    """
    for idx, pt in enumerate(points):
        pos = pt["position"]
        rot = pt["rotation"]
        px_old, py_old, pz = pos
        
        # Transform position
        px_new, py_new = flip_position(
            px_old, py_old, min_x, max_x, min_y, max_y, flip_x=flip_x, flip_y=flip_y, negate=negate_xy
        )
        pt["position"] = [px_new, py_new, pz]
        
        # Transform rotation
        if idx == len(points) - 1:
            # Last point: reset rotation
            pt["rotation"] = [0.0, 0.0, 0.0, 1.0]
        else:
            # Extract yaw, add pi, convert back to quaternion
            yaw = yaw_from_quaternion(*rot)
            yaw_new = yaw + math.pi
            if yaw_new > math.pi:
                yaw_new -= 2 * math.pi
            
            _, _, qz_tmp, qw_tmp = quaternion_from_yaw(yaw_new)
            # Rotate quaternion: put qz to qx and negate
            qx = -qz_tmp
            qy = 0.0
            qz = 0.0
            qw = qw_tmp
            pt["rotation"] = [qx, qy, qz, qw]


def process_scene(
    scene_dir: Path,
    map_root: Path,
    flip_x: bool,
    flip_y: bool,
    negate_xy: bool,
    skip_existing: bool = True,
) -> int:
    """Process a single scene directory.
    
    Args:
        scene_dir: Scene directory path
        map_root: Root directory containing semantic map files
        flip_x: Whether to mirror horizontally
        flip_y: Whether to mirror vertically
        negate_xy: Whether to negate coordinates
        skip_existing: Whether to skip already transformed files
    
    Returns:
        Number of transformed trajectory files
    """
    scene_id = scene_dir.name
    
    # Find trajectory files
    traj_files = [
        f for f in scene_dir.iterdir()
        if f.is_file()
        and f.name.startswith("trajectories_")
        and f.name.endswith(".json")
        and "_trans" not in f.name
    ]
    
    if not traj_files:
        print(f"[SKIP] {scene_id}: No trajectory files found (excluding _trans files)")
        return 0
    
    # Find corresponding 2D map file
    map_candidates = [
        f for f in map_root.iterdir()
        if f.is_file() and f.suffix == ".json" and f"_{scene_id}_" in f.name
    ]
    
    if not map_candidates:
        print(f"[WARN] {scene_id}: No corresponding 2D map file found")
        return 0
    
    map_file = map_candidates[0]
    
    # Load map and extract bounds
    try:
        with map_file.open("r", encoding="utf-8") as f:
            map_data = json.load(f)
        bounds = extract_map_bounds(map_data)
        if bounds is None:
            print(f"[ERROR] {scene_id}: Failed to extract map bounds from {map_file.name}")
            return 0
        min_x, max_x, min_y, max_y = bounds
        print(f"[{scene_id}] Map bounds X:[{min_x:.2f}, {max_x:.2f}], Y:[{min_y:.2f}, {max_y:.2f}]")
    except Exception as e:
        print(f"[ERROR] {scene_id}: Failed to load map file {map_file.name}: {e}")
        return 0
    
    # Process each trajectory file
    transformed_count = 0
    for traj_file in traj_files:
        trans_file = traj_file.with_name(traj_file.name.replace(".json", "_trans.json"))
        
        if skip_existing and trans_file.exists():
            print(f"[SKIP] {traj_file.name}: Transformed file already exists")
            continue
        
        try:
            # Load trajectory data
            with traj_file.open("r", encoding="utf-8") as f:
                traj_data = json.load(f)
            
            # Transform trajectory points
            for scene in traj_data.get("scenes", []):
                for sample in scene.get("samples", []):
                    points = sample.get("points", [])
                    if points:
                        transform_trajectory_points(
                            points, min_x, max_x, min_y, max_y, flip_x, flip_y, negate_xy
                        )
            
            # Save transformed trajectory
            with trans_file.open("w", encoding="utf-8") as f:
                json.dump(traj_data, f, indent=2, ensure_ascii=False)
            
            print(f"[SAVED] {trans_file.name}")
            transformed_count += 1
            
        except Exception as e:
            print(f"[ERROR] {scene_id}: Failed to transform {traj_file.name}: {e}")
            continue
    
    return transformed_count


# ==================== CLI Interface ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transform 2D navigation trajectories to 3D space coordinates."
    )
    
    parser.add_argument(
        "--traj-root",
        type=Path,
        default=DEFAULT_TRAJ_ROOT,
        help="Root directory containing trajectory scene folders",
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=DEFAULT_MAP_ROOT,
        help="Root directory containing 2D semantic map JSON files",
    )
    parser.add_argument(
        "--flip-x",
        action="store_true",
        default=DEFAULT_FLIP_X,
        help="Mirror horizontally (default: True)",
    )
    parser.add_argument(
        "--no-flip-x",
        dest="flip_x",
        action="store_false",
        help="Disable horizontal mirroring",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        default=DEFAULT_FLIP_Y,
        help="Mirror vertically (default: True)",
    )
    parser.add_argument(
        "--no-flip-y",
        dest="flip_y",
        action="store_false",
        help="Disable vertical mirroring",
    )
    parser.add_argument(
        "--negate-xy",
        action="store_true",
        default=DEFAULT_NEGATE_XY,
        help="Negate x and y coordinates after mirroring (default: True)",
    )
    parser.add_argument(
        "--no-negate-xy",
        dest="negate_xy",
        action="store_false",
        help="Disable coordinate negation",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only process specified scene folders",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transform even if _trans files already exist",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Validate paths
    if not args.traj_root.exists():
        print(f"[ERROR] Trajectory root does not exist: {args.traj_root}")
        return
    
    if not args.map_root.exists():
        print(f"[ERROR] Map root does not exist: {args.map_root}")
        return
    
    # Collect scene directories
    scene_dirs = sorted([d for d in args.traj_root.iterdir() if d.is_dir()])
    if args.only:
        scene_dirs = [d for d in scene_dirs if d.name in args.only]
    
    print(f"[INFO] Found {len(scene_dirs)} scene directories")
    print(f"[INFO] Transformation flags: flip_x={args.flip_x}, flip_y={args.flip_y}, negate_xy={args.negate_xy}")
    
    # Process each scene
    total_transformed = 0
    for scene_dir in scene_dirs:
        count = process_scene(
            scene_dir, args.map_root, args.flip_x, args.flip_y, args.negate_xy, skip_existing=not args.force
        )
        total_transformed += count
    
    print(f"\n[COMPLETE] Total transformed: {total_transformed} trajectory files")


if __name__ == "__main__":
    main()









