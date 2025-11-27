#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Action GroundTruth: Generate action sequences for NaVILA training.

This script processes trajectory data and generates action sequences with sampled points
for training navigation models.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ==================== Default Configuration ====================

DEFAULT_INPUT_DIR = Path("Data/training_data/trajectories/train")
DEFAULT_OUTPUT_DIR = Path("Data/training_data/actions")
DEFAULT_CONFIG_PRESET = "vlnce"
DEFAULT_MAX_WORKERS = None
DEFAULT_LIMIT = None


# ==================== Configuration Presets ====================


def get_preset_config(preset: str = "vlnce") -> Dict[str, Any]:
    """Get preset configuration.

    Args:
        preset: Configuration preset type
            - "vlnce": VLN-CE/Habitat standard (0.25m, 15°)
            - "navila_small": NaVILA small step (0.25m, 10°)
            - "navila_large": NaVILA large step (0.75m, 15°)
            - "custom_small": Custom small step (0.50m, 30°)

    Returns:
        Configuration dictionary
    """
    presets = {
        "vlnce": {
            "move_distance_per_action": 0.25,
            "turn_angle_per_action": 15,
            "max_actions_per_trajectory": 50,
        },
        "navila_small": {
            "move_distance_per_action": 0.35,
            "turn_angle_per_action": 30,
            "max_actions_per_trajectory": 50,
        },
        "navila_large": {
            "move_distance_per_action": 0.75,
            "turn_angle_per_action": 15,
            "max_actions_per_trajectory": 30,
        },
        "custom_small": {
            "move_distance_per_action": 0.50,
            "turn_angle_per_action": 30,
            "max_actions_per_trajectory": 60,
        },
    }

    base_config = {
        "use_smart_sampling": True,
        "straight_sample_interval": 5,
        "turn_sample_interval": 1,
        "turn_detection_threshold": 0.1,
        "min_distance_threshold": 0.05,
        "smooth_window": 2,
    }

    if preset in presets:
        base_config.update(presets[preset])
        return base_config
    else:
        raise ValueError(f"Unknown preset: {preset}. Available presets: {list(presets.keys())}")


# ==================== Batch Action Generator ====================


class BatchActionGenerator:
    """Batch action groundtruth generator for NaVILA training."""

    def __init__(self, input_dir: Path, output_dir: Path, config: Dict[str, Any] | None = None):
        """Initialize action generator.

        Args:
            input_dir: Input directory containing scene folders with trajectory files
            output_dir: Output directory for generated action groundtruth files
            config: Action generation configuration dictionary
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Action generation config - use VLN-CE standard preset
        self.config = config or get_preset_config(DEFAULT_CONFIG_PRESET)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Action generator initialized")
        print(f"[INFO] Input directory: {self.input_dir}")
        print(f"[INFO] Output directory: {self.output_dir}")
        print(f"[INFO] Action config: MOVE_FORWARD={self.config['move_distance_per_action']}m, "
              f"TURN_LEFT/RIGHT={self.config['turn_angle_per_action']}°")

    def yaw_from_quaternion(self, quaternion: List[float]) -> float:
        """Extract yaw angle from quaternion.

        In this environment, left/right turns are based on x-axis rotation
        (original z-axis rotation mapped to x-axis).

        Args:
            quaternion: Quaternion components [qx, qy, qz, qw]

        Returns:
            Yaw angle in radians
        """
        qx, qy, qz, qw = quaternion
        # According to mapping: qx = -qz_original, where qz_original = sin(yaw/2)
        # So qx = -sin(yaw/2), qw = cos(yaw/2)
        yaw = 2.0 * math.atan2(-qx, qw)
        return yaw

    def quaternion_to_euler(self, quaternion: List[float]) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles, mainly extracting yaw.

        Args:
            quaternion: Quaternion components

        Returns:
            (roll, pitch, yaw) tuple in radians
        """
        yaw = self.yaw_from_quaternion(quaternion)
        return 0.0, 0.0, yaw  # Only care about yaw

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π].

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def detect_trajectory_features(self, points: List[Dict]) -> List[Dict]:
        """Detect trajectory features: straight segments, turning segments.

        Args:
            points: List of trajectory points

        Returns:
            Points with feature annotations
        """
        if len(points) < 3:
            return points

        featured_points = []

        for i in range(len(points)):
            point = points[i].copy()

            if i == 0:
                point["feature"] = "start"
            elif i == len(points) - 1:
                point["feature"] = "end"
            else:
                # Analyze angle changes
                if i >= 1 and i < len(points) - 1:
                    prev_yaw = self.quaternion_to_euler(points[i - 1]["rotation"])[2]
                    curr_yaw = self.quaternion_to_euler(points[i]["rotation"])[2]
                    next_yaw = self.quaternion_to_euler(points[i + 1]["rotation"])[2]

                    angle_change = abs(self.normalize_angle(next_yaw - prev_yaw))

                    if angle_change > self.config["turn_detection_threshold"]:
                        point["feature"] = "turning"
                    else:
                        point["feature"] = "straight"
                else:
                    point["feature"] = "straight"

            featured_points.append(point)

        return featured_points

    def smart_sample_trajectory(self, points: List[Dict]) -> List[Dict]:
        """Intelligent trajectory point sampling.

        Args:
            points: List of trajectory points

        Returns:
            Sampled points
        """
        if not self.config["use_smart_sampling"] or len(points) < 3:
            return points

        # Detect trajectory features
        featured_points = self.detect_trajectory_features(points)

        # Intelligent sampling
        sampled_points = [featured_points[0]]  # Always include start point

        i = 1
        while i < len(featured_points) - 1:
            current_point = featured_points[i]
            feature = current_point["feature"]

            if feature == "turning":
                # Turning segment: dense sampling
                interval = self.config["turn_sample_interval"]
            elif feature == "straight":
                # Straight segment: sparse sampling
                interval = self.config["straight_sample_interval"]
            else:
                interval = 2

            if i % interval == 0:
                sampled_points.append(current_point)

            i += 1

        # Always include end point
        sampled_points.append(featured_points[-1])

        return sampled_points

    def _generate_actions_from_sampled_points(self, sampled_points: List[Dict]) -> List[str]:
        """Generate action sequence from sampled points.

        Args:
            sampled_points: List of sampled trajectory points

        Returns:
            List of action strings
        """
        actions = []
        accumulated_distance = 0.0

        for i in range(len(sampled_points) - 1):
            current_point = sampled_points[i]
            next_point = sampled_points[i + 1]

            # Calculate position and angle changes
            pos1 = current_point["position"]
            pos2 = next_point["position"]
            distance = math.sqrt(
                (pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2 + (pos2[2] - pos1[2]) ** 2
            )

            _, _, yaw1 = self.quaternion_to_euler(current_point["rotation"])
            _, _, yaw2 = self.quaternion_to_euler(next_point["rotation"])
            angle_change = self.normalize_angle(yaw2 - yaw1)

            # Handle turning first (if significant angle change)
            turn_threshold = math.radians(15)
            if abs(angle_change) > turn_threshold:
                # Calculate number of turns needed
                angle_per_turn = math.radians(self.config["turn_angle_per_action"])
                num_turns = max(1, int(abs(angle_change) / angle_per_turn))
                num_turns = min(num_turns, 4)  # Max 4 turns

                for _ in range(num_turns):
                    if angle_change > 0:
                        actions.append("TURN_LEFT")
                    else:
                        actions.append("TURN_RIGHT")

            # Handle movement (accumulate distance)
            accumulated_distance += distance

            # Generate move actions when accumulated distance is sufficient
            move_threshold = self.config["move_distance_per_action"]
            moves_to_add = int(accumulated_distance / move_threshold)
            moves_to_add = min(moves_to_add, 5)  # Max 5 moves at once

            for _ in range(moves_to_add):
                actions.append("MOVE_FORWARD")
                accumulated_distance -= move_threshold

            # Limit total action count
            if len(actions) >= min(50, self.config["max_actions_per_trajectory"] * 2):
                break

        # Handle remaining movement distance
        move_threshold = self.config["move_distance_per_action"]
        if accumulated_distance > move_threshold * 0.5:
            actions.append("MOVE_FORWARD")

        # Add stop action
        actions.append("STOP")

        return actions

    def _slerp_quaternions(self, q1: List[float], q2: List[float], t: float) -> List[float]:
        """Spherical linear interpolation of quaternions.

        Args:
            q1: First quaternion
            q2: Second quaternion
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated quaternion
        """
        # Ensure quaternions are unit quaternions
        def normalize_quat(q):
            norm = math.sqrt(sum(x * x for x in q))
            return [x / norm for x in q] if norm > 0 else q

        q1 = normalize_quat(q1)
        q2 = normalize_quat(q2)

        # Calculate dot product
        dot = sum(a * b for a, b in zip(q1, q2))

        # If dot product is negative, flip one quaternion to take shorter path
        if dot < 0:
            q2 = [-x for x in q2]
            dot = -dot

        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = [a + t * (b - a) for a, b in zip(q1, q2)]
            return normalize_quat(result)

        # Spherical linear interpolation
        theta_0 = math.acos(abs(dot))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * t
        sin_theta = math.sin(theta)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return [s0 * a + s1 * b for a, b in zip(q1, q2)]

    def _interpolate_points_for_actions(
        self, original_sampled_points: List[Dict], actions: List[str]
    ) -> List[Dict]:
        """Interpolate points for actions to ensure one-to-one correspondence.

        Args:
            original_sampled_points: Original sampled points
            actions: Generated action sequence

        Returns:
            Interpolated points matching action count
        """
        if len(actions) == len(original_sampled_points):
            return original_sampled_points

        interpolated_points = []
        action_idx = 0

        # Iterate through segments between original sampled points
        for i in range(len(original_sampled_points) - 1):
            current_point = original_sampled_points[i]
            next_point = original_sampled_points[i + 1]

            # Add current point
            interpolated_points.append(copy.deepcopy(current_point))
            action_idx += 1

            # Calculate how many actions are needed between these two points
            pos1 = current_point["position"]
            pos2 = next_point["position"]
            distance = math.sqrt(
                (pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2 + (pos2[2] - pos1[2]) ** 2
            )

            _, _, yaw1 = self.quaternion_to_euler(current_point["rotation"])
            _, _, yaw2 = self.quaternion_to_euler(next_point["rotation"])
            angle_change = self.normalize_angle(yaw2 - yaw1)

            # Count actions needed for this segment
            turn_threshold = math.radians(15)
            num_turns = 0
            num_moves = 0

            if abs(angle_change) > turn_threshold:
                angle_per_turn = math.radians(self.config["turn_angle_per_action"])
                num_turns = max(1, int(abs(angle_change) / angle_per_turn))
                num_turns = min(num_turns, 4)

            move_threshold = self.config["move_distance_per_action"]
            num_moves = int(distance / move_threshold)
            num_moves = min(num_moves, 5)

            total_segment_actions = num_turns + num_moves

            # If multiple actions needed, interpolate intermediate points
            if total_segment_actions > 1:
                for j in range(1, total_segment_actions):
                    # Calculate interpolation ratio
                    ratio = j / total_segment_actions

                    # Position interpolation
                    interp_pos = [
                        pos1[0] + (pos2[0] - pos1[0]) * ratio,
                        pos1[1] + (pos2[1] - pos1[1]) * ratio,
                        pos1[2] + (pos2[2] - pos1[2]) * ratio,
                    ]

                    # Angle interpolation (spherical quaternion interpolation)
                    interp_rotation = self._slerp_quaternions(
                        current_point["rotation"], next_point["rotation"], ratio
                    )

                    # Create interpolated point
                    interp_point = copy.deepcopy(current_point)
                    interp_point["position"] = interp_pos
                    interp_point["rotation"] = interp_rotation

                    # Generate interpolated point_id if original has point field
                    if "point" in current_point:
                        interp_point["point"] = f"{current_point['point']}_interp_{j}"

                    interpolated_points.append(interp_point)
                    action_idx += 1

            # Prevent exceeding action count
            if action_idx >= len(actions) - 1:
                break

        # Add last point
        interpolated_points.append(copy.deepcopy(original_sampled_points[-1]))

        # Adjust to exactly match action count
        while len(interpolated_points) < len(actions):
            interpolated_points.append(copy.deepcopy(interpolated_points[-1]))

        while len(interpolated_points) > len(actions):
            interpolated_points.pop(-2)

        return interpolated_points[: len(actions)]

    def generate_actions_from_trajectory_with_sampling(
        self, points: List[Dict]
    ) -> Tuple[List[str], List[Dict]]:
        """Generate action sequence and sampled points from trajectory.

        Args:
            points: List of trajectory points

        Returns:
            (actions, sampled_points) tuple
        """
        if len(points) < 2:
            return ["STOP"], points

        # Intelligent sampling: sparse for straight, dense for turning
        if self.config["use_smart_sampling"]:
            original_sampled_points = self.smart_sample_trajectory(points)
        else:
            # Simple uniform sampling as fallback
            num_samples = min(max(15, len(points) // 5), len(points))
            if num_samples >= len(points):
                original_sampled_points = points
            else:
                indices = [int(i * (len(points) - 1) / (num_samples - 1)) for i in range(num_samples)]
                original_sampled_points = [points[i] for i in indices]

        # Generate action sequence
        actions = self._generate_actions_from_sampled_points(original_sampled_points)

        # Interpolate points to match action count
        final_sampled_points = self._interpolate_points_for_actions(original_sampled_points, actions)

        # Verify count match
        assert len(actions) == len(
            final_sampled_points
        ), f"Action count ({len(actions)}) must equal sampled point count ({len(final_sampled_points)})"

        return actions, final_sampled_points

    def find_all_trajectory_files(self) -> List[Dict[str, str]]:
        """Scan all trajectory files in input directory.

        Returns:
            List of file info dictionaries
        """
        all_files = []

        if not self.input_dir.exists():
            print(f"[ERROR] Input directory does not exist: {self.input_dir}")
            return all_files

        print(f"\n[INFO] Scanning input directory: {self.input_dir}")

        # Get all scene directories
        scene_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        print(f"[INFO] Found {len(scene_dirs)} scene directories")

        for scene_dir in sorted(scene_dirs):
            scene_id = scene_dir.name

            # Find trajectory JSON files (support multiple naming patterns)
            traj_files = list(scene_dir.glob("train_trajectories_*.json"))
            if not traj_files:
                traj_files = list(scene_dir.glob("val_trajectories_*.json"))
            if not traj_files:
                traj_files = list(scene_dir.glob("test_trajectories_*.json"))
            if not traj_files:
                traj_files = list(scene_dir.glob("trajectories_overall_*.json"))
            if not traj_files:
                traj_files = list(scene_dir.glob("*.json"))

            if not traj_files:
                print(f"[WARN] Scene {scene_id}: No trajectory files found")
                continue

            for traj_file in traj_files:
                file_info = {
                    "scene_id": scene_id,
                    "traj_file": str(traj_file),
                    "output_file": str(self.output_dir / scene_id / "action_groundtruth.json"),
                }
                all_files.append(file_info)

            print(f"[INFO] Scene {scene_id}: {len(traj_files)} trajectory files")

        print(f"\n[INFO] Total found {len(all_files)} files to process")
        return all_files

    def process_single_file(self, file_info: Dict[str, str]) -> bool:
        """Process a single trajectory file.

        Args:
            file_info: File information dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            scene_id = file_info["scene_id"]
            print(f"\n[PROCESS] Scene: {scene_id}")
            print(f"  Trajectory file: {Path(file_info['traj_file']).name}")

            # Create output directory
            output_file = Path(file_info["output_file"])
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Load trajectory data
            with Path(file_info["traj_file"]).open("r", encoding="utf-8") as f:
                traj_data = json.load(f)

            scene_info = traj_data["scenes"][0]
            scene_id = scene_info["scene_id"]

            all_groundtruth = []

            for sample in scene_info["samples"]:
                trajectory_id = sample["trajectory_id"]
                points = sample["points"]
                instructions = sample["instructions"]

                # Generate action sequence and sampled points for this trajectory
                actions, sampled_points = self.generate_actions_from_trajectory_with_sampling(points)

                # Create groundtruth for each instruction
                for inst_idx, instruction in enumerate(instructions):
                    groundtruth_item = {
                        "scene_id": scene_id,
                        "trajectory_id": trajectory_id,
                        "instruction_index": inst_idx,
                        "instruction": instruction,
                        "action_sequence": actions,
                        "sampled_points": [
                            {
                                "point_id": point.get("point", str(i)),
                                "position": point["position"],
                                "rotation": point["rotation"],
                            }
                            for i, point in enumerate(sampled_points)
                        ],
                        "trajectory_stats": {
                            "total_points": len(points),
                            "final_sampled_points": len(sampled_points),
                            "total_distance": self._calculate_total_distance(points),
                            "total_actions": len(actions),
                            "action_breakdown": self._analyze_actions(actions),
                            "data_consistency": {
                                "actions_match_points": len(actions) == len(sampled_points),
                                "interpolation_applied": True,
                            },
                        },
                    }

                    all_groundtruth.append(groundtruth_item)

            # Prepare result data
            result = {
                "scene_id": scene_id,
                "scene_name": scene_info.get("scene_name", scene_id),
                "total_trajectories": len(scene_info["samples"]),
                "total_instructions": sum(len(s["instructions"]) for s in scene_info["samples"]),
                "groundtruth_data": all_groundtruth,
                "generation_config": self.config,
                "source_file": file_info["traj_file"],
            }

            # Verify data consistency
            consistency_issues = []
            for item in all_groundtruth:
                action_count = len(item["action_sequence"])
                point_count = len(item["sampled_points"])
                if action_count != point_count:
                    consistency_issues.append(
                        {
                            "trajectory_id": item["trajectory_id"],
                            "instruction_index": item["instruction_index"],
                            "actions": action_count,
                            "points": point_count,
                        }
                    )

            if consistency_issues:
                print(f"  [WARN] Data consistency issues found:")
                for issue in consistency_issues:
                    print(
                        f"    Trajectory {issue['trajectory_id']}, instruction {issue['instruction_index']}: "
                        f"{issue['actions']} actions vs {issue['points']} points"
                    )
                return False

            # Save result
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"  [SUCCESS] Generated {len(all_groundtruth)} training samples")
            print(f"  Action stats: {self._get_action_summary(all_groundtruth)}")

            return True

        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _calculate_total_distance(self, points: List[Dict]) -> float:
        """Calculate total trajectory distance.

        Args:
            points: List of trajectory points

        Returns:
            Total distance in meters
        """
        total_distance = 0.0
        for i in range(len(points) - 1):
            pos1 = np.array(points[i]["position"])
            pos2 = np.array(points[i + 1]["position"])
            total_distance += np.linalg.norm(pos2 - pos1)
        return total_distance

    def _analyze_actions(self, actions: List[str]) -> Dict[str, int]:
        """Analyze action sequence composition.

        Args:
            actions: List of action strings

        Returns:
            Dictionary of action counts
        """
        action_counts = Counter(actions)
        return dict(action_counts)

    def _get_action_summary(self, groundtruth_data: List[Dict]) -> str:
        """Get action statistics summary.

        Args:
            groundtruth_data: List of groundtruth items

        Returns:
            Summary string
        """
        all_actions = []
        for item in groundtruth_data:
            all_actions.extend(item["action_sequence"])

        action_counts = Counter(all_actions)

        summary_parts = []
        for action in ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]:
            if action in action_counts:
                summary_parts.append(f"{action}:{action_counts[action]}")

        return " ".join(summary_parts)

    def run_batch_processing(self, max_workers: int | None = None, limit: int | None = None) -> None:
        """Run batch processing.

        Args:
            max_workers: Maximum number of worker processes
            limit: Limit number of files to process (for testing)
        """
        # Scan all files
        all_files = self.find_all_trajectory_files()

        if not all_files:
            print("[ERROR] No files found to process")
            return

        # Apply limit
        if limit is not None and limit > 0:
            all_files = all_files[:limit]
            print(f"[INFO] Limiting processing to {limit} files")

        print(f"\n[INFO] Starting batch processing of {len(all_files)} files...")

        # Determine worker count
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)

        print(f"[INFO] Using {max_workers} worker processes")

        success_count = 0
        failed_count = 0

        # Use process pool for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_info): file_info for file_info in all_files
            }

            # Collect results
            for i, future in enumerate(future_to_file, 1):
                file_info = future_to_file[future]
                print(f"\n[PROGRESS] {i}/{len(all_files)} - Scene: {file_info['scene_id']}")

                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"  [ERROR] Execution exception: {e}")
                    failed_count += 1

        # Generate batch summary
        self._generate_batch_summary(all_files, success_count, failed_count)

        print(f"\n[COMPLETE] Batch processing finished!")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(all_files)}")

    def _generate_batch_summary(self, all_files: List[Dict], success_count: int, failed_count: int) -> None:
        """Generate batch processing summary.

        Args:
            all_files: List of all file info dictionaries
            success_count: Number of successful files
            failed_count: Number of failed files
        """
        summary = {
            "batch_processing_summary": {
                "total_files": len(all_files),
                "success_count": success_count,
                "failed_count": failed_count,
                "success_rate": success_count / len(all_files) if all_files else 0,
            },
            "generation_config": self.config,
            "processed_summary": {},
            "file_details": [],
            "overall_action_statistics": {},
        }

        # Overall statistics
        summary["processed_summary"] = {
            "total_files": len(all_files),
            "unique_scenes": len(set(f["scene_id"] for f in all_files)),
        }

        # Statistics for all actions
        overall_actions: Dict[str, int] = {}
        total_samples = 0

        # File details and action statistics
        for file_info in all_files:
            output_file = Path(file_info["output_file"])
            file_detail = {
                "scene_id": file_info["scene_id"],
                "trajectory_file": file_info["traj_file"],
                "output_file": str(output_file),
                "groundtruth_exists": output_file.exists(),
            }

            if output_file.exists():
                try:
                    with output_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)

                    file_detail["total_instructions"] = data.get("total_instructions", 0)
                    file_detail["total_trajectories"] = data.get("total_trajectories", 0)

                    # Statistics for actions
                    for item in data.get("groundtruth_data", []):
                        total_samples += 1
                        for action in item["action_sequence"]:
                            overall_actions[action] = overall_actions.get(action, 0) + 1

                except Exception as e:
                    print(f"  [WARN] Failed to read {output_file}: {e}")
                    file_detail["total_instructions"] = 0
                    file_detail["total_trajectories"] = 0

            summary["file_details"].append(file_detail)

        summary["overall_action_statistics"] = {
            "total_samples": total_samples,
            "action_distribution": overall_actions,
            "average_actions_per_sample": sum(overall_actions.values()) / total_samples
            if total_samples > 0
            else 0,
        }

        # Save summary
        summary_file = self.output_dir / "action_generation_summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Processing summary saved: {summary_file}")

        # Print statistics
        if overall_actions:
            print(f"\n[STATS] Overall action statistics:")
            print(f"  Total samples: {total_samples}")
            for action, count in sorted(overall_actions.items()):
                percentage = count / sum(overall_actions.values()) * 100
                print(f"  {action}: {count} ({percentage:.1f}%)")


# ==================== CLI Interface ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate action groundtruth for NaVILA training.")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input directory containing scene folders with trajectory files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated action groundtruth files",
    )
    parser.add_argument(
        "--config-preset",
        choices=["vlnce", "navila_small", "navila_large", "custom_small"],
        default=DEFAULT_CONFIG_PRESET,
        help="Configuration preset (default: vlnce)",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Optional configuration file (overrides preset)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of worker processes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Limit number of files to process (for testing)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("       Action GroundTruth Generation for NaVILA Training")
    print("=" * 80)

    # Load configuration
    config = get_preset_config(args.config_preset)
    print(f"[INFO] Using config preset: {args.config_preset}")

    # Override with config file if provided
    if args.config_file and args.config_file.exists():
        with args.config_file.open("r", encoding="utf-8") as f:
            custom_config = json.load(f)
        config.update(custom_config)
        print(f"[INFO] Applied config file override: {args.config_file}")

    print(f"[INFO] Input directory: {args.input_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")

    try:
        # Create generator
        generator = BatchActionGenerator(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=config,
        )

        # Run batch processing
        generator.run_batch_processing(max_workers=args.max_workers, limit=args.limit)

    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

