#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Images: Generate training images from trajectories using Isaac Sim.

This script processes trajectory data and generates RGB images at sampled waypoints
using Isaac Sim for rendering.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil
from PIL import Image

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.sensor import Camera
from pxr import Gf, UsdGeom, UsdLux

# ==================== Default Configuration ====================

DEFAULT_INPUT_DIR = Path("Data/training_data/trajectories/train")
DEFAULT_USD_ROOT = Path("/path/to/data/InteriorGS_usda")
DEFAULT_OUTPUT_DIR = Path("Data/training_data/images")
DEFAULT_ACTION_ROOT = Path("Data/training_data/actions")

# Camera settings
CAMERA_RESOLUTION = (1024, 768)
CAMERA_FOCAL_LENGTH = 8.0
CAMERA_HEIGHT = 1.2  # meters

# Processing settings
WORLD_STEP_COUNT = 5
RENDER_STEP_COUNT = 3
MEMORY_WARNING_THRESHOLD = 80
MEMORY_CRITICAL_THRESHOLD = 85


# ==================== Sequential Fast Image Generator ====================


class SequentialFastImageGenerator:
    """Fast sequential image generator with scene reuse optimization."""

    def __init__(
        self,
        input_dir: Path,
        usd_root: Path,
        output_dir: Path,
        action_root: Path | None = None,
        force: bool = False,
        max_trajectories_per_scene: int | None = None,
        instance_id: int = 0,
        total_instances: int = 1,
    ):
        """Initialize image generator.

        Args:
            input_dir: Input directory containing scene folders with trajectory files
            usd_root: Root directory containing USD scene files (.usda)
            output_dir: Output directory for generated images
            action_root: Root directory containing action groundtruth files
            force: Force reprocess even if already processed
            max_trajectories_per_scene: Maximum trajectories to process per scene
            instance_id: Instance ID for distributed processing (0-indexed)
            total_instances: Total number of instances for distributed processing
        """
        self.input_dir = Path(input_dir)
        self.usd_root = Path(usd_root)
        self.output_dir = Path(output_dir)
        self.action_root = Path(action_root) if action_root else None
        self.force = force
        self.max_trajectories_per_scene = max_trajectories_per_scene

        # Distributed processing parameters
        self.instance_id = instance_id
        self.total_instances = total_instances

        # Validate distributed parameters
        if not (0 <= instance_id < total_instances):
            raise ValueError(f"instance_id({instance_id}) must be in range [0, {total_instances})")

        if self.total_instances > 1:
            print(f"[INFO] Distributed processing mode: Instance {self.instance_id + 1}/{self.total_instances}")
        else:
            print(f"[INFO] Single instance processing mode")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Image generator initialized")
        print(f"[INFO] Input directory: {self.input_dir}")
        print(f"[INFO] USD root: {self.usd_root}")
        print(f"[INFO] Output directory: {self.output_dir}")
        print(f"[INFO] Action root: {self.action_root}")
        if self.max_trajectories_per_scene:
            print(f"[INFO] Max trajectories per scene: {self.max_trajectories_per_scene}")

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

            # Distributed processing: scene-level sharding
            if self.total_instances > 1:
                scene_hash = hash(scene_id) % self.total_instances
                if scene_hash != self.instance_id:
                    continue  # Skip scenes not assigned to this instance

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

            # Check for USD file
            usd_file = self.usd_root / f"{scene_id}.usda"
            if not usd_file.exists():
                print(f"[WARN] Scene {scene_id}: USD file not found: {usd_file}")
                continue

            for traj_file in traj_files:
                file_info = {
                    "scene_id": scene_id,
                    "traj_file": str(traj_file),
                    "usd_file": str(usd_file),
                    "output_dir": str(self.output_dir / scene_id),
                }
                all_files.append(file_info)

            print(f"[INFO] Scene {scene_id}: {len(traj_files)} trajectory files")

        if self.total_instances > 1:
            print(f"\n[INFO] Total found {len(all_files)} files to process (Instance {self.instance_id + 1}/{self.total_instances})")
            print(f"[INFO] Note: Each instance processes different scenes, total files may differ per instance")
        else:
            print(f"\n[INFO] Total found {len(all_files)} files to process")
        return all_files

    def _load_action_sampled_points(
        self, file_info: Dict[str, str], trajectory_id: str, instruction_index: int
    ) -> List[Dict] | None:
        """Load sampled points from action groundtruth file.

        Args:
            file_info: File information dictionary
            trajectory_id: Trajectory ID
            instruction_index: Instruction index

        Returns:
            List of sampled points or None if not found
        """
        try:
            if self.action_root:
                action_file = self.action_root / file_info["scene_id"] / "action_groundtruth.json"
            else:
                # Try relative to output directory
                action_file = Path(file_info["output_dir"]).parent / "actions" / file_info["scene_id"] / "action_groundtruth.json"

            if not action_file.exists():
                return None

            with action_file.open("r", encoding="utf-8") as f:
                action_data = json.load(f)

            # Find matching groundtruth item
            for groundtruth_item in action_data.get("groundtruth_data", []):
                if (
                    groundtruth_item["trajectory_id"] == str(trajectory_id)
                    and groundtruth_item["instruction_index"] == instruction_index
                ):
                    sampled_points = []
                    for sp in groundtruth_item.get("sampled_points", []):
                        point = {
                            "point": sp["point_id"],
                            "position": sp["position"],
                            "rotation": sp["rotation"],
                        }
                        sampled_points.append(point)

                    return sampled_points

            return None

        except Exception as e:
            print(f"[ERROR] Failed to load sampled points: {e}")
            return None

    def _check_if_already_processed(self, file_info: Dict[str, str]) -> tuple[bool, str]:
        """Check if file has already been processed.

        Args:
            file_info: File information dictionary

        Returns:
            (is_processed, reason) tuple
        """
        output_dir = Path(file_info["output_dir"])
        images_dir = output_dir / "images"
        metadata_file = output_dir / "image_metadata.json"

        if not metadata_file.exists():
            return False, "Metadata file does not exist"

        try:
            # Read metadata
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Check trajectory data
            with Path(file_info["traj_file"]).open("r", encoding="utf-8") as f:
                traj_data = json.load(f)

            scene_info = traj_data["scenes"][0]
            samples_to_check = scene_info["samples"]
            if self.max_trajectories_per_scene:
                samples_to_check = samples_to_check[: self.max_trajectories_per_scene]

            # Count expected sequences
            expected_sequences = 0
            unique_trajectories = {}
            for sample in samples_to_check:
                trajectory_id = sample["trajectory_id"]
                if trajectory_id not in unique_trajectories:
                    unique_trajectories[trajectory_id] = sample
                    expected_sequences += len(sample["instructions"])

            # Check actual image files
            actual_images = 0
            if images_dir.exists():
                for trajectory_id in unique_trajectories.keys():
                    traj_dir = images_dir / f"trajectory_{trajectory_id}"
                    if traj_dir.exists():
                        image_files = list(traj_dir.glob("*.jpg"))
                        if len(image_files) > 0:
                            traj_instruction_count = len(unique_trajectories[trajectory_id]["instructions"])
                            actual_images += traj_instruction_count

            # Check if complete
            if actual_images >= expected_sequences:
                return True, f"Complete: {actual_images}/{expected_sequences} sequences"
            else:
                return False, f"Incomplete: {actual_images}/{expected_sequences} sequences"

        except Exception as e:
            return False, f"Check failed: {e}"

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

            if self.force:
                print(f"  [FORCE] Force reprocessing")
            else:
                print(f"  [INFO] Starting processing")

            # Create output directory
            output_dir = Path(file_info["output_dir"])
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Load trajectory data
            with Path(file_info["traj_file"]).open("r", encoding="utf-8") as f:
                traj_data = json.load(f)

            scene_info = traj_data["scenes"][0]

            # Load USD scene
            print("  [INFO] Loading USD scene...")
            omni.usd.get_context().close_stage()
            gc.collect()

            if not open_stage(usd_path=file_info["usd_file"]):
                print(f"[ERROR] Failed to open USD scene: {file_info['usd_file']}")
                return False

            stage = omni.usd.get_context().get_stage()

            # Add environment light
            if not stage.GetPrimAtPath("/World/EnvLight"):
                dome = UsdLux.DomeLight.Define(stage, "/World/EnvLight")
                dome.CreateIntensityAttr(30000.0)
                dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

            # Initialize world
            print("  [INFO] Initializing Isaac Sim world...")
            world = World()
            world.reset()
            for _ in range(WORLD_STEP_COUNT):
                world.step(render=True)

            # Create camera (reused)
            print("  [INFO] Creating camera...")
            sensor_cam_path = "/World/NaVILACamera"
            cam = Camera(prim_path=sensor_cam_path, frequency=30, resolution=CAMERA_RESOLUTION)
            cam.initialize()

            cam_prim = stage.GetPrimAtPath(sensor_cam_path)
            usd_cam = UsdGeom.Camera(cam_prim)
            usd_cam.GetFocalLengthAttr().Set(CAMERA_FOCAL_LENGTH)

            print("  [INFO] Scene initialization complete")

            try:
                # Collect trajectories
                samples_to_process = scene_info["samples"]
                if self.max_trajectories_per_scene:
                    print(
                        f"  [INFO] Limiting trajectories: {len(samples_to_process)} → {self.max_trajectories_per_scene}"
                    )
                    samples_to_process = samples_to_process[: self.max_trajectories_per_scene]

                # Deduplicate trajectories
                unique_trajectories = {}
                for sample in samples_to_process:
                    trajectory_id = sample["trajectory_id"]
                    if trajectory_id not in unique_trajectories:
                        unique_trajectories[trajectory_id] = {
                            "trajectory_id": trajectory_id,
                            "points": sample["points"],
                            "instructions": sample["instructions"],
                        }

                print(f"  [INFO] Processing {len(unique_trajectories)} unique trajectories")

                all_image_data = []
                total_images_generated = 0
                processed_trajectories = 0

                # Process each trajectory sequentially
                for i, trajectory_data in enumerate(unique_trajectories.values(), 1):
                    trajectory_id = trajectory_data["trajectory_id"]
                    instructions = trajectory_data["instructions"]

                    print(f"  [INFO] Processing trajectory {i}/{len(unique_trajectories)}: {trajectory_id}")

                    # Load sampled points
                    trajectory_sampled_points = self._load_action_sampled_points(file_info, trajectory_id, 0)

                    if trajectory_sampled_points is None:
                        print(f"    [WARN] No sampled points for trajectory {trajectory_id}, skipping")
                        continue

                    print(f"    [INFO] Found {len(trajectory_sampled_points)} sampled points")
                    processed_trajectories += 1

                    # Create trajectory folder
                    traj_dir = images_dir / f"trajectory_{trajectory_id}"
                    traj_dir.mkdir(parents=True, exist_ok=True)

                    # Generate images
                    frame_filenames = []
                    traj_images = 0
                    total_points = len(trajectory_sampled_points)

                    print(f"    [INFO] Generating {total_points} images...")

                    for frame_idx, point in enumerate(trajectory_sampled_points):
                        if frame_idx % 10 == 0:
                            progress = (frame_idx + 1) / total_points * 100
                            print(f"      Progress: {frame_idx+1}/{total_points} ({progress:.1f}%)")
                            gc.collect()

                        filename = f"{scene_id}_{trajectory_id}_{frame_idx:03d}.jpg"

                        # Set camera pose
                        position = np.array(point["position"], dtype=np.float32)
                        position[2] = CAMERA_HEIGHT
                        cam.set_world_pose(
                            position=position, orientation=np.array(point["rotation"], dtype=np.float32)
                        )

                        # Render and capture
                        try:
                            for _ in range(RENDER_STEP_COUNT):
                                world.step(render=True)

                            img = cam.get_rgba()
                            if img is not None and img.size > 0:
                                # Save image and release memory immediately
                                img_array = img[:, :, :3].copy()
                                Image.fromarray(img_array).save(traj_dir / filename)
                                del img_array
                                del img
                                frame_filenames.append(filename)
                                traj_images += 1
                            else:
                                # Placeholder image
                                placeholder = Image.new("RGB", CAMERA_RESOLUTION, color=(0, 0, 0))
                                placeholder.save(traj_dir / filename)
                                placeholder.close()
                                frame_filenames.append(filename)

                        except Exception as e:
                            print(f"      [ERROR] Failed to generate image {frame_idx}: {e}")
                            try:
                                placeholder = Image.new("RGB", CAMERA_RESOLUTION, color=(0, 0, 0))
                                placeholder.save(traj_dir / filename)
                                placeholder.close()
                                frame_filenames.append(filename)
                            except Exception as save_error:
                                print(f"      [ERROR] Failed to save placeholder: {save_error}")

                        # Periodic memory cleanup
                        if frame_idx % 5 == 0:
                            gc.collect()

                    total_images_generated += traj_images

                    # Record data for each instruction
                    for inst_idx, instruction in enumerate(instructions):
                        image_record = {
                            "scene_id": scene_id,
                            "trajectory_id": trajectory_id,
                            "instruction_index": inst_idx,
                            "instruction": instruction,
                            "frame_filenames": frame_filenames,
                            "trajectory_sampled_points": [
                                {
                                    "point_id": point["point"],
                                    "position": point["position"],
                                    "rotation": point["rotation"],
                                }
                                for point in trajectory_sampled_points
                            ],
                            "sampling_info": {
                                "sampled_points_count": len(trajectory_sampled_points),
                                "generated_images_count": len(frame_filenames),
                                "data_source": "sequential_fast_processing",
                            },
                        }
                        all_image_data.append(image_record)

                    print(f"    [INFO] Trajectory {trajectory_id}: {traj_images} images")

                    # Check memory usage
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > MEMORY_WARNING_THRESHOLD:
                        print(f"    [WARN] High memory usage: {memory_percent:.1f}%, consider restarting process")
                        gc.collect()
                        gc.collect()

                print(f"  [INFO] Scene processing result: {processed_trajectories}/{len(unique_trajectories)} trajectories processed")
                if processed_trajectories == 0:
                    print(f"  [WARN] No trajectories were successfully processed!")

            finally:
                # Cleanup resources
                print("  [INFO] Cleaning up resources...")
                try:
                    # Cleanup camera
                    if "cam" in locals():
                        cam = None

                    # Cleanup world
                    if "world" in locals():
                        world.clear()
                        world = None

                    # Close USD stage
                    try:
                        omni.usd.get_context().close_stage()
                    except Exception as e:
                        print(f"  [WARN] Failed to close stage: {e}")

                    # Force garbage collection
                    gc.collect()
                    gc.collect()

                    print("  [INFO] Resource cleanup complete")

                except Exception as cleanup_error:
                    print(f"  [ERROR] Error during cleanup: {cleanup_error}")

            # Save metadata
            print(f"  [INFO] Saving metadata to: {output_dir / 'image_metadata.json'}")
            print(f"  [INFO] Data stats: {len(all_image_data)} sequences, {total_images_generated} images")

            if len(all_image_data) == 0:
                print(f"  [WARN] No data generated, but saving empty metadata")

            try:
                self._save_image_metadata(all_image_data, output_dir, file_info)
                print(f"  [INFO] Metadata saved successfully")
            except Exception as e:
                print(f"  [ERROR] Failed to save metadata: {e}")
                import traceback

                traceback.print_exc()
                return False

            print(f"  [SUCCESS] Generated {total_images_generated} images")
            print(f"    Processed trajectories: {len(unique_trajectories)}")
            print(f"    Instruction sequences: {len(all_image_data)}")

            return True

        except Exception as e:
            import traceback

            error_msg = str(e)
            print(f"  [ERROR] Processing failed: {error_msg}")

            # Check for memory-related errors
            if any(keyword in error_msg.lower() for keyword in ["corrupted", "malloc", "free"]):
                print(f"  [WARN] Detected memory corruption error, consider restarting process!")
                print(f"  [TIP] Use --max-trajectories to limit trajectories per scene")

            print(f"  [INFO] Error details:")
            traceback.print_exc()

            # Emergency cleanup
            try:
                print(f"  [INFO] Attempting emergency cleanup...")
                gc.collect()
                omni.usd.get_context().close_stage()
            except Exception:
                pass

            return False

    def _save_image_metadata(self, image_data: List[Dict], output_dir: Path, file_info: Dict[str, str]) -> None:
        """Save image metadata.

        Args:
            image_data: List of image data dictionaries
            output_dir: Output directory
            file_info: File information dictionary
        """
        metadata_file = output_dir / "image_metadata.json"

        metadata = {
            "scene_id": file_info["scene_id"],
            "scene_name": image_data[0]["scene_id"] if image_data else file_info["scene_id"],
            "total_image_sequences": len(image_data),
            "frames_per_sequence": "variable_based_on_action_sampling",
            "image_resolution": list(CAMERA_RESOLUTION),
            "camera_settings": {
                "focal_length": CAMERA_FOCAL_LENGTH,
                "height": CAMERA_HEIGHT,
            },
            "sequences": image_data,
            "source_files": {
                "trajectory_file": file_info["traj_file"],
                "usd_file": file_info["usd_file"],
            },
            "processing_limits": {
                "max_trajectories_per_scene": self.max_trajectories_per_scene,
                "trajectory_limit_applied": self.max_trajectories_per_scene is not None,
            },
            "processing_mode": {
                "type": "sequential_fast",
                "scene_reuse": True,
                "single_camera": True,
            },
        }

        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def run_batch_processing(self, limit: int | None = None) -> None:
        """Run batch processing.

        Args:
            limit: Limit number of files to process (for testing)
        """
        all_files = self.find_all_trajectory_files()

        if not all_files:
            print("[ERROR] No files found to process")
            return

        # Apply limit
        if limit is not None and limit > 0:
            all_files = all_files[:limit]
            print(f"[INFO] Limiting processing to {limit} files")

        if self.total_instances > 1:
            print(f"\n[INFO] Starting batch processing of {len(all_files)} files (Instance {self.instance_id + 1}/{self.total_instances})...")
            print(f"[INFO] Note: Each instance processes different scenes")
        else:
            print(f"\n[INFO] Starting batch processing of {len(all_files)} files...")

        success_count = 0
        failed_count = 0
        skipped_count = 0

        for i, file_info in enumerate(all_files, 1):
            print(f"\n[PROGRESS] {i}/{len(all_files)}")

            # Check if already processed (unless force mode)
            if not self.force:
                is_processed, reason = self._check_if_already_processed(file_info)
                if is_processed:
                    print(f"  [SKIP] {file_info['scene_id']}: {reason}")
                    skipped_count += 1
                    continue

            start_time = time.time()

            # Check memory before processing
            memory_before = psutil.virtual_memory().percent

            if self.process_single_file(file_info):
                success_count += 1
                end_time = time.time()
                memory_after = psutil.virtual_memory().percent
                print(
                    f"  [INFO] Time: {end_time - start_time:.1f}s, Memory: {memory_before:.1f}% → {memory_after:.1f}%"
                )

                # Warn if memory usage is high
                if memory_after > MEMORY_CRITICAL_THRESHOLD:
                    print(f"  [WARN] High memory usage: {memory_after:.1f}%, consider restarting process")
            else:
                failed_count += 1

        print(f"\n[COMPLETE] Batch processing finished!")
        if self.total_instances > 1:
            print(f"  Instance {self.instance_id + 1}/{self.total_instances} statistics:")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Total: {len(all_files)}")

        # Distributed processing status
        if self.total_instances > 1:
            print(f"\n[INFO] Distributed processing status:")
            print(f"  Current instance: {self.instance_id + 1}/{self.total_instances}")
            print(f"  Scenes assigned to this instance: {len(all_files)}")
            print(f"  Scenes actually processed: {success_count + failed_count}")
            print(f"[INFO] Tip: The sum of all instances is the complete processing result")
            print(f"[INFO] Tip: Different instances having different scene counts is normal (hash sharding)")


# ==================== CLI Interface ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate training images from trajectories using Isaac Sim.")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input directory containing scene folders with trajectory files",
    )
    parser.add_argument(
        "--usd-root",
        type=Path,
        default=DEFAULT_USD_ROOT,
        help="Root directory containing USD scene files (.usda)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--action-root",
        type=Path,
        default=DEFAULT_ACTION_ROOT,
        help="Root directory containing action groundtruth files",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        help="Maximum trajectories to process per scene",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess even if already processed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        default=0,
        help="Instance ID for distributed processing (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--total-instances",
        type=int,
        default=1,
        help="Total number of instances for distributed processing (default: 1)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("       Image Generation for NaVILA Training (Isaac Sim)")
    print("=" * 80)

    print(f"[INFO] Input directory: {args.input_dir}")
    print(f"[INFO] USD root: {args.usd_root}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Action root: {args.action_root}")
    if args.max_trajectories:
        print(f"[INFO] Max trajectories per scene: {args.max_trajectories}")

    # Distributed processing configuration
    if args.total_instances > 1:
        print(f"\n[INFO] Distributed processing configuration:")
        print(f"  Instance ID: {args.instance_id + 1}/{args.total_instances}")
        print(f"[INFO] Tip: Launch {args.total_instances} instances with instance-id from 0 to {args.total_instances - 1}")
        print(f"[INFO] Tip: Each instance will process different scenes to avoid duplicate work")

        # Validate parameters
        if not (0 <= args.instance_id < args.total_instances):
            print(f"[ERROR] Invalid parameters: instance-id({args.instance_id}) must be in range [0, {args.total_instances})")
            return

    # Validate paths
    if not args.input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {args.input_dir}")
        return

    if not args.usd_root.exists():
        print(f"[ERROR] USD root directory does not exist: {args.usd_root}")
        return

    try:
        generator = SequentialFastImageGenerator(
            input_dir=args.input_dir,
            usd_root=args.usd_root,
            output_dir=args.output_dir,
            action_root=args.action_root,
            force=args.force,
            max_trajectories_per_scene=args.max_trajectories,
            instance_id=args.instance_id,
            total_instances=args.total_instances,
        )

        generator.run_batch_processing(limit=args.limit)

    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

