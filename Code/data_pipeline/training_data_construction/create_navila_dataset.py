#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create NaVILA Dataset: Convert SAGE-Bench data to NaVILA training format with multi-part support.

This script converts action groundtruth and image data into NaVILA training format,
supporting multi-part file splitting for efficient loading and parallel processing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# ==================== Default Configuration ====================

DEFAULT_ACTIONS_DIR = Path("Data/training_data/actions/train")
DEFAULT_IMAGES_DIR = Path("Data/training_data/images/train")
DEFAULT_OUTPUT_DIR = Path("Data/training_data/navila_dataset/train")
DEFAULT_SAMPLES_PER_PART = 10000

# Action format configuration (based on navila_small config)
ACTION_FORMAT_CONFIG = {
    "MOVE_FORWARD": "move forward 0.35 meter",
    "TURN_LEFT": "turn left 30 degree",
    "TURN_RIGHT": "turn right 30 degree",
    "STOP": "stop",
}


# ==================== Helper Functions ====================


def load_action_data(action_file: str) -> Dict[str, Any]:
    """Load action groundtruth data.

    Args:
        action_file: Path to action groundtruth JSON file

    Returns:
        Action data dictionary
    """
    with open(action_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_metadata(image_metadata_file: str) -> Dict[str, Any]:
    """Load image metadata.

    Args:
        image_metadata_file: Path to image metadata JSON file

    Returns:
        Image metadata dictionary
    """
    with open(image_metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


def format_action_output(action: str) -> str:
    """Format action to NaVILA output format.

    Args:
        action: Action string (e.g., "MOVE_FORWARD")

    Returns:
        Formatted action string (e.g., "The next action is move forward 0.35 meter.")
    """
    formatted_action = ACTION_FORMAT_CONFIG.get(action, action.lower())
    return f"The next action is {formatted_action}."


def create_sliding_window_samples(
    instruction: str, action_sequence: List[str], frame_files: List[str], video_id: str
) -> List[Dict[str, Any]]:
    """Create training samples using sliding window approach.

    Each sample contains:
    - All frames from start to current step (sliding window)
    - Next action at current step

    Args:
        instruction: Instruction text
        action_sequence: List of action strings
        frame_files: List of frame file paths (relative)
        video_id: Video ID prefix

    Returns:
        List of training samples
    """
    samples = []

    # Ensure sufficient frames and actions
    min_length = min(len(frame_files), len(action_sequence))
    if min_length < 1:
        return samples

    # Create training sample for each step
    for step_idx in range(min_length):
        # Get all frames from start to current step (sliding window)
        window_frames = frame_files[: step_idx + 1]

        # Get next action
        next_action = action_sequence[step_idx]
        formatted_action = format_action_output(next_action)

        # Create training sample
        sample = {
            "video_id": f"{video_id}-{step_idx}",
            "q": instruction,
            "a": formatted_action,
            "frames": window_frames,
        }

        samples.append(sample)

    return samples


def process_scene_data(
    actions_dir: Path, images_dir: Path, scene_id: str
) -> List[Dict[str, Any]]:
    """Process data for a single scene.

    Args:
        actions_dir: Directory containing action groundtruth files
        images_dir: Directory containing image data
        scene_id: Scene ID

    Returns:
        List of training samples for this scene
    """
    # Check if data files exist
    action_file = actions_dir / scene_id / "action_groundtruth.json"
    image_metadata_file = images_dir / scene_id / "image_metadata.json"

    if not action_file.exists():
        print(f"[WARN] Action file not found: {action_file}")
        return []

    if not image_metadata_file.exists():
        print(f"[WARN] Image metadata file not found: {image_metadata_file}")
        return []

    # Load data
    action_data = load_action_data(str(action_file))
    image_data = load_image_metadata(str(image_metadata_file))

    # Create image sequence index
    image_sequences = {}
    for seq in image_data.get("sequences", []):
        key = f"{seq['trajectory_id']}_{seq['instruction_index']}"
        image_sequences[key] = seq

    all_samples = []

    # Process each action sequence
    for action_item in action_data.get("groundtruth_data", []):
        trajectory_id = action_item["trajectory_id"]
        instruction_index = action_item["instruction_index"]
        key = f"{trajectory_id}_{instruction_index}"

        # Find corresponding image sequence
        if key not in image_sequences:
            print(f"[WARN] Image sequence not found: {scene_id}_{key}")
            continue

        image_seq = image_sequences[key]
        
        # Handle instruction format (can be string or dict with 'generated_instruction' key)
        instruction_raw = action_item.get("instruction", "")
        if isinstance(instruction_raw, dict):
            instruction = instruction_raw.get("generated_instruction", "")
        else:
            instruction = instruction_raw
        
        action_sequence = action_item["action_sequence"]

        # Build frame file paths (relative paths)
        frame_files = []
        for frame_name in image_seq["frame_filenames"]:
            # Check if file exists
            full_frame_path = images_dir / scene_id / "images" / f"trajectory_{trajectory_id}" / frame_name
            if full_frame_path.exists():
                # Use relative path format: scene_id/images/trajectory_id/frame_name
                relative_path = f"{scene_id}/images/trajectory_{trajectory_id}/{frame_name}"
                frame_files.append(relative_path)
            else:
                print(f"[WARN] Image file not found: {full_frame_path}")
                break

        # Ensure frame count matches action count
        if len(frame_files) != len(action_sequence):
            print(
                f"[WARN] Frame count ({len(frame_files)}) != action count ({len(action_sequence)}) for {scene_id}_{key}"
            )
            # Take minimum length
            min_len = min(len(frame_files), len(action_sequence))
            frame_files = frame_files[:min_len]
            action_sequence = action_sequence[:min_len]

        # Create sliding window samples for this trajectory
        video_id = f"{scene_id}_{trajectory_id}_{instruction_index}"
        sliding_samples = create_sliding_window_samples(instruction, action_sequence, frame_files, video_id)
        all_samples.extend(sliding_samples)

    return all_samples


def save_data_in_parts(
    all_samples: List[Dict[str, Any]], output_dir: Path, samples_per_part: int = 10000
) -> List[str]:
    """Save data split into multiple part files.

    Args:
        all_samples: All training samples
        output_dir: Output directory
        samples_per_part: Number of samples per part file

    Returns:
        List of part file names
    """
    if not all_samples:
        return []

    # Calculate number of parts needed
    num_parts = math.ceil(len(all_samples) / samples_per_part)
    part_files = []

    print(f"[INFO] Splitting {len(all_samples)} samples into {num_parts} part files")

    for part_idx in range(num_parts):
        start_idx = part_idx * samples_per_part
        end_idx = min((part_idx + 1) * samples_per_part, len(all_samples))

        part_samples = all_samples[start_idx:end_idx]
        part_filename = f"annotations_part_{part_idx:03d}.json"
        part_file = output_dir / part_filename

        with part_file.open("w", encoding="utf-8") as f:
            json.dump(part_samples, f, indent=2, ensure_ascii=False)

        part_files.append(part_filename)
        print(f"[INFO] Part {part_idx:03d}: {len(part_samples)} samples -> {part_filename}")

    return part_files


def create_dataset_info(
    output_dir: Path,
    part_files: List[str],
    total_samples: int,
    actions_dir: Path,
    images_dir: Path,
    samples_per_part: int,
    max_scenes: int | None = None,
) -> None:
    """Create dataset info file for training use.

    Args:
        output_dir: Output directory
        part_files: List of part file names
        total_samples: Total number of samples
        actions_dir: Actions directory path
        images_dir: Images directory path
        samples_per_part: Samples per part
        max_scenes: Maximum scenes processed (if limited)
    """
    dataset_info = {
        "dataset_name": "SAGE-Bench_NaVILA",
        "total_samples": total_samples,
        "num_parts": len(part_files),
        "samples_per_part": samples_per_part,
        "part_files": part_files,
        "data_format": {
            "move_distance": "0.35 meter",
            "turn_angle": "30 degree",
            "action_format": "The next action is {action}.",
        },
        "conversion_params": {
            "actions_dir": str(actions_dir),
            "images_dir": str(images_dir),
            "max_scenes": max_scenes,
        },
    }

    info_file = output_dir / "dataset_info.json"
    with info_file.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Dataset info saved to: {info_file}")


# ==================== Main Processing Class ====================


class NaVILADatasetCreator:
    """NaVILA dataset creator with multi-part support."""

    def __init__(
        self,
        actions_dir: Path,
        images_dir: Path,
        output_dir: Path,
        samples_per_part: int = DEFAULT_SAMPLES_PER_PART,
        max_scenes: int | None = None,
        create_images_link: bool = True,
    ):
        """Initialize dataset creator.

        Args:
            actions_dir: Directory containing action groundtruth files
            images_dir: Directory containing image data
            output_dir: Output directory for NaVILA dataset
            samples_per_part: Number of samples per part file
            max_scenes: Maximum number of scenes to process (for testing)
            create_images_link: Whether to create symbolic link to images directory
        """
        self.actions_dir = Path(actions_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.samples_per_part = samples_per_part
        self.max_scenes = max_scenes
        self.create_images_link = create_images_link

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] NaVILA dataset creator initialized")
        print(f"[INFO] Actions directory: {self.actions_dir}")
        print(f"[INFO] Images directory: {self.images_dir}")
        print(f"[INFO] Output directory: {self.output_dir}")
        print(f"[INFO] Samples per part: {self.samples_per_part}")

    def create_images_symbolic_link(self) -> None:
        """Create symbolic link to images directory."""
        images_link_dir = self.output_dir / "images"
        if images_link_dir.exists():
            if images_link_dir.is_symlink():
                print(f"[INFO] Images symbolic link already exists: {images_link_dir}")
                return
            else:
                print(f"[WARN] Images directory exists but is not a symlink: {images_link_dir}")
                return

        try:
            images_link_dir.symlink_to(self.images_dir.absolute())
            print(f"[INFO] Created images symbolic link: {images_link_dir} -> {self.images_dir}")
        except Exception as e:
            print(f"[WARN] Failed to create symbolic link: {e}")

    def find_common_scenes(self) -> List[str]:
        """Find common scenes in both actions and images directories.

        Returns:
            List of common scene IDs
        """
        if not self.actions_dir.exists():
            print(f"[ERROR] Actions directory does not exist: {self.actions_dir}")
            return []

        if not self.images_dir.exists():
            print(f"[ERROR] Images directory does not exist: {self.images_dir}")
            return []

        # Get scene IDs from both directories
        action_scenes = {d.name for d in self.actions_dir.iterdir() if d.is_dir()}
        image_scenes = {d.name for d in self.images_dir.iterdir() if d.is_dir()}
        common_scenes = sorted(action_scenes.intersection(image_scenes))

        if self.max_scenes:
            common_scenes = common_scenes[: self.max_scenes]

        print(f"[INFO] Found {len(common_scenes)} common scenes")
        return common_scenes

    def process_all_scenes(self) -> List[Dict[str, Any]]:
        """Process all scenes and generate training samples.

        Returns:
            List of all training samples
        """
        common_scenes = self.find_common_scenes()
        if not common_scenes:
            print("[ERROR] No common scenes found")
            return []

        all_samples = []

        for scene_id in tqdm(common_scenes, desc="Processing scenes"):
            scene_samples = process_scene_data(self.actions_dir, self.images_dir, scene_id)
            all_samples.extend(scene_samples)

            if len(scene_samples) > 0:
                print(f"[INFO] Scene {scene_id}: Generated {len(scene_samples)} samples")

        return all_samples

    def run(self) -> None:
        """Run dataset creation process."""
        print("=" * 80)
        print("       NaVILA Dataset Creation")
        print("=" * 80)

        # Create images symbolic link if needed
        if self.create_images_link:
            self.create_images_symbolic_link()

        # Process all scenes
        all_samples = self.process_all_scenes()

        if not all_samples:
            print("[ERROR] No samples generated")
            return

        # Randomly shuffle data
        print("[INFO] Shuffling samples...")
        random.shuffle(all_samples)

        # Save data in parts
        part_files = save_data_in_parts(all_samples, self.output_dir, self.samples_per_part)

        # Create dataset info file
        create_dataset_info(
            self.output_dir,
            part_files,
            len(all_samples),
            self.actions_dir,
            self.images_dir,
            self.samples_per_part,
            self.max_scenes,
        )

        print(f"\n[SUCCESS] Conversion complete!")
        print(f"[INFO] Total samples: {len(all_samples)}")
        print(f"[INFO] Part files: {len(part_files)}")
        print(f"[INFO] Samples per part: {self.samples_per_part}")

        # Show sample example
        if all_samples:
            print(f"\n[INFO] NaVILA format sample example:")
            sample = all_samples[0]
            print(f"  Video ID: {sample['video_id']}")
            print(f"  Q: {sample['q'][:60]}...")
            print(f"  A: {sample['a']}")
            print(f"  Frames: {len(sample['frames'])} frames")
            if sample["frames"]:
                print(f"  Frame range: {sample['frames'][0]} -> {sample['frames'][-1]}")


# ==================== CLI Interface ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert SAGE-Bench data to NaVILA training format (multi-part version)"
    )

    parser.add_argument(
        "--actions-dir",
        type=Path,
        default=DEFAULT_ACTIONS_DIR,
        help="Directory containing action groundtruth files",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing image data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for NaVILA dataset",
    )
    parser.add_argument(
        "--samples-per-part",
        type=int,
        default=DEFAULT_SAMPLES_PER_PART,
        help="Number of samples per part file (default: 10000)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Maximum number of scenes to process (for testing)",
    )
    parser.add_argument(
        "--no-images-link",
        action="store_true",
        help="Do not create symbolic link to images directory",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    creator = NaVILADatasetCreator(
        actions_dir=args.actions_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        samples_per_part=args.samples_per_part,
        max_scenes=args.max_scenes,
        create_images_link=not args.no_images_link,
    )

    creator.run()


if __name__ == "__main__":
    main()

