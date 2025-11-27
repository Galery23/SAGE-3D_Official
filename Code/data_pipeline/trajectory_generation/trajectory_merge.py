#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Merge: Merge part-based 3D trajectory data files.

This script merges trajectory data files that are split into multiple parts (e.g., _part2, _part3)
into a single consolidated trajectory file, and organizes corresponding visualization images.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

# ==================== Default Configuration ====================

DEFAULT_SOURCE_DIR = Path("Data/trajectories")
DEFAULT_OUTPUT_DIR = Path("Data/trajectories_merged")


# ==================== Trajectory Merger Class ====================


class TrajectoryMerger:
    """Merger for trajectory data files."""

    def __init__(self, source_dir: Path, output_dir: Path):
        """Initialize trajectory merger.

        Args:
            source_dir: Source directory containing scene folders with trajectory files
            output_dir: Output directory for merged trajectory data
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_scene_info(self, filename: str) -> Tuple[str | None, str | None]:
        """Extract scene information from filename.

        Args:
            filename: Filename like trajectories_0033_839873_trans.json

        Returns:
            (prefix, scene_id) tuple or (None, None) if extraction fails
        """
        pattern = r"trajectories_(\d+)_(\d+)(?:_part\d+)?_trans\.json"
        match = re.match(pattern, filename)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def get_part_number(self, filename: str) -> int:
        """Extract part number from filename.

        Args:
            filename: Filename that may contain _partN_

        Returns:
            Part number, defaults to 1 if no part marker found
        """
        if "_part" in filename:
            pattern = r"_part(\d+)_"
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        return 1

    def merge_trajectory_data(self, scene_folder: Path) -> Dict | None:
        """Merge trajectory data files for a single scene.

        Args:
            scene_folder: Scene folder containing trajectory files

        Returns:
            Merged trajectory data dictionary or None if merge fails
        """
        trans_files = list(scene_folder.glob("*_trans.json"))

        if not trans_files:
            print(f"[WARN] {scene_folder.name}: No _trans.json files found")
            return None

        # Sort by part number
        trans_files.sort(key=lambda x: self.get_part_number(x.name))

        print(f"[{scene_folder.name}] Found {len(trans_files)} trajectory files")

        merged_data = None
        current_trajectory_id = 0

        for file_path in trans_files:
            print(f"  Processing: {file_path.name}")

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                if merged_data is None:
                    # First file: use as base structure
                    merged_data = data.copy()
                    # Set scene_id to match scene_name
                    if "scenes" in merged_data and merged_data["scenes"]:
                        scene_name = merged_data["scenes"][0].get("scene_name", "")
                        merged_data["scenes"][0]["scene_id"] = scene_name
                else:
                    # Merge samples from subsequent files
                    if "scenes" in data and data["scenes"] and "samples" in data["scenes"][0]:
                        merged_data["scenes"][0]["samples"].extend(data["scenes"][0]["samples"])

                # Renumber trajectory IDs
                if "scenes" in merged_data and merged_data["scenes"] and "samples" in merged_data["scenes"][0]:
                    samples = merged_data["scenes"][0]["samples"]
                    for i in range(len(samples)):
                        if i >= current_trajectory_id:
                            samples[i]["trajectory_id"] = str(current_trajectory_id)
                            current_trajectory_id += 1

            except Exception as e:
                print(f"  [ERROR] Failed to process {file_path.name}: {e}")
                continue

        return merged_data

    def organize_visualization_images(
        self, scene_folder: Path, output_scene_folder: Path
    ) -> bool:
        """Organize visualization images for a scene.

        Args:
            scene_folder: Source scene folder
            output_scene_folder: Output scene folder

        Returns:
            True if successful, False otherwise
        """
        vis_folders = list(scene_folder.glob("recollected_nav_vis*"))

        if not vis_folders:
            print(f"  [WARN] {scene_folder.name}: No visualization folders found")
            return False

        # Sort by part number
        vis_folders.sort(key=lambda x: self.get_part_number(x.name + "_"))

        print(f"  Organizing {len(vis_folders)} visualization folders")

        # Extract prefix and scene_id from first trajectory file
        trans_files = list(scene_folder.glob("*_trans.json"))
        if not trans_files:
            return False

        prefix, scene_id = self.extract_scene_info(trans_files[0].name)
        if not prefix or not scene_id:
            print(f"  [ERROR] Failed to extract scene info from {trans_files[0].name}")
            return False

        # Create visualization output folder
        vis_output_folder = output_scene_folder / f"trajectory_visualization_{prefix}_{scene_id}"
        vis_output_folder.mkdir(parents=True, exist_ok=True)

        current_trajectory_id = 0

        for vis_folder in vis_folders:
            print(f"    Processing: {vis_folder.name}")

            # Get and sort image files by trajectory ID
            image_files = list(vis_folder.glob("*.png"))

            def extract_traj_id(filename: Path) -> int:
                pattern = r"_traj_(\d+)_vis\.png"
                match = re.search(pattern, filename.name)
                return int(match.group(1)) if match else 0

            image_files.sort(key=extract_traj_id)

            # Copy and rename images
            for image_file in image_files:
                new_name = f"trajectory_visualization_{prefix}_{scene_id}_{current_trajectory_id}.png"
                output_path = vis_output_folder / new_name

                try:
                    shutil.copy2(image_file, output_path)
                    current_trajectory_id += 1
                except Exception as e:
                    print(f"      [ERROR] Failed to copy {image_file.name}: {e}")

        return True

    def process_single_scene(self, scene_folder: Path, skip_existing: bool = True) -> bool:
        """Process a single scene directory.

        Args:
            scene_folder: Scene folder path
            skip_existing: Whether to skip if output already exists

        Returns:
            True if successful, False otherwise
        """
        scene_name = scene_folder.name
        print(f"\n[{scene_name}] Processing scene")

        # Create output scene folder
        output_scene_folder = self.output_dir / scene_name
        output_scene_folder.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        existing_files = list(output_scene_folder.glob("trajectories_overall_*.json"))
        if skip_existing and existing_files:
            print(f"  [SKIP] Output file already exists: {existing_files[0].name}")
            return True

        # Merge trajectory data
        merged_data = self.merge_trajectory_data(scene_folder)
        if merged_data is None:
            print(f"  [ERROR] Failed to merge trajectory data")
            return False

        # Save merged trajectory data
        trans_files = list(scene_folder.glob("*_trans.json"))
        if trans_files:
            prefix, scene_id = self.extract_scene_info(trans_files[0].name)
            if prefix and scene_id:
                output_filename = f"trajectories_overall_{prefix}_{scene_id}.json"
                output_path = output_scene_folder / output_filename

                try:
                    with output_path.open("w", encoding="utf-8") as f:
                        json.dump(merged_data, f, ensure_ascii=False, indent=2)

                    # Count trajectories
                    trajectory_count = 0
                    if "scenes" in merged_data and merged_data["scenes"] and "samples" in merged_data["scenes"][0]:
                        trajectory_count = len(merged_data["scenes"][0]["samples"])

                    print(f"  [SAVED] {output_filename} ({trajectory_count} trajectories)")

                except Exception as e:
                    print(f"  [ERROR] Failed to save trajectory data: {e}")
                    return False

        # Organize visualization images
        if not self.organize_visualization_images(scene_folder, output_scene_folder):
            print(f"  [WARN] Failed to organize visualization images")

        return True

    def process_all_scenes(self, only_scenes: list[str] | None = None, skip_existing: bool = True) -> None:
        """Process all scene directories.

        Args:
            only_scenes: Optional list of scene names to process (if None, process all)
            skip_existing: Whether to skip scenes with existing output
        """
        print(f"[INFO] Starting trajectory merge")
        print(f"[INFO] Source: {self.source_dir}")
        print(f"[INFO] Output: {self.output_dir}")

        # Get scene folders
        scene_folders = sorted([d for d in self.source_dir.iterdir() if d.is_dir()])
        if only_scenes:
            scene_folders = [d for d in scene_folders if d.name in only_scenes]

        print(f"[INFO] Found {len(scene_folders)} scene folders")

        success_count = 0
        error_count = 0
        skipped_count = 0

        for scene_folder in scene_folders:
            try:
                output_scene_folder = self.output_dir / scene_folder.name
                existing_files = (
                    list(output_scene_folder.glob("trajectories_overall_*.json"))
                    if output_scene_folder.exists()
                    else []
                )

                if skip_existing and existing_files:
                    skipped_count += 1
                    continue

                if self.process_single_scene(scene_folder, skip_existing=skip_existing):
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {scene_folder.name}: {e}")
                error_count += 1

        print(f"\n[STATS] Success: {success_count}, Skipped: {skipped_count}, Errors: {error_count}, Total: {len(scene_folders)}")


# ==================== CLI Interface ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge part-based trajectory data files.")

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Source directory containing scene folders with trajectory files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for merged trajectory data",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only process specified scene folders",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-merge even if output files already exist",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate source directory
    if not args.source_dir.exists():
        print(f"[ERROR] Source directory does not exist: {args.source_dir}")
        return

    # Create merger and process
    merger = TrajectoryMerger(args.source_dir, args.output_dir)
    merger.process_all_scenes(only_scenes=args.only, skip_existing=not args.force)


if __name__ == "__main__":
    main()









