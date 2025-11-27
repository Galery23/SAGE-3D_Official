#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain-Aware Trajectory Data Splitter: Split VLN trajectory data into train/val/test sets.

This script performs domain-aware three-level data splitting (Scene-Unseen, Trajectory-Unseen, 
Instruction-Unseen) with balanced distribution across home and non-home scene types.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ==================== Default Configuration ====================

DEFAULT_TRAJ_ROOT = Path("Data/trajectories_merged")
DEFAULT_SCENE_TYPE_FILE = Path("Data/scene_type.json")
DEFAULT_OUTPUT_DIR = Path("Data/trajectories_merged")

DEFAULT_TARGET_SCENES_PER_TEST = 15
DEFAULT_TARGET_PAIRS_PER_TEST = 1000
DEFAULT_VAL_SCENES = 20
DEFAULT_VAL_PAIRS = 50000
DEFAULT_TRAJECTORY_UNSEEN_SCENES = 15
DEFAULT_INSTRUCTION_UNSEEN_SCENES = 15
DEFAULT_RANDOM_SEED = 42
DEFAULT_PROGRESS_INTERVAL = 20


# ==================== Domain-Aware Trajectory Data Splitter ====================


class DomainAwareTrajectoryDataSplitter:
    """Domain-aware trajectory data splitter with balanced scene type distribution."""

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED, progress_interval: int = DEFAULT_PROGRESS_INTERVAL):
        """Initialize splitter.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.all_scenes: Dict[str, Dict] = {}
        self.scene_type_map: Dict[str, str] = {}  # scene_id -> scene type (home/non_home)
        self.scene_design_type_map: Dict[str, str] = {}  # scene_id -> design type
        self.progress_interval = max(1, progress_interval)
        random.seed(random_seed)

    def load_scene_types(self, scene_type_file: Path) -> Tuple[List[str], List[str], Dict[str, int]]:
        """Load scene type information.

        Args:
            scene_type_file: Path to scene_type.json file

        Returns:
            (home_scenes, non_home_scenes, design_type_counts) tuple
        """
        print("Loading scene type information...")

        with scene_type_file.open("r", encoding="utf-8") as f:
            scene_types = json.load(f)

        # Build world_id to design_type mapping
        design_type_map: Dict[str, str] = {}
        for item in scene_types:
            world_id = str(item.get("world_id", ""))
            design_type = item.get("design_type", "")
            if world_id and design_type:
                design_type_map[world_id] = design_type

        # Classify scenes
        home_scenes: List[str] = []
        non_home_scenes: List[str] = []
        design_type_counts: Dict[str, int] = {}

        for scene_id in self.all_scenes.keys():
            if scene_id in design_type_map:
                design_type = design_type_map[scene_id]
                self.scene_type_map[scene_id] = "non_home"
                self.scene_design_type_map[scene_id] = design_type
                non_home_scenes.append(scene_id)
                design_type_counts[design_type] = design_type_counts.get(design_type, 0) + 1
            else:
                self.scene_type_map[scene_id] = "home"
                self.scene_design_type_map[scene_id] = "Home"
                home_scenes.append(scene_id)

        print(f"Scene type distribution:")
        print(f"  Home scenes: {len(home_scenes)}")
        print(f"  Non-home scenes: {len(non_home_scenes)}")
        print(f"  Non-home design types: {len(design_type_counts)}")

        return home_scenes, non_home_scenes, design_type_counts

    def load_all_scenes(self, base_path: Path) -> Dict[str, Dict]:
        """Load all scene data and statistics.

        Args:
            base_path: Base directory containing scene folders

        Returns:
            Dictionary mapping scene_id to scene data and statistics
        """
        print("Loading scene data...")
        all_scenes: Dict[str, Dict] = {}

        scene_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        total_dirs = len(scene_dirs)

        for idx, scene_dir in enumerate(scene_dirs, 1):
            scene_id = scene_dir.name

            # Find trajectory and statistic files
            trajectory_files = list(scene_dir.glob("trajectories_overall_*.json"))
            statistic_files = list(scene_dir.glob("trajectories_statistic_*.json"))

            if trajectory_files and statistic_files:
                try:
                    # Load trajectory data
                    with trajectory_files[0].open("r", encoding="utf-8") as f:
                        trajectory_data = json.load(f)

                    # Load statistics
                    with statistic_files[0].open("r", encoding="utf-8") as f:
                        statistics_data = json.load(f)

                    all_scenes[scene_id] = {
                        "trajectory_data": trajectory_data,
                        "statistics": statistics_data,
                    }

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[WARN] Failed to load scene {scene_id}: {e}")
                    continue

            if idx % self.progress_interval == 0 or idx == total_dirs:
                print(f"[LOAD] Processed {idx}/{total_dirs} scene folders, {len(all_scenes)} valid scenes")

        print(f"Loaded {len(all_scenes)} scenes")
        self.all_scenes = all_scenes
        return all_scenes

    def calculate_total_instruction_pairs(self, scene_data: Dict) -> int:
        """Calculate total instruction pairs for a scene.

        Args:
            scene_data: Scene data dictionary

        Returns:
            Total number of instruction pairs
        """
        return scene_data["statistics"]["scene_summary"]["instruction_stats"]["total_instructions"]

    def select_balanced_scenes(
        self, available_scenes: List[str], target_scenes: int, target_pairs: int, split_name: str
    ) -> Tuple[List[str], Dict]:
        """Select balanced scene set (considering home/non-home balance).

        Args:
            available_scenes: List of available scene IDs
            target_scenes: Target number of scenes
            target_pairs: Target number of instruction pairs
            split_name: Name of the split (for logging)

        Returns:
            (selected_scenes, scene_details) tuple
        """
        print(f"\n=== Selecting {split_name} scenes (target: {target_scenes} scenes, {target_pairs:,} pairs) ===")

        # Group by scene type
        home_scenes = [s for s in available_scenes if self.scene_type_map[s] == "home"]
        non_home_scenes = [s for s in available_scenes if self.scene_type_map[s] == "non_home"]

        print(f"Available scenes: {len(home_scenes)} home, {len(non_home_scenes)} non-home")

        # Calculate proportional allocation
        total_available = len(available_scenes)
        home_ratio = len(home_scenes) / total_available if total_available > 0 else 0
        non_home_ratio = len(non_home_scenes) / total_available if total_available > 0 else 0

        target_home_scenes = max(1, int(target_scenes * home_ratio))
        target_non_home_scenes = max(1, target_scenes - target_home_scenes)

        # Ensure we don't exceed available counts
        target_home_scenes = min(target_home_scenes, len(home_scenes))
        target_non_home_scenes = min(target_non_home_scenes, len(non_home_scenes))

        print(f"Target allocation: {target_home_scenes} home, {target_non_home_scenes} non-home")

        # Select scenes by size (prefer medium-sized scenes)
        def select_scenes_by_size(scene_list: List[str], target_count: int) -> List[str]:
            if not scene_list or target_count <= 0:
                return []

            # Sort by instruction pair count
            scene_pairs = []
            for scene_id in scene_list:
                pair_count = self.calculate_total_instruction_pairs(self.all_scenes[scene_id])
                scene_pairs.append((scene_id, pair_count))

            scene_pairs.sort(key=lambda x: x[1])

            # Select medium-sized scenes
            total_scenes = len(scene_pairs)
            if total_scenes <= target_count:
                return [s[0] for s in scene_pairs]

            start_idx = max(0, total_scenes // 4)
            end_idx = min(total_scenes, start_idx + target_count * 2)

            candidates = scene_pairs[start_idx:end_idx]
            random.shuffle(candidates)

            return [s[0] for s in candidates[:target_count]]

        selected_home = select_scenes_by_size(home_scenes, target_home_scenes)
        selected_non_home = select_scenes_by_size(non_home_scenes, target_non_home_scenes)

        selected_scenes = selected_home + selected_non_home

        # If not enough, add from remaining scenes
        if len(selected_scenes) < target_scenes:
            remaining_scenes = [s for s in available_scenes if s not in selected_scenes]
            additional_needed = target_scenes - len(selected_scenes)
            additional_scenes = random.sample(
                remaining_scenes, min(additional_needed, len(remaining_scenes))
            )
            selected_scenes.extend(additional_scenes)

        # Calculate actual instruction pairs
        total_pairs = 0
        scene_details: Dict[str, Dict] = {}

        for scene_id in selected_scenes:
            pair_count = self.calculate_total_instruction_pairs(self.all_scenes[scene_id])
            total_pairs += pair_count
            scene_details[scene_id] = {
                "scene_id": scene_id,
                "total_instruction_pairs": pair_count,
                "scene_type": self.scene_type_map[scene_id],
                "design_type": self.scene_design_type_map[scene_id],
            }

        # Count final distribution
        final_home = len([s for s in selected_scenes if self.scene_type_map[s] == "home"])
        final_non_home = len([s for s in selected_scenes if self.scene_type_map[s] == "non_home"])

        print(f"Final selection: {len(selected_scenes)} scenes, {total_pairs:,} instruction pairs")
        print(f"Type distribution: {final_home} home, {final_non_home} non-home")

        return selected_scenes, scene_details

    def allocate_trajectory_unseen_from_train_scenes(
        self, train_scenes: List[str], target_pairs: int, num_scenes: int = DEFAULT_TRAJECTORY_UNSEEN_SCENES
    ) -> Dict:
        """Allocate Trajectory-Unseen test data from training scenes.

        Args:
            train_scenes: List of training scene IDs
            target_pairs: Target number of instruction pairs
            num_scenes: Number of scenes to use

        Returns:
            Dictionary mapping scene_id to trajectory details
        """
        print(f"\n=== Allocating Trajectory-Unseen data from training scenes (target: {target_pairs:,} pairs) ===")

        # Randomly select training scenes
        random.shuffle(train_scenes)
        selected_scenes = train_scenes[:num_scenes]

        trajectory_details: Dict[str, Dict] = {}
        current_pairs = 0

        for scene_id in selected_scenes:
            if current_pairs >= target_pairs:
                break

            scene_data = self.all_scenes[scene_id]
            trajectory_details_data = scene_data["statistics"]["trajectory_details"]

            # Allocate target pairs for this scene
            remaining_pairs = target_pairs - current_pairs
            scene_target_pairs = min(remaining_pairs, target_pairs // len(selected_scenes))

            # Randomly select trajectories (about 30%)
            trajectory_ids = list(trajectory_details_data.keys())
            random.shuffle(trajectory_ids)

            num_trajectories_for_test = max(1, len(trajectory_ids) // 3)
            test_trajectory_ids = trajectory_ids[:num_trajectories_for_test]

            selected_trajectories = []
            scene_pairs = 0

            for traj_id in test_trajectory_ids:
                if scene_pairs >= scene_target_pairs:
                    break

                traj_data = trajectory_details_data[traj_id]
                traj_instructions = traj_data["total_instructions"]

                selected_trajectories.append(
                    {
                        "trajectory_id": traj_id,
                        "instruction_count": traj_instructions,
                        "length_category": traj_data["length_category"],
                        "instruction_types": traj_data["instruction_types_count"],
                        "test_type": "trajectory_unseen",
                    }
                )

                scene_pairs += traj_instructions

                if scene_pairs >= scene_target_pairs:
                    break

            if selected_trajectories:
                trajectory_details[scene_id] = {
                    "scene_id": scene_id,
                    "trajectories": selected_trajectories,
                    "total_instruction_pairs": scene_pairs,
                    "test_type": "trajectory_unseen",
                    "scene_type": self.scene_type_map[scene_id],
                    "design_type": self.scene_design_type_map[scene_id],
                }
                current_pairs += scene_pairs

        print(f"Trajectory-Unseen: Selected trajectories from {len(trajectory_details)} scenes, {current_pairs:,} pairs")
        return trajectory_details

    def allocate_instruction_unseen_from_train_scenes(
        self,
        train_scenes: List[str],
        trajectory_unseen_details: Dict,
        target_pairs: int,
        num_scenes: int = DEFAULT_INSTRUCTION_UNSEEN_SCENES,
    ) -> Dict:
        """Allocate Instruction-Unseen test data from training scenes.

        Args:
            train_scenes: List of training scene IDs
            trajectory_unseen_details: Trajectory-unseen details (to avoid conflicts)
            target_pairs: Target number of instruction pairs
            num_scenes: Number of scenes to use

        Returns:
            Dictionary mapping scene_id to instruction details
        """
        print(f"\n=== Allocating Instruction-Unseen data from training scenes (target: {target_pairs:,} pairs) ===")

        # Exclude scenes already used by Trajectory-Unseen
        trajectory_unseen_scenes = set(trajectory_unseen_details.keys())
        available_scenes = [s for s in train_scenes if s not in trajectory_unseen_scenes]

        # Randomly select scenes
        random.shuffle(available_scenes)
        selected_scenes = available_scenes[:num_scenes]

        instruction_details: Dict[str, Dict] = {}
        current_pairs = 0

        for scene_id in selected_scenes:
            if current_pairs >= target_pairs:
                break

            scene_data = self.all_scenes[scene_id]
            trajectory_details_data = scene_data["statistics"]["trajectory_details"]

            # Allocate target pairs for this scene
            remaining_pairs = target_pairs - current_pairs
            scene_target_pairs = min(remaining_pairs, target_pairs // len(selected_scenes))

            selected_instructions = []
            scene_pairs = 0

            trajectory_ids = list(trajectory_details_data.keys())
            random.shuffle(trajectory_ids)

            for traj_id in trajectory_ids:
                if scene_pairs >= scene_target_pairs:
                    break

                traj_data = trajectory_details_data[traj_id]
                total_instructions = traj_data["total_instructions"]

                # Select about 30% of instructions
                num_instructions_to_select = min(total_instructions // 3 + 1, scene_target_pairs - scene_pairs)

                if num_instructions_to_select > 0:
                    instruction_indices = list(range(total_instructions))
                    random.shuffle(instruction_indices)
                    selected_indices = instruction_indices[:num_instructions_to_select]

                    selected_instructions.append(
                        {
                            "trajectory_id": traj_id,
                            "selected_instruction_indices": sorted(selected_indices),
                            "instruction_count": len(selected_indices),
                            "length_category": traj_data["length_category"],
                            "instruction_types": traj_data["instruction_types_count"],
                            "test_type": "instruction_unseen",
                        }
                    )

                    scene_pairs += len(selected_indices)

            if selected_instructions:
                instruction_details[scene_id] = {
                    "scene_id": scene_id,
                    "trajectories": selected_instructions,
                    "total_instruction_pairs": scene_pairs,
                    "test_type": "instruction_unseen",
                    "scene_type": self.scene_type_map[scene_id],
                    "design_type": self.scene_design_type_map[scene_id],
                }
                current_pairs += scene_pairs

        print(f"Instruction-Unseen: Selected instructions from {len(instruction_details)} scenes, {current_pairs:,} pairs")
        return instruction_details

    def create_train_details_with_exclusions(
        self, train_scenes: List[str], trajectory_unseen_details: Dict, instruction_unseen_details: Dict
    ) -> Dict:
        """Create training set details, excluding trajectories and instructions used by test sets.

        Args:
            train_scenes: List of training scene IDs
            trajectory_unseen_details: Trajectory-unseen details
            instruction_unseen_details: Instruction-unseen details

        Returns:
            Dictionary mapping scene_id to training details
        """
        print(f"\n=== Creating training set (excluding test data) ===")

        # Collect excluded trajectories and instructions
        excluded_trajectories: Dict[str, Set[str]] = defaultdict(set)
        excluded_instructions: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))

        # Process Trajectory-Unseen trajectories
        for scene_id, scene_data in trajectory_unseen_details.items():
            for traj_data in scene_data["trajectories"]:
                excluded_trajectories[scene_id].add(traj_data["trajectory_id"])

        # Process Instruction-Unseen instructions
        for scene_id, scene_data in instruction_unseen_details.items():
            for traj_data in scene_data["trajectories"]:
                traj_id = traj_data["trajectory_id"]
                excluded_indices = set(traj_data["selected_instruction_indices"])
                excluded_instructions[scene_id][traj_id] = excluded_indices

        train_details: Dict[str, Dict] = {}

        for scene_id in train_scenes:
            scene_data = self.all_scenes[scene_id]
            trajectory_details_data = scene_data["statistics"]["trajectory_details"]

            scene_trajectories = []

            for traj_id, traj_data in trajectory_details_data.items():
                # Skip if trajectory is used by Trajectory-Unseen
                if traj_id in excluded_trajectories[scene_id]:
                    continue

                # Exclude instructions used by Instruction-Unseen
                excluded_instruction_indices = excluded_instructions[scene_id][traj_id]
                total_instructions = traj_data["total_instructions"]

                if excluded_instruction_indices:
                    available_indices = [
                        i for i in range(total_instructions) if i not in excluded_instruction_indices
                    ]
                else:
                    available_indices = list(range(total_instructions))

                if available_indices:
                    scene_trajectories.append(
                        {
                            "trajectory_id": traj_id,
                            "available_instruction_indices": available_indices,
                            "instruction_count": len(available_indices),
                            "length_category": traj_data["length_category"],
                            "instruction_types": traj_data["instruction_types_count"],
                        }
                    )

            if scene_trajectories:
                train_details[scene_id] = {
                    "scene_id": scene_id,
                    "trajectories": scene_trajectories,
                    "total_instruction_pairs": sum(t["instruction_count"] for t in scene_trajectories),
                    "scene_type": self.scene_type_map[scene_id],
                    "design_type": self.scene_design_type_map[scene_id],
                }

        train_pairs = sum(s["total_instruction_pairs"] for s in train_details.values())
        print(f"Training set: {len(train_details)} scenes, {train_pairs:,} instruction pairs")

        return train_details

    def create_domain_balanced_splits(
        self,
        target_scenes_per_test: int = DEFAULT_TARGET_SCENES_PER_TEST,
        target_pairs_per_test: int = DEFAULT_TARGET_PAIRS_PER_TEST,
        val_scenes: int = DEFAULT_VAL_SCENES,
        val_pairs: int = DEFAULT_VAL_PAIRS,
    ) -> Dict:
        """Create domain-balanced data splits.

        Args:
            target_scenes_per_test: Target number of scenes per test set
            target_pairs_per_test: Target number of instruction pairs per test set
            val_scenes: Target number of validation scenes
            val_pairs: Target number of validation instruction pairs

        Returns:
            Dictionary containing all splits
        """
        print(f"\n{'=' * 60}")
        print("    Creating domain-balanced data splits")
        print("=" * 60)

        # Calculate total data
        total_pairs = sum(
            self.calculate_total_instruction_pairs(data) for data in self.all_scenes.values()
        )
        all_scene_ids = list(self.all_scenes.keys())

        print(f"Total scenes: {len(all_scene_ids)}")
        print(f"Total instruction pairs: {total_pairs:,}")

        # 1. Select Scene-Unseen test set (independent scenes)
        scene_unseen_scenes, scene_unseen_details = self.select_balanced_scenes(
            all_scene_ids, target_scenes_per_test, target_pairs_per_test * 2, "Scene-Unseen test"
        )

        # 2. Remaining scenes as training pool
        train_scene_pool = [s for s in all_scene_ids if s not in scene_unseen_scenes]

        print(f"\nTraining scene pool: {len(train_scene_pool)} scenes")

        # 3. Select validation scenes from training pool
        val_scenes_list, val_details = self.select_balanced_scenes(
            train_scene_pool, val_scenes, val_pairs, "Validation"
        )

        # 4. Remaining training scenes
        pure_train_scenes = [s for s in train_scene_pool if s not in val_scenes_list]

        # 5. Allocate Trajectory-Unseen and Instruction-Unseen from training scenes
        trajectory_unseen_details = self.allocate_trajectory_unseen_from_train_scenes(
            pure_train_scenes, target_pairs_per_test
        )
        instruction_unseen_details = self.allocate_instruction_unseen_from_train_scenes(
            pure_train_scenes, trajectory_unseen_details, target_pairs_per_test
        )

        # 6. Create training set details
        train_details = self.create_train_details_with_exclusions(
            pure_train_scenes, trajectory_unseen_details, instruction_unseen_details
        )

        print(f"\nFinal allocation:")
        print(f"Scene-Unseen: {len(scene_unseen_scenes)} independent scenes")
        print(f"Training: {len(pure_train_scenes)} scenes")
        print(f"Validation: {len(val_scenes_list)} scenes")
        print(f"Total: {len(scene_unseen_scenes) + len(pure_train_scenes) + len(val_scenes_list)} scenes")

        return {
            "train": train_details,
            "val": val_details,
            "scene_unseen": scene_unseen_details,
            "trajectory_unseen": trajectory_unseen_details,
            "instruction_unseen": instruction_unseen_details,
        }

    def calculate_split_statistics(self, split_data: Dict, split_name: str) -> Dict:
        """Calculate split statistics (including scene type distribution).

        Args:
            split_data: Split data dictionary
            split_name: Name of the split

        Returns:
            Statistics dictionary
        """
        total_pairs = 0
        length_distribution = Counter()
        instruction_type_distribution = Counter()
        scene_type_distribution = Counter()
        design_type_distribution = Counter()

        for scene_id, scene_data in split_data.items():
            # Statistics by scene type
            scene_type = scene_data.get("scene_type", self.scene_type_map.get(scene_id, "unknown"))
            design_type = scene_data.get("design_type", self.scene_design_type_map.get(scene_id, "unknown"))

            if "trajectories" in scene_data:
                # Has trajectory details
                for traj_data in scene_data["trajectories"]:
                    instruction_count = traj_data["instruction_count"]
                    total_pairs += instruction_count

                    # Length distribution
                    length_category = traj_data["length_category"]
                    length_distribution[length_category] += instruction_count

                    # Instruction type distribution
                    for inst_type, count in traj_data["instruction_types"].items():
                        proportion = instruction_count / sum(traj_data["instruction_types"].values())
                        instruction_type_distribution[inst_type] += int(count * proportion)

                scene_type_distribution[scene_type] += scene_data["total_instruction_pairs"]
                design_type_distribution[design_type] += scene_data["total_instruction_pairs"]
            else:
                # Scene-unseen case
                total_pairs += scene_data["total_instruction_pairs"]
                scene_type_distribution[scene_type] += scene_data["total_instruction_pairs"]
                design_type_distribution[design_type] += scene_data["total_instruction_pairs"]

                # Get distribution from statistics
                scene_stats = self.all_scenes[scene_id]["statistics"]["scene_summary"]
                length_dist = scene_stats["length_categories"]
                inst_type_dist = scene_stats["most_common_instruction_types"]

                for category, count in length_dist.items():
                    length_distribution[category] += count

                for inst_type, count in inst_type_dist.items():
                    instruction_type_distribution[inst_type] += count

        return {
            "total_scenes": len(split_data),
            "total_instruction_pairs": total_pairs,
            "length_distribution": dict(length_distribution),
            "instruction_type_distribution": dict(instruction_type_distribution),
            "scene_type_distribution": dict(scene_type_distribution),
            "design_type_distribution": dict(design_type_distribution),
        }

    def save_split_files(self, all_splits: Dict, output_dir: Path):
        """Save all split files.

        Args:
            all_splits: Dictionary containing all splits
            output_dir: Output directory
        """
        print(f"\n=== Saving split files ===")

        split_configs = [
            ("train", "GSNav-Bench_Train_Split_Domain.json"),
            ("val", "GSNav-Bench_Val_Split_Domain.json"),
            ("scene_unseen", "GSNav-Bench_Test_Scene_Unseen_Split_Domain.json"),
            ("trajectory_unseen", "GSNav-Bench_Test_Trajectory_Unseen_Split_Domain.json"),
            ("instruction_unseen", "GSNav-Bench_Test_Instruction_Unseen_Split_Domain.json"),
        ]

        for split_name, filename in split_configs:
            split_data = {
                "split_type": split_name,
                "scenes": all_splits[split_name],
                "statistics": self.calculate_split_statistics(all_splits[split_name], split_name),
            }

            file_path = output_dir / filename
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"Saved: {filename}")

    def print_final_statistics(self, all_splits: Dict):
        """Print final statistics.

        Args:
            all_splits: Dictionary containing all splits
        """
        print(f"\n{'=' * 80}")
        print("                    Final Split Statistics (Domain-Aware)")
        print("=" * 80)

        for split_name in ["train", "val", "scene_unseen", "trajectory_unseen", "instruction_unseen"]:
            stats = self.calculate_split_statistics(all_splits[split_name], split_name)

            scene_type_dist = stats["scene_type_distribution"]
            home_pairs = scene_type_dist.get("home", 0)
            non_home_pairs = scene_type_dist.get("non_home", 0)

            print(
                f"{split_name:20s}: {stats['total_scenes']:>3} scenes, {stats['total_instruction_pairs']:>8,} pairs "
                f"(home: {home_pairs:,}, non-home: {non_home_pairs:,})"
            )

        print("=" * 80)

        # Print scene type balance analysis
        print(f"\n=== Scene Type Balance Analysis ===")
        for split_name in ["train", "val", "scene_unseen", "trajectory_unseen", "instruction_unseen"]:
            stats = self.calculate_split_statistics(all_splits[split_name], split_name)
            scene_type_dist = stats["scene_type_distribution"]
            total = sum(scene_type_dist.values())

            if total > 0:
                home_ratio = scene_type_dist.get("home", 0) / total * 100
                non_home_ratio = scene_type_dist.get("non_home", 0) / total * 100
                print(f"{split_name:20s}: home {home_ratio:.1f}%, non-home {non_home_ratio:.1f}%")


# ==================== CLI Interface ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Domain-aware trajectory data splitter.")

    parser.add_argument(
        "--traj-root",
        type=Path,
        default=DEFAULT_TRAJ_ROOT,
        help="Root directory containing merged trajectory scene folders",
    )
    parser.add_argument(
        "--scene-type-file",
        type=Path,
        default=DEFAULT_SCENE_TYPE_FILE,
        help="Path to scene_type.json file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for split files",
    )
    parser.add_argument(
        "--target-scenes-per-test",
        type=int,
        default=DEFAULT_TARGET_SCENES_PER_TEST,
        help="Target number of scenes per test set",
    )
    parser.add_argument(
        "--target-pairs-per-test",
        type=int,
        default=DEFAULT_TARGET_PAIRS_PER_TEST,
        help="Target number of instruction pairs per test set",
    )
    parser.add_argument(
        "--val-scenes",
        type=int,
        default=DEFAULT_VAL_SCENES,
        help="Target number of validation scenes",
    )
    parser.add_argument(
        "--val-pairs",
        type=int,
        default=DEFAULT_VAL_PAIRS,
        help="Target number of validation instruction pairs",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Print progress after processing N scene folders",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("       Domain-Aware Three-Level Data Splitting")
    print("=" * 80)

    # Validate paths
    if not args.traj_root.exists():
        print(f"[ERROR] Trajectory root does not exist: {args.traj_root}")
        return

    if not args.scene_type_file.exists():
        print(f"[ERROR] Scene type file does not exist: {args.scene_type_file}")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create splitter
    splitter = DomainAwareTrajectoryDataSplitter(
        random_seed=args.random_seed,
        progress_interval=args.progress_interval,
    )

    # 1. Load data
    splitter.load_all_scenes(args.traj_root)

    # 2. Load scene type information
    splitter.load_scene_types(args.scene_type_file)

    # 3. Create domain-balanced splits
    all_splits = splitter.create_domain_balanced_splits(
        target_scenes_per_test=args.target_scenes_per_test,
        target_pairs_per_test=args.target_pairs_per_test,
        val_scenes=args.val_scenes,
        val_pairs=args.val_pairs,
    )

    # 4. Save files
    splitter.save_split_files(all_splits, args.output_dir)

    # 5. Print final statistics
    splitter.print_final_statistics(all_splits)


if __name__ == "__main__":
    main()

