#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Statistics: Analyze merged VLN trajectory datasets.

This script computes per-scene and global statistics for trajectory files produced after
merging (e.g., `trajectories_overall_*.json`). Results support downstream dataset splits.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

# ==================== Default Configuration ====================

DEFAULT_DATA_DIR = Path("Data/trajectories_merged")
DEFAULT_GLOBAL_SUMMARY = "global_trajectory_summary.json"


# ==================== Statistic Helper Class ====================


class TrajectoryStatistics:
    """Trajectory dataset statistic analyzer."""

    def __init__(self, data_dir: Path):
        """Initialize analyzer."""
        self.data_dir = data_dir
        self.global_stats = {
            "total_scenes": 0,
            "total_trajectories": 0,
            "trajectory_lengths": [],
            "start_end_pairs": Counter(),
            "instruction_types": Counter(),
            "unique_starts": set(),
            "unique_ends": set(),
            "scene_trajectory_counts": [],
        }

    # ---------- Helpers ----------

    def extract_trajectory_info(self, sample: Dict) -> Dict:
        """Extract details from a single trajectory sample."""
        traj_info = {
            "trajectory_id": sample.get("trajectory_id", ""),
            "start_end_pairs": [],
            "instruction_types_count": {},
            "path_length": len(sample.get("points", [])),
            "unique_starts": set(),
            "unique_ends": set(),
            "instruction_word_counts": [],
            "total_instructions": 0,
        }

        instructions = sample.get("instructions", [])
        traj_info["total_instructions"] = len(instructions)

        for instruction in instructions:
            start = instruction.get("start", "")
            end = instruction.get("end", "")
            inst_type = instruction.get("instruction_type", "")
            text = instruction.get("generated_instruction", "")

            if start and end:
                pair = f"{start} -> {end}"
                traj_info["start_end_pairs"].append(pair)
                traj_info["unique_starts"].add(start)
                traj_info["unique_ends"].add(end)

            if inst_type:
                traj_info["instruction_types_count"][inst_type] = (
                    traj_info["instruction_types_count"].get(inst_type, 0) + 1
                )

            if text:
                traj_info["instruction_word_counts"].append(len(text.split()))

        return traj_info

    def calculate_length_thresholds(self, lengths: List[int]) -> Dict[str, int]:
        """Compute thresholds for length categorization."""
        if not lengths:
            return {"short": 10, "long": 50}

        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        short_th = sorted_lengths[n // 3] if n >= 3 else min(sorted_lengths)
        long_th = sorted_lengths[2 * n // 3] if n >= 3 else max(sorted_lengths)
        return {"short": short_th, "long": long_th}

    def categorize_length(self, length: int, thresholds: Dict[str, int]) -> str:
        """Categorize trajectory length."""
        if length <= thresholds["short"]:
            return "short"
        if length <= thresholds["long"]:
            return "middle"
        return "long"

    # ---------- Scene-Level Analysis ----------

    def analyze_scene(self, scene_folder: Path) -> Dict | None:
        """Analyze a single scene directory."""
        scene_name = scene_folder.name
        print(f"[Scene] {scene_name}")

        trajectory_files = list(scene_folder.glob("trajectories_overall_*.json"))
        if not trajectory_files:
            print(f"  [WARN] No trajectories_overall_*.json found")
            return None

        trajectory_file = trajectory_files[0]
        try:
            with trajectory_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  [ERROR] Failed to read {trajectory_file.name}: {exc}")
            return None

        scene_stats = {
            "scene_summary": {
                "scene_name": scene_name,
                "scene_id": data.get("scenes", [{}])[0].get("scene_id", ""),
                "total_trajectories": 0,
                "trajectory_length_stats": {
                    "avg_length": 0,
                    "min_length": 0,
                    "max_length": 0,
                    "median_length": 0,
                    "std_length": 0,
                },
                "length_categories": {"short": 0, "middle": 0, "long": 0},
                "instruction_stats": {
                    "avg_instructions_per_trajectory": 0,
                    "total_instructions": 0,
                    "avg_words_per_instruction": 0,
                    "total_words": 0,
                },
                "location_stats": {
                    "unique_starts": [],
                    "unique_ends": [],
                    "unique_start_count": 0,
                    "unique_end_count": 0,
                    "unique_pairs_count": 0,
                },
                "most_common_pairs": {},
                "most_common_instruction_types": {},
            },
            "trajectory_details": {},
        }

        temp_data = {
            "start_end_pairs": Counter(),
            "instruction_types": Counter(),
            "unique_starts": set(),
            "unique_ends": set(),
            "trajectory_lengths": [],
            "instructions_per_traj": [],
            "instruction_word_counts": [],
            "unique_pairs": set(),
        }

        scenes = data.get("scenes", [])
        if scenes:
            samples = scenes[0].get("samples", [])
            scene_stats["scene_summary"]["total_trajectories"] = len(samples)

            for sample in samples:
                traj_info = self.extract_trajectory_info(sample)
                traj_id = traj_info["trajectory_id"]

                scene_stats["trajectory_details"][traj_id] = {
                    "trajectory_id": traj_id,
                    "path_length": traj_info["path_length"],
                    "total_instructions": traj_info["total_instructions"],
                    "instruction_types_count": traj_info["instruction_types_count"],
                    "instruction_word_counts": traj_info["instruction_word_counts"],
                    "avg_words_per_instruction": (
                        sum(traj_info["instruction_word_counts"])
                        / len(traj_info["instruction_word_counts"])
                        if traj_info["instruction_word_counts"]
                        else 0
                    ),
                    "total_words": sum(traj_info["instruction_word_counts"]),
                    "start_end_pairs": traj_info["start_end_pairs"],
                    "unique_starts": list(traj_info["unique_starts"]),
                    "unique_ends": list(traj_info["unique_ends"]),
                }

                for pair in traj_info["start_end_pairs"]:
                    temp_data["start_end_pairs"][pair] += 1
                    temp_data["unique_pairs"].add(pair)

                for inst_type, count in traj_info["instruction_types_count"].items():
                    temp_data["instruction_types"][inst_type] += count

                temp_data["unique_starts"].update(traj_info["unique_starts"])
                temp_data["unique_ends"].update(traj_info["unique_ends"])
                temp_data["trajectory_lengths"].append(traj_info["path_length"])
                temp_data["instructions_per_traj"].append(traj_info["total_instructions"])
                temp_data["instruction_word_counts"].extend(traj_info["instruction_word_counts"])

        # Length statistics and categorization
        if temp_data["trajectory_lengths"]:
            lengths = temp_data["trajectory_lengths"]
            summary = scene_stats["scene_summary"]["trajectory_length_stats"]
            summary.update(
                {
                    "avg_length": statistics.mean(lengths),
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                    "median_length": statistics.median(lengths),
                    "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0,
                }
            )

            thresholds = self.calculate_length_thresholds(lengths)
            for length in lengths:
                category = self.categorize_length(length, thresholds)
                scene_stats["scene_summary"]["length_categories"][category] += 1

            for traj_id, traj_data in scene_stats["trajectory_details"].items():
                traj_data["length_category"] = self.categorize_length(
                    traj_data["path_length"], thresholds
                )

        # Instruction stats
        if temp_data["instructions_per_traj"]:
            scene_stats["scene_summary"]["instruction_stats"].update(
                {
                    "avg_instructions_per_trajectory": statistics.mean(
                        temp_data["instructions_per_traj"]
                    ),
                    "total_instructions": sum(temp_data["instructions_per_traj"]),
                }
            )

        if temp_data["instruction_word_counts"]:
            scene_stats["scene_summary"]["instruction_stats"].update(
                {
                    "avg_words_per_instruction": statistics.mean(
                        temp_data["instruction_word_counts"]
                    ),
                    "total_words": sum(temp_data["instruction_word_counts"]),
                }
            )

        # Location stats
        scene_stats["scene_summary"]["location_stats"].update(
            {
                "unique_starts": list(temp_data["unique_starts"]),
                "unique_ends": list(temp_data["unique_ends"]),
                "unique_start_count": len(temp_data["unique_starts"]),
                "unique_end_count": len(temp_data["unique_ends"]),
                "unique_pairs_count": len(temp_data["unique_pairs"]),
            }
        )

        scene_stats["scene_summary"]["most_common_pairs"] = dict(
            temp_data["start_end_pairs"].most_common(10)
        )
        scene_stats["scene_summary"]["most_common_instruction_types"] = dict(
            temp_data["instruction_types"].most_common()
        )

        return scene_stats

    # ---------- Saving & Global Aggregation ----------

    def save_scene_statistics(self, scene_stats: Dict, output_path: Path) -> None:
        """Save per-scene statistics to disk."""
        try:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(scene_stats, f, ensure_ascii=False, indent=2, default=str)
            print(f"  [SAVED] {output_path.name}")
        except Exception as exc:
            print(f"  [ERROR] Failed to save {output_path.name}: {exc}")

    def update_global_stats(self, scene_stats: Dict) -> None:
        """Update global counters with scene statistics."""
        if not scene_stats:
            return

        summary = scene_stats["scene_summary"]
        details = scene_stats["trajectory_details"]

        self.global_stats["total_scenes"] += 1
        self.global_stats["total_trajectories"] += summary["total_trajectories"]
        self.global_stats["scene_trajectory_counts"].append(summary["total_trajectories"])

        for traj_data in details.values():
            self.global_stats["trajectory_lengths"].append(traj_data["path_length"])

        for pair, count in summary["most_common_pairs"].items():
            self.global_stats["start_end_pairs"][pair] += count

        for inst_type, count in summary["most_common_instruction_types"].items():
            self.global_stats["instruction_types"][inst_type] += count

        self.global_stats["unique_starts"].update(summary["location_stats"]["unique_starts"])
        self.global_stats["unique_ends"].update(summary["location_stats"]["unique_ends"])

    def generate_global_summary(self) -> Dict:
        """Generate overall dataset summary."""
        global_summary = {
            "total_scenes": self.global_stats["total_scenes"],
            "total_trajectories": self.global_stats["total_trajectories"],
            "avg_trajectories_per_scene": 0,
            "trajectory_length_stats": {},
            "scene_trajectory_distribution": {},
            "global_start_end_pairs": {},
            "global_instruction_types": {},
            "unique_locations": {
                "starts": list(self.global_stats["unique_starts"]),
                "ends": list(self.global_stats["unique_ends"]),
                "start_count": len(self.global_stats["unique_starts"]),
                "end_count": len(self.global_stats["unique_ends"]),
            },
        }

        if self.global_stats["total_scenes"] > 0:
            global_summary["avg_trajectories_per_scene"] = (
                self.global_stats["total_trajectories"] / self.global_stats["total_scenes"]
            )

        lengths = self.global_stats["trajectory_lengths"]
        if lengths:
            stats = {
                "mean": statistics.mean(lengths),
                "median": statistics.median(lengths),
                "min": min(lengths),
                "max": max(lengths),
                "std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
                "total_count": len(lengths),
            }
            thresholds = self.calculate_length_thresholds(lengths)
            categories = {"short": 0, "middle": 0, "long": 0}
            for length in lengths:
                categories[self.categorize_length(length, thresholds)] += 1
            stats["categories"] = categories
            stats["thresholds"] = thresholds
            global_summary["trajectory_length_stats"] = stats

        counts = self.global_stats["scene_trajectory_counts"]
        if counts:
            global_summary["scene_trajectory_distribution"] = {
                "mean": statistics.mean(counts),
                "median": statistics.median(counts),
                "min": min(counts),
                "max": max(counts),
                "std": statistics.stdev(counts) if len(counts) > 1 else 0,
            }

        global_summary["global_start_end_pairs"] = {
            "total_unique_pairs": len(self.global_stats["start_end_pairs"]),
            "most_common": dict(self.global_stats["start_end_pairs"].most_common(20)),
        }
        global_summary["global_instruction_types"] = {
            "total_types": len(self.global_stats["instruction_types"]),
            "distribution": dict(self.global_stats["instruction_types"]),
        }

        return global_summary

    # ---------- Main Routine ----------

    def analyze_all_scenes(
        self, only: List[str] | None = None, skip_existing: bool = True, summary_name: str = DEFAULT_GLOBAL_SUMMARY
    ) -> None:
        """Analyze all scene folders under data_dir."""
        print(f"[INFO] Data directory: {self.data_dir}")

        scene_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if only:
            scene_folders = [d for d in scene_folders if d.name in only]

        print(f"[INFO] Found {len(scene_folders)} scene folders")

        success = error = skipped = 0

        for scene_folder in scene_folders:
            try:
                traj_files = list(scene_folder.glob("trajectories_overall_*.json"))
                if traj_files:
                    parts = traj_files[0].stem.replace("trajectories_overall_", "")
                    stats_filename = f"trajectories_statistic_{parts}.json"
                else:
                    stats_filename = f"trajectories_statistic_{scene_folder.name}.json"

                stats_path = scene_folder / stats_filename

                if skip_existing and stats_path.exists():
                    print(f"[Scene] {scene_folder.name}")
                    print(f"  [SKIP] Statistic file exists: {stats_filename}")
                    skipped += 1

                    try:
                        with stats_path.open("r", encoding="utf-8") as f:
                            existing_stats = json.load(f)
                        self.update_global_stats(existing_stats)
                        success += 1
                    except Exception as exc:
                        print(f"  [WARN] Failed to load existing stats: {exc}")
                        error += 1
                    continue

                scene_stats = self.analyze_scene(scene_folder)
                if scene_stats:
                    self.update_global_stats(scene_stats)
                    self.save_scene_statistics(scene_stats, stats_path)
                    success += 1

                    summary = scene_stats["scene_summary"]
                    print(
                        f"  Trajectory count: {summary['total_trajectories']}, "
                        f"Avg length: {summary['trajectory_length_stats']['avg_length']:.1f}, "
                        f"Length distribution: short {summary['length_categories']['short']}, "
                        f"middle {summary['length_categories']['middle']}, "
                        f"long {summary['length_categories']['long']}"
                    )
                else:
                    error += 1
            except Exception as exc:
                print(f"[ERROR] Failed to process {scene_folder.name}: {exc}")
                error += 1

        # Global summary
        print("\n[INFO] Generating global summary...")
        global_summary = self.generate_global_summary()
        summary_path = self.data_dir / summary_name
        try:
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(global_summary, f, ensure_ascii=False, indent=2, default=str)
            print(f"[SAVED] Global summary: {summary_path}")
        except Exception as exc:
            print(f"[ERROR] Failed to save global summary: {exc}")

        print(
            f"\n[STATS] Success: {success}, Skipped: {skipped}, Errors: {error}, "
            f"Total scenes: {len(scene_folders)}, Processed scenes: {global_summary['total_scenes']}, "
            f"Total trajectories: {global_summary['total_trajectories']}, "
            f"Avg trajectories/scene: {global_summary.get('avg_trajectories_per_scene', 0):.1f}"
        )
        if global_summary.get("trajectory_length_stats"):
            length_stats = global_summary["trajectory_length_stats"]
            print(
                f"Length stats - mean: {length_stats['mean']:.1f}, "
                f"range: {length_stats['min']} - {length_stats['max']}, "
                f"categories: {length_stats['categories']}"
            )
        print(
            f"Unique locations - starts: {global_summary['unique_locations']['start_count']}, "
            f"ends: {global_summary['unique_locations']['end_count']}"
        )


# ==================== CLI Interface ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute statistics for merged VLN trajectories.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing merged trajectory scene folders",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only analyze specified scene names",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute statistics even if per-scene files already exist",
    )
    parser.add_argument(
        "--global-summary-name",
        type=str,
        default=DEFAULT_GLOBAL_SUMMARY,
        help="Filename for the global summary JSON (saved under data-dir)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.data_dir.exists():
        print(f"[ERROR] Data directory does not exist: {args.data_dir}")
        return

    analyzer = TrajectoryStatistics(args.data_dir)
    analyzer.analyze_all_scenes(
        only=args.only,
        skip_existing=not args.force,
        summary_name=args.global_summary_name,
    )


if __name__ == "__main__":
    main()









