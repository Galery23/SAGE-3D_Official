#!/usr/bin/env python3
"""
Episode Adapter Module for SAGE-3D Benchmark.

Converts GVLN trajectory JSON data into standardized episode dictionaries
used by the benchmark evaluator.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


def adapt_gvln_to_episodes(
    gvln_json_path: str,
    scene_usd_path: str,
    goal_radius: float = 0.5,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Convert GVLN JSON into a list of episode dicts used by the evaluator.

    Args:
        gvln_json_path: Path to GVLN trajectory JSON file
        scene_usd_path: Absolute path to the scene USD/USDA file
        goal_radius: Success radius in meters (default: 0.5)
        verbose: Whether to print debug information

    Returns:
        List of episode dictionaries
    """
    gvln_json = Path(gvln_json_path)
    scene_usd = Path(scene_usd_path)
    assert gvln_json.exists(), f"GVLN JSON not found: {gvln_json}"
    assert scene_usd.exists(), f"Scene USD not found: {scene_usd}"

    with open(gvln_json, "r") as f:
        data = json.load(f)

    assert "scenes" in data and len(data["scenes"]) > 0, "Malformed GVLN JSON: missing scenes"
    scene = data["scenes"][0]
    episodes: List[Dict[str, Any]] = []

    for sample in scene.get("samples", []):
        points = sample.get("points", [])
        assert len(points) > 0, "Sample has no points"

        if verbose:
            print(f"[DEBUG] Loading trajectory {sample.get('trajectory_id', '0')} with {len(points)} points")

        gt_locations = [p["position"] for p in points]
        start_position = points[0]["position"]
        start_rotation = points[0]["rotation"]
        goal_position = points[-1]["position"]

        if verbose:
            print(f"[DEBUG] Start position: {start_position}")
            print(f"[DEBUG] Goal position: {goal_position}")
            print(f"[DEBUG] Total GT locations: {len(gt_locations)}")

        # Process both old and new instruction formats
        processed_instructions = _parse_instructions(sample.get("instructions", [""]))

        if verbose:
            print(f"[DEBUG] Total instructions found: {len(processed_instructions)}")

        # Create separate episode for each instruction
        for instr_idx, instruction_text, instruction_type, start_item, end_item in processed_instructions:
            if verbose:
                display_text = instruction_text[:50] + "..." if len(instruction_text) > 50 else instruction_text
                print(f"[DEBUG] Creating episode for instruction {instr_idx}: {display_text}")

            episodes.append({
                "scene_usd": str(scene_usd.resolve()),
                "scene_id": scene.get("scene_id", 0),
                "scene_name": scene.get("scene_name", "scene"),
                "episode_id": f"{sample.get('trajectory_id', '0')}-{instr_idx}",
                "trajectory_id": sample.get("trajectory_id", "0"),
                "instruction_index": instr_idx,
                "instruction": {"instruction_text": instruction_text},
                "instruction_type": instruction_type,
                "start_item": start_item,
                "end_item": end_item,
                "start_position": start_position,
                "start_rotation": start_rotation,
                "goals": [{"radius": goal_radius, "position": goal_position}],
                "gt_locations": gt_locations,
                "reference_path": gt_locations,
            })

    return episodes


def _parse_instructions(instr_list: List[Any]) -> List[Tuple[int, str, str, str, str]]:
    """Parse instruction list supporting both old and new formats.

    Args:
        instr_list: List of instructions (strings or dicts)

    Returns:
        List of tuples: (index, text, type, start_item, end_item)
    """
    if not instr_list:
        return [(0, "", "", "", "")]

    first_instr = instr_list[0]

    if isinstance(first_instr, str):
        # Old format: instructions are string array
        return [(i, instr, "", "", "") for i, instr in enumerate(instr_list)]

    elif isinstance(first_instr, dict) and "generated_instruction" in first_instr:
        # New format: instructions are object array
        processed = []
        for i, instr_obj in enumerate(instr_list):
            if isinstance(instr_obj, dict):
                instruction_text = instr_obj.get("generated_instruction", "")
                instruction_type = instr_obj.get("instruction_type", "")
                start_item = instr_obj.get("start", "")
                end_item = instr_obj.get("end", "")
                processed.append((i, instruction_text, instruction_type, start_item, end_item))
        return processed if processed else [(0, "", "", "", "")]

    else:
        # Unknown format, use defaults
        return [(0, "", "", "", "")]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert GVLN JSON to episode format for benchmark evaluation"
    )
    parser.add_argument(
        "--gvln-json", type=str, required=True,
        help="Path to GVLN trajectory JSON file"
    )
    parser.add_argument(
        "--scene-usd", type=str, required=True,
        help="Path to scene USD/USDA file"
    )
    parser.add_argument(
        "--goal-radius", type=float, default=0.5,
        help="Success radius in meters (default: 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: print to stdout)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose debug output"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI usage."""
    args = parse_args()

    episodes = adapt_gvln_to_episodes(
        gvln_json_path=args.gvln_json,
        scene_usd_path=args.scene_usd,
        goal_radius=args.goal_radius,
        verbose=args.verbose
    )

    print(f"[INFO] Converted {len(episodes)} episodes")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(episodes, f, indent=2)
        print(f"[INFO] Saved episodes to {output_path}")
    else:
        print(json.dumps(episodes, indent=2))


if __name__ == "__main__":
    main()








