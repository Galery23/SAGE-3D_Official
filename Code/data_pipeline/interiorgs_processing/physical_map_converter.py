#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert InteriorGS labels.json files to physical-map scene.json files.

The script scans every scene folder under the given source root, reads
the original 3DGS labels, and writes compact bounding-box summaries that
match the legacy scene.json format expected by downstream components.

Command-line arguments expose input/output directories as well as common
options such as overwrite behavior and scene filtering so the script can
be imported as a module or invoked directly from the CLI.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple



@dataclass
class Bounds:
    """Axis-aligned bounding box built from eight 3D corner points."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @classmethod
    def from_points(cls, points: Sequence[dict]) -> "Bounds":
        """Create bounds from a list of dicts containing x/y/z keys."""
        if not points:
            raise ValueError("bounding_box is empty; cannot compute bounds.")

        try:
            xs = [float(p["x"]) for p in points]
            ys = [float(p["y"]) for p in points]
            zs = [float(p["z"]) for p in points]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bounding_box point data: {exc}") from exc

        return cls(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def to_string(self, decimals: int = 2) -> str:
        """Return '(min_xyz),(max_xyz)' rounded to the requested decimals."""
        fmt = f"{{:.{decimals}f}}"
        min_part = f"({fmt.format(self.min_x)},{fmt.format(self.min_y)},{fmt.format(self.min_z)})"
        max_part = f"({fmt.format(self.max_x)},{fmt.format(self.max_y)},{fmt.format(self.max_z)})"
        return f"{min_part},{max_part}"


def build_scene_entries(items: Sequence[dict]) -> Tuple[Dict[str, str], Dict[str, int], int]:
    """
    Convert label entries to the scene.json dict format.

    Returns
    -------
    scene_entries:
        Mapping of "label_index" -> "(min_xyz),(max_xyz)" strings.
    label_counts:
        Per-label counters that help build stable suffixes.
    skipped:
        Count of entries ignored due to missing or invalid bounding boxes.
    """
    scene_entries: Dict[str, str] = {}
    label_counts: Dict[str, int] = defaultdict(int)
    skipped = 0

    for item in items:
        bbox_points = item.get("bounding_box")
        if not isinstance(bbox_points, Sequence):
            skipped += 1
            continue

        label = (item.get("label") or "unknown").strip() or "unknown"
        label_counts[label] += 1
        entry_name = f"{label}_{label_counts[label]}"

        try:
            bounds = Bounds.from_points(bbox_points)
        except ValueError:
            skipped += 1
            continue

        scene_entries[entry_name] = bounds.to_string()

    return scene_entries, label_counts, skipped


def convert_one_scene(labels_path: Path, scene_path: Path, overwrite: bool) -> Tuple[int, int]:
    """Convert a single labels.json file and write its scene.json counterpart."""
    with labels_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict) and "labels" in data:
        items = data["labels"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"{labels_path} must contain a list or dict with a 'labels' key.")

    scene_entries, _, skipped = build_scene_entries(items)

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    if scene_path.exists() and not overwrite:
        raise FileExistsError(f"{scene_path} already exists. Use --overwrite to regenerate.")

    with scene_path.open("w", encoding="utf-8") as fh:
        json.dump(scene_entries, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return len(scene_entries), skipped


def iter_scene_dirs(src_root: Path, only: Iterable[str] | None = None) -> List[Path]:
    """Return sorted scene folders, optionally filtered by explicit names."""
    if only:
        return [src_root / name for name in only]
    return sorted(p for p in src_root.iterdir() if p.is_dir())


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    overwrite: bool = False,
    limit: int | None = None,
    only: Iterable[str] | None = None,
) -> None:
    """High-level helper that processes multiple scene folders in batch."""
    if not src_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)
    scene_dirs = iter_scene_dirs(src_root, only)
    if limit is not None:
        scene_dirs = scene_dirs[:limit]

    if not scene_dirs:
        print(f"[WARN] No scene directories found under {src_root}")
        return

    processed = 0
    skipped_instances = 0

    for scene_dir in scene_dirs:
        labels_path = scene_dir / "labels.json"
        if not labels_path.exists():
            print(f"[WARN] {scene_dir.name}: labels.json not found, skipping.")
            continue

        dst_scene_path = dst_root / scene_dir.name / "scene.json"

        try:
            count, skipped = convert_one_scene(labels_path, dst_scene_path, overwrite=overwrite)
        except FileExistsError as exc:
            print(f"[SKIP] {scene_dir.name}: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[FAIL] {scene_dir.name}: {exc}")
            continue

        processed += 1
        skipped_instances += skipped
        print(f"[OK] {scene_dir.name}: wrote {count} entries, skipped {skipped} -> {dst_scene_path}")

    print(
        f"\nFinished: processed {processed} scenes with {skipped_instances} skipped instances."
        f" Output root: {dst_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert InteriorGS labels.json files into compact physical-map scene.json files."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        help=f"Root directory containing scene subfolders with labels.json (default: {DEFAULT_SRC_ROOT})",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        help=f"Directory where scene.json outputs will be saved (default: {DEFAULT_DST_ROOT})",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional list of scene folder names to process (e.g., 0001_839920 0002_839955).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many scenes (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene.json files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        src_root=args.src_root.expanduser(),
        dst_root=args.dst_root.expanduser(),
        overwrite=args.overwrite,
        limit=args.limit,
        only=args.only,
    )


if __name__ == "__main__":
    main()










