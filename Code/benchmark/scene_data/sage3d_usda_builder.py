#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAGE-3D USDA Builder
====================

This utility converts a directory of USDZ scenes into USDA files by cloning a
template stage and swapping scene-specific references.

Key features
------------
- Fully configurable input/output directories through CLI arguments.
- Customizable template placeholder ID with safety checks on occurrence count.
- Optional overrides for the USDZ reference path and the collision payload path.
- Automatic adjustment of the authoring layer to match the generated filename.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def read_template(template_path: Path, base_id: str, expected_count: int) -> str:
    """Load and validate template content."""
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    content = template_path.read_text(encoding="utf-8")
    actual_count = content.count(base_id)
    if actual_count != expected_count:
        raise ValueError(
            f"Placeholder '{base_id}' appears {actual_count} time(s), expected {expected_count}."
        )
    
    # Validate that placeholder strings exist in template
    if "@usdz_root" not in content:
        print("[WARN] Template does not contain '@usdz_root' placeholder")
    if "@collision_root@" not in content:
        print("[WARN] Template does not contain '@collision_root@' placeholder")
    
    return content


def iter_usdz_files(usdz_dir: Path) -> Iterable[Path]:
    """Yield USDZ files."""
    if not usdz_dir.exists():
        raise FileNotFoundError(f"USDZ directory not found: {usdz_dir}")

    for usdz_path in sorted(usdz_dir.glob("*.usdz")):
        yield usdz_path


def replace_placeholder(
    content: str,
    placeholder: str,
    replacement: str,
    label: str,
) -> str:
    """Replace a placeholder string within content.

    Args:
        content: Text to modify.
        placeholder: Placeholder string to search for (e.g., "@usdz_root[gauss.usda]@").
        replacement: Replacement string.
        label: Human readable name for logging.

    Returns:
        Updated content string.
    """
    if placeholder not in content:
        print(f"[WARN] {label} placeholder '{placeholder}' not found in template")
        return content

    occurrences = content.count(placeholder)
    content = content.replace(placeholder, replacement)

    if occurrences > 1:
        print(
            f"[WARN] {label} placeholder found {occurrences} times; "
            "all occurrences were replaced."
        )
    return content


def build_usda_content(
    template_text: str,
    scene_id: str,
    base_id: str,
    usdz_placeholder: str,
    usdz_path_template: str,
    collision_placeholder: str,
    collision_path_template: str,
) -> str:
    """Generate the USDA content for a single scene.
    
    Args:
        template_text: Template file content.
        scene_id: Scene ID to generate USDA for.
        base_id: Base placeholder ID in template (e.g., "839920").
        usdz_placeholder: Placeholder string in template (e.g., "@usdz_root[gauss.usda]@").
        usdz_path_template: Template for USDZ path with {scene_id} placeholder.
        collision_placeholder: Placeholder string in template (e.g., "@collision_root@").
        collision_path_template: Template for collision path with {scene_id} placeholder.
    
    Returns:
        Generated USDA content string.
    """
    content = template_text
    
    # Extract world_id from scene_id (e.g. 0001_839920 -> 839920) for collision path matching
    world_id = scene_id.split("_")[-1] if "_" in scene_id else scene_id
    
    # Replace base_id placeholder (for authoring_layer and any other occurrences)
    content = content.replace(base_id, scene_id)

    # Replace USDZ reference placeholder
    # Support both {scene_id} and {world_id} in templates
    usdz_path = usdz_path_template.format(scene_id=scene_id, world_id=world_id)
    content = replace_placeholder(
        content,
        usdz_placeholder,
        usdz_path,
        label="USDZ reference",
    )

    # Replace collision payload placeholder
    collision_path = collision_path_template.format(scene_id=scene_id, world_id=world_id)
    content = replace_placeholder(
        content,
        collision_placeholder,
        collision_path,
        label="Collision payload",
    )

    # Ensure authoring layer follows "./<scene_id>.usda"
    authoring_search = f'string authoring_layer = "./{scene_id}.usda"'
    if authoring_search not in content:
        authoring_base = f'string authoring_layer = "./{base_id}.usda"'
        replacement = f'string authoring_layer = "./{scene_id}.usda"'
        if authoring_base in content:
            content = content.replace(authoring_base, replacement, 1)
        else:
            print(f"[WARN] Authoring layer token not found for scene {scene_id}")

    return content


def generate_usda_files(
    usdz_dir: Path,
    out_dir: Path,
    template_path: Path,
    base_id: str,
    expected_count: int,
    usdz_placeholder: str,
    usdz_path_template: str,
    collision_placeholder: str,
    collision_path_template: str,
    overwrite: bool,
    limit: Optional[int] = None,
) -> None:
    """Generate USDA files by cloning the template for each USDZ scene.
    
    Args:
        usdz_dir: Directory containing input .usdz files.
        out_dir: Output directory for generated .usda files.
        template_path: Path to template USDA file.
        base_id: Base placeholder ID in template.
        expected_count: Expected count of base_id in template.
        usdz_placeholder: Placeholder string for USDZ reference (e.g., "@usdz_root[gauss.usda]@").
        usdz_path_template: Template for USDZ path with {scene_id} placeholder.
        collision_placeholder: Placeholder string for collision (e.g., "@collision_root@").
        collision_path_template: Template for collision path with {scene_id} placeholder.
        overwrite: Whether to overwrite existing files.
        limit: Limit number of files to generate (for testing).
    """
    template_text = read_template(template_path, base_id, expected_count)

    out_dir.mkdir(parents=True, exist_ok=True)
    usdz_files = list(iter_usdz_files(usdz_dir))
    if not usdz_files:
        print(f"[WARN] No USDZ files found in {usdz_dir}")
        return

    generated = 0
    skipped = 0

    for usdz_path in usdz_files:
        scene_id = usdz_path.stem
        out_path = out_dir / f"{scene_id}.usda"

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path} already exists (use --overwrite to replace)")
            skipped += 1
            continue

        content = build_usda_content(
            template_text,
            scene_id=scene_id,
            base_id=base_id,
            usdz_placeholder=usdz_placeholder,
            usdz_path_template=usdz_path_template,
            collision_placeholder=collision_placeholder,
            collision_path_template=collision_path_template,
        )

        out_path.write_text(content, encoding="utf-8")
        generated += 1
        print(f"[OK] Generated USDA: {out_path}")

        if limit and generated >= limit:
            print(f"[INFO] Limit reached ({limit} files). Stopping early.")
            break

    print(
        f"[SUMMARY] Generated: {generated}, Skipped: {skipped}, "
        f"Output directory: {out_dir}"
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert USDZ assets into USDA files using a template."
    )

    parser.add_argument(
        "--usdz-dir",
        type=Path,
        default=Path("/path/to/data/InteriorGS_usdz"),
        help="Directory containing input .usdz files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/path/to/data/InteriorGS_usda"),
        help="Directory to store generated .usda files.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("Data/template.usda"),
        help="Path to the USDA template file containing the placeholder ID.",
    )
    parser.add_argument(
        "--base-id",
        type=str,
        default="839920",
        help="Placeholder ID present in the template (e.g., 839920) for scene ID replacement.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=1,
        help="Expected number of occurrences of the placeholder ID in the template (for authoring_layer).",
    )
    parser.add_argument(
        "--usdz-placeholder",
        type=str,
        default="@usdz_root[gauss.usda]@",
        help="Placeholder string in template for USDZ reference (e.g., '@usdz_root[gauss.usda]@').",
    )
    parser.add_argument(
        "--usdz-path-template",
        type=str,
        default="/path/to/data/InteriorGS_usdz/{scene_id}.usdz[gauss.usda]",
        help="Template for the USDZ reference path. Use {scene_id} as placeholder. Note: '@' symbols will be added automatically.",
    )
    parser.add_argument(
        "--collision-placeholder",
        type=str,
        default="@collision_root@",
        help="Placeholder string in template for collision payload (e.g., '@collision_root@').",
    )
    parser.add_argument(
        "--collision-path-template",
        type=str,
        default="/path/to/data/InteriorGS_Collision/Collision/{scene_id}/{scene_id}_collision.usd",
        help="Template for the collision payload path. Use {scene_id} as placeholder. Note: '@' symbols will be added automatically.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing USDA files in the output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of files to generate (for testing).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure path templates have '@' symbols if not provided
    usdz_path_template = args.usdz_path_template
    if not usdz_path_template.startswith("@"):
        usdz_path_template = "@" + usdz_path_template
    if not usdz_path_template.endswith("@"):
        usdz_path_template = usdz_path_template + "@"

    collision_path_template = args.collision_path_template
    if not collision_path_template.startswith("@"):
        collision_path_template = "@" + collision_path_template
    if not collision_path_template.endswith("@"):
        collision_path_template = collision_path_template + "@"

    generate_usda_files(
        usdz_dir=args.usdz_dir,
        out_dir=args.out_dir,
        template_path=args.template,
        base_id=args.base_id,
        expected_count=args.expected_count,
        usdz_placeholder=args.usdz_placeholder,
        usdz_path_template=usdz_path_template,
        collision_placeholder=args.collision_placeholder,
        collision_path_template=collision_path_template,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()


