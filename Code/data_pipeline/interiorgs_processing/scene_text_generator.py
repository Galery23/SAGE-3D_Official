#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert InteriorGS physical maps (scene.json) into textual 2D scene descriptions.

This script loads bounding-box summaries produced by physical_map_converter.py,
feeds them into an LLM via the standard OpenAI Chat Completions API, and stores
the generated semantic narratives as plain-text files.

All critical knobs (input/output roots, prompt file, OpenAI endpoint, model,
API key, etc.) are exposed via CLI arguments and .env variables so the script
can be reused without hunting through the code.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests


DEFAULT_PHYSICAL_ROOT = Path("/path/to/data/InteriorGS/0_physical_map")
DEFAULT_OUTPUT_ROOT = Path("Data/InteriorGS/semantic_text")
DEFAULT_PROMPT_FILE = Path("prompts/prompt_phy_to_sem.json")
DEFAULT_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def load_prompt_template(prompt_file: Path) -> Sequence[Dict[str, Any]]:
    """Load the chat prompt template from JSON."""
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with prompt_file.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_messages(prompt_template: Sequence[Dict[str, Any]], json_payload: str) -> List[Dict[str, Any]]:
    """Replace the {json} placeholder in the last prompt entry and return messages."""
    if not prompt_template:
        raise ValueError("Prompt template is empty.")
    messages: List[Dict[str, Any]] = []

    for entry in prompt_template[:-1]:
        messages.append(entry)

    final_entry = prompt_template[-1].copy()
    final_entry_content = final_entry.get("content", "")
    final_entry["content"] = final_entry_content.replace("{json}", json_payload)
    messages.append(final_entry)
    return messages


def call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, Any]],
    temperature: float,
    timeout: int,
) -> str:
    """Send a chat completion request to the OpenAI-compatible endpoint."""
    if not api_key:
        raise ValueError("Missing API key. Provide --api-key or set OPENAI_API_KEY.")

    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")

    data = response.json()
    choices = data.get("choices")
    if not choices:
        raise RuntimeError("OpenAI API returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content.strip():
        raise RuntimeError("OpenAI API returned empty content.")
    return content


def write_output_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(text)


def process_scene(
    scene_path: Path,
    output_root: Path,
    prompt_template: Sequence[Dict[str, Any]],
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_retries: int,
    timeout: int,
) -> bool:
    """Convert a single scene.json file into text using the configured LLM."""
    folder_name = scene_path.parent.name
    output_path = output_root / f"semantic_map_{folder_name}.txt"

    json_payload = scene_path.read_text(encoding="utf-8")
    messages = build_messages(prompt_template, json_payload)

    for attempt in range(max_retries + 1):
        try:
            text = call_chat_completion(
                base_url=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            write_output_text(output_path, text)
            if attempt > 0:
                print(f"[RETRY-SUCCESS] {folder_name} after {attempt} retries -> {output_path}")
            else:
                print(f"[OK] {folder_name} -> {output_path}")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            if attempt < max_retries:
                print(f"[RETRY] {folder_name} failed attempt {attempt + 1}/{max_retries}: {exc}")
            else:
                print(f"[FAIL] {folder_name} after {max_retries} retries: {exc}")
    return False


def collect_scene_files(physical_root: Path, limit: int | None = None) -> List[Path]:
    """Gather scene.json files under the given root."""
    if not physical_root.exists():
        raise FileNotFoundError(f"Physical map root does not exist: {physical_root}")

    scene_files: List[Path] = []
    for scene_dir in sorted(p for p in physical_root.iterdir() if p.is_dir()):
        candidate = scene_dir / "scene.json"
        if candidate.is_file():
            scene_files.append(candidate)
    if limit is not None:
        scene_files = scene_files[:limit]
    return scene_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate textual 2D scene descriptions from InteriorGS physical maps via OpenAI APIs."
    )
    parser.add_argument(
        "--physical-map-root",
        type=Path,
        default=DEFAULT_PHYSICAL_ROOT,
        help=f"Directory containing subfolders with scene.json files (default: {DEFAULT_PHYSICAL_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory where semantic map text files will be saved (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=f"Prompt template JSON (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"OpenAI-compatible base URL, typically https://api.openai.com/v1 (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key for the OpenAI-compatible endpoint (default: OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to call (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of concurrent threads.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Process at most this many scene folders (useful for testing).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Number of retries per scene when API calls fail.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout (seconds) for each chat completion call.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Shortcut for --max-scenes 10 and --max-workers 4 to run quick sanity checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_workers = 4 if args.test else args.max_workers
    max_scenes = 10 if args.test else args.max_scenes

    prompt_template = load_prompt_template(args.prompt_file.expanduser())
    scene_files = collect_scene_files(args.physical_map_root.expanduser(), limit=max_scenes)
    if not scene_files:
        print(f"[WARN] No scene.json files found under {args.physical_map_root}")
        return

    print(f"[INFO] Processing {len(scene_files)} scenes with {max_workers} workers.")
    output_root = args.output_root.expanduser()

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                process_scene,
                scene_path,
                output_root,
                prompt_template,
                args.api_base.rstrip("/"),
                args.api_key,
                args.model,
                args.temperature,
                args.max_retries,
                args.timeout,
            ): scene_path
            for scene_path in scene_files
        }

        for future in as_completed(future_map):
            scene_path = future_map[future]
            try:
                if future.result():
                    completed += 1
                else:
                    failed += 1
            except Exception as exc:  # pragma: no cover - defensive
                failed += 1
                print(f"[ERROR] Unexpected failure for {scene_path.parent.name}: {exc}")

    print(f"[SUMMARY] Completed: {completed}, Failed: {failed}. Outputs -> {output_root}")


if __name__ == "__main__":
    main()

