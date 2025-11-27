#!/usr/bin/env python3
"""
SAGE-3D Benchmark Runner.

Main script for running VLN (Vision-and-Language Navigation) benchmark evaluation
on SAGE-3D dataset using Isaac Sim environment.

Usage:
    python run_benchmark.py --batch-dir /path/to/test/data --scene-folder /path/to/scenes \\
        --out-root /path/to/output --vlm-host localhost --vlm-port 54321

For distributed testing:
    python run_benchmark.py --batch-dir /path/to/data --scene-folder /path/to/scenes \\
        --out-root /path/to/output --vlm-host localhost --vlm-port 54321 \\
        --instance-id 0 --total-instances 4
"""

import os
import sys


def _init_global_silent_mode():
    """Initialize global silent mode before any module imports.
    
    Filters empty lines and debug output from print statements.
    """
    print(f"[INIT] Enabling smart print filter - filtering empty lines and debug info")
    
    # Enable basic empty line filtering regardless of --silent-logging
    original_print = print
    def smart_filtered_print(*args, **kwargs):
        # Skip if no args or all args are empty
        if not args or all(str(arg).strip() == '' for arg in args):
            return
        
        # Check for pure empty line
        if len(args) == 1 and str(args[0]).strip() == '':
            return
            
        # If silent-logging enabled, apply stricter filtering
        if '--silent-logging' in sys.argv:
            if args:
                msg = str(args[0])
                # Filter debug tags
                debug_tags = ['[COLLISION_2D]', '[PHYSICS]', '[CAMERA_UPDATE]',
                            '[RGB_CAPTURE]', '[COLLISION_VIS]', '[YAW_UPDATE]',
                            '[COORD_TRANSFORM]', '[POSITION]', '[VELOCITY]',
                            '[DEBUG_ENV]', '[EPISODE_RESET]', '[SUCCESS]',
                            '[ORACLE_SUCCESS]', '[CSR]', '[OBJECT_SUCCESS]']
                if any(tag in msg for tag in debug_tags):
                    return
        
        # Print other content normally
        original_print(*args, **kwargs)
    
    # Replace global print function
    import builtins
    builtins.print = smart_filtered_print
    
    if '--silent-logging' in sys.argv:
        print(f"[INIT] Detected --silent-logging argument, enabling strict filter mode")
        os.environ['SILENT_LOGGING_MODE'] = 'True'
    else:
        print(f"[INIT] Basic filter mode: only filtering empty lines")


# Execute immediately
_init_global_silent_mode()

import io
import json
import math
import argparse
import logging
import glob
import time
import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

# For trajectory visualization
try:
    pass
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not available, trajectory visualization will be disabled")


class ProgressTracker:
    """Progress tracker - real-time display of test progress info"""
    
    def __init__(self, total_episodes: int, model_name: str = "Unknown", enable_live_display: bool = True):
        self.total_episodes = total_episodes
        self.model_name = model_name
        self.completed_episodes = 0
        self.failed_episodes = 0
        self.skipped_episodes = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.episode_times = []
        self.enable_live_display = enable_live_display
        self.last_displayed_episode = 0
        self.global_episode_counter = 0
        
        # Save original terminal settings
        import sys
        self.stdout = sys.stdout
        
        # Display initial progress
        if self.enable_live_display:
            pass
            self._display_progress_header()
        
    def start_episode(self, episode_id: str, scene_name: str, episode_idx: int):
        """Start processing episode"""
        self.global_episode_counter += 1
        self.current_episode_start = time.time()
        self.current_episode_id = episode_id
        self.current_scene_name = scene_name
        self.current_episode_idx = episode_idx
        
        # Real-time progress update (more frequent updates to ensure visibility)
        if self.enable_live_display and (self.global_episode_counter % 5 == 0 or self.global_episode_counter <= 3 or self.global_episode_counter == self.total_episodes):
            self._update_live_progress(self.global_episode_counter)
    
    def complete_episode(self, success: bool = True, skipped: bool = False):
        """Complete episode processing"""
        episode_time = time.time() - self.current_episode_start
        self.episode_times.append(episode_time)
        
        if skipped:
            pass
            self.skipped_episodes += 1
            status_char = "⏭️"
        elif success:
            pass
            self.completed_episodes += 1
            status_char = "✅"
        else:
            pass
            self.failed_episodes += 1
            status_char = "❌"
        
        # Compact episode completion info (doesn't interfere with main progress display)
        compact_status = f"[{self.global_episode_counter:4d}/{self.total_episodes}] {status_char} {self.current_scene_name}/{self.current_episode_id} ({self._format_duration(episode_time)})"
        print(compact_status, flush=True)
        
        # Ensure progress info is displayed promptly (force write to stdout)
        import sys
        sys.stdout.flush()
        
        # Keep last 20 episode times for ETA calculation
        if len(self.episode_times) > 20:
            pass
            self.episode_times = self.episode_times[-20:]
            
        # Show detailed progress every 10 episodes (more frequent)
        if self.enable_live_display and self.global_episode_counter % 10 == 0:
            self._update_live_progress(self.global_episode_counter, force_display=True)
    
    def _display_progress_header(self):
        """Display progress bar header"""
        print(f"\n{'='*100}")
        print(f"🚀 SAGE-Bench Test Progress - Model: {self.model_name}")
        print(f"📊 Total Episodes: {self.total_episodes}")
        print(f"⏰ Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")
        print("[Progress] [Status] Episode Info")
        print("-" * 100)
    
    def _update_live_progress(self, episode_idx: int, force_display: bool = False):
        """Update live progress display"""
        # Avoid too frequent updates (reduced interval for responsiveness)
        current_time = time.time()
        if not force_display and current_time - self.last_update_time < 3:  # 3 second interval
            return
            
        self.last_update_time = current_time
        
        # Calculate progress
        progress_pct = (episode_idx / self.total_episodes) * 100
        
        # Calculate time info
        elapsed_time = current_time - self.start_time
        if self.episode_times:
            pass
            avg_time = sum(self.episode_times) / len(self.episode_times)
            remaining_episodes = self.total_episodes - episode_idx
            eta_seconds = avg_time * remaining_episodes
            eta_str = self._format_duration(eta_seconds)
        else:
            pass
            eta_str = "Calculating..."
        
        # Progress bar
        bar_length = 50
        filled_length = int(bar_length * episode_idx // self.total_episodes)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print colored progress info
        print(f"\n{'='*100}")
        print(f"📊 Progress: [{bar}] {progress_pct:.1f}% ({episode_idx}/{self.total_episodes})")
        print(f"⏱️  Elapsed: {self._format_duration(elapsed_time)} | ETA: {eta_str}")
        print(f"📈 Success: {self.completed_episodes} | Failed: {self.failed_episodes} | Skipped: {self.skipped_episodes}")
        if self.episode_times:
            pass
            success_rate = (self.completed_episodes / max(1, self.completed_episodes + self.failed_episodes)) * 100
            avg_time_str = self._format_duration(sum(self.episode_times) / len(self.episode_times))
            print(f"⚡ Success Rate: {success_rate:.1f}% | Avg Time: {avg_time_str}/episode")
        print(f"{'='*100}\n", flush=True)
    
    def final_summary(self):
        """Display final summary"""
        total_time = time.time() - self.start_time
        total_str = self._format_duration(total_time)
        
        print(f"\n{'🎉 SAGE-Bench Test Complete!':<50}")
        print(f"{'='*100}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📊 Total Episodes: {self.total_episodes}")
        print(f"✅ Completed: {self.completed_episodes}")
        print(f"❌ Failed: {self.failed_episodes}")
        print(f"⏭️  Skipped: {self.skipped_episodes}")
        
        # Calculate success rate (excluding skipped)
        tested_episodes = self.completed_episodes + self.failed_episodes
        if tested_episodes > 0:
            pass
            success_rate = (self.completed_episodes / tested_episodes) * 100
            print(f"📈 Success Rate: {success_rate:.1f}% (based on {tested_episodes} actual tests)")
        
        print(f"⏱️  Total Time: {total_str}")
        if self.episode_times:
            pass
            avg_time = sum(self.episode_times) / len(self.episode_times)
            print(f"⚡ Avg Time: {self._format_duration(avg_time)}/episode")
            
            # Performance analysis
            total_test_time = tested_episodes * avg_time
            efficiency = (total_test_time / total_time) * 100 if total_time > 0 else 0
            print(f"🔧 Test Efficiency: {efficiency:.1f}% (actual test time ratio)")
            
        print(f"🏁 End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format time display"""
        if seconds < 60:
            pass
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            pass
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        else:
            pass
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def update_model_name(self, model_name: str):
        """Update model name"""
        self.model_name = model_name
    
    def force_progress_update(self):
        """Force progress update"""
        if hasattr(self, 'current_episode_idx'):
            pass
            self._update_live_progress(self.current_episode_idx, force_display=True)


# Add current script directory to Python path to ensure related modules can be found
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Try relative import (when running as a package)
    from .episodes_adapter import adapt_gvln_to_episodes
    from .measures import default_measures, nogoal_measures
    from .vlm_client_modular import query_vlm, set_log_function
    from .simple_env import SimpleVLNEnv
    from .task_types import TaskTypeManager, adapt_episode_for_task
except ImportError:
    # Fallback to absolute import (when running directly)
    try:
        from episodes_adapter import adapt_gvln_to_episodes
        from measures import default_measures, nogoal_measures
        from vlm_client_modular import query_vlm, set_log_function
        from simple_env import SimpleVLNEnv
        from task_types import TaskTypeManager, adapt_episode_for_task
    except ImportError as e:
        print(f"[ERROR] Failed to import required modules: {e}")
        print(f"[ERROR] Script directory: {current_dir}")
        print("[ERROR] Please ensure all required modules are in the same directory")
        sys.exit(1)


def find_test_json_files(batch_dir: str, pattern: str = "test_*.json") -> List[str]:
    """
    Scan directory and subdirectories to find all JSON files matching pattern
    
    Args:
        batch_dir: Batch test directory
        pattern: File pattern, default "test_*.json"
        
    Returns:
        List of matching JSON file paths
    """
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        pass
        print(f"[ERROR] Batch test directory does not exist: {batch_dir}")
        return []
    
    # Recursively search for all matching JSON files
    json_files = []
    for root, dirs, files in os.walk(batch_path):
        for file in files:
            if file.startswith("test_") and file.endswith(".json"):
                pass
                file_path = os.path.join(root, file)
                json_files.append(file_path)
    
    json_files.sort()  # Sort by filename
    print(f"[INFO] Found {len(json_files)} test JSON files:")
    for i, file_path in enumerate(json_files, 1):
        rel_path = os.path.relpath(file_path, batch_dir)
        print(f"[INFO]   {i:3d}. {rel_path}")
    
    return json_files


def get_scene_name_from_json(json_file_path: str) -> str:
    """
    Extract scene_name from JSON file
    
    Args:
        json_file_path: JSON trajectory file path
        
    Returns:
        scene_name string, empty string if not found
    """
    try:
        pass
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract scene_name from JSON data
        if "scenes" in data and len(data["scenes"]) > 0:
            scene_name = data["scenes"][0].get("scene_name", "")
            return scene_name
        else:
            print(f"[WARN] Cannot get scene_name from JSON file: {json_file_path}")
            return ""
    except Exception as e:
        print(f"[ERROR] Failed to read JSON file: {json_file_path}, error: {e}")
        return ""


def check_episode_completed(out_root: Path, scene_name: str, episode_id: str) -> bool:
    """
    Check if specified episode has completed testing (by checking measurements file existence)
    
    Args:
        out_root: Output root directory
        scene_name: Scene name
        episode_id: Trajectory-instruction pair ID
        
    Returns:
        True if completed, False if not
    """
    measurements_file = out_root / scene_name / episode_id / "measurements" / f"{episode_id}.json"
    exists = measurements_file.exists()
    
    if exists:
        pass
        # Additional validation: check if file is valid JSON with required fields
        try:
            pass
            with open(measurements_file, 'r') as f:
                data = json.load(f)
            # Check required fields
            required_fields = ["success", "spl", "path_length"]
            has_required_fields = all(field in data for field in required_fields)
            if has_required_fields:
                print(f"[CHECKPOINT] ✅ Episode {scene_name}/{episode_id} completed, skipping")
                return True
            else:
                print(f"[CHECKPOINT] ⚠️ Episode {scene_name}/{episode_id} measurements incomplete, re-testing")
                return False
        except (json.JSONDecodeError, Exception) as e:
            print(f"[CHECKPOINT] ⚠️ Episode {scene_name}/{episode_id} measurements corrupted, re-testing: {e}")
            return False
    else:
        print(f"[CHECKPOINT] ⏭️  Episode {scene_name}/{episode_id} not completed, starting test")
        return False


def find_matching_scene_file(json_file_path: str, scene_folder: str, scene_name: str = None) -> str:
    """
    Auto-match corresponding scene USDA file based on JSON file
    
    Args:
        json_file_path: JSON trajectory file path
        scene_folder: Scene folder path
        scene_name: Optional scene_name, will read from JSON file if not provided
        
    Returns:
        Matching scene file path, empty string if not found
    """
    if not scene_folder or not os.path.exists(scene_folder):
        pass
        print(f"[WARN] Scene folder does not exist or not specified: {scene_folder}")
        return ""
    
    # Read scene_name from JSON file if not provided
    if not scene_name:
        pass
        scene_name = get_scene_name_from_json(json_file_path)
    
    if not scene_name:
        pass
        print(f"[WARN] scene_name not found: {json_file_path}")
        return ""
    
    print(f"[SCENE_MATCH] Finding scene file for '{scene_name}'...")
    
    # Search for matching scene file
    # Scene file format: scene_name.usda or scene_name.usd
    scene_patterns = [
        f"{scene_name}.usda",
        f"{scene_name}.usd"
    ]
    
    for pattern in scene_patterns:
        scene_file = os.path.join(scene_folder, pattern)
        if os.path.exists(scene_file):
            pass
            print(f"[SCENE_MATCH] ✓ Found matching scene: {os.path.basename(scene_file)}")
            return scene_file
    
    # Try fuzzy matching if exact match not found
    print(f"[SCENE_MATCH] Trying fuzzy match...")
    for file in os.listdir(scene_folder):
        if (file.endswith(".usda") or file.endswith(".usd")) and scene_name in file:
            pass
            scene_file = os.path.join(scene_folder, file)
            print(f"[SCENE_MATCH] ✓ Fuzzy match found scene: {file}")
            return scene_file
    
    print(f"[SCENE_MATCH] ✗ No matching scene file found for '{scene_name}'")
    return ""


def find_matching_map_file(json_file_path: str, map_folder: str, scene_name: str = None) -> str:
    """
    Auto-match corresponding 2D semantic map file based on JSON file
    
    Args:
        json_file_path: JSON trajectory file path
        map_folder: Map folder path
        scene_name: Optional scene_name, will read from JSON file if not provided
        
    Returns:
        Matching map file path, empty string if not found
    """
    if not map_folder or not os.path.exists(map_folder):
        pass
        print(f"[WARN] Map folder does not exist or not specified: {map_folder}")
        return ""
    
    # Read scene_name from JSON file if not provided
    if not scene_name:
        pass
        scene_name = get_scene_name_from_json(json_file_path)
    
    if not scene_name:
        pass
        print(f"[WARN] scene_name not found: {json_file_path}")
        return ""
    
    print(f"[MAP_MATCH] Finding map file for '{scene_name}'...")
    
    # Search for matching map file
    # Map file format: 2D_Semantic_Map_xxxx_scene_name_Complete.json
    map_patterns = [
        f"2D_Semantic_Map_*_{scene_name}_Complete.json",
        f"2D_Semantic_Map_{scene_name}_Complete.json",
        f"*_{scene_name}_Complete.json",
        f"*{scene_name}*.json"
    ]
    
    for pattern in map_patterns:
        search_pattern = os.path.join(map_folder, pattern)
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            pass
            # Select first match if multiple files found
            map_file = matching_files[0]
            print(f"[MAP_MATCH] ✓ Found matching map: {os.path.basename(map_file)}")
            return map_file
    
    # Try fuzzy matching if exact match not found
    print(f"[MAP_MATCH] Trying fuzzy match...")
    for file in os.listdir(map_folder):
        if file.endswith(".json") and scene_name in file:
            pass
            map_file = os.path.join(map_folder, file)
            print(f"[MAP_MATCH] ✓ Fuzzy match found map: {file}")
            return map_file
    
    print(f"[MAP_MATCH] ✗ No matching map file found for '{scene_name}'")
    return ""


def save_batch_summary(batch_results: List[Dict[str, Any]], output_root: Path, model_info: str) -> None:
    """
    Save batch test summary results
    
    Args:
        batch_results: List of batch test results
        output_root: Output root directory
        model_info: Model info string
    """
    summary_file = output_root / "batch_test_summary.json"
    
    # Calculate summary statistics
    total_files = len(batch_results)
    total_episodes = sum(r["total_episodes"] for r in batch_results)
    total_successful = sum(r["successful_episodes"] for r in batch_results)
    total_failed = sum(r["failed_episodes"] for r in batch_results)
    
    overall_success_rate = total_successful / total_episodes if total_episodes > 0 else 0.0
    
    summary = {
        "model_info": model_info,
        "batch_summary": {
            "total_json_files": total_files,
            "total_episodes": total_episodes,
            "successful_episodes": total_successful,
            "failed_episodes": total_failed,
            "overall_success_rate": overall_success_rate
        },
        "file_results": batch_results
    }
    
    # Save summary results
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[BATCH_SUMMARY] ===== Batch Test Summary =====")
    print(f"[BATCH_SUMMARY] Model Info: {model_info}")
    print(f"[BATCH_SUMMARY] Test Files: {total_files}")
    print(f"[BATCH_SUMMARY] Total Episodes: {total_episodes}")
    print(f"[BATCH_SUMMARY] Successful: {total_successful}")
    print(f"[BATCH_SUMMARY] Failed: {total_failed}")
    print(f"[BATCH_SUMMARY] Overall Success Rate: {overall_success_rate:.2%}")
    print(f"[BATCH_SUMMARY] Summary saved to: {summary_file}")
    print(f"[BATCH_SUMMARY] =============================\n")


def run_single_json_test(episodes: List[Dict[str, Any]], args, out_root: Path, json_file: str, model_info: str, map_path: str = "", scene_usd_path: str = "", close_env_on_finish: bool = True, shared_env=None, progress_tracker=None) -> tuple:
    """
    Run test for a single JSON file
    
    Args:
        episodes: Episode list
        args: Command line arguments
        out_root: Output root directory
        json_file: JSON file path
        model_info: Model info string
        map_path: 2D semantic map file path
        scene_usd_path: Optional scene file path, overrides episode scene_usd if provided
        
    Returns:
        (successful_episodes, failed_episodes) tuple
    """
    # Initialize shared environment for all episodes
    if len(episodes) == 0:
        pass
        print("[ERROR] No episodes to process", flush=True)
        return 0, 0
    
    first_episode = episodes[0]
    
    # Determine scene file path to use
    if scene_usd_path:
        pass
        # Use provided scene_usd_path parameter
        actual_scene_path = scene_usd_path
        print(f"[INFO] Using provided scene path: {actual_scene_path}", flush=True)
    else:
        pass
        # Use default path from episode
        actual_scene_path = first_episode["scene_usd"]
        print(f"[INFO] Using scene path from episode: {actual_scene_path}", flush=True)
    
    # Use passed map_path instead of args.map_path
    actual_map_path = map_path if map_path else args.map_path
    print(f"[INFO] Using map file: {actual_map_path if actual_map_path else 'None'}", flush=True)
    
    # Use shared environment or create new one
    if shared_env is not None:
        pass
        env = shared_env
        print(f"[INFO] Using shared environment", flush=True)
        # Load new scene for shared environment
        if hasattr(env, 'load_scene'):
            pass
            print(f"[INFO] Switching to new scene: {actual_scene_path}", flush=True)
            env.load_scene(actual_scene_path)
        else:
            pass
            print(f"[WARN] Shared environment does not support scene switching, using current scene", flush=True)
        
        # Dynamically update map
        if hasattr(env, 'update_map') and actual_map_path:
            pass
            print(f"[INFO] Updating 2D semantic map: {actual_map_path}", flush=True)
            env.update_map(actual_map_path)
        else:
            pass
            if not actual_map_path:
                pass
                print(f"[WARN] No map path provided, skipping map update", flush=True)
            else:
                pass
                print(f"[WARN] Shared environment does not support map update", flush=True)
    else:
        pass
        env = SimpleVLNEnv(scene_usd_path=actual_scene_path, headless=True, hz=args.hz, map_json_path=actual_map_path)
        print(f"[INFO] New environment initialized successfully", flush=True)
    
    successful_episodes = 0
    failed_episodes = 0
    
    try:
        pass
        for i, ep in enumerate(episodes):
            # Progress tracking
            if progress_tracker:
                pass
                progress_tracker.start_episode(ep['episode_id'], ep['scene_name'], i + 1)
            else:
                pass
            print(f"[INFO] ===== Processing Episode {i+1}/{len(episodes)} =====", flush=True)
            print(f"Running scene {ep['scene_name']} episode {ep['episode_id']}...", flush=True)
            
            # Checkpoint check: see if episode already completed (if checkpoint enabled)
            if args.skip_completed and check_episode_completed(out_root, ep['scene_name'], ep['episode_id']):
                pass
                successful_episodes += 1  # Count completed episode as success
                if progress_tracker:
                    pass
                    progress_tracker.complete_episode(success=True, skipped=True)
                continue  # Skip completed episode
            
            try:
                pass
                # Prepare task config
                task_config = {
                    "goal_radius": args.goal_radius,
                    "max_episode_time": 80.0,  # 80-second time limit for no-goal tasks
                    "collision_penalty": True,
                    "min_exploration_coverage": 0.25
                }
                
                run_episode(ep, out_root, args.vlm_host, args.vlm_port, env, hz=args.hz, max_steps=args.max_steps, 
                           map_path=actual_map_path, disable_collision=args.disable_collision, 
                           disable_autopilot=args.disable_autopilot, model_type=args.model_type,
                           input_type=getattr(args, 'input_type', None), 
                           output_type=getattr(args, 'output_type', None),
                           protocol=getattr(args, 'protocol', None),
                           task_type=args.task_type, task_config=task_config, args=args)
                successful_episodes += 1
                if progress_tracker:
                    pass
                    progress_tracker.complete_episode(success=True, skipped=False)
                else:
                    pass
                print(f"[SUCCESS] Episode {ep['episode_id']} completed successfully!", flush=True)
            except Exception as e:
                pass
                failed_episodes += 1
                if progress_tracker:
                    pass
                    progress_tracker.complete_episode(success=False, skipped=False)
                else:
                    pass
                print(f"[ERROR] Episode {ep['episode_id']} failed with error: {e}", flush=True)
                print(f"[ERROR] Continuing with next episode...", flush=True)
                import traceback
                traceback.print_exc()
    
    finally:
        # Close shared environment after all episodes (only if not in batch mode)
        if close_env_on_finish and shared_env is None:
            pass
            print(f"[INFO] Closing shared environment...", flush=True)
            try:
                pass
                env.close()
                print(f"[INFO] Shared environment closed successfully", flush=True)
            except Exception as e:
                pass
                print(f"[ERROR] Failed to close environment: {e}", flush=True)
        else:
            pass
            print(f"[INFO] Keeping environment open for subsequent use", flush=True)
    
    print(f"[INFO] ===== File Summary =====", flush=True)
    print(f"[INFO] JSON file: {os.path.basename(json_file)}", flush=True)
    print(f"[INFO] Total episodes processed: {len(episodes)}", flush=True)
    print(f"[INFO] Successful episodes: {successful_episodes}", flush=True)
    print(f"[INFO] Failed episodes: {failed_episodes}", flush=True)
    success_rate = successful_episodes / len(episodes) if len(episodes) > 0 else 0.0
    print(f"[INFO] Success rate: {success_rate:.2%}", flush=True)
    
    return successful_episodes, failed_episodes


def _closest_waypoint(ep: Dict[str, Any], pos: np.ndarray) -> np.ndarray:
    gts = np.asarray(ep["gt_locations"], dtype=np.float32)
    if gts.shape[0] == 0:
        pass
        return pos
    d = np.linalg.norm(gts - pos[None, :], axis=1)
    idx = int(np.argmin(d))
    return gts[idx]


def reverse_position_mapping(px_3d, py_3d, map_data, flip_x=True, flip_y=True, negate_xy=True):
    """
    Reverse mapping: convert 3D trajectory coordinates back to 2D for visualization
    This is the inverse of the original mapping code
    
    Args:
        px_3d, py_3d: Coordinates in 3D trajectory
        map_data: Map data for getting bounds
        flip_x, flip_y, negate_xy: Mapping parameters, should match original mapping
    
    Returns:
        (px_2d, py_2d): Converted 2D coordinates
    """
    # Get map bounds (keeping original correct calculation)
    all_y = [float(y) for inst in map_data for y, x in inst.get('mask_coords_m', [])]
    all_x = [float(x) for inst in map_data for y, x in inst.get('mask_coords_m', [])]
    min_y, max_y = min(all_y), max(all_y)
    min_x, max_x = min(all_x), max(all_x)
    
    # Reverse mapping process (opposite order of original mapping)
    px, py = px_3d, py_3d
    
    # 1. If original negated overall, reverse it
    if negate_xy:
        pass
        px = -px
        py = -py
    
    # 2. If original mirrored, reverse it
    if flip_x:
        pass
        px = (min_x + max_x) - px
    if flip_y:
        pass
        py = (min_y + max_y) - py
        
    return px, py


def visualize_trajectory(episode: Dict[str, Any], trajectory_positions: List[np.ndarray], 
                         map_path: str, output_dir: Path) -> None:
    """
    Visualize the VLM agent's trajectory on 2D semantic map
    
    Args:
        episode: Episode data containing scene info
        trajectory_positions: List of agent positions during execution
        map_path: Path to 2D semantic map JSON
        output_dir: Directory to save visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        pass
        print("[WARN] matplotlib not available, skipping trajectory visualization", flush=True)
        return
    
    if not map_path or not os.path.exists(map_path):
        pass
        print(f"[WARN] Map file not found: {map_path}, skipping trajectory visualization", flush=True)
        return
    
    try:
        pass
        print("[INFO] Loading 2D semantic map...", flush=True)
        # Load 2D semantic map
        with open(map_path, 'r') as f:
            map_data = json.load(f)
        
        print(f"[INFO] Loaded map data with {len(map_data)} instances", flush=True)
        
        # Get map bounds
        all_y = [float(y) for inst in map_data for y, x in inst.get('mask_coords_m', [])]
        all_x = [float(x) for inst in map_data for y, x in inst.get('mask_coords_m', [])]
        
        if not all_x or not all_y:
            pass
            print("[WARN] No valid coordinates in map data", flush=True)
            return
            
        min_y, max_y = min(all_y), max(all_y)
        min_x, max_x = min(all_x), max(all_x)
        print(f"[INFO] Map bounds: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]", flush=True)
        
        # Create color map image for background (similar to your approach)
        print("[INFO] Creating color map for background...", flush=True)
        map_width = int((max_x - min_x) * 10) + 20  # 10 pixels per meter + padding
        map_height = int((max_y - min_y) * 10) + 20
        color_map_img = np.ones((map_height, map_width, 3), dtype=np.float32) * 0.9  # Light gray background
        
        # Fill obstacles
        for inst in map_data:
            category = str(inst.get('category_label', '')).lower()
            if category in ['wall', 'unable area']:
                pass
                coords = inst.get('mask_coords_m', [])
                if coords:
                    pass
                    for y, x in coords:
                        try:
                            pass
                            # Convert world coordinates to image coordinates (ensure float conversion)
                            x_float = float(x)
                            y_float = float(y)
                            img_x = int((x_float - min_x) * 10) + 10
                            img_y = int((y_float - min_y) * 10) + 10
                            if 0 <= img_x < map_width and 0 <= img_y < map_height:
                                pass
                                if category == 'wall':
                                    pass
                                    color_map_img[img_y, img_x] = [0.6, 0.8, 1.0]  # Light blue for walls
                                else:  # unable area
                                    color_map_img[img_y, img_x] = [1.0, 0.4, 0.4]  # Light red for unable areas
                        except (ValueError, TypeError) as e:
                            pass
                            # Skip invalid coordinates
                            continue
        
        # Create visualization
        print("[INFO] Creating matplotlib figure...", flush=True)
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
        
        # Set background color and display map image
        bg_color = (0.9, 0.9, 0.9)
        ax.set_facecolor(bg_color)
        img_extent = [min_x - 1, max_x + 1, min_y - 1, max_y + 1]
        ax.imshow(color_map_img, extent=img_extent, origin='lower', interpolation='nearest', alpha=0.8)
        
        # Apply reverse mapping to convert 3D coordinates back to 2D for visualization
        print("[INFO] Applying reverse mapping from 3D to 2D coordinates...", flush=True)
        
        # Plot ground truth path (red like your code) - with reverse mapping
        if episode.get("gt_locations"):
            pass
            gt_positions_3d = np.array(episode["gt_locations"])
            print(f"[INFO] GT path original 3D positions: {len(gt_positions_3d)}", flush=True)
            
            # Apply reverse mapping to each GT position
            gt_positions_2d = []
            for pos_3d in gt_positions_3d:
                pos_2d_x, pos_2d_y = reverse_position_mapping(pos_3d[0], pos_3d[1], map_data)
                gt_positions_2d.append([pos_2d_x, pos_2d_y])
            gt_positions_2d = np.array(gt_positions_2d)
            
            print(f"[INFO] GT path first 3D point: {gt_positions_3d[0][:2]} -> 2D: {gt_positions_2d[0]}", flush=True)
            print(f"[INFO] GT path last 3D point: {gt_positions_3d[-1][:2]} -> 2D: {gt_positions_2d[-1]}", flush=True)
            
            if len(gt_positions_2d) >= 2:
                pass
                ax.plot(gt_positions_2d[:, 0], gt_positions_2d[:, 1], '-', color='red', linewidth=3, alpha=0.9)
                ax.scatter([gt_positions_2d[0, 0], gt_positions_2d[-1, 0]], 
                          [gt_positions_2d[0, 1], gt_positions_2d[-1, 1]], color='red', s=80)
        
        # Plot VLM agent trajectory (blue) - with reverse mapping
        if trajectory_positions:
            pass
            traj_array_3d = np.array(trajectory_positions)
            print(f"[INFO] Agent trajectory original 3D positions: {len(traj_array_3d)}", flush=True)
            
            # Apply reverse mapping to each agent position
            traj_array_2d = []
            for pos_3d in traj_array_3d:
                pos_2d_x, pos_2d_y = reverse_position_mapping(pos_3d[0], pos_3d[1], map_data)
                traj_array_2d.append([pos_2d_x, pos_2d_y])
            traj_array_2d = np.array(traj_array_2d)
            
            print(f"[INFO] Agent trajectory first 3D point: {traj_array_3d[0][:2]} -> 2D: {traj_array_2d[0]}", flush=True)
            print(f"[INFO] Agent trajectory last 3D point: {traj_array_3d[-1][:2]} -> 2D: {traj_array_2d[-1]}", flush=True)
            
            if len(traj_array_2d) >= 2:
                pass
                # Check if all points are the same (agent stuck)
                first_point = traj_array_2d[0]
                all_same = np.allclose(traj_array_2d, first_point, atol=0.01)  # 1cm tolerance
                
                if all_same:
                    pass
                    # All points same, display as large blue dot
                    ax.scatter(first_point[0], first_point[1], color='blue', s=200, alpha=0.9, 
                              marker='o', edgecolors='darkblue', linewidth=2, label='Agent Stuck')
                    print(f"[INFO] Agent stuck at position: {first_point} ({len(traj_array_2d)} steps)")
                else:
                    pass
                    # Normal trajectory, draw lines
                    ax.plot(traj_array_2d[:, 0], traj_array_2d[:, 1], '-', color='blue', linewidth=4, alpha=0.9)
                    ax.scatter([traj_array_2d[0, 0], traj_array_2d[-1, 0]], 
                              [traj_array_2d[0, 1], traj_array_2d[-1, 1]], color='blue', s=100)
            elif len(traj_array_2d) == 1:
                pass
                # Single point, display as large blue dot
                ax.scatter(traj_array_2d[0, 0], traj_array_2d[0, 1], color='blue', s=200, alpha=0.9, 
                          marker='o', edgecolors='darkblue', linewidth=2, label='Agent Position')
                print(f"[INFO] Agent stayed at single position: {traj_array_2d[0]}")
            else:
                pass
                print(f"[WARN] No valid agent trajectory points to plot")
        
        # Plot start and goal with stars (using GT path start/end points) - with reverse mapping
        # Use GT locations for accurate start/goal positions
        if episode.get("gt_locations") and len(episode["gt_locations"]) >= 2:
            pass
            gt_positions_3d = np.array(episode["gt_locations"])
            start_pos_3d = gt_positions_3d[0]  # First GT point
            goal_pos_3d = gt_positions_3d[-1]   # Last GT point
            
            # Apply reverse mapping to start and goal positions
            start_pos_2d = reverse_position_mapping(start_pos_3d[0], start_pos_3d[1], map_data)
            goal_pos_2d = reverse_position_mapping(goal_pos_3d[0], goal_pos_3d[1], map_data)
        else:
            pass
            # Fallback to episode data if GT not available (also apply reverse mapping)
            start_pos_3d = episode.get("start_position", [0, 0, 0])
            goal_pos_3d = episode.get("goals", [{}])[0].get("position", [0, 0, 0])
            start_pos_2d = reverse_position_mapping(start_pos_3d[0], start_pos_3d[1], map_data)
            goal_pos_2d = reverse_position_mapping(goal_pos_3d[0], goal_pos_3d[1], map_data)
        
        print(f"[INFO] Start 3D: {start_pos_3d[:2]} -> 2D: {start_pos_2d}", flush=True)
        print(f"[INFO] Goal 3D: {goal_pos_3d[:2]} -> 2D: {goal_pos_2d}", flush=True)
        
        ax.scatter(start_pos_2d[0], start_pos_2d[1], c='orange', s=200, marker='*', 
                  edgecolors='black', linewidth=2)
        ax.scatter(goal_pos_2d[0], goal_pos_2d[1], c='green', s=200, marker='*', 
                  edgecolors='black', linewidth=2)
        
        # Add text annotations (like your code)
        ax.text(start_pos_2d[0], start_pos_2d[1] + 0.5, "START", color='yellow', fontsize=12,
                ha='center', va='center', fontweight='bold')
        ax.text(goal_pos_2d[0], goal_pos_2d[1] + 0.5, "GOAL", color='yellow', fontsize=12,
                ha='center', va='center', fontweight='bold')
        
        # Customize plot (no legend, similar to your style)
        print("[INFO] Customizing plot appearance...", flush=True)
        ax.set_xlim(min_x - 1, max_x + 1)
        ax.set_ylim(min_y - 1, max_y + 1)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'2D Navigation Map - Scene {episode["scene_name"]} Episode {episode["episode_id"]}')
        # No legend, no grid for cleaner look like your code
        ax.set_aspect('equal')
        
        # Save visualization
        vis_path = output_dir / f"trajectory_visualization_{episode['scene_name']}_{episode['episode_id']}.png"
        print(f"[INFO] Saving visualization to: {vis_path}", flush=True)
        plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[INFO] Trajectory visualization saved successfully to: {vis_path}", flush=True)
        
    except Exception as e:
        pass
        print(f"[ERROR] Failed to create trajectory visualization: {e}", flush=True)
        import traceback
        traceback.print_exc()
        plt.close('all')  # Clean up any open figures


def run_episode(ep: Dict[str, Any], out_root: Path, vlm_host: str, vlm_port: int, env: SimpleVLNEnv,
                 hz: int = 30, max_steps: int = 200, fps: int = 10, map_path: str = "", 
                 disable_collision: bool = False, disable_autopilot: bool = False, 
                 model_type: str = "navdp", input_type: str = None, output_type: str = None, 
                 protocol: str = None, task_type: str = "vln", task_config: Dict[str, Any] = None, 
                 args: Any = None) -> None:
                     # Earliest debug info - confirm function called
    print(f"[DEBUG_ENTRY] run_episode function starting: {ep['episode_id']}", flush=True)
    # Write directly to stderr to ensure output not filtered
    import sys
    sys.stderr.write(f"[STDERR_DEBUG] run_episode function starting: {ep['episode_id']}\n")
    sys.stderr.flush()
    
    # Safe args parameter handling - use global variable for log_and_print access
    global perf_opts
    perf_opts = {
        'batch_logging': getattr(args, 'batch_logging', False) if args else False,
        'minimal_logging': getattr(args, 'minimal_logging', False) if args else False,
        'low_res': getattr(args, 'low_res', False) if args else False,
        'save_debug_files': getattr(args, 'save_debug_files', False) if args else False,
        'save_videos': getattr(args, 'save_videos', False) if args else False,
        'save_vlm_inputs': getattr(args, 'save_vlm_inputs', False) if args else False,
        'fast_mode': getattr(args, 'fast_mode', False) if args else False,
        'ultra_fast': getattr(args, 'ultra_fast', False) if args else False,
        'enable_vlm_cache': getattr(args, 'enable_vlm_cache', False) if args else False,
        'adaptive_timeout': getattr(args, 'adaptive_timeout', False) if args else False,
        'silent_logging': getattr(args, 'silent_logging', False) if args else False,
        'terminal_only': getattr(args, 'terminal_only', False) if args else False,
    }
    
    
    # Initialize task type system
    if task_config is None:
        pass
        task_config = {
            "goal_radius": 0.5,
            "max_episode_time": 80.0,  # 80 second time limit for no-goal tasks
            "collision_penalty": True,
            "min_exploration_coverage": 0.25
        }
    
    # Adapt episode data for specified task type
    adapted_episode = adapt_episode_for_task(ep, task_type)
    
    # Create task instance
    navigation_task = TaskTypeManager.create_task(task_type, task_config)
    
    result_dir = out_root / str(adapted_episode["scene_name"]) / str(adapted_episode["episode_id"])
    meas_dir = result_dir / "measurements"
    vid_dir = result_dir / "videos"
    log_path = result_dir / "episode.log"
    meas_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    # Create task ID and display episode info
    task_id = f"{adapted_episode['scene_name']}_Trajectory_{adapted_episode['episode_id']}"
    
    # Get instruction using task system (supports different task types)
    instruction = navigation_task.get_instruction(adapted_episode, step=0)
    
    print(f"[INFO] ===== Starting Episode =====", flush=True)
    print(f"[INFO] Task ID: {task_id}", flush=True)
    print(f"[INFO] Task Type: {task_type.upper()}", flush=True)
    print(f"[INFO] Scene: {adapted_episode['scene_name']}", flush=True)
    print(f"[INFO] Episode: {adapted_episode['episode_id']}", flush=True)
    print(f"[INFO] Instruction: {instruction}", flush=True)
    
    # Display task-specific info
    goal_pos = navigation_task.get_goal_position(adapted_episode)
    goal_radius = navigation_task.get_goal_radius(adapted_episode)
    print(f"[INFO] Goal Position: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})", flush=True)
    print(f"[INFO] Goal Radius: {goal_radius:.2f}m", flush=True)
    print(f"[INFO] Start pos: {adapted_episode.get('start_position', ep.get('start_position', 'N/A'))}", flush=True)
    print(f"[INFO] Original Goal pos: {ep.get('goals', [{}])[0].get('position', 'N/A')}", flush=True)
    print(f"[INFO] GT path length: {len(ep.get('gt_locations', []))} waypoints", flush=True)
    print(f"[INFO] ============================", flush=True)
    
    # Open log file for the entire function execution
    logf = open(log_path, "w")
    
    # Configure logging module to also write to episode.log
    # Clear existing handlers to avoid duplicate config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create file handler to write all logging messages to episode.log
    file_handler = logging.FileHandler(str(log_path), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Create console handler to maintain console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    print(f"[DEBUG] Logging configured to: {log_path}", flush=True)
    logging.info("[DEBUG] Logging configuration complete")
    
    # Write initial info to log file
    logf.write(f"episode_id={ep['episode_id']} scene={ep['scene_name']}\n")
    logf.write(f"task_id={task_id}\n")
    logf.write(f"instruction={instruction}\n")
    
    # Add additional info records
    if 'instruction_type' in ep and ep['instruction_type']:
        pass
        logf.write(f"instruction_type={ep['instruction_type']}\n")
    if 'instruction_index' in ep:
        pass
        logf.write(f"instruction_index={ep['instruction_index']}\n")
    if 'trajectory_id' in ep:
        pass
        logf.write(f"trajectory_id={ep['trajectory_id']}\n")
    if 'start_item' in ep and ep['start_item']:
        pass
        logf.write(f"start_item={ep['start_item']}\n")
    if 'end_item' in ep and ep['end_item']:
        pass
        logf.write(f"end_item={ep['end_item']}\n")
        
    logf.write(f"start_position={ep['start_position']}\n")
    logf.write(f"start_rotation={ep['start_rotation']}\n")
    logf.write(f"goal_radius={ep['goals'][0]['radius']}\n")
    logf.write(f"max_steps={max_steps}\n")
    logf.write(f"hz={hz}\n")
    logf.write("="*50 + "\n")  # Separator
    logf.flush()

    # Environment is now passed as parameter - reset it for this episode
    print(f"[DEBUG] Resetting environment for episode {ep['episode_id']}")
    # Note: Environment is already initialized and passed from main function
    
    # Set environment log function
    def env_log_func(msg):
        logf.write(msg + "\n")
        logf.flush()
    env.set_log_function(env_log_func)
    if disable_collision:
        pass
        env.set_collision_detection(False)
        print(f"[CONFIG] Collision detection disabled, agent can move freely")
    
    # Silent mode: redirect all print output
    if perf_opts['silent_logging']:
        pass
        # Set environment variable so other modules know silent mode is enabled
        import os
        os.environ['SILENT_LOGGING_MODE'] = 'True'
        import sys
        import io
        
        class SilentPrintFilter:
            def __init__(self, original_stdout):
                self.original_stdout = original_stdout
                # Consecutive empty line compression flag
                self._last_was_newline = False
                self.excluded_keywords = [
                    '[OBJECT_SUCCESS]', '[RGB_CAPTURE]', '[COLLISION_VIS]', '[CAMERA_UPDATE]',
                    '[COLLISION_2D]', '[PHYSICS]', '[DEPTH_CAPTURE]', '[PERF]',
                    '[YAW_UPDATE]', '[COORD_TRANSFORM]', '[POSITION]', '[VELOCITY]'
                ]
                self.important_keywords = [
                    '[ERROR]', '[WARN]', '✅', '❌', '⏭️',
                    '===== Processing Episode', 'Episode completed', 'Episode failed',
                    '===== Starting Episode', '[CHECKPOINT]', '[BATCH]',
                    'Model:', 'Progress:', 'Elapsed:', 'Success Rate:', 'Test Complete', 'SAGE-Bench', '======',
                    '🚀', '📊', '⏱️', '📈', '⚡', '🎉',  # Progress bar emoji
                    'SAGE-Bench Test Progress', 'Total Episodes:', 'Start Time:', 'Est. Remaining:', 'Avg. Time:',
                    '[Progress]', '[Status]', 'Episode Info'  # Progress bar related text
                ]
            
            def write(self, text):
                # Non-empty text
                if text.strip():
                    # Priority check if contains important info
                    if any(keyword in text for keyword in self.important_keywords):
                        self.original_stdout.write(text)
                        self.original_stdout.flush()
                        self._last_was_newline = False
                    # Check if it's debug info that needs to be excluded
                    elif any(keyword in text for keyword in self.excluded_keywords):
                        # Don't display in terminal
                        pass
                    # Use regex to match debug tags with square brackets (general filtering)
                    elif self._is_debug_log(text):
                        # Don't display in terminal
                        pass
                    else:
                        # Other content displayed normally
                        self.original_stdout.write(text)
                        self.original_stdout.flush()
                        self._last_was_newline = False
                else:
                    # Blank output: compress consecutive empty lines, keep only one newline
                    if not self._last_was_newline:
                        self.original_stdout.write("\n")
                        self._last_was_newline = True
                return len(text)

            
            def _is_debug_log(self, text):
                """Check if it's a debug log (format: [Tag] content)"""
                import re
                # Match debug logs starting with [Tag]
                debug_pattern = r'^\[([A-Z_]+)\]'
                return re.match(debug_pattern, text.strip())
            
            def flush(self):
                self.original_stdout.flush()
        
        # Replace stdout but keep original reference for important info
        original_stdout = sys.stdout
        sys.stdout = SilentPrintFilter(original_stdout)
    
    log_and_print(f"[DEBUG_MAIN] Preparing to set starting pose...")
    env.set_start_pose(ep["start_position"], ep["start_rotation"])
    
    # Reset episode time (especially important for no-goal tasks)
    if hasattr(env, 'reset_episode_time'):
        env.reset_episode_time()
        log_and_print(f"[DEBUG_MAIN] Episode time has been reset")
    
    log_and_print(f"[DEBUG_MAIN] Starting pose set successfully")
    
    # Initialize measure manager - select different metrics based on task type
    if task_type.lower() == "nogoalnav":
        measure_manager = nogoal_measures(adapted_episode)
        log_and_print(f"[DEBUG_MAIN] Using no-goal task specific metrics")
    else:
        measure_manager = default_measures(adapted_episode)
        log_and_print(f"[DEBUG_MAIN] Using default VLN task metrics")
    log_and_print(f"[DEBUG_MAIN] Preparing to reset measure_manager...")
    try:
        pass
        measure_manager.reset(env)
        log_and_print(f"[DEBUG_MAIN] measure_manager reset successful")
    except Exception as e:
        pass
        log_and_print(f"[ERROR] measure_manager reset failed: {e}")
        import traceback
        log_and_print(f"[ERROR] Stack trace: {traceback.format_exc()}")
        return  # Early exit

    frames: List[np.ndarray] = []
    images_for_vlm: List[Image.Image] = []
    trajectory_positions: List[np.ndarray] = []  # Track agent positions
    instr = ep["instruction"]["instruction_text"]
    
    # 🚀 Simple VLM response cache (for performance optimization)
    vlm_response_cache = {}
    last_vlm_response = None

    # 🚀 Optimized Warm-up: avoid apply_cmd_for infinite loop
    log_and_print("[INFO] Optimized Warm-up capture - avoiding apply_cmd_for")
    sys.stderr.write(f"[STDERR_DEBUG] Starting optimized warm-up\n")
    warm_tries = 0
    first_rgb = None
    max_warm_tries = 10  # Reduced number of attempts
    
    while warm_tries < max_warm_tries and first_rgb is None:
        sys.stderr.write(f"[STDERR_DEBUG] warm-up attempt {warm_tries}/{max_warm_tries}: Getting RGB directly...\n")
        first_rgb = env.get_rgb()
        if first_rgb is None:
            pass
            sys.stderr.write(f"[STDERR_DEBUG] RGB capture failed, waiting briefly before retry...\n")
            sys.stderr.flush()
            # Use simple time wait instead of apply_cmd_for, avoid camera update loop
            import time
            time.sleep(0.1)  # 100ms wait
            warm_tries += 1
        else:
            pass
            sys.stderr.write(f"[STDERR_DEBUG] RGB capture successful! shape={first_rgb.shape}\n")
            sys.stderr.flush()
            break
    
    # If still failed, create a dummy RGB image to continue execution
    if first_rgb is None:
        pass
        sys.stderr.write(f"[STDERR_DEBUG] warm-up failed, creating dummy RGB image\n")
        sys.stderr.flush()
        # Create dummy RGB image (480x640x3)
        first_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        log_and_print("[WARN] Warm-up failed, using dummy RGB image to continue")
    if first_rgb is not None:
        pass
        images_for_vlm.append(Image.fromarray(first_rgb))
        frames.append(first_rgb)
        log_and_print("[INFO] First RGB captured")
    else:
        pass
        log_and_print("[WARN] No RGB during warm-up; proceeding")
    
    # Record initial position
    log_and_print(f"[DEBUG_MAIN] Preparing to get initial position...")
    try:
        pass
        initial_pos = env.get_agent_pos()
        trajectory_positions.append(initial_pos)
        log_and_print(f"[INFO] Initial position: {initial_pos}")
    except Exception as e:
        pass
        log_and_print(f"[ERROR] Failed to get initial position: {e}")
        import traceback
        log_and_print(f"[ERROR] Stack trace: {traceback.format_exc()}")
        return  # Early exit

    steps_run = 0
    turn_only_count = 0
    stop_override_count = 0  # Count STOP command overrides (only for no-goal tasks)
    
    # Only non-no-goal tasks need to get initial distance
    prev_dist = None
    if task_type != "nogoalnav":
        log_and_print(f"[DEBUG_MAIN] Preparing to get initial distance...")
        try:
            prev_dist = measure_manager.measures["distance_to_goal"].get()
            log_and_print(f"[DEBUG_MAIN] Initial distance: {prev_dist}")
        except Exception as e:
            log_and_print(f"[ERROR] Failed to get initial distance: {e}")
            import traceback
            log_and_print(f"[ERROR] Stack trace: {traceback.format_exc()}")
            return  # Early exit
    else:
        log_and_print(f"[DEBUG_MAIN] No-goal task, skipping distance initialization")
    
    # 🔍 Debug: Check if entering main loop
    log_and_print(f"[DEBUG_MAIN] About to start main loop, max_steps={max_steps}")
    
    for step in range(max_steps):
        # 🕒 Update environment time state (for no-goal tasks)
        env.update_time_and_reset_collision()
        
        # 🚫 No-goal task special handling: check time limit and collision termination
        if task_type.lower() == "nogoalnav":
            current_time = env._current_time
            episode_time = current_time - env._episode_start_time
            
            # Check if timeout (80 second limit)
            max_episode_time = task_config.get("max_episode_time", 80.0)
            if episode_time >= max_episode_time:
                log_and_print(f"[NOGOAL] Episode timeout termination ({episode_time:.1f}s >= {max_episode_time}s)")
                env.is_stop_called = True
                break
            
            # Check if collision occurred
            if env._collision_detected:
                log_and_print(f"[NOGOAL] Collision detected, Episode terminated immediately (time: {episode_time:.1f}s)")
                env.is_stop_called = True
                break
            
            log_and_print(f"[NOGOAL] Exploration in progress... Time: {episode_time:.1f}s/{max_episode_time}s")
        
        log_and_print(f"[DEBUG_MAIN] Step {step}: About to query VLM")
        log_and_print(f"[INFO] Step {step}")
        
        # 🔧 Smart image capture based on input type (performance optimized)
        rgb = None
        depth = None
        
        # Decide what data to get based on modular config
        need_depth = False
        if input_type == "rgbd":
            need_depth = True
            if not perf_opts['minimal_logging']:
                pass  # Can add logging or debug code here
            log_and_print(f"[IMAGE_INPUT] RGB-D mode: Need to get RGB and depth")
        elif input_type == "rgb":
            need_depth = False
            if not perf_opts['minimal_logging']:
                pass
            log_and_print(f"[IMAGE_INPUT] RGB mode: Only need to get RGB image")
        elif model_type and model_type in ["navdp"]:
            need_depth = True
            if not perf_opts['minimal_logging']:
                pass
            log_and_print(f"[IMAGE_INPUT] Predefined model {model_type}: Need RGB-D")
        elif model_type and model_type in ["navila", "navid"]:
            need_depth = False
            if not perf_opts['minimal_logging']:
                pass
            log_and_print(f"[IMAGE_INPUT] Predefined model {model_type}: Only need RGB")
        else:
            need_depth = True  # Default fallback
            if not perf_opts['minimal_logging']:
                pass
            log_and_print(f"[IMAGE_INPUT] Default mode: Getting RGB-D")
        
        # 🚀 Performance optimization: frame skip detection
        # If no significant movement for several steps, reuse image to reduce rendering overhead
        skip_rendering = False
        if step > 0 and len(trajectory_positions) >= 2:
            last_pos = trajectory_positions[-1]
            second_last_pos = trajectory_positions[-2] if len(trajectory_positions) >= 2 else last_pos
            movement_distance = np.linalg.norm(last_pos - second_last_pos)
            if movement_distance < 0.05 and not perf_opts['fast_mode']:  # 5cm movement threshold
                skip_rendering = False  # Temporarily disable frame skip for stability
                if not perf_opts['minimal_logging']:
                    log_and_print(f"[PERF] Minimal movement ({movement_distance:.3f}m), considering image reuse")
        
        # Get RGB image (always needed)
        if hasattr(env, 'get_rgb'):
            pass
            rgb = env.get_rgb()
            if rgb is not None:
                pass
                # 🚀 Performance optimization: low resolution processing
                if perf_opts['low_res'] and rgb.shape[:2] != (240, 320):
                    pass
                    try:
                        pass
                        import cv2
                        rgb = cv2.resize(rgb, (320, 240))
                    except ImportError:
                        pass
                        # fallback to PIL if cv2 not available
                        rgb_pil = Image.fromarray(rgb).resize((320, 240))
                        rgb = np.array(rgb_pil)
                
                images_for_vlm.append(Image.fromarray(rgb))
                frames.append(rgb)
                if not perf_opts['minimal_logging']:
                    pass
                log_and_print(f"[DEBUG] ✓ Got RGB image: shape={rgb.shape}")
            else:
                pass
                if not perf_opts['minimal_logging']:
                    pass
                log_and_print("[WARN] ✗ Failed to get RGB")
        
        # Only get depth map when needed
        if need_depth:
            pass
            if hasattr(env, 'get_depth'):
                pass
                depth = env.get_depth()
                if depth is not None:
                    pass
                    # 🚀 Performance optimization: low resolution processing
                    if perf_opts['low_res'] and depth.shape[:2] != (240, 320):
                        pass
                        try:
                            pass
                            import cv2
                            depth = cv2.resize(depth, (320, 240))
                        except ImportError:
                            pass
                            # fallback to PIL for depth
                            depth_pil = Image.fromarray(depth).resize((320, 240))
                            depth = np.array(depth_pil)
                    
                    if not perf_opts['minimal_logging']:
                        pass
                    log_and_print(f"[DEBUG] ✓ 获取到深度图: shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]m")
                else:
                    pass
                    if not perf_opts['minimal_logging']:
                        pass
                    log_and_print("[WARN] ✗ Failed to get depth")
            else:
                pass
                if not perf_opts['minimal_logging']:
                    pass
                log_and_print("[WARN] Environment does not support depth capture")
        else:
            pass
            if not perf_opts['minimal_logging']:
                pass
            log_and_print(f"[DEBUG] RGB mode: Skipping depth capture")

        # Measure current step
        measure_manager.update(env)
        if task_type != "nogoalnav" and "distance_to_goal" in measure_manager.measures:
            d = measure_manager.measures["distance_to_goal"].get()
            log_and_print(f"[INFO] DistanceToGoal(before): {d:.3f}")
        else:
            d = None

        # Query VLM
        try:
            pass
            
            # Get task instruction for current step
            current_instruction = navigation_task.get_instruction(adapted_episode, step=steps_run)
            
            # 🚀 Performance optimization: VLM response cache check
            # Create simple cache key for similar scenes and instructions
            cache_hit = False
            cache_key = None
            if perf_opts.get('enable_vlm_cache', False) and len(trajectory_positions) >= 2:
                current_pos = trajectory_positions[-1]
                cache_key = f"{current_instruction}_{current_pos[0]:.1f}_{current_pos[1]:.1f}"
                if cache_key in vlm_response_cache:
                    resp = vlm_response_cache[cache_key]
                    cache_hit = True
                    log_and_print(f"[CACHE] 🎯 Cache hit, reusing VLM response")
            
            if not cache_hit:
                # No cache hit, need to query VLM
                pass
            else:
                # Cache hit, skip VLM query and image saving
                log_and_print(f"[INFO] VLM resp (cached): {resp}")
                
            # 🎥 Conditional VLM input image saving - based on debug file settings (forced off in fast-mode)
            if not cache_hit:
                should_save_vlm_inputs = (
                    perf_opts.get('save_debug_files', False) or perf_opts.get('save_vlm_inputs', False)
                ) and not perf_opts.get('fast_mode', False)
                
                if should_save_vlm_inputs:
                    debug_img_path = vid_dir / "vlm_inputs" / f"step_{step:03d}_input.png"
                    debug_depth_path = vid_dir / "vlm_inputs" / f"step_{step:03d}_depth.png"
                    debug_img_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if rgb is not None:
                        Image.fromarray(rgb).save(debug_img_path)
                        if not perf_opts['minimal_logging']:
                            log_and_print(f"[DEBUG] Saved VLM input RGB: {debug_img_path}")
                else:
                    if step == 0 and not perf_opts['minimal_logging']:  # Only prompt once on first step
                        log_and_print(f"[PERF] Skipping VLM input image saving for performance (debug file disabled)")
                
                log_and_print(f"[VLM_INPUT] Task: {task_type.upper()} | Sending to VLM: \"{current_instruction}\"")
                log_and_print(f"[VLM_DEBUG] VLM server: {vlm_host}:{vlm_port}")
                log_and_print(f"[VLM_DEBUG] Image count: {len(images_for_vlm)}")
                
                # Use VLM model, supports multiple configuration methods
                current_yaw = env.get_yaw()
                depth_images = [depth] if depth is not None else None
                log_and_print(f"[VLM_DEBUG] Preparing VLM query, yaw: {current_yaw:.3f}")
                
                # 🚀 Performance optimization: Dynamic VLM timeout
                # Adjust timeout based on network conditions and historical response time
                import time as time_module
                vlm_start_time = time_module.time()
                
                # Prefer modular config if complete parameters provided
                log_and_print(f"[DEBUG] 检查模块化配置参数: input_type={input_type}, output_type={output_type}, protocol={protocol}, vlm_host={vlm_host}, vlm_port={vlm_port}")
                if input_type and output_type and protocol:
                    pass
                    log_and_print(f"[VLM_CONFIG] Using modular config: {input_type} + {output_type} + {protocol}")
                    log_and_print(f"[VLM_DEBUG] Starting VLM query...")
                    resp = query_vlm(images_for_vlm, current_instruction, vlm_host, vlm_port, 
                                    current_yaw=current_yaw, depth_images=depth_images,
                                    input_type=input_type, output_type=output_type, protocol=protocol)
                    log_and_print(f"[VLM_DEBUG] VLM query completed")
                # Fallback to predefined model type
                elif model_type:
                    pass
                    log_and_print(f"[VLM_CONFIG] Using predefined model type: {model_type}")
                    log_and_print(f"[VLM_DEBUG] Starting VLM query...")
                    resp = query_vlm(images_for_vlm, current_instruction, vlm_host, vlm_port, 
                                    current_yaw=current_yaw, depth_images=depth_images, model_type=model_type)
                    log_and_print(f"[VLM_DEBUG] VLM查询完成")
                else:
                    pass
                    log_and_print(f"[ERROR] Must provide model_type or (input_type, output_type, protocol)")
                    resp = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration_s": 0.0, "stop": True}
                
                log_and_print(f"[INFO] VLM resp: {resp}")
                
                # 🚀 Performance stats: record VLM response time
                vlm_end_time = time_module.time()
                vlm_duration = vlm_end_time - vlm_start_time
                if not perf_opts.get('minimal_logging', False):
                    log_and_print(f"[PERF] VLM response time: {vlm_duration:.2f}s")
                
                # Save response to cache
                if perf_opts.get('enable_vlm_cache', False) and cache_key is not None:
                    vlm_response_cache[cache_key] = resp
                last_vlm_response = resp
            
        except Exception as e:
            pass
            log_and_print(f"[ERROR] Failed to query VLM: {e}")
            # 🚀 Smart fallback: if previous response exists, use modified version
            if last_vlm_response and not perf_opts.get('fast_mode', False):
                # Use modified version of last response as fallback
                resp = {
                    "vx": last_vlm_response.get("vx", 0.1) * 0.5,  # Reduce speed
                    "vy": last_vlm_response.get("vy", 0.0) * 0.5,
                    "yaw_rate": last_vlm_response.get("yaw_rate", 0.0) * 0.5,
                    "duration_s": 1.0,
                    "stop": False
                }
                log_and_print(f"[FALLBACK] Using modified previous VLM response")
            else:
                # Don't stop immediately, continue with a default action
                resp = {"vx": 0.1, "vy": 0.0, "yaw_rate": 0.0, "duration_s": 1.0, "stop": False}

        # 🚫 No-goal task special handling: STOP not allowed, convert to exploration command
        if resp.get("stop", False):
            if task_type.lower() == "nogoalnav":
                # In no-goal task, convert STOP to exploration command
                stop_override_count += 1
                log_and_print(f"[NOGOAL_OVERRIDE] STOP command detected, converting to exploration command (#{stop_override_count})")
                resp["stop"] = False
                
                # Use step count to create some variation, avoid repetitive behavior
                import random
                random.seed(steps_run)  # Use step count as seed for reproducibility
                
                # Randomly select exploration behavior: forward, turn left, turn right
                action_type = random.choice(['forward', 'turn_left', 'turn_right'])
                
                if action_type == 'forward':
                    resp["vx"] = 0.2
                    resp["vy"] = 0.0
                    resp["yaw_rate"] = 0.0
                elif action_type == 'turn_left':
                    resp["vx"] = 0.1
                    resp["vy"] = 0.0
                    resp["yaw_rate"] = 0.5  # Turn left
                else:  # turn_right
                    resp["vx"] = 0.1
                    resp["vy"] = 0.0
                    resp["yaw_rate"] = -0.5  # Turn right
                
                resp["duration_s"] = 1.0
                log_and_print(f"[NOGOAL_OVERRIDE] 新命令({action_type}): vx={resp['vx']}, vy={resp['vy']}, yaw_rate={resp['yaw_rate']}, duration={resp['duration_s']}")
            else:
                # Other tasks handle STOP normally
                env.is_stop_called = True
        
        # execute
        try:
            pass
            # Log command details for debugging
            cmd_vx = resp.get("vx", 0.0)
            cmd_vy = resp.get("vy", 0.0)
            cmd_yaw_rate = resp.get("yaw_rate", 0.0)
            cmd_duration = resp.get("duration_s", 0.0)
            
            log_and_print(f"[DEBUG] Executing command: vx={cmd_vx:.3f}, vy={cmd_vy:.3f}, yaw_rate={cmd_yaw_rate:.3f}, duration={cmd_duration:.3f}")
            
            # Execute movement command
            log_and_print(f"[DEBUG] About to call env.apply_cmd_for...")
            env.apply_cmd_for(cmd_vx, cmd_vy, cmd_yaw_rate, cmd_duration)
            log_and_print(f"[DEBUG] Finished env.apply_cmd_for")
            
            log_and_print(f"[DEBUG] About to update measures...")
            measure_manager.update(env)
            log_and_print(f"[DEBUG] Finished updating measures")
            
        except Exception as e:
            pass
            log_and_print(f"[ERROR] Failed to execute command or update measures: {e}")
            import traceback
            tb = traceback.format_exc()
            log_and_print(f"[ERROR] Full traceback: {tb}")
            env.is_stop_called = True
            break

        steps_run += 1
        if task_type != "nogoalnav" and "distance_to_goal" in measure_manager.measures:
            new_dist = measure_manager.measures["distance_to_goal"].get()
            log_and_print(f"[INFO] DistanceToGoal(after): {new_dist:.3f}")
            prev_dist = new_dist
        else:
            new_dist = None
        
        # Record current position for trajectory visualization
        current_pos = env.get_agent_pos()
        trajectory_positions.append(current_pos)
        
        # 🎯 Task-specific success check
        if task_type.lower() == "nogoalnav":
            # No-goal task: use task instance's success judgment
            current_pos = env.get_agent_pos()
            episode_time = env._current_time - env._episode_start_time
            exploration_coverage = 0.0
            if "exploration_coverage" in measure_manager.measures:
                exploration_coverage = measure_manager.measures["exploration_coverage"].get()
            
            task_success = navigation_task.is_success(
                current_pos, adapted_episode, 
                collision_detected=env._collision_detected,
                episode_time=episode_time,
                exploration_coverage=exploration_coverage
            )
            
            if task_success:
                log_and_print(f"[NOGOAL] Exploration task complete! Time: {episode_time:.1f}s, Coverage: {exploration_coverage:.2%}")
                env.is_stop_called = True
        else:
            # Traditional VLN task: distance check
            goal_radius = 0.5  # default radius
            if ep.get("goals") and len(ep["goals"]) > 0 and "radius" in ep["goals"][0]:
                goal_radius = ep["goals"][0]["radius"]
            if new_dist < goal_radius:
                log_and_print(f"[INFO] Within goal radius ({goal_radius}); stopping")
                env.is_stop_called = True
        
        if env.is_stop_called:
            log_and_print("[INFO] Stop called; breaking")
            break
    
    # Save results even if we break early
    log_and_print(f"[INFO] Episode ended after {steps_run} steps")

    try:
        pass
        log_and_print("[INFO] Saving measurements...")
        meas_path = meas_dir / f"{ep['episode_id']}.json"
        
        # Get original measurements
        measurements = measure_manager.dump()
        
        # Add episode details
        measurements["episode_info"] = {
            "episode_id": ep['episode_id'],
            "scene_name": ep['scene_name'],
            "task_id": task_id,
            "instruction": instruction,
            "start_position": ep['start_position'],
            "start_rotation": ep['start_rotation'],
            "goal_radius": ep['goals'][0]['radius'],
            "max_steps": max_steps,
            "hz": hz
        }
        
        # Add special stats for no-goal tasks
        if task_type.lower() == "nogoalnav":
            measurements["episode_info"]["stop_override_count"] = stop_override_count
            log_and_print(f"[INFO] No-goal task stats: STOP command overridden {stop_override_count} times")
        
        # Add new details
        if 'instruction_type' in ep and ep['instruction_type']:
            pass
            measurements["episode_info"]["instruction_type"] = ep['instruction_type']
        if 'instruction_index' in ep:
            pass
            measurements["episode_info"]["instruction_index"] = ep['instruction_index']
        if 'trajectory_id' in ep:
            pass
            measurements["episode_info"]["trajectory_id"] = ep['trajectory_id']
        if 'start_item' in ep and ep['start_item']:
            pass
            measurements["episode_info"]["start_item"] = ep['start_item']
        if 'end_item' in ep and ep['end_item']:
            pass
            measurements["episode_info"]["end_item"] = ep['end_item']
        
        with open(meas_path, "w") as f:
            json.dump(measurements, f, indent=2)
        log_and_print(f"[INFO] ✅ Measurements saved (key output): {meas_path}")
    except Exception as e:
        pass
        log_and_print(f"[ERROR] Failed to save measurements: {e}")

     # 🎥 Conditional video saving - based on debug file settings (forced off in fast-mode)
    should_save_video = (
        perf_opts.get('save_debug_files', False) or perf_opts.get('save_videos', False)
    ) and not perf_opts.get('fast_mode', False)
     
    if should_save_video:
        try:
            log_and_print("[INFO] Saving episode video...")
            # save video (use placeholder if empty)
            video_path = (vid_dir / f"{ep['episode_id']}.mp4").resolve()
            if len(frames) == 0:
                log_and_print("[WARN] No frames captured; writing a placeholder")
                frames = [np.zeros((240, 320, 3) if perf_opts['low_res'] else (480, 640, 3), dtype=np.uint8)]
            SimpleVLNEnv.write_video(frames, str(video_path), fps=fps)
            log_and_print(f"[INFO] Saved video to: {video_path}")
        except Exception as e:
            log_and_print(f"[ERROR] Failed to save video: {e}")
    else:
        # Processing when not saving video
        log_and_print("[PERF] Skipping video saving for performance (debug file disabled)")
    
    if task_type != "nogoalnav" and "distance_to_goal" in measure_manager.measures:
        final_dist = measure_manager.measures['distance_to_goal'].get()
        log_and_print(f"[INFO] Steps run: {steps_run}, Final distance_to_goal: {final_dist:.3f}")
    else:
        log_and_print(f"[INFO] Steps run: {steps_run} (No-goal task)")
    
    # 📊 2D trajectory visualization - key output, always keep
    try:
        pass
        if map_path:
            pass
            log_and_print("[INFO] Creating trajectory visualization...")
            visualize_trajectory(ep, trajectory_positions, map_path, result_dir)
            log_and_print("[INFO] ✅ 2D trajectory visualization saved (key output)")
        else:
            pass
            log_and_print("[INFO] No map path provided, skipping trajectory visualization")
    except Exception as e:
        pass
        log_and_print(f"[ERROR] Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # 🚀 Ensure all log buffers are flushed
    try:
        pass
        flush_log_buffer()
    except Exception as e:
        pass
        print(f"[WARN] Failed to flush log buffer: {e}")
    
    # 🔇 Restore original stdout, print function and env vars (if silent mode was used)
    if perf_opts['silent_logging']:
        pass
        if 'original_stdout' in locals():
            pass
            sys.stdout = original_stdout
        # Clean up environment variables
        import os
        if 'SILENT_LOGGING_MODE' in os.environ:
            pass
            del os.environ['SILENT_LOGGING_MODE']
    
        # Environment will be closed by main function after all episodes
        log_and_print("[INFO] Episode completed, environment remains open for next episode")
        
        # Clean up logging handlers
        try:
            pass
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
            log_and_print("[DEBUG] Logging handlers cleared")
        except Exception as e:
            pass
            log_and_print(f"[WARN] Failed to clear logging handlers: {e}")
        
        # Close log file
        try:
            pass
            logf.close()
            log_and_print("[INFO] Log file closed")
        except Exception as e:
            pass
            log_and_print(f"[WARN] Failed to close log file: {e}")


# 🚀 Performance optimization: batch log buffering (moved to run_episode internally)

# Helper function to log to both console and file (performance optimized)
def log_and_print(msg: str):
    """Print to console and write to log file (supports silent log mode)"""
    global perf_opts
    
    # Inline debug log detection function
    def _is_debug_log_msg_inline(msg: str) -> bool:
        """Check if it's a debug log (format: [Tag] content)"""
        import re
        debug_pattern = r'^\[([A-Z_]+)\]'
        return bool(re.match(debug_pattern, msg.strip()))
    
    # 🔇 Silent log mode: only write to log, print to terminal only for key info
    # Safety check: perf_opts may not exist (when called outside function)
    try:
        pass
        silent_logging = perf_opts['silent_logging']
    except (NameError, KeyError):
        pass
        silent_logging = False
    
    if silent_logging:
        # Only print truly important info to terminal, filter out detailed debug info
        important_keywords = [
            '[ERROR]', '[WARN]',
            '✅', '❌', '⏭️',  # episode状态
            '===== Processing Episode',  # episode开始
            '===== Starting Episode', '[CHECKPOINT]', '[BATCH]',  # episode状态
            'Episode completed', 'Episode failed',  # episode结果
            '模型:', '进度:', '已用时间:', '成功率:',  # 进度信息
            '测试完成', 'SAGE-Bench',  # 测试状态
            '======',  # 进度条分隔符
            '🚀', '📊', '⏱️', '📈', '⚡', '🎉',  # 进度条emoji
            'SAGE-Bench 测试进度', '总Episodes:', '开始时间:', '预计剩余:', '平均耗时:',
            '[进度]', '[状态]', 'Episode信息'  # 进度条相关文本
        ]

        # 排除详细调试信息
        excluded_keywords = [
            '[OBJECT_SUCCESS]', '[RGB_CAPTURE]', '[COLLISION_VIS]', '[CAMERA_UPDATE]',
            '[QUERY_VLM]', '[MODULAR_CLIENT]', '[SOCKET_CLIENT]', '[DEBUG]',
            '[COLLISION_2D]', '[EPISODE_RESET]', '[PHYSICS]', '[VLM]',
            '[DIRECT_MOVE]', '[DEPTH_CAPTURE]', '[IMAGE_INPUT]', '[PERF]',
            '[YAW_UPDATE]', '[SOCKET_PROTOCOL]', '[TEXT_PARSER]', '[MOVEMENT]',
            '[SAFE_MOVE]', '[MOVEMENT_RESULT]', '[COORD_TRANSFORM]', '[ACTION]',
            '[STEP]', '[ROTATION]', '[POSITION]', '[VELOCITY]', '[CONTROL]',
            '[SUCCESS]', '[ORACLE_SUCCESS]', '[CSR]', '[COL', '[INFO]'
        ]

        # 优先检查是否包含重要信息
        if any(keyword in msg for keyword in important_keywords):
            print(msg, flush=True)
        # 检查是否是需要排除的调试信息
        elif any(keyword in msg for keyword in excluded_keywords):
            pass  # 不打印，只写入日志
        # 使用正则表达式匹配所有带方括号的调试标签（通用过滤）
        elif _is_debug_log_msg_inline(msg):
            pass  # 不打印，只写入日志
        else:
            # 其他内容正常打印
            print(msg, flush=True)

        # 但所有日志都写入文件
        try:
            terminal_only = perf_opts.get('terminal_only', False)
        except (NameError, KeyError):
            terminal_only = False

        if terminal_only:
            # 仅终端模式：只打印不写日志
            print(msg, flush=True)
            return  # 直接返回，不写入日志文件
        else:
            # 默认模式：既打印又写日志
            print(msg, flush=True)

        if not terminal_only:
            try:
                batch_logging = perf_opts.get('batch_logging', False)

                # 安全检测运行期是否定义了批量日志缓冲
                has_log_buffer = 'log_buffer' in globals()
                has_buffer_size = 'log_buffer_size' in globals()
                has_flush = 'flush_log_buffer' in globals()
                has_log_file = 'logf' in globals()

                if batch_logging and has_log_buffer and has_buffer_size and has_flush:
                    try:
                        log_buffer.append(msg)
                        if len(log_buffer) >= log_buffer_size:
                            flush_log_buffer()
                    except Exception:
                        # 回退到直接写文件
                        if has_log_file:
                            try:
                                logf.write(msg + "\n")
                                logf.flush()
                                import os
                                os.fsync(logf.fileno())
                            except Exception:
                                pass
                else:
                    # 非批量/无缓冲定义：直接写日志文件（若可用）
                    if has_log_file:
                        try:
                            logf.write(msg + "\n")
                            logf.flush()
                            import os
                            os.fsync(logf.fileno())
                        except Exception:
                            pass
            except Exception:
                # 静默忽略日志写入失败，避免刷屏
                pass


# 设置VLM客户端的日志函数
set_log_function(log_and_print)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_usd_path", "--scene-path", type=str, required=True, help="Path to scene USD/USDA file (single file) or folder (batch mode)")
    parser.add_argument("--traj_json_path", "--episodes-path", type=str, help="Path to episodes JSON file (single file mode)")
    parser.add_argument("--batch_test_dir", "--batch-test-dir", type=str, help="Directory containing multiple JSON files for batch testing")
    parser.add_argument("--json_pattern", "--json-pattern", type=str, default="test_*.json", help="Pattern to match JSON files (default: test_*.json)")
    parser.add_argument("--output_root", "--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--goal-radius", "--goal_radius", type=float, default=0.5, help="Success radius in meters")
    parser.add_argument("--map_path", "--map-path", type=str, default="", help="Path to 2D semantic map JSON (single file) or folder (batch mode)")
    parser.add_argument("--vlm-host", "--vlm_host", type=str, default="localhost", help="VLM server host")
    parser.add_argument("--vlm-port", "--vlm_port", type=int, default=8888, help="VLM server port (NavDP default: 8888, NaVILA default: 54321)")
    parser.add_argument("--vlm-timeout", type=float, default=60.0, help="VLM server timeout in seconds")
    parser.add_argument("--model-type", "--model_type", type=str, default=None, 
                       help="Predefined VLM model type (e.g., navdp, navila) or use modular config params")
    
    # Modular configuration parameters
    parser.add_argument("--input-type", "--input_type", type=str, choices=["rgb", "rgbd"], 
                       help="Input type: rgb (RGB image sequence) or rgbd (RGB-D image)")
    parser.add_argument("--output-type", "--output_type", type=str, choices=["trajectory", "text"], 
                       help="Output type: trajectory (waypoints) or text (text action)")
    parser.add_argument("--protocol", type=str, choices=["http", "socket"], 
                       help="Communication protocol: http or socket")
    
    # Task type parameters
    parser.add_argument("--task-type", "--task_type", type=str, default="vln", 
                       choices=["vln", "objectnav", "pointnav", "imgnav", "nogoalnav"],
                       help="Navigation task type: vln, objectnav, pointnav, imgnav, nogoalnav")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--disable-collision", action="store_true", help="Disable collision detection for debugging")
    parser.add_argument("--disable-autopilot", action="store_true", help="Disable auto-alignment, execute VLM commands directly")
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=-1, help="Maximum number of episodes to run")
    parser.add_argument("--start-idx", "--start_idx", type=int, default=0, help="Start index in episodes list")
    parser.add_argument("--num-episodes", "--num_episodes", type=int, default=-1, help="Number of episodes to run")
    parser.add_argument("--hz", type=int, default=30, help="Simulation frequency")
    parser.add_argument("--max-steps", "--max_steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--skip-completed", "--skip_completed", action="store_true", default=True, help="Skip episodes that already have measurements files (enable checkpoint/resume functionality)")
    parser.add_argument("--no-skip-completed", "--no_skip_completed", dest="skip_completed", action="store_false", help="Disable checkpoint functionality, re-run all episodes")
    
    # Performance optimization parameters
    parser.add_argument("--fast-mode", action="store_true", help="Enable fast mode: disable debug output, reduce I/O, improve speed")
    parser.add_argument("--low-res", action="store_true", help="Use low resolution images (320x240) for faster processing")
    parser.add_argument("--minimal-logging", action="store_true", help="Minimize log output, keep only key info")
    parser.add_argument("--batch-logging", action="store_true", default=True, help="Enable batch log writing")
    parser.add_argument("--ultra-fast", action="store_true", help="Ultra-fast mode: enable all optimizations (may affect accuracy)")
    parser.add_argument("--enable-vlm-cache", action="store_true", help="Enable VLM response caching (experimental)")
    parser.add_argument("--adaptive-timeout", action="store_true", help="Enable adaptive VLM timeout")
    
    # Debug output control parameters
    parser.add_argument("--save-debug-files", action="store_true", default=False, help="Save debug files (videos, vlm_inputs) - disabled by default for performance")
    parser.add_argument("--no-debug-files", dest="save_debug_files", action="store_false", help="Disable debug file saving (default behavior)")
    parser.add_argument("--save-videos", action="store_true", help="Save episode video files")
    parser.add_argument("--save-vlm-inputs", action="store_true", help="Save VLM input images")
    
    # Progress display control parameters
    parser.add_argument("--quiet-progress", action="store_true", help="Quiet mode: simplify progress display, reduce terminal output")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display, keep only basic logs")
    parser.add_argument("--silent-logging", action="store_true", help="Silent logging mode: write to log file only, no terminal detail logs (keep progress bar)")
    parser.add_argument("--terminal-only", action="store_true", help="Terminal only mode: display only in terminal, no log file")
    
    # Backward compatible old parameters (deprecated but kept)
    parser.add_argument("--no-debug-images", action="store_true", help="[Deprecated] Use --no-debug-files instead")
    parser.add_argument("--no-videos", action="store_true", help="[Deprecated] Use --no-debug-files instead")
    
    args = parser.parse_args()

    # 验证参数：必须提供单文件路径或批量测试目录之一
    if not args.traj_json_path and not args.batch_test_dir:
        pass
        print("[ERROR] 必须提供 --traj_json_path 或 --batch_test_dir 参数之一")
        parser.print_help()
        return
    
    if args.traj_json_path and args.batch_test_dir:
        pass
        print("[ERROR] --traj_json_path 和 --batch_test_dir 不能同时提供，请选择其中一个")
        parser.print_help()
        return
    
    out_root = Path(args.output_root).resolve()
    
    # 显示断点继续功能状态
    if args.skip_completed:
        pass
        print("[INFO] 🔄 断点继续功能已启用：将跳过已完成的episode（有measurements文件）")
    else:
        pass
        print("[INFO] ⚠️  断点继续功能已禁用：将重新运行所有episode")
    
    # 🚀 超高速模式处理
    if args.ultra_fast:
        print("[INFO] ⚡ 超高速模式已启用：")
        # 启用所有性能优化
        args.fast_mode = True
        args.minimal_logging = True
        args.save_debug_files = False
        args.low_res = True
        args.silent_logging = True
        args.enable_vlm_cache = True
        args.adaptive_timeout = True
        print("     - 启用所有性能优化选项")
        print("     - 实验性VLM缓存")
        print("     - 自适应超时时间")
        print("     - ⚠️  可能影响测试准确性")
    
    # 🚀 快速模式处理
    if args.fast_mode:
        pass
        print("[INFO] 🚀 快速模式已启用：")
        # 自动启用相关的性能优化选项
        args.minimal_logging = True
        args.save_debug_files = False  # 快速模式下禁用调试文件
        args.low_res = True
        args.silent_logging = True  # 快速模式下启用静默日志
        print("     - 最小化日志输出")
        print("     - 禁用调试文件保存 (videos + vlm_inputs)")
        print("     - 使用低分辨率图像(320x240)")
        print("     - 启用静默日志模式 (减少终端输出，保留进度条)")
        print("     - 保留关键输出 (measurements + episode.log + 2D轨迹可视化)")
    
    # 🎥 调试输出状态显示
    if args.save_debug_files or args.save_videos or args.save_vlm_inputs:
        print("[INFO] 🎥 调试输出已启用：")
        if args.save_debug_files:
            print("     - 保存所有调试文件 (videos + vlm_inputs)")
        else:
            if args.save_videos:
                print("     - 保存episode视频")
            if args.save_vlm_inputs:
                print("     - 保存VLM输入图像")
    else:
        print("[INFO] 🚫 调试文件已禁用 (默认) - 仅保存关键输出")
    
    # 显示其他性能优化状态
    if args.low_res:
        pass
        print("[INFO] 📷 低分辨率模式：图像分辨率 320x240")
    if args.minimal_logging:
        pass
        print("[INFO] 📝 最小化日志模式：减少详细输出")
    
    # 🔇 日志模式状态显示
    if args.silent_logging:
        pass
        print("[INFO] 🔇 静默日志模式：终端仅显示进度条和关键信息，详细日志仍保存到文件")
    elif args.terminal_only:
        pass
        print("[INFO] 🖥️ 仅终端模式：只在终端显示，不写入日志文件")
    elif not args.silent_logging:
        pass
        print("[INFO] 📄 标准日志模式：终端显示+文件记录")
    
    # 向后兼容处理
    if args.no_debug_images or args.no_videos:
        pass
        print("[INFO] ⚠️  检测到已废弃参数，建议使用新的调试控制参数")
        if args.no_debug_images:
            pass
            args.save_vlm_inputs = False
        if args.no_videos:
            pass
            args.save_videos = False
    
    # 确定模型信息字符串用于汇总
    if args.model_type:
        pass
        model_info = f"predefined:{args.model_type}"
    elif args.input_type and args.output_type and args.protocol:
        pass
        model_info = f"modular:{args.input_type}+{args.output_type}+{args.protocol}"
    else:
        pass
        model_info = "default:navdp"
    
    # 批量测试模式
    if args.batch_test_dir:
        pass
        print(f"[INFO] ===== 批量测试模式 =====")
        print(f"[INFO] 批量测试目录: {args.batch_test_dir}")
        print(f"[INFO] JSON文件模式: {args.json_pattern}")
        print(f"[INFO] 输出根目录: {out_root}")
        print(f"[INFO] 模型配置: {model_info}")
        print(f"[INFO] ========================\n")
        
        # 扫描所有测试JSON文件
        json_files = find_test_json_files(args.batch_test_dir, args.json_pattern)
        if not json_files:
            pass
            print("[ERROR] 未找到匹配的JSON文件")
            return
        
        batch_results = []
        shared_env = None  # 共享环境
        
        # 🚀 初始化总体进度跟踪器
        total_episodes_count = 0
        for json_file in json_files:
            try:
                pass
                temp_episodes = adapt_gvln_to_episodes(json_file, "", goal_radius=args.goal_radius)
                if args.max_episodes > 0:
                    pass
                    temp_episodes = temp_episodes[:args.max_episodes]
                if args.start_idx > 0:
                    pass
                    temp_episodes = temp_episodes[args.start_idx:]
                if args.num_episodes > 0:
                    pass
                    temp_episodes = temp_episodes[:args.num_episodes]
                total_episodes_count += len(temp_episodes)
            except:
                pass
                pass  # 忽略无法解析的文件
        
        # 根据用户参数决定进度显示模式
        enable_progress = not args.no_progress
        enable_live_display = enable_progress and not args.quiet_progress
        progress_tracker = ProgressTracker(total_episodes_count, model_info, enable_live_display=enable_live_display) if enable_progress else None
        
        try:
            pass
            # 为批量测试创建第一个环境
            print(f"[BATCH] 初始化共享Isaac Sim环境...")
            first_json = json_files[0]
            first_scene_path = find_matching_scene_file(first_json, args.scene_usd_path) if os.path.isdir(args.scene_usd_path) else args.scene_usd_path
            # 初始化时不设置地图，稍后在每个文件处理时动态设置
            shared_env = SimpleVLNEnv(scene_usd_path=first_scene_path, headless=True, hz=args.hz, map_json_path="")
            print(f"[BATCH] 共享环境初始化成功")
        except Exception as e:
            pass
            print(f"[BATCH_ERROR] 共享环境初始化失败: {e}")
            return
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n[BATCH] ===== 处理文件 {i}/{len(json_files)} =====")
            print(f"[BATCH] 文件: {os.path.basename(json_file)}")
            print(f"[BATCH] 路径: {json_file}")
            print(f"[BATCH] 进度: {i}/{len(json_files)} ({i/len(json_files)*100:.1f}%)")
            
            try:
                pass
                # 自动匹配场景文件
                current_scene_path = ""
                print(f"[BATCH] 开始场景匹配...")
                if os.path.isdir(args.scene_usd_path):
                    pass
                    # 批量模式：自动匹配场景文件
                    current_scene_path = find_matching_scene_file(json_file, args.scene_usd_path)
                    print(f"[BATCH] 场景匹配结果: {current_scene_path if current_scene_path else '未找到'}")
                    if not current_scene_path:
                        pass
                        print(f"[BATCH_ERROR] 未找到匹配的场景文件，跳过: {json_file}")
                        batch_results.append({
                            "json_file": json_file,
                            "scene_file": "",
                            "map_file": "",
                            "status": "failed",
                            "reason": "scene_file_not_found",
                            "total_episodes": 0,
                            "successful_episodes": 0,
                            "failed_episodes": 0
                        })
                        continue
                else:
                    pass
                    # 单文件模式：直接使用指定的场景文件
                    current_scene_path = args.scene_usd_path
                
                print(f"[BATCH] 使用场景文件: {current_scene_path}")
                
                # 为每个JSON文件适配episodes
                episodes = adapt_gvln_to_episodes(json_file, current_scene_path, goal_radius=args.goal_radius)
                
                # slice episodes (应用到每个文件)
                original_episode_count = len(episodes)
                if args.max_episodes > 0:
                    pass
                    episodes = episodes[:args.max_episodes]
                if args.start_idx > 0:
                    pass
                    episodes = episodes[args.start_idx:]
                if args.num_episodes > 0:
                    pass
                    episodes = episodes[:args.num_episodes]
                
                print(f"[BATCH] Episodes: {len(episodes)} (原始: {original_episode_count})")
                
                if len(episodes) == 0:
                    pass
                    print(f"[BATCH] 跳过空文件: {json_file}")
                    batch_results.append({
                        "json_file": json_file,
                        "scene_file": current_scene_path,
                        "map_file": "",
                        "status": "skipped",
                        "reason": "no_episodes",
                        "total_episodes": 0,
                        "successful_episodes": 0,
                        "failed_episodes": 0
                    })
                    continue
                
                # 自动匹配地图文件
                current_map_path = ""
                if args.map_path:
                    pass
                    if os.path.isdir(args.map_path):
                        pass
                        # 批量模式：自动匹配地图文件
                        current_map_path = find_matching_map_file(json_file, args.map_path)
                        if not current_map_path:
                            pass
                            print(f"[BATCH_WARN] 未找到匹配的地图文件，将使用空地图路径")
                    else:
                        pass
                        # 单文件模式：直接使用指定的地图文件
                        current_map_path = args.map_path
                
                print(f"[BATCH] 使用地图文件: {current_map_path if current_map_path else '无'}")
                
                # 运行这个文件的测试
                print(f"[BATCH] 开始处理 {len(episodes)} 个episodes...")
                successful, failed = run_single_json_test(episodes, args, out_root, json_file, model_info, current_map_path, current_scene_path, close_env_on_finish=False, shared_env=shared_env, progress_tracker=progress_tracker)
                print(f"[BATCH] 文件处理完成: 成功={successful}, 失败={failed}")
                
                batch_results.append({
                    "json_file": json_file,
                    "scene_file": current_scene_path,
                    "map_file": current_map_path,
                    "status": "completed",
                    "total_episodes": len(episodes),
                    "successful_episodes": successful,
                    "failed_episodes": failed,
                    "success_rate": successful / len(episodes) if len(episodes) > 0 else 0.0
                })
                
            except Exception as e:
                pass
                print(f"[BATCH_ERROR] 处理文件失败: {json_file}")
                print(f"[BATCH_ERROR] 错误类型: {type(e).__name__}")
                print(f"[BATCH_ERROR] 错误信息: {e}")
                import traceback
                print(f"[BATCH_ERROR] 详细错误堆栈:")
                traceback.print_exc()
                
                batch_results.append({
                    "json_file": json_file,
                    "scene_file": current_scene_path if 'current_scene_path' in locals() else "",
                    "map_file": "",
                    "status": "failed",
                    "reason": f"{type(e).__name__}: {str(e)}",
                    "total_episodes": 0,
                    "successful_episodes": 0,
                    "failed_episodes": 0
                })
                
                print(f"[BATCH] 继续处理下一个文件...")
            
            print(f"[BATCH] 文件 {i}/{len(json_files)} 处理完毕")
            
            # 强制垃圾回收，释放内存
            try:
                pass
                import gc
                gc.collect()
                print(f"[BATCH] 内存清理完成")
            except Exception:
                pass
                pass
        
        # 关闭共享环境
        if shared_env is not None:
            pass
            print(f"[BATCH] 关闭共享Isaac Sim环境...")
            try:
                pass
                shared_env.close()
                print(f"[BATCH] 共享环境已关闭")
            except Exception as e:
                pass
                print(f"[BATCH_ERROR] 关闭共享环境失败: {e}")
        
        # 🎉 显示最终进度总结
        if 'progress_tracker' in locals():
            pass
            progress_tracker.final_summary()
        
        # 保存批量测试汇总
        save_batch_summary(batch_results, out_root, model_info)
        return
    
    # 单文件测试模式（原有逻辑）
    else:
        pass
        print(f"[INFO] ===== 单文件测试模式 =====")
        print(f"[INFO] JSON文件: {args.traj_json_path}")
        print(f"[INFO] 输出目录: {out_root}")
        print(f"[INFO] 模型配置: {model_info}")
        print(f"[INFO] ========================\n")
        
        episodes = adapt_gvln_to_episodes(args.traj_json_path, args.scene_usd_path, goal_radius=args.goal_radius)
        # slice episodes
        if args.max_episodes > 0:
            pass
            episodes = episodes[:args.max_episodes]
        if args.start_idx > 0:
            pass
            episodes = episodes[args.start_idx:]
        if args.num_episodes > 0:
            pass
            episodes = episodes[:args.num_episodes]

        print(f"[INFO] Total episodes to run: {len(episodes)}", flush=True)
        
        # 🚀 为单个文件测试创建进度跟踪器
        enable_progress = not args.no_progress
        enable_live_display = enable_progress and not args.quiet_progress
        progress_tracker = ProgressTracker(len(episodes), model_info, enable_live_display=enable_live_display) if enable_progress else None
        
        successful, failed = run_single_json_test(episodes, args, out_root, args.traj_json_path, model_info, args.map_path, progress_tracker=progress_tracker)
        
        # 🎉 显示最终总结
        if progress_tracker:
            pass
            progress_tracker.final_summary()



if __name__ == "__main__":
    pass
    main()
