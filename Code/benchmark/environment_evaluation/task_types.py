#!/usr/bin/env python3
"""
Navigation Task Types for SAGE-3D Benchmark.

Supports multiple navigation tasks: VLN, ObjectNav, PointNav, ImgNav, NoGoalNav.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from PIL import Image


class NavigationTask(ABC):
    """Base class for navigation tasks."""

    def __init__(self, task_config: Dict[str, Any]):
        self.task_config = task_config
        self.task_type = self.__class__.__name__.lower().replace('task', '')

    @abstractmethod
    def get_instruction(self, episode: Dict[str, Any], step: int = 0) -> str:
        """Get navigation instruction for current step."""
        pass

    @abstractmethod
    def get_goal_position(self, episode: Dict[str, Any]) -> np.ndarray:
        """Get goal position."""
        pass

    @abstractmethod
    def get_goal_radius(self, episode: Dict[str, Any]) -> float:
        """Get success radius."""
        pass

    @abstractmethod
    def is_success(self, current_pos: np.ndarray, episode: Dict[str, Any], **kwargs) -> bool:
        """Determine if task is successfully completed."""
        pass

    @abstractmethod
    def get_task_specific_metrics(self) -> List[str]:
        """Get task-specific evaluation metrics."""
        pass

    def get_progress_info(self, current_pos: np.ndarray, episode: Dict[str, Any], step: int = 0) -> str:
        """Get task progress info (for debugging)."""
        goal_pos = self.get_goal_position(episode)
        distance = np.linalg.norm(current_pos - goal_pos)
        return f"Distance to goal: {distance:.2f}m"


class VLNTask(NavigationTask):
    """Vision-Language Navigation task."""

    def get_instruction(self, episode: Dict[str, Any], step: int = 0) -> str:
        """Get VLN instruction."""
        instructions = episode.get("instructions", [])
        if not instructions:
            # Fallback to original instruction format
            instruction_obj = episode.get("instruction", {})
            return instruction_obj.get("instruction_text", "Navigate to the destination")

        # Can select different instruction based on step, or randomly
        instruction_idx = step % len(instructions) if len(instructions) > 1 else 0
        selected_instruction = instructions[instruction_idx]

        # Handle both old and new formats
        if isinstance(selected_instruction, dict):
            # New format: extract generated_instruction from dict
            return selected_instruction.get("generated_instruction", "Navigate to the destination")
        else:
            # Old format: return string directly
            return selected_instruction

    def get_goal_position(self, episode: Dict[str, Any]) -> np.ndarray:
        """Get VLN goal position (last point of trajectory)."""
        points = episode.get("points", [])
        if not points:
            return np.array([0.0, 0.0, 0.0])

        last_point = points[-1]
        return np.array(last_point["position"])

    def get_goal_radius(self, episode: Dict[str, Any]) -> float:
        """VLN task success radius."""
        return self.task_config.get("goal_radius", 0.5)

    def is_success(self, current_pos: np.ndarray, episode: Dict[str, Any], **kwargs) -> bool:
        """VLN success: reached goal position."""
        goal_pos = self.get_goal_position(episode)
        distance = np.linalg.norm(current_pos - goal_pos)
        return distance < self.get_goal_radius(episode)

    def get_task_specific_metrics(self) -> List[str]:
        """VLN-specific metrics."""
        return ["instruction_following_score", "semantic_alignment"]

    def get_progress_info(self, current_pos: np.ndarray, episode: Dict[str, Any], step: int = 0) -> str:
        base_info = super().get_progress_info(current_pos, episode, step)
        instruction = self.get_instruction(episode, step)
        return f"{base_info} | Instruction: {instruction[:50]}..."


class ObjectNavTask(NavigationTask):
    """Object Navigation task."""

    def get_instruction(self, episode: Dict[str, Any], step: int = 0) -> str:
        """Generate ObjectNav instruction."""
        target_object = episode.get("target_object", "unknown object")
        return f"Find the {target_object}"

    def get_goal_position(self, episode: Dict[str, Any]) -> np.ndarray:
        """Get target object position."""
        target_position = episode.get("target_object_position", episode.get("goal_position", [0, 0, 0]))
        return np.array(target_position)

    def get_goal_radius(self, episode: Dict[str, Any]) -> float:
        """ObjectNav success radius (typically smaller, need to get close to object)."""
        return self.task_config.get("goal_radius", 1.0)

    def is_success(self, current_pos: np.ndarray, episode: Dict[str, Any], **kwargs) -> bool:
        """ObjectNav success: reached target object and can see it."""
        goal_pos = self.get_goal_position(episode)
        distance = np.linalg.norm(current_pos - goal_pos)

        if distance > self.get_goal_radius(episode):
            return False

        # Can add visual detection: whether target object is visible
        return True

    def get_task_specific_metrics(self) -> List[str]:
        """ObjectNav-specific metrics."""
        return ["object_detection_accuracy", "view_success_rate"]


class PointNavTask(NavigationTask):
    """Point Navigation task."""

    def get_instruction(self, episode: Dict[str, Any], step: int = 0) -> str:
        """Generate PointNav instruction."""
        goal_pos = self.get_goal_position(episode)
        return f"Navigate to coordinates ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})"

    def get_goal_position(self, episode: Dict[str, Any]) -> np.ndarray:
        """Get target coordinate point."""
        goal_position = episode.get("goal_position", [0, 0, 0])
        return np.array(goal_position)

    def get_goal_radius(self, episode: Dict[str, Any]) -> float:
        """PointNav success radius."""
        return self.task_config.get("goal_radius", 0.2)  # Point nav requires more precision

    def is_success(self, current_pos: np.ndarray, episode: Dict[str, Any], **kwargs) -> bool:
        """PointNav success: reached specified coordinate point."""
        goal_pos = self.get_goal_position(episode)
        distance = np.linalg.norm(current_pos - goal_pos)
        return distance < self.get_goal_radius(episode)

    def get_task_specific_metrics(self) -> List[str]:
        """PointNav-specific metrics."""
        return ["coordinate_accuracy", "path_efficiency"]


class ImgNavTask(NavigationTask):
    """Image Navigation task."""

    def get_instruction(self, episode: Dict[str, Any], step: int = 0) -> str:
        """Generate ImgNav instruction."""
        return "Navigate to the location that matches the target image"

    def get_goal_position(self, episode: Dict[str, Any]) -> np.ndarray:
        """Get position corresponding to target image."""
        target_position = episode.get("target_image_position", episode.get("goal_position", [0, 0, 0]))
        return np.array(target_position)

    def get_goal_radius(self, episode: Dict[str, Any]) -> float:
        """ImgNav success radius."""
        return self.task_config.get("goal_radius", 1.0)

    def is_success(self, current_pos: np.ndarray, episode: Dict[str, Any], **kwargs) -> bool:
        """ImgNav success: current view similar to target image."""
        goal_pos = self.get_goal_position(episode)
        distance = np.linalg.norm(current_pos - goal_pos)

        if distance > self.get_goal_radius(episode):
            return False

        # Image similarity check (requires current observation image)
        current_image = kwargs.get("current_image")
        target_image = episode.get("target_image")

        if current_image is not None and target_image is not None:
            similarity = self._compute_image_similarity(current_image, target_image)
            return similarity > self.task_config.get("similarity_threshold", 0.8)

        return True  # If no image info, only rely on position

    def _compute_image_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute image similarity (simplified version)."""
        # Can use CLIP, SSIM, or other image similarity algorithms
        return 0.85  # Placeholder

    def get_task_specific_metrics(self) -> List[str]:
        """ImgNav-specific metrics."""
        return ["image_similarity_score", "visual_alignment"]


class NoGoalNavTask(NavigationTask):
    """No-Goal Navigation task (exploration task)."""

    def __init__(self, task_config: Dict[str, Any]):
        super().__init__(task_config)
        self.visited_positions = []
        self.start_time = None
        self.max_episode_time = task_config.get("max_episode_time", 80.0)
        self.collision_penalty = task_config.get("collision_penalty", True)

    def get_instruction(self, episode: Dict[str, Any], step: int = 0) -> str:
        """Get NoGoalNav instruction."""
        instructions = episode.get("instructions", [])
        if instructions and isinstance(instructions[0], dict):
            instruction_idx = step % len(instructions) if len(instructions) > 1 else 0
            return instructions[instruction_idx].get(
                "generated_instruction",
                "Explore this environment as much as possible, but avoid colliding with objects, walls, etc., and maintain safe navigation."
            )
        else:
            return "Explore this environment as much as possible, but avoid colliding with objects, walls, etc., and maintain safe navigation."

    def get_goal_position(self, episode: Dict[str, Any]) -> np.ndarray:
        """NoGoalNav has no fixed goal, return start position."""
        points = episode.get("points", [])
        if points:
            start_point = points[0]
            return np.array(start_point["position"])
        return np.array([0, 0, 0.5])

    def get_goal_radius(self, episode: Dict[str, Any]) -> float:
        """In NoGoalNav, goal_radius represents robot safety/collision detection radius."""
        return self.task_config.get("goal_radius", 0.5)

    def is_success(self, current_pos: np.ndarray, episode: Dict[str, Any], **kwargs) -> bool:
        """NoGoalNav success: based on time and exploration coverage, collision means failure."""
        # Check for collision
        if self.collision_penalty and kwargs.get("collision_detected", False):
            return False

        # Check timeout
        episode_time = kwargs.get("episode_time", 0.0)

        # If reached max time, consider successful (completed exploration)
        if episode_time >= self.max_episode_time:
            return True

        # Based on exploration coverage
        exploration_coverage = kwargs.get("exploration_coverage", 0.0)
        min_coverage = self.task_config.get("min_exploration_coverage", 0.25)

        return exploration_coverage >= min_coverage

    def should_terminate_episode(self, **kwargs) -> bool:
        """Determine if episode should terminate."""
        # Collision terminates immediately
        if self.collision_penalty and kwargs.get("collision_detected", False):
            return True

        # Timeout terminates
        episode_time = kwargs.get("episode_time", 0.0)
        if episode_time >= self.max_episode_time:
            return True

        return False

    def update_exploration_state(self, current_pos: np.ndarray, step: int):
        """Update exploration state."""
        self.visited_positions.append(current_pos.copy())

    def calculate_exploration_coverage(self, visited_positions: List[np.ndarray], grid_size: float = 0.5) -> float:
        """Calculate exploration coverage."""
        if not visited_positions:
            return 0.0

        # Discretize visited positions to grid
        visited_cells = set()
        for pos in visited_positions:
            cell_x = int(pos[0] / grid_size)
            cell_y = int(pos[1] / grid_size)
            visited_cells.add((cell_x, cell_y))

        if len(visited_cells) == 0:
            return 0.0

        # Estimate total explorable area
        estimated_total_cells = 400  # Assume 20x20 grid
        coverage = len(visited_cells) / estimated_total_cells
        return min(coverage, 1.0)

    def get_task_specific_metrics(self) -> List[str]:
        """NoGoalNav-specific metrics."""
        return ["episode_time", "explored_areas", "exploration_coverage", "collision_count"]

    def get_progress_info(self, current_pos: np.ndarray, episode: Dict[str, Any], step: int = 0) -> str:
        return f"Steps: {step} | Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}) | Visited: {len(self.visited_positions)}"


class TaskTypeManager:
    """Task type manager."""

    TASK_TYPES = {
        "vln": VLNTask,
        "objectnav": ObjectNavTask,
        "pointnav": PointNavTask,
        "imgnav": ImgNavTask,
        "nogoalnav": NoGoalNavTask
    }

    @classmethod
    def create_task(cls, task_type: str, task_config: Dict[str, Any] = None) -> NavigationTask:
        """Create navigation task of specified type."""
        if task_config is None:
            task_config = {}

        task_type_lower = task_type.lower()
        if task_type_lower not in cls.TASK_TYPES:
            raise ValueError(f"Unsupported task type: {task_type}. Supported types: {list(cls.TASK_TYPES.keys())}")

        task_class = cls.TASK_TYPES[task_type_lower]
        return task_class(task_config)

    @classmethod
    def infer_task_type(cls, episode: Dict[str, Any]) -> str:
        """Infer task type from episode data."""
        # Check if no-goal task
        if "task_type" in episode and episode["task_type"] == "no_goal_exploration":
            return "nogoalnav"

        # Check if instructions contain Goal-less type
        if "instructions" in episode and episode["instructions"]:
            instructions = episode["instructions"]
            if isinstance(instructions[0], dict):
                instruction_type = instructions[0].get("instruction_type", "")
                if instruction_type == "Goal-less":
                    return "nogoalnav"

        # If has instructions field, likely VLN
        if "instructions" in episode and episode["instructions"]:
            return "vln"

        # If has target_object, is ObjectNav
        if "target_object" in episode:
            return "objectnav"

        # If has target_image, is ImgNav
        if "target_image" in episode:
            return "imgnav"

        # If only has goal_position, is PointNav
        if "goal_position" in episode:
            return "pointnav"

        # Default to VLN
        return "vln"

    @classmethod
    def get_supported_tasks(cls) -> List[str]:
        """Get list of supported task types."""
        return list(cls.TASK_TYPES.keys())


def adapt_episode_for_task(episode: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """Adapt episode data for specified task type."""
    adapted_episode = episode.copy()

    if task_type.lower() == "vln":
        # VLN task is already standard format
        pass

    elif task_type.lower() == "objectnav":
        # Extract target object from VLN instruction
        instructions = episode.get("instructions", [])
        if instructions:
            first_instruction = instructions[0].lower() if isinstance(instructions[0], str) else ""
            # Simplified object extraction (can use NLP model in production)
            for obj_keyword, obj_name in [
                ("folder", "folder"), ("window", "window"), ("chair", "chair"),
                ("screen", "projection_screen"), ("projection", "projection_screen"),
                ("notebook", "notebook"), ("cup", "cup")
            ]:
                if obj_keyword in first_instruction:
                    adapted_episode["target_object"] = obj_name
                    break
            else:
                adapted_episode["target_object"] = "unknown"
        else:
            instruction_obj = episode.get("instruction", {})
            instruction_text = instruction_obj.get("instruction_text", "").lower()
            for obj_keyword, obj_name in [
                ("folder", "folder"), ("window", "window"), ("chair", "chair"),
                ("screen", "projection_screen")
            ]:
                if obj_keyword in instruction_text:
                    adapted_episode["target_object"] = obj_name
                    break
            else:
                adapted_episode["target_object"] = "unknown"

        # Set target object position
        if "points" in episode and episode["points"]:
            adapted_episode["target_object_position"] = episode["points"][-1]["position"]

    elif task_type.lower() == "pointnav":
        # Use trajectory endpoint as target coordinate
        if "points" in episode and episode["points"]:
            adapted_episode["goal_position"] = episode["points"][-1]["position"]
        else:
            adapted_episode["goal_position"] = [0, 0, 0]

    elif task_type.lower() == "imgnav":
        # Set target image position
        if "points" in episode and episode["points"]:
            adapted_episode["target_image_position"] = episode["points"][-1]["position"]

    elif task_type.lower() == "nogoalnav":
        # Set start position
        if "points" in episode and episode["points"]:
            adapted_episode["start_position"] = episode["points"][0]["position"]
        else:
            adapted_episode["start_position"] = [0, 0, 0]

    return adapted_episode


if __name__ == "__main__":
    # Example episode data
    example_episode = {
        "instructions": [
            "Retrieve a folder from storage while you're closing the curtain by the window.",
            "Move from the window to the folders stored along the side wall."
        ],
        "points": [
            {"position": [-0.385, 2.985, 0.5], "rotation": [0, 0, 0, 1]},
            {"position": [1.234, 3.567, 0.5], "rotation": [0, 0, 0, 1]}
        ]
    }

    print("Supported task types:", TaskTypeManager.get_supported_tasks())

    # Auto-infer task type
    inferred_type = TaskTypeManager.infer_task_type(example_episode)
    print(f"Inferred task type: {inferred_type}")

    # Create different task types
    for task_type in ["vln", "objectnav", "pointnav"]:
        print(f"\n=== {task_type.upper()} Task ===")

        # Adapt data
        adapted_episode = adapt_episode_for_task(example_episode, task_type)

        # Create task
        task = TaskTypeManager.create_task(task_type, {"goal_radius": 0.5})

        # Get instruction and goal
        instruction = task.get_instruction(adapted_episode)
        goal_pos = task.get_goal_position(adapted_episode)

        print(f"Instruction: {instruction}")
        print(f"Goal position: {goal_pos}")
        print(f"Success radius: {task.get_goal_radius(adapted_episode)}")

        # Test success determination
        current_pos = np.array([1.2, 3.5, 0.5])
        success = task.is_success(current_pos, adapted_episode)
        print(f"At position {current_pos}, success: {success}")








