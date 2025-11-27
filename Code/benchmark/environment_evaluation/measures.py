#!/usr/bin/env python3
"""
Evaluation Measures for SAGE-3D Benchmark.

Provides various metrics for evaluating VLN (Vision-and-Language Navigation) tasks,
including Success Rate (SR), SPL, Oracle Success Rate (OSR), and 3DGS-specific metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import math
from scipy.spatial import KDTree

try:
    from .object_based_success import ObjectBasedSuccessEvaluator
except ImportError:
    from object_based_success import ObjectBasedSuccessEvaluator


def euclidean_distance(a, b) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(b) - np.array(a), ord=2))


class BaseMeasure:
    """Base class for all evaluation measures."""

    def __init__(self, episode: Dict, manager: "MeasureManager") -> None:
        self.ep = episode
        self.mm = manager
        self._metric = None

    def uuid(self) -> str:
        """Return unique identifier for this measure."""
        raise NotImplementedError

    def reset(self, env) -> None:
        """Reset measure state at episode start."""
        raise NotImplementedError

    def update(self, env) -> None:
        """Update measure based on current environment state."""
        raise NotImplementedError

    def get(self):
        """Get current metric value."""
        return self._metric


class MeasureManager:
    """Manager for coordinating multiple evaluation measures."""

    def __init__(self) -> None:
        self.measures: Dict[str, BaseMeasure] = {}

    def register(self, m: BaseMeasure) -> None:
        """Register a measure with the manager."""
        self.measures[m.uuid()] = m

    def reset(self, env) -> None:
        """Reset all measures."""
        for m in self.measures.values():
            m.reset(env)

    def update(self, env) -> None:
        """Update all measures."""
        for m in self.measures.values():
            m.update(env)

    def dump(self) -> Dict[str, float]:
        """Dump all metric values as dictionary."""
        return {k: float(v.get()) for k, v in self.measures.items()}


class PathLength(BaseMeasure):
    """Measure: Total path length traveled by agent."""

    def uuid(self) -> str:
        return "path_length"

    def reset(self, env) -> None:
        self.prev = env.get_agent_pos()
        self._metric = 0.0

    def update(self, env) -> None:
        cur = env.get_agent_pos()
        self._metric += euclidean_distance(cur, self.prev)
        self.prev = cur


class DistanceToGoal(BaseMeasure):
    """Measure: Current distance to goal position."""

    def uuid(self) -> str:
        return "distance_to_goal"

    def reset(self, env) -> None:
        self.update(env)

    def update(self, env) -> None:
        cur = env.get_agent_pos()
        # Use straight-line distance to goal point
        goal_pos = self.ep["goals"][0]["position"] if self.ep.get("goals") else [0, 0, 0]
        self._metric = euclidean_distance(cur, goal_pos)


class Success(BaseMeasure):
    """Measure: Success Rate (SR) - whether agent reached goal."""

    def uuid(self) -> str:
        return "success"

    def reset(self, env) -> None:
        # Try to initialize object-based success evaluation
        try:
            if hasattr(env, 'semantic_map_path') and env.semantic_map_path:
                self.object_evaluator = ObjectBasedSuccessEvaluator(env.semantic_map_path, verbose=False)
                print(f"[SUCCESS] ✓ Initialized object-based success evaluation")
            else:
                self.object_evaluator = None
                print(f"[SUCCESS] ⚠ No semantic_map_path found, using traditional distance evaluation")
        except Exception as e:
            self.object_evaluator = None
            print(f"[SUCCESS] ✗ Object-based success init failed: {e}")

        self.update(env)

    def update(self, env) -> None:
        # Try object-based success evaluation
        if self.object_evaluator is not None:
            try:
                current_pos = env.get_agent_pos()
                success, info = self.object_evaluator.evaluate_success(
                    current_pos,
                    self.ep,
                    expansion_radius=1.0  # Optimized expansion radius
                )
                self._metric = 1.0 if success else 0.0

                if hasattr(self, '_debug_info'):
                    self._debug_info = info

                return
            except Exception as e:
                print(f"[SUCCESS] ⚠ Object-based evaluation failed, falling back to traditional: {e}")

        # Traditional distance-based success evaluation (fallback)
        d = self.mm.measures["distance_to_goal"].get()
        r = float(self.ep["goals"][0]["radius"]) if self.ep.get("goals") else 0.5
        self._metric = 1.0 if d < r else 0.0


class SPL(BaseMeasure):
    """Measure: Success weighted by Path Length."""

    def uuid(self) -> str:
        return "spl"

    def reset(self, env) -> None:
        self.prev = env.get_agent_pos()
        # Calculate shortest path length (straight-line distance from start to goal)
        start_pos = env.get_agent_pos()
        goal_pos = self.ep["goals"][0]["position"] if self.ep.get("goals") else [0, 0, 0]
        self.shortest_path_length = euclidean_distance(start_pos, goal_pos)
        self.pl = 0.0
        self.update(env)

    def update(self, env) -> None:
        cur = env.get_agent_pos()
        self.pl += euclidean_distance(cur, self.prev)
        self.prev = cur
        suc = self.mm.measures["success"].get()
        # Standard SPL: Success × (shortest_path / max(shortest_path, actual_path))
        if self.shortest_path_length > 0:
            self._metric = float(suc * (self.shortest_path_length / max(self.shortest_path_length, self.pl)))
        else:
            self._metric = float(suc)  # If start equals goal, SPL equals Success


class NavigationError(BaseMeasure):
    """Measure: Final navigation error (distance to goal at episode end)."""

    def uuid(self) -> str:
        return "navigation_error"

    def reset(self, env) -> None:
        self.update(env)

    def update(self, env) -> None:
        d = self.mm.measures["distance_to_goal"].get()
        self._metric = float(d)


class OracleSuccess(BaseMeasure):
    """Measure: Oracle Success Rate (OSR) - whether agent ever entered goal region."""

    def uuid(self) -> str:
        return "oracle_success"

    def reset(self, env) -> None:
        self._metric = 0.0

        # Try to initialize object-based success evaluation
        try:
            if hasattr(env, 'semantic_map_path') and env.semantic_map_path:
                self.object_evaluator = ObjectBasedSuccessEvaluator(env.semantic_map_path, verbose=False)
                print(f"[ORACLE_SUCCESS] ✓ Initialized object-based success evaluation")
            else:
                self.object_evaluator = None
                print(f"[ORACLE_SUCCESS] ⚠ No semantic_map_path found, using traditional distance evaluation")
        except Exception as e:
            self.object_evaluator = None
            print(f"[ORACLE_SUCCESS] ✗ Object-based success init failed: {e}")

        self.update(env)

    def update(self, env) -> None:
        # OSR: Whether agent ever entered goal region during episode
        # Once successful, maintain success state
        if self._metric >= 1.0:
            return

        # Try object-based success evaluation
        if self.object_evaluator is not None:
            try:
                cur = env.get_agent_pos()
                success, info = self.object_evaluator.evaluate_success(
                    cur, self.ep, expansion_radius=1.2  # More lenient radius for oracle
                )
                if success:
                    self._metric = 1.0
                return
            except Exception as e:
                pass  # Fallback to traditional method

        # Traditional distance-based method
        d = self.mm.measures["distance_to_goal"].get()
        r = float(self.ep["goals"][0]["radius"]) if self.ep.get("goals") else 0.5

        # Use expanded radius for oracle success
        oracle_radius = max(r * 3.0, 1.5)  # At least 1.5 meters

        if d < oracle_radius:
            self._metric = 1.0


class ContinuousSuccessRatio(BaseMeasure):
    """Measure: CSR - Ratio of time spent in success region."""

    def uuid(self) -> str:
        return "continuous_success_ratio"

    def reset(self, env) -> None:
        self.total_steps = 0
        self.success_steps = 0

        # Try to initialize object-based success evaluation
        try:
            if hasattr(env, 'semantic_map_path') and env.semantic_map_path:
                self.object_evaluator = ObjectBasedSuccessEvaluator(env.semantic_map_path, verbose=False)
                print(f"[CSR] ✓ Initialized object-based success evaluation")
            else:
                self.object_evaluator = None
                print(f"[CSR] ⚠ No semantic_map_path found, using traditional distance evaluation")
        except Exception as e:
            self.object_evaluator = None
            print(f"[CSR] ✗ Object-based success init failed: {e}")

        self.update(env)

    def update(self, env) -> None:
        cur = env.get_agent_pos()
        self.total_steps += 1

        # Try object-based success region evaluation
        if self.object_evaluator is not None:
            try:
                success, info = self.object_evaluator.evaluate_success(
                    cur, self.ep, expansion_radius=1.5  # Larger success region for continuous evaluation
                )
                if success:
                    self.success_steps += 1
                self._metric = float(self.success_steps / self.total_steps) if self.total_steps > 0 else 0.0
                return
            except Exception as e:
                pass  # Silent fallback to traditional method

        # Traditional distance-based success region
        goal_pos = self.ep["goals"][0]["position"] if self.ep.get("goals") else [0, 0, 0]
        base_radius = float(self.ep["goals"][0]["radius"]) if self.ep.get("goals") else 0.5

        # Use larger expanded radius as continuous success region
        success_radius = max(base_radius * 4.0, 2.0)  # At least 2 meters
        distance = euclidean_distance(cur, goal_pos)

        if distance <= success_radius:
            self.success_steps += 1

        self._metric = float(self.success_steps / self.total_steps) if self.total_steps > 0 else 0.0


class IntegratedCollisionPenalty(BaseMeasure):
    """Measure: ICP - Integrated collision penalty (ratio of collision time)."""

    def uuid(self) -> str:
        return "integrated_collision_penalty"

    def reset(self, env) -> None:
        self.total_steps = 0
        self.collision_steps = 0
        self.collision_recovery_frames = 0  # Remaining recovery frames after collision
        self.update(env)

    def update(self, env) -> None:
        self.total_steps += 1

        # Check current collision state from environment
        current_collision = False
        if hasattr(env, 'consecutive_collisions') and env.consecutive_collisions > 0:
            current_collision = True
        elif hasattr(env, '_last_collision_detected') and env._last_collision_detected:
            current_collision = True
        elif hasattr(env, 'collisions') and env.step_idx < len(env.collisions):
            current_collision = env.collisions[env.step_idx]

        # Collision recovery mechanism: frames after collision also count as collision impact
        if current_collision:
            self.collision_recovery_frames = 3  # 3 frames recovery period

        if self.collision_recovery_frames > 0:
            self.collision_steps += 1
            self.collision_recovery_frames -= 1

        # Calculate integrated collision penalty (0-1, 0=no collision, 1=always colliding)
        self._metric = float(self.collision_steps / self.total_steps) if self.total_steps > 0 else 0.0


class PathSmoothness(BaseMeasure):
    """Measure: PS - Path smoothness based on velocity change rate."""

    def uuid(self) -> str:
        return "path_smoothness"

    def reset(self, env) -> None:
        self.positions = [env.get_agent_pos().copy()]
        self.update(env)

    def update(self, env) -> None:
        current_pos = env.get_agent_pos()
        self.positions.append(current_pos.copy())

        if len(self.positions) < 3:
            self._metric = 1.0  # Initially assume smooth
            return

        # Calculate velocity sequence (position differences)
        velocities = []
        for i in range(len(self.positions) - 1):
            vel = np.array(self.positions[i+1]) - np.array(self.positions[i])
            vel_magnitude = np.linalg.norm(vel[:2])  # Only consider XY plane
            if vel_magnitude > 1e-6:  # Avoid division by zero
                velocities.append(vel[:2])

        if len(velocities) < 2:
            self._metric = 1.0
            return

        # Calculate acceleration sequence (velocity differences)
        accelerations = []
        for i in range(len(velocities) - 1):
            acc = velocities[i+1] - velocities[i]
            acc_magnitude = np.linalg.norm(acc)
            accelerations.append(acc_magnitude)

        if len(accelerations) == 0:
            self._metric = 1.0
            return

        # Calculate smoothness: 1 / (1 + mean_acceleration_change)
        mean_acceleration = np.mean(accelerations)
        self._metric = float(1.0 / (1.0 + mean_acceleration * 10.0))  # Scale factor adjusts sensitivity


class EpisodeTime(BaseMeasure):
    """Measure: Episode duration (for no-goal tasks)."""

    def uuid(self) -> str:
        return "episode_time"

    def reset(self, env) -> None:
        self.start_time = getattr(env, '_episode_start_time', 0.0)
        self._metric = 0.0

    def update(self, env) -> None:
        current_time = getattr(env, '_current_time', 0.0)
        self._metric = float(current_time - self.start_time)


class ExploredAreas(BaseMeasure):
    """Measure: Number of explored areas (for no-goal tasks)."""

    def uuid(self) -> str:
        return "explored_areas"

    def reset(self, env) -> None:
        self.visited_cells = set()
        self.grid_size = 0.5  # 0.5 meter grid
        self._metric = 0.0

    def update(self, env) -> None:
        current_pos = env.get_agent_pos()
        # Discretize position to grid
        cell_x = int(current_pos[0] / self.grid_size)
        cell_y = int(current_pos[1] / self.grid_size)
        self.visited_cells.add((cell_x, cell_y))
        self._metric = float(len(self.visited_cells))


class ExplorationCoverage(BaseMeasure):
    """Measure: Exploration coverage ratio (for no-goal tasks)."""

    def uuid(self) -> str:
        return "exploration_coverage"

    def reset(self, env) -> None:
        self.visited_cells = set()
        self.grid_size = 0.5  # 0.5 meter grid
        self.estimated_total_cells = 400  # Estimated total explorable cells (20x20)
        self._metric = 0.0

    def update(self, env) -> None:
        current_pos = env.get_agent_pos()
        cell_x = int(current_pos[0] / self.grid_size)
        cell_y = int(current_pos[1] / self.grid_size)
        self.visited_cells.add((cell_x, cell_y))
        coverage = len(self.visited_cells) / self.estimated_total_cells
        self._metric = float(min(coverage, 1.0))  # Clamp to 0-1


class CollisionCount(BaseMeasure):
    """Measure: CR (Collision Rate) - Total collision count during episode.
    
    This metric counts the total number of collisions detected during navigation.
    Unlike IntegratedCollisionPenalty which measures collision time ratio,
    this directly counts collision events.
    """

    def uuid(self) -> str:
        return "collision_count"

    def reset(self, env) -> None:
        self._metric = 0.0
        # Reset environment collision counter if available
        if hasattr(env, '_total_collision_count'):
            env._total_collision_count = 0

    def update(self, env) -> None:
        # Get total collision count from environment
        if hasattr(env, 'get_collision_count'):
            self._metric = float(env.get_collision_count())
        elif hasattr(env, '_total_collision_count'):
            self._metric = float(env._total_collision_count)
        # Fallback: increment on collision detection
        elif hasattr(env, '_collision_detected') and env._collision_detected:
            self._metric += 1.0


def default_measures(episode: Dict) -> MeasureManager:
    """Create MeasureManager with default VLN metrics.

    Args:
        episode: Episode dictionary

    Returns:
        MeasureManager with registered measures
    """
    mm = MeasureManager()
    # Registration order ensures dependencies are ready

    # Core VLN metrics: SR, OSR, SPL
    mm.register(DistanceToGoal(episode, mm))  # Base distance calculation
    mm.register(Success(episode, mm))         # SR - Success Rate
    mm.register(OracleSuccess(episode, mm))   # OSR - Oracle Success Rate
    mm.register(PathLength(episode, mm))      # Path length
    mm.register(SPL(episode, mm))             # SPL - Success weighted by Path Length
    mm.register(NavigationError(episode, mm)) # Final navigation error
    mm.register(CollisionCount(episode, mm))            # CR - Collision Count (total collisions)

    # 3DGS VLN continuous advantage metrics
    mm.register(ContinuousSuccessRatio(episode, mm))    # CSR - Continuous Success Ratio
    mm.register(IntegratedCollisionPenalty(episode, mm)) # ICP - Integrated Collision Penalty
    mm.register(PathSmoothness(episode, mm))            # PS - Path Smoothness

    return mm


def nogoal_measures(episode: Dict) -> MeasureManager:
    """Create MeasureManager for no-goal navigation tasks.

    Args:
        episode: Episode dictionary

    Returns:
        MeasureManager with registered measures
    """
    mm = MeasureManager()

    # No-goal task specific metrics
    mm.register(EpisodeTime(episode, mm))           # Exploration time
    mm.register(ExploredAreas(episode, mm))         # Explored area count
    mm.register(ExplorationCoverage(episode, mm))   # Exploration coverage
    mm.register(CollisionCount(episode, mm))        # Collision count

    # General metrics
    mm.register(PathLength(episode, mm))            # Path length
    mm.register(PathSmoothness(episode, mm))        # Path smoothness

    return mm








