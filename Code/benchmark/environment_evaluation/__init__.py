#!/usr/bin/env python3
"""
SAGE-3D Benchmark Environment Evaluation Package.

This package provides evaluation infrastructure for VLN (Vision-and-Language Navigation)
benchmarks using Isaac Sim environments.
"""

from .episodes_adapter import adapt_gvln_to_episodes
from .measures import (
    default_measures,
    nogoal_measures,
    MeasureManager,
    BaseMeasure,
    Success,
    SPL,
    OracleSuccess,
    PathLength,
    DistanceToGoal,
    NavigationError,
    ContinuousSuccessRatio,
    IntegratedCollisionPenalty,
    PathSmoothness,
)
from .task_types import (
    TaskTypeManager,
    NavigationTask,
    VLNTask,
    ObjectNavTask,
    PointNavTask,
    ImgNavTask,
    NoGoalNavTask,
    adapt_episode_for_task,
)
from .vlm_client_modular import (
    query_vlm,
    create_vlm_client,
    ModularVLMClient,
    set_log_function,
    PREDEFINED_CONFIGS,
)
from .object_based_success import ObjectBasedSuccessEvaluator
from .collision_detector import SemanticMap2DCollisionDetector

__all__ = [
    # Episode adapter
    "adapt_gvln_to_episodes",
    # Measures
    "default_measures",
    "nogoal_measures",
    "MeasureManager",
    "BaseMeasure",
    "Success",
    "SPL",
    "OracleSuccess",
    "PathLength",
    "DistanceToGoal",
    "NavigationError",
    "ContinuousSuccessRatio",
    "IntegratedCollisionPenalty",
    "PathSmoothness",
    # Task types
    "TaskTypeManager",
    "NavigationTask",
    "VLNTask",
    "ObjectNavTask",
    "PointNavTask",
    "ImgNavTask",
    "NoGoalNavTask",
    "adapt_episode_for_task",
    # VLM client
    "query_vlm",
    "create_vlm_client",
    "ModularVLMClient",
    "set_log_function",
    "PREDEFINED_CONFIGS",
    # Success evaluation
    "ObjectBasedSuccessEvaluator",
    # Collision detection
    "SemanticMap2DCollisionDetector",
]

# Note: SimpleVLNEnv requires Isaac Sim and is not imported by default
# Import it explicitly when needed:
# from .simple_env import SimpleVLNEnv








