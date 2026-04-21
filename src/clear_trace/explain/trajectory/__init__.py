"""Agent trajectory attribution — why the agent chose each tool."""

from clear_trace.explain.trajectory.attribution import TrajectoryAttributor
from clear_trace.explain.trajectory.bridge import (
    cassette_to_decisions,
    cassette_to_trajectory,
    load_cleartrace_cassette,
)

__all__ = [
    "TrajectoryAttributor",
    "load_cleartrace_cassette",
    "cassette_to_decisions",
    "cassette_to_trajectory",
]
