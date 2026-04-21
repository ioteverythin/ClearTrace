"""Counterfactual generation — smallest prompt change that flips output."""

from clear_trace.explain.counterfactual.generator import CounterfactualGenerator
from clear_trace.explain.counterfactual.analyzer import (
    find_minimal_flip,
    robustness_score,
    sensitivity_profile,
)

__all__ = [
    "CounterfactualGenerator",
    "sensitivity_profile",
    "find_minimal_flip",
    "robustness_score",
]
