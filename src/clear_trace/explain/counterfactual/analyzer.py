"""Counterfactual analysis utilities.

Functions for analyzing and comparing sets of counterfactual
explanations across prompts or model versions.
"""

from __future__ import annotations

from typing import Dict, List

from clear_trace.explain.core.types import CounterfactualResult


def sensitivity_profile(results: List[CounterfactualResult]) -> Dict[str, float]:
    """Compute a sensitivity profile from counterfactual results.

    Returns a dict mapping change types to average semantic distance,
    showing which types of changes the model is most sensitive to.
    """
    type_scores: Dict[str, List[float]] = {}

    for r in results:
        # Classify the change type from description
        desc = r.change_description.lower()
        if "removed sentence" in desc:
            change_type = "sentence_removal"
        elif "flipped" in desc:
            change_type = "instruction_flip"
        elif "redacted" in desc:
            change_type = "token_redaction"
        elif "llm" in desc:
            change_type = "llm_guided"
        else:
            change_type = "other"

        if change_type not in type_scores:
            type_scores[change_type] = []
        type_scores[change_type].append(r.semantic_distance)

    return {k: sum(v) / len(v) for k, v in type_scores.items() if v}


def find_minimal_flip(results: List[CounterfactualResult]) -> CounterfactualResult | None:
    """Find the counterfactual with the smallest edit distance that still causes a flip."""
    flips = [r for r in results if r.is_flip]
    if not flips:
        return None
    return min(flips, key=lambda r: r.edit_distance)


def robustness_score(results: List[CounterfactualResult]) -> float:
    """Compute a robustness score in [0, 1].

    1.0 = model is completely robust (no perturbation flips the output).
    0.0 = model is maximally sensitive (every perturbation flips).
    """
    if not results:
        return 1.0
    flips = sum(1 for r in results if r.is_flip)
    return 1.0 - (flips / len(results))
