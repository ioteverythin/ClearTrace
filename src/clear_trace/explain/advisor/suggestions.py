"""Data types for prompt improvement suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SuggestionType(str, Enum):
    """Category of prompt improvement."""

    REWRITE = "rewrite"           # Full sentence rewrite
    ADD = "add"                   # Add a missing instruction
    REMOVE = "remove"             # Remove a harmful/useless part
    STRENGTHEN = "strengthen"     # Make an instruction more specific
    RESTRUCTURE = "restructure"   # Change prompt structure/order
    FORMAT = "format"             # Add output format constraints
    EXAMPLE = "example"           # Add few-shot examples
    PERSONA = "persona"           # Adjust persona/role framing


class ImpactLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Suggestion:
    """A single actionable suggestion for improving a prompt.

    Attributes:
        type: Category of change (rewrite, add, remove, etc.).
        target: Which part of the prompt this applies to (sentence text or 'overall').
        problem: What's wrong with the current prompt (the diagnosis).
        fix: The specific change to make (the prescription).
        improved_text: The rewritten version of the target.
        impact: Expected impact level.
        confidence: How confident we are this will help (0-1).
        evidence: What analysis led to this suggestion (LIME, counterfactual, etc.).
    """

    type: SuggestionType
    target: str
    problem: str
    fix: str
    improved_text: str = ""
    impact: ImpactLevel = ImpactLevel.MEDIUM
    confidence: float = 0.5
    evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptReport:
    """Full prompt improvement report.

    Contains the analysis, all suggestions, and a proposed rewritten prompt
    with expected improvements.

    Attributes:
        original_prompt: The prompt as-is.
        original_output: What the LLM produced.
        desired_output_description: What the user WANTED (if provided).
        diagnosis: High-level assessment of the prompt's effectiveness.
        suggestions: Ordered list of improvement suggestions.
        improved_prompt: The full rewritten prompt incorporating all suggestions.
        improved_output: The LLM's output for the improved prompt (if tested).
        improvement_score: How much better the new output is (0-1 scale).
        score_before: Estimated effectiveness of original prompt (0-1).
        score_after: Estimated effectiveness of improved prompt (0-1).
    """

    original_prompt: str
    original_output: str = ""
    desired_output_description: str = ""
    diagnosis: str = ""
    suggestions: List[Suggestion] = field(default_factory=list)
    improved_prompt: str = ""
    improved_output: str = ""
    improvement_score: float = 0.0
    score_before: float = 0.0
    score_after: float = 0.0
    # Intermediate analysis (used by MatrixReport)
    _lime_result: Any = field(default=None, repr=False)
    _cf_result: Any = field(default=None, repr=False)
    _concepts: Any = field(default_factory=list, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_suggestions(self) -> int:
        return len(self.suggestions)

    def high_impact(self) -> List[Suggestion]:
        """Return only high-impact suggestions."""
        return [s for s in self.suggestions if s.impact == ImpactLevel.HIGH]
