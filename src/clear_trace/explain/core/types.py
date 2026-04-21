"""Core types and data structures for Prism explanations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class ImportanceLevel(str, Enum):
    """Qualitative importance levels for explanation elements."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class TokenImportance:
    """Importance score for an individual token in a prompt.

    Attributes:
        token: The raw token string.
        position: Index position in the tokenized prompt.
        score: Normalized importance score in [-1.0, 1.0].
            Positive = pushes output toward observed response.
            Negative = pushes output away from observed response.
        level: Qualitative importance bucket.
        metadata: Optional extra data (e.g., token ID, attention weight).
    """

    token: str
    position: int
    score: float
    level: ImportanceLevel = ImportanceLevel.MEDIUM
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        abs_score = abs(self.score)
        if abs_score >= 0.8:
            self.level = ImportanceLevel.CRITICAL
        elif abs_score >= 0.6:
            self.level = ImportanceLevel.HIGH
        elif abs_score >= 0.3:
            self.level = ImportanceLevel.MEDIUM
        elif abs_score >= 0.1:
            self.level = ImportanceLevel.LOW
        else:
            self.level = ImportanceLevel.NEGLIGIBLE


@dataclass
class SentenceImportance:
    """Importance score for a sentence / segment in a prompt.

    Attributes:
        text: The sentence text.
        index: Sentence index in the segmented prompt.
        score: Normalized importance in [-1.0, 1.0].
        tokens: Optional per-token breakdown within the sentence.
    """

    text: str
    index: int
    score: float
    level: ImportanceLevel = ImportanceLevel.MEDIUM
    reason: str = ""
    tokens: List[TokenImportance] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        abs_score = abs(self.score)
        if abs_score >= 0.8:
            self.level = ImportanceLevel.CRITICAL
        elif abs_score >= 0.6:
            self.level = ImportanceLevel.HIGH
        elif abs_score >= 0.3:
            self.level = ImportanceLevel.MEDIUM
        elif abs_score >= 0.1:
            self.level = ImportanceLevel.LOW
        else:
            self.level = ImportanceLevel.NEGLIGIBLE


@dataclass
class CounterfactualResult:
    """A counterfactual explanation: minimal change that alters the output.

    Attributes:
        original_prompt: The original input prompt.
        modified_prompt: The minimally modified prompt.
        original_output: The LLM output for the original prompt.
        modified_output: The LLM output for the modified prompt.
        change_description: Human-readable summary of the change.
        edit_distance: Character-level edit distance between prompts.
        semantic_distance: Semantic similarity between outputs (0=same, 1=opposite).
        changed_tokens: List of (position, old_token, new_token) tuples.
    """

    original_prompt: str
    modified_prompt: str
    original_output: str
    modified_output: str
    change_description: str = ""
    edit_distance: int = 0
    semantic_distance: float = 0.0
    changed_tokens: List[tuple] = field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_flip(self) -> bool:
        """Whether the counterfactual caused a meaningful output change."""
        return self.semantic_distance > 0.5


@dataclass
class ToolDecision:
    """A single tool/action decision within an agent trajectory.

    Attributes:
        step: Step number in the trajectory.
        tool_name: Name of the tool/action chosen.
        alternatives: Other tools that could have been chosen.
        attribution_scores: {factor: score} explaining why this tool was picked.
        context_snippet: The relevant context that led to this decision.
        confidence: Model's implicit confidence in this choice.
    """

    step: int
    tool_name: str
    alternatives: List[str] = field(default_factory=list)
    attribution_scores: Dict[str, float] = field(default_factory=dict)
    context_snippet: str = ""
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryExplanation:
    """Full explanation of an agent's decision trajectory.

    Attributes:
        decisions: Ordered list of tool/action decisions.
        critical_decision_indices: Which steps were most impactful.
        trajectory_summary: Human-readable summary of the trajectory.
        alternative_trajectories: Hypothetical different paths.
    """

    decisions: List[ToolDecision] = field(default_factory=list)
    critical_decision_indices: List[int] = field(default_factory=list)
    trajectory_summary: str = ""
    alternative_trajectories: List[List[ToolDecision]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_decisions(self) -> int:
        return len(self.decisions)

    def get_critical_decisions(self) -> List[ToolDecision]:
        """Return only the decisions flagged as critical."""
        return [self.decisions[i] for i in self.critical_decision_indices if i < len(self.decisions)]


@dataclass
class ConceptAttribution:
    """Attribution of an LLM response to high-level human concepts.

    Attributes:
        concept: The human-readable concept name (e.g. "politeness", "safety").
        score: How much this concept influenced the output [-1.0, 1.0].
        evidence_tokens: Tokens in the prompt that activated this concept.
        description: Explanation of what this concept represents.
    """

    concept: str
    score: float
    evidence_tokens: List[str] = field(default_factory=list)
    description: str = ""
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Explanation:
    """Top-level container for any Prism explanation.

    Provides a unified interface regardless of the explanation method used.
    """

    method: str  # e.g. "prompt_lime", "counterfactual", "trajectory", "concept"
    model_name: str = ""
    prompt: str = ""
    output: str = ""

    # Perturbation results
    token_importances: List[TokenImportance] = field(default_factory=list)
    sentence_importances: List[SentenceImportance] = field(default_factory=list)

    # Counterfactual results
    counterfactuals: List[CounterfactualResult] = field(default_factory=list)

    # Trajectory results
    trajectory: Optional[TrajectoryExplanation] = None

    # Concept results
    concept_attributions: List[ConceptAttribution] = field(default_factory=list)

    # Generic
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def top_tokens(self, n: int = 10) -> List[TokenImportance]:
        """Return the top-n most important tokens by absolute score."""
        return sorted(self.token_importances, key=lambda t: abs(t.score), reverse=True)[:n]

    def top_sentences(self, n: int = 5) -> List[SentenceImportance]:
        """Return the top-n most important sentences by absolute score."""
        return sorted(self.sentence_importances, key=lambda s: abs(s.score), reverse=True)[:n]

    def top_concepts(self, n: int = 5) -> List[ConceptAttribution]:
        """Return the top-n most influential concepts."""
        return sorted(self.concept_attributions, key=lambda c: abs(c.score), reverse=True)[:n]

    def flipped_counterfactuals(self) -> List[CounterfactualResult]:
        """Return only counterfactuals that caused a meaningful output change."""
        return [c for c in self.counterfactuals if c.is_flip]
