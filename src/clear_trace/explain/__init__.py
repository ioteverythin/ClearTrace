"""TraceOps Explain — Explainability engine for LLM agents.

Break the black box into understandable components. Explain why an LLM said
what it said, why an agent chose that tool, and what prompt changes would
improve quality.

Quick start:
    from clear_trace.explain import PromptLIME, LLMClient

    llm = LLMClient(call_fn=my_llm_function)
    explainer = PromptLIME(llm=llm, n_samples=50)
    explanation = explainer.explain(prompt="...", output="...")
"""

# Core types and base classes
from clear_trace.explain.core import (
    BaseExplainer,
    ConceptAttribution,
    CounterfactualResult,
    Explanation,
    LLMClient,
    SentenceImportance,
    TokenImportance,
    TrajectoryExplanation,
)

# Perturbation analysis
from clear_trace.explain.perturbation import PromptLIME, SentenceExplainer, TokenExplainer

# Counterfactual generation
from clear_trace.explain.counterfactual import CounterfactualGenerator

# Agent trajectory attribution
from clear_trace.explain.trajectory import TrajectoryAttributor

# Concept mapping
from clear_trace.explain.concepts import ConceptExtractor, ConceptMapper

# Reasoning engine
from clear_trace.explain.reasoning import ReasoningEngine

# Prompt & tool advisors
from clear_trace.explain.advisor import (
    MatrixReport,
    PromptAdvisor,
    PromptReport,
    ToolAdvisor,
    ToolDefinition,
    ToolMatrixReport,
    ToolReport,
    ToolTestCase,
)

# Visualization
from clear_trace.explain.visualization import (
    AdvisorReport,
    ConsoleReport,
    HTMLReport,
)

__all__ = [
    # Core
    "BaseExplainer",
    "LLMClient",
    "Explanation",
    "TokenImportance",
    "SentenceImportance",
    "CounterfactualResult",
    "TrajectoryExplanation",
    "ConceptAttribution",
    # Perturbation
    "PromptLIME",
    "TokenExplainer",
    "SentenceExplainer",
    # Counterfactual
    "CounterfactualGenerator",
    # Trajectory
    "TrajectoryAttributor",
    # Concepts
    "ConceptExtractor",
    "ConceptMapper",
    # Reasoning
    "ReasoningEngine",
    # Advisor
    "PromptAdvisor",
    "PromptReport",
    "MatrixReport",
    "ToolAdvisor",
    "ToolDefinition",
    "ToolReport",
    "ToolTestCase",
    "ToolMatrixReport",
    # Visualization
    "HTMLReport",
    "ConsoleReport",
    "AdvisorReport",
]
