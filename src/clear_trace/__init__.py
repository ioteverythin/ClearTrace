"""TraceOps — Record, replay, explain, and improve LLM agent behavior.

Deterministic regression testing + black-box explainability in one library.

Record:
    from clear_trace import Recorder, Replayer

    with Recorder(save_to="cassettes/test.yaml") as rec:
        result = agent.run("What is 2+2?")

    with Replayer("cassettes/test.yaml"):
        result = agent.run("What is 2+2?")  # zero API calls

Explain:
    from clear_trace.explain import PromptLIME, LLMClient

    llm = LLMClient(call_fn=my_llm_function)
    explainer = PromptLIME(llm=llm)
    explanation = explainer.explain(prompt="...", output="...")

Advise:
    from clear_trace.explain import PromptAdvisor

    advisor = PromptAdvisor(llm=llm)
    report = advisor.analyze(prompt="...", context="...")
    improved = advisor.improve(report)
"""

from clear_trace._types import (
    EventType,
    Trace,
    TraceEvent,
    TraceMetadata,
)
from clear_trace.assertions import (
    AgentLoopError,
    BudgetExceededError,
    assert_cost_under,
    assert_max_llm_calls,
    assert_no_loops,
    assert_tokens_under,
)
from clear_trace.cassette import (
    CassetteMismatchError,
    CassetteNotFoundError,
    load_cassette,
    save_cassette,
)
from clear_trace.diff import TraceDiff, assert_trace_unchanged, diff_traces
from clear_trace.normalize import (
    NormalizedResponse,
    NormalizedToolCall,
    normalize_for_comparison,
    normalize_response,
)
from clear_trace.recorder import Recorder
from clear_trace.replayer import Replayer
from clear_trace.reporters.cost_dashboard import CostDashboard, CostSummary

# RAG add-on (graceful degradation if not installed)
try:
    from clear_trace.rag.assertions import (
        RAGAssertionError,
        assert_chunk_count,
        assert_min_relevance_score,
        assert_no_retrieval_drift,
        assert_rag_scores,
        assert_retrieval_latency,
    )
    from clear_trace.rag.context_analysis import analyze_context_usage
    from clear_trace.rag.diff import RAGDiffResult, diff_rag
    from clear_trace.rag.scorers import DeepEvalScorer, RagasScorer
    from clear_trace.rag.snapshot import RetrieverSnapshot
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

# MCP add-on
try:
    from clear_trace.mcp.diff import MCPDiffResult, diff_mcp
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

# Semantic add-on
try:
    from clear_trace.semantic.assertions import SemanticRegressionError, assert_semantic_similarity
    from clear_trace.semantic.similarity import SemanticDiffResult, semantic_similarity
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

# Export add-on
try:
    from clear_trace.export.finetune import to_anthropic_finetune, to_openai_finetune
    _EXPORT_AVAILABLE = True
except ImportError:
    _EXPORT_AVAILABLE = False

# Behavioral analysis (always available — no extra deps)
from clear_trace.analysis import (
    BehavioralGap,
    GapAnalyzer,
    GapReport,
    PatternDetector,
    PatternReport,
    SkillsGenerator,
)
from clear_trace.github import PRDiff, PRFetcher

# Explain engine (graceful degradation if numpy not installed)
try:
    from clear_trace.explain import (
        PromptLIME,
        TokenExplainer,
        SentenceExplainer,
        CounterfactualGenerator,
        TrajectoryAttributor,
        ConceptExtractor,
        ConceptMapper,
        ReasoningEngine,
        PromptAdvisor,
        ToolAdvisor,
        LLMClient,
    )
    _EXPLAIN_AVAILABLE = True
except ImportError:
    _EXPLAIN_AVAILABLE = False

__version__ = "0.7.0"

__all__ = [
    # Core
    "Recorder",
    "Replayer",
    # Types
    "Trace",
    "TraceEvent",
    "TraceMetadata",
    "EventType",
    # Cassette
    "save_cassette",
    "load_cassette",
    "CassetteNotFoundError",
    "CassetteMismatchError",
    # Diff
    "TraceDiff",
    "diff_traces",
    "assert_trace_unchanged",
    # Normalization
    "NormalizedToolCall",
    "NormalizedResponse",
    "normalize_response",
    "normalize_for_comparison",
    # Assertions
    "assert_cost_under",
    "assert_tokens_under",
    "assert_max_llm_calls",
    "assert_no_loops",
    "BudgetExceededError",
    "AgentLoopError",
    # Reporters
    "CostDashboard",
    "CostSummary",
    # RAG (available when clear_trace[rag] installed)
    "diff_rag",
    "RAGDiffResult",
    "RAGAssertionError",
    "assert_chunk_count",
    "assert_retrieval_latency",
    "assert_min_relevance_score",
    "assert_no_retrieval_drift",
    "assert_rag_scores",
    "RagasScorer",
    "DeepEvalScorer",
    "RetrieverSnapshot",
    "analyze_context_usage",
    # MCP
    "diff_mcp",
    "MCPDiffResult",
    # Semantic
    "semantic_similarity",
    "SemanticDiffResult",
    "SemanticRegressionError",
    "assert_semantic_similarity",
    # Export / fine-tune
    "to_openai_finetune",
    "to_anthropic_finetune",
    # Behavioral analysis (v0.6.0, inspired by agent-pr-replay)
    "PatternDetector",
    "PatternReport",
    "GapAnalyzer",
    "GapReport",
    "BehavioralGap",
    "SkillsGenerator",
    # GitHub integration
    "PRFetcher",
    "PRDiff",
    # Explain engine (available when clear_trace[explain] installed)
    "LLMClient",
    "PromptLIME",
    "TokenExplainer",
    "SentenceExplainer",
    "CounterfactualGenerator",
    "TrajectoryAttributor",
    "ConceptExtractor",
    "ConceptMapper",
    "ReasoningEngine",
    "PromptAdvisor",
    "ToolAdvisor",
]
