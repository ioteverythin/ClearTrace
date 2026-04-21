"""Agent trajectory attribution — explain WHY an agent made specific decisions.

This module analyzes an agent's execution trajectory (sequence of tool calls,
LLM invocations, and decisions) and attributes each decision to factors in
the context.

Key questions this module answers:
    - Why did the agent choose tool A instead of tool B?
    - Which part of the conversation context drove this decision?
    - What was the agent's implicit confidence at each step?
    - What would have happened with a different decision? (counterfactual trajectories)

Attribution methods:
    1. **Context window analysis**: Identify which parts of the context window
       most influenced the tool selection (via perturbation).
    2. **Decision point analysis**: At each tool call, analyze the "decision prompt"
       to determine what signaled the agent to use that specific tool.
    3. **Alternative trajectory generation**: Propose what would have happened
       if the agent had made a different choice at a critical decision point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from clear_trace.explain.core.base import BaseExplainer, LLMClient
from clear_trace.explain.core.types import (
    Explanation,
    ToolDecision,
    TrajectoryExplanation,
)
from clear_trace.explain.core.utils import cosine_similarity_text


class TrajectoryAttributor(BaseExplainer):
    """Explain an agent's decision trajectory.

    Analyzes why an agent chose specific tools/actions at each step
    and identifies the most critical decision points.

    Can work with:
        - Manually provided trajectories (list of decisions)
        - TraceOps cassettes (via the traceops_bridge module)

    Args:
        llm: LLMClient used for generating attribution analysis.
        available_tools: List of all tools the agent could have chosen from.
        use_llm_analysis: Whether to use the LLM for deeper decision analysis.

    Example:
        >>> attributor = TrajectoryAttributor(llm=llm, available_tools=["search", "read_file", "write_file"])
        >>> explanation = attributor.explain_trajectory(
        ...     decisions=[
        ...         {"tool": "search", "context": "User asked to find bugs in auth module"},
        ...         {"tool": "read_file", "context": "Search returned auth.py as relevant"},
        ...         {"tool": "write_file", "context": "Found a bug, need to fix it"},
        ...     ]
        ... )
        >>> for d in explanation.trajectory.get_critical_decisions():
        ...     print(f"Step {d.step}: {d.tool_name} — {d.attribution_scores}")
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        available_tools: Optional[List[str]] = None,
        use_llm_analysis: bool = True,
        **config: Any,
    ):
        super().__init__(llm=llm, **config)
        self.available_tools = available_tools or []
        self.use_llm_analysis = use_llm_analysis

    def _explain_impl(self, prompt: str, output: str, **kwargs: Any) -> Explanation:
        """Not used directly — use explain_trajectory() instead."""
        decisions = kwargs.get("decisions", [])
        return self.explain_trajectory(decisions).to_explanation()

    def explain_trajectory(
        self,
        decisions: List[Dict[str, Any]],
        full_context: str = "",
    ) -> _TrajectoryResult:
        """Analyze a sequence of agent decisions.

        Args:
            decisions: List of dicts with keys:
                - "tool": name of the tool used
                - "context": the context/prompt at that decision point
                - "output": (optional) the tool's output
                - "alternatives": (optional) list of other tools considered
            full_context: The complete conversation/context for the agent run.

        Returns:
            A _TrajectoryResult with the full analysis.
        """
        tool_decisions: List[ToolDecision] = []

        for i, dec in enumerate(decisions):
            tool_name = dec.get("tool", "unknown")
            context = dec.get("context", "")
            alternatives = dec.get("alternatives", self._infer_alternatives(tool_name))

            # Compute attribution scores for this decision
            attribution = self._attribute_decision(
                step=i,
                tool_name=tool_name,
                context=context,
                alternatives=alternatives,
            )

            # Estimate confidence
            confidence = self._estimate_confidence(context, tool_name)

            tool_decisions.append(
                ToolDecision(
                    step=i,
                    tool_name=tool_name,
                    alternatives=alternatives,
                    attribution_scores=attribution,
                    context_snippet=context[:500],
                    confidence=confidence,
                )
            )

        # Identify critical decisions (highest variance in attribution)
        critical_indices = self._find_critical_decisions(tool_decisions)

        # Generate summary
        summary = self._generate_summary(tool_decisions, critical_indices)

        trajectory = TrajectoryExplanation(
            decisions=tool_decisions,
            critical_decision_indices=critical_indices,
            trajectory_summary=summary,
        )

        return _TrajectoryResult(trajectory=trajectory)

    def _attribute_decision(
        self,
        step: int,
        tool_name: str,
        context: str,
        alternatives: List[str],
    ) -> Dict[str, float]:
        """Attribute a tool decision to factors in the context.

        Uses keyword matching and (optionally) LLM analysis to
        determine what in the context signaled this tool choice.
        """
        attribution: Dict[str, float] = {}

        # Heuristic: look for tool-name-like keywords in context
        context_lower = context.lower()
        tool_lower = tool_name.lower()

        # Direct mention score
        if tool_lower in context_lower or tool_lower.replace("_", " ") in context_lower:
            attribution["direct_mention"] = 0.9
        else:
            attribution["direct_mention"] = 0.0

        # Keyword signals
        keyword_signals = {
            "search": ["find", "look for", "search", "query", "locate"],
            "read_file": ["read", "file", "content", "open", "check"],
            "write_file": ["write", "create", "fix", "update", "modify", "edit"],
            "execute": ["run", "execute", "test", "try"],
            "browse": ["url", "http", "website", "page", "browse"],
        }

        for signal_tool, keywords in keyword_signals.items():
            if signal_tool in tool_lower:
                matches = sum(1 for kw in keywords if kw in context_lower)
                attribution[f"keyword_match_{signal_tool}"] = min(matches / len(keywords), 1.0)

        # Context recency: later context is usually more relevant
        attribution["context_length"] = min(len(context) / 1000, 1.0)

        # LLM-based deep analysis
        if self.use_llm_analysis and self.llm:
            llm_attribution = self._llm_analyze_decision(
                step, tool_name, context, alternatives
            )
            attribution.update(llm_attribution)

        return attribution

    def _llm_analyze_decision(
        self,
        step: int,
        tool_name: str,
        context: str,
        alternatives: List[str],
    ) -> Dict[str, float]:
        """Use the LLM to analyze why a specific tool was chosen."""
        alt_str = ", ".join(alternatives) if alternatives else "none specified"
        analysis_prompt = (
            f"An AI agent at step {step} chose to use the tool '{tool_name}' "
            f"(alternatives: {alt_str}).\n\n"
            f"Context at decision point:\n{context[:800]}\n\n"
            "Rate these factors on how much they influenced the tool choice (0.0-1.0):\n"
            "1. explicit_instruction: Was the user explicitly asking for this action?\n"
            "2. implicit_need: Did the context implicitly require this tool?\n"
            "3. logical_sequence: Was this the natural next step in the workflow?\n"
            "4. information_gap: Was there missing information that this tool fills?\n\n"
            "Reply in format: explicit=X.X implicit=X.X sequence=X.X gap=X.X"
        )

        try:
            response = self.llm(analysis_prompt)
            scores = {}
            for part in response.strip().split():
                if "=" in part:
                    key, val = part.split("=", 1)
                    try:
                        scores[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
            return {f"llm_{k}": min(max(v, 0.0), 1.0) for k, v in scores.items()}
        except Exception:
            return {}

    def _estimate_confidence(self, context: str, tool_name: str) -> float:
        """Estimate the agent's implicit confidence in this tool choice.

        Based on:
        - Clarity of the context (shorter, more specific = higher confidence)
        - Presence of hedging language
        - Whether the tool name appears in context
        """
        score = 0.5  # baseline

        context_lower = context.lower()

        # Boost if tool is directly mentioned
        if tool_name.lower() in context_lower:
            score += 0.2

        # Penalize hedging language
        hedging = ["maybe", "perhaps", "might", "could", "not sure", "unclear"]
        hedges = sum(1 for h in hedging if h in context_lower)
        score -= hedges * 0.05

        # Boost for clear directives
        directives = ["must", "need to", "should", "please", "required"]
        boosts = sum(1 for d in directives if d in context_lower)
        score += boosts * 0.05

        return min(max(score, 0.0), 1.0)

    def _find_critical_decisions(self, decisions: List[ToolDecision]) -> List[int]:
        """Identify the most critical decision points.

        A decision is critical if:
        1. It has high attribution variance (many competing signals)
        2. It has low confidence
        3. It has viable alternatives
        """
        scores = []
        for d in decisions:
            attrs = list(d.attribution_scores.values())
            if not attrs:
                scores.append(0.0)
                continue
            # Variance in attribution = more complex decision
            import statistics

            variance = statistics.variance(attrs) if len(attrs) > 1 else 0.0
            alt_count = len(d.alternatives)
            inv_confidence = 1.0 - d.confidence

            criticality = variance * 0.4 + (alt_count / 10) * 0.3 + inv_confidence * 0.3
            scores.append(criticality)

        # Return indices of top critical decisions
        if not scores:
            return []
        threshold = sorted(scores, reverse=True)[min(2, len(scores) - 1)]
        return [i for i, s in enumerate(scores) if s >= threshold]

    def _infer_alternatives(self, tool_name: str) -> List[str]:
        """Infer alternative tools from the available tools list."""
        return [t for t in self.available_tools if t != tool_name]

    def _generate_summary(
        self, decisions: List[ToolDecision], critical_indices: List[int]
    ) -> str:
        """Generate a human-readable trajectory summary."""
        if not decisions:
            return "Empty trajectory."

        lines = [f"Agent trajectory: {len(decisions)} decisions"]
        path = " → ".join(d.tool_name for d in decisions)
        lines.append(f"Path: {path}")

        if critical_indices:
            lines.append(f"Critical decision points: steps {critical_indices}")
            for idx in critical_indices:
                d = decisions[idx]
                top_attr = sorted(
                    d.attribution_scores.items(), key=lambda x: x[1], reverse=True
                )[:2]
                attr_str = ", ".join(f"{k}={v:.2f}" for k, v in top_attr)
                lines.append(
                    f"  Step {idx} ({d.tool_name}): confidence={d.confidence:.2f}, "
                    f"top factors: {attr_str}"
                )

        return "\n".join(lines)


class _TrajectoryResult:
    """Internal result wrapper for trajectory analysis."""

    def __init__(self, trajectory: TrajectoryExplanation):
        self.trajectory = trajectory

    def to_explanation(self) -> Explanation:
        return Explanation(
            method="trajectory_attribution",
            trajectory=self.trajectory,
            summary=self.trajectory.trajectory_summary,
        )
