"""ReasoningEngine — turns numerical Prism scores into causal 'why' explanations.

Prism's core explainers produce attribution scores (what matters, how much).
The ReasoningEngine adds the missing layer: **why** the model behaved that way,
by asking the LLM to reflect on its own sensitivity patterns.

Works as a post-processor on any Explanation object:

    explanation = lime.explain(prompt, output)
    reasoner = ReasoningEngine(llm)
    reasoner.add_reasons(explanation)  # mutates in-place, fills .reason fields

Or integrated into the explain pipeline via ``generate_reasons=True``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from clear_trace.explain.core.base import LLMClient
from clear_trace.explain.core.types import (
    ConceptAttribution,
    CounterfactualResult,
    Explanation,
    SentenceImportance,
    ToolDecision,
    TrajectoryExplanation,
)


# ── Prompt templates ──────────────────────────────────────────────

_SENTENCE_REASON_PROMPT = """\
You are an AI explainability analyst.

An LLM was given this prompt:
\"\"\"{prompt}\"\"\"

It produced this output:
\"\"\"{output}\"\"\"

We ran a perturbation analysis (LIME) and found the following sentence \
importance scores:
{scored_sentences}

For EACH sentence, write one concise sentence explaining WHY it has that \
level of influence on the output. Focus on the causal mechanism — what does \
that sentence tell the model to do, and how does removing it change behavior?

Format: one line per sentence, starting with the sentence index:
0: <reason>
1: <reason>
..."""

_COUNTERFACTUAL_REASON_PROMPT = """\
You are an AI explainability analyst.

Original prompt:
\"\"\"{original_prompt}\"\"\"

Original output (first 200 chars):
\"\"\"{original_output}\"\"\"

A minimal change was made to the prompt:
Change: {change_description}

Modified prompt:
\"\"\"{modified_prompt}\"\"\"

Modified output (first 200 chars):
\"\"\"{modified_output}\"\"\"

The semantic distance between outputs is {semantic_distance:.2f} \
(0=identical, 1=completely different). \
This change {flip_word} flip the output significantly.

In 1-2 sentences, explain WHY this edit caused (or didn't cause) the output \
to change. Focus on what the removed/changed element contributed to the \
model's reasoning."""

_CONCEPT_REASON_PROMPT = """\
You are an AI explainability analyst.

An LLM was given this prompt:
\"\"\"{prompt}\"\"\"

It produced this output:
\"\"\"{output}\"\"\"

We detected these concepts in the prompt and measured their influence \
by ablation (removing concept markers and re-querying):
{scored_concepts}

For EACH concept, write one concise sentence explaining WHY it had that \
level of influence. What role does that concept play in shaping the response?

Format: one line per concept:
<concept_name>: <reason>"""

_TRAJECTORY_REASON_PROMPT = """\
You are an AI explainability analyst.

An AI agent executed this trajectory:
{trajectory_steps}

Available tools were: {available_tools}

For EACH step, write one concise sentence explaining WHY the agent chose \
that specific tool over the alternatives, based on the context at that point.

Format:
step 0: <reason>
step 1: <reason>
..."""


class ReasoningEngine:
    """Generate causal 'why' explanations for Prism attribution results.

    Uses the LLM itself to reflect on its sensitivity patterns and
    produce human-readable reasoning for each finding.

    Args:
        llm: LLMClient to use for generating reasons.
        max_tokens: Maximum tokens for reasoning responses.

    Example:
        >>> reasoner = ReasoningEngine(llm)
        >>> explanation = lime.explain(prompt, output)
        >>> reasoner.add_reasons(explanation)
        >>> for s in explanation.top_sentences():
        ...     print(f"{s.score:+.3f}  {s.text}")
        ...     print(f"  WHY: {s.reason}")
    """

    def __init__(self, llm: LLMClient, max_tokens: int = 800):
        self.llm = llm
        self.max_tokens = max_tokens

    def add_reasons(self, explanation: Explanation) -> Explanation:
        """Add 'why' reasoning to all elements of an Explanation (mutates in-place).

        Dispatches to the appropriate reasoning method based on which
        fields are populated in the Explanation.
        """
        if explanation.sentence_importances:
            self._reason_sentences(explanation)

        if explanation.counterfactuals:
            self._reason_counterfactuals(explanation)

        if explanation.concept_attributions:
            self._reason_concepts(explanation)

        if explanation.trajectory and explanation.trajectory.decisions:
            self._reason_trajectory(explanation)

        return explanation

    # ── Sentence importance reasoning ────────────────────────────

    def _reason_sentences(self, explanation: Explanation) -> None:
        """Generate reasons for sentence importance scores."""
        scored = "\n".join(
            f"  [{i}] score={s.score:+.3f} ({s.level.value}): \"{s.text}\""
            for i, s in enumerate(explanation.sentence_importances)
        )

        prompt = _SENTENCE_REASON_PROMPT.format(
            prompt=explanation.prompt[:500],
            output=explanation.output[:300],
            scored_sentences=scored,
        )

        response = self._call_llm(prompt)
        reasons = self._parse_indexed_response(response)

        for s in explanation.sentence_importances:
            if s.index in reasons:
                s.reason = reasons[s.index]

    # ── Counterfactual reasoning ─────────────────────────────────

    def _reason_counterfactuals(self, explanation: Explanation) -> None:
        """Generate reasons for each counterfactual result."""
        # Batch up to 5 counterfactuals in one call to save LLM queries
        for cf in explanation.counterfactuals[:5]:
            prompt = _COUNTERFACTUAL_REASON_PROMPT.format(
                original_prompt=cf.original_prompt[:300],
                original_output=cf.original_output[:200],
                change_description=cf.change_description,
                modified_prompt=cf.modified_prompt[:300],
                modified_output=cf.modified_output[:200],
                semantic_distance=cf.semantic_distance,
                flip_word="DID" if cf.is_flip else "did NOT",
            )
            response = self._call_llm(prompt)
            cf.reason = response.strip()

    # ── Concept reasoning ────────────────────────────────────────

    def _reason_concepts(self, explanation: Explanation) -> None:
        """Generate reasons for concept attribution scores."""
        scored = "\n".join(
            f"  {c.concept}: score={c.score:+.3f}, "
            f"evidence={c.evidence_tokens}, desc=\"{c.description}\""
            for c in explanation.concept_attributions
        )

        prompt = _CONCEPT_REASON_PROMPT.format(
            prompt=explanation.prompt[:500],
            output=explanation.output[:300],
            scored_concepts=scored,
        )

        response = self._call_llm(prompt)
        reasons = self._parse_named_response(response)

        for c in explanation.concept_attributions:
            if c.concept in reasons:
                c.reason = reasons[c.concept]

    # ── Trajectory reasoning ─────────────────────────────────────

    def _reason_trajectory(self, explanation: Explanation) -> None:
        """Generate reasons for each trajectory decision."""
        traj = explanation.trajectory
        if not traj:
            return

        steps_text = "\n".join(
            f"  Step {d.step}: tool={d.tool_name}, "
            f"context=\"{d.context_snippet[:150]}\", "
            f"alternatives={d.alternatives}, "
            f"confidence={d.confidence:.2f}"
            for d in traj.decisions
        )

        # Collect available tools from alternatives
        all_tools = set()
        for d in traj.decisions:
            all_tools.add(d.tool_name)
            all_tools.update(d.alternatives)

        prompt = _TRAJECTORY_REASON_PROMPT.format(
            trajectory_steps=steps_text,
            available_tools=", ".join(sorted(all_tools)),
        )

        response = self._call_llm(prompt)
        reasons = self._parse_step_response(response)

        for d in traj.decisions:
            if d.step in reasons:
                d.reason = reasons[d.step]

    # ── Helpers ──────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the reasoning prompt."""
        try:
            return self.llm(prompt)
        except Exception as e:
            return f"(reasoning unavailable: {e})"

    @staticmethod
    def _parse_indexed_response(response: str) -> dict[int, str]:
        """Parse '0: reason\\n1: reason' format into {index: reason}."""
        reasons: dict[int, str] = {}
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Try patterns: "0: reason", "0. reason", "[0] reason"
            for sep in [":", ".", "]"]:
                if sep in line:
                    parts = line.split(sep, 1)
                    idx_str = parts[0].strip().lstrip("[").strip()
                    try:
                        idx = int(idx_str)
                        reasons[idx] = parts[1].strip()
                        break
                    except ValueError:
                        continue
        return reasons

    @staticmethod
    def _parse_named_response(response: str) -> dict[str, str]:
        """Parse 'concept_name: reason' format into {name: reason}."""
        reasons: dict[str, str] = {}
        for line in response.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                name, reason = line.split(":", 1)
                name = name.strip().lower().replace(" ", "_")
                reasons[name] = reason.strip()
        return reasons

    @staticmethod
    def _parse_step_response(response: str) -> dict[int, str]:
        """Parse 'step 0: reason' format into {step: reason}."""
        reasons: dict[int, str] = {}
        for line in response.strip().split("\n"):
            line = line.strip().lower()
            if line.startswith("step"):
                rest = line[4:].strip()
                for sep in [":", ".", ")"]:
                    if sep in rest:
                        parts = rest.split(sep, 1)
                        try:
                            idx = int(parts[0].strip())
                            # Get original-cased reason
                            orig_line = response.strip().split("\n")[
                                [l.strip().lower() for l in response.strip().split("\n")].index(line.strip())
                            ]
                            _, reason = orig_line.split(sep, 1)
                            reasons[idx] = reason.strip()
                            break
                        except (ValueError, IndexError):
                            continue
        return reasons
