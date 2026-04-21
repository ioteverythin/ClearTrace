"""Concept mapper — measure how concepts influence LLM output.

Given a prompt, its output, and a set of concepts, the mapper:
    1. Creates concept-ablated prompts (removing concept markers).
    2. Queries the LLM with each ablated prompt.
    3. Measures how much the output changes.
    4. Attributes the output change to the concept.

This answers: "How much did the concept of POLITENESS change the output?"
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from clear_trace.explain.core.base import BaseExplainer, LLMClient
from clear_trace.explain.core.types import ConceptAttribution, Explanation
from clear_trace.explain.core.utils import cosine_similarity_text
from clear_trace.explain.concepts.extractor import ConceptExtractor, DEFAULT_CONCEPTS


class ConceptMapper(BaseExplainer):
    """Map LLM output changes to high-level concepts via ablation.

    For each concept detected in the prompt, removes concept-related
    tokens and measures the output impact.

    Args:
        llm: LLMClient for re-querying the model.
        concepts: Concept definitions. Defaults to DEFAULT_CONCEPTS.
        similarity_fn: Function to compare outputs.

    Example:
        >>> mapper = ConceptMapper(llm=llm)
        >>> explanation = mapper.explain(
        ...     "You are an expert. Please explain quantum physics step by step.",
        ...     output="Step 1: Quantum physics is..."
        ... )
        >>> for c in explanation.top_concepts(3):
        ...     print(f"{c.concept}: {c.score:+.3f}")
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        concepts: Optional[Dict[str, Dict[str, Any]]] = None,
        similarity_fn: Optional[Callable] = None,
        **config: Any,
    ):
        super().__init__(llm=llm, **config)
        self.concepts = concepts or DEFAULT_CONCEPTS
        self.similarity_fn = similarity_fn or cosine_similarity_text
        self.extractor = ConceptExtractor(concepts=self.concepts, llm=llm)

    def _explain_impl(self, prompt: str, output: str, **kwargs: Any) -> Explanation:
        # First, detect which concepts are present
        detected = self.extractor.extract(prompt)

        if not detected:
            return Explanation(
                method="concept_mapping",
                summary="No recognized concepts detected in the prompt.",
            )

        # For each detected concept, ablate its markers and measure impact
        attributions: List[ConceptAttribution] = []

        for concept_attr in detected:
            concept_name = concept_attr.concept
            concept_def = self.concepts.get(concept_name, {})
            markers = concept_def.get("positive_markers", []) + concept_def.get(
                "negative_markers", []
            )

            if not markers:
                attributions.append(concept_attr)
                continue

            # Create ablated prompt: remove all markers for this concept
            ablated_prompt = prompt
            for marker in markers:
                pattern = re.compile(re.escape(marker), re.IGNORECASE)
                ablated_prompt = pattern.sub("", ablated_prompt)
            ablated_prompt = re.sub(r"\s+", " ", ablated_prompt).strip()

            # Skip if ablation didn't change anything
            if ablated_prompt == prompt:
                attributions.append(concept_attr)
                continue

            # Query LLM with ablated prompt
            if self.llm:
                ablated_output = self.llm(ablated_prompt)
            else:
                ablated_output = ""

            # Measure output change
            sim = self.similarity_fn(output, ablated_output)
            impact = 1.0 - sim  # higher impact = concept was more influential

            attributions.append(
                ConceptAttribution(
                    concept=concept_name,
                    score=impact,
                    evidence_tokens=concept_attr.evidence_tokens,
                    description=concept_attr.description,
                    metadata={
                        "detection_score": concept_attr.score,
                        "ablation_similarity": sim,
                        "ablated_prompt_preview": ablated_prompt[:200],
                    },
                )
            )

        attributions.sort(key=lambda c: abs(c.score), reverse=True)

        return Explanation(
            method="concept_mapping",
            concept_attributions=attributions,
            summary=self._build_summary(attributions),
            metadata={"num_concepts_detected": len(detected)},
        )

    def _build_summary(self, attributions: List[ConceptAttribution]) -> str:
        if not attributions:
            return "No concept attributions computed."

        lines = ["Concept influence on LLM output:"]
        for i, a in enumerate(attributions[:5], 1):
            lines.append(
                f"  {i}. {a.concept} ({a.score:+.3f}): {a.description}"
            )
            if a.evidence_tokens:
                lines.append(f"     Evidence: {', '.join(a.evidence_tokens[:5])}")
        return "\n".join(lines)
