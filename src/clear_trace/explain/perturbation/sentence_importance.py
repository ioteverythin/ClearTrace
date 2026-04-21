"""Sentence-level importance using ablation.

Ablates (removes) each sentence and measures the impact on
the LLM output. Simpler and faster than full LIME when you
just want a quick ranking of sentence importance.
"""

from __future__ import annotations

from typing import Any, List, Optional

from clear_trace.explain.core.base import BaseExplainer, LLMClient
from clear_trace.explain.core.types import Explanation, SentenceImportance
from clear_trace.explain.core.utils import (
    cosine_similarity_text,
    normalize_scores,
    segment_sentences,
)


class SentenceExplainer(BaseExplainer):
    """Sentence-level ablation explainer.

    Removes each sentence from the prompt one at a time and
    measures how much the LLM output changes.

    Args:
        llm: LLMClient instance.
        similarity_fn: Output comparison function.

    Example:
        >>> explainer = SentenceExplainer(llm=llm)
        >>> explanation = explainer.explain(
        ...     "You are an expert. Be concise. Answer in bullet points.",
        ...     output="• Point 1\\n• Point 2"
        ... )
        >>> for s in explanation.top_sentences():
        ...     print(f"{s.score:+.3f}  {s.text}")
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        similarity_fn: Any = None,
        **config: Any,
    ):
        super().__init__(llm=llm, **config)
        self.similarity_fn = similarity_fn or cosine_similarity_text

    def _explain_impl(self, prompt: str, output: str, **kwargs: Any) -> Explanation:
        sentences = segment_sentences(prompt)
        if not sentences:
            return Explanation(method="sentence_ablation", summary="Empty prompt.")

        drop_scores: List[float] = []

        for i in range(len(sentences)):
            # Remove sentence i
            remaining = [s for j, s in enumerate(sentences) if j != i]
            perturbed_prompt = " ".join(remaining) if remaining else "[EMPTY]"

            if self.llm:
                perturbed_output = self.llm(perturbed_prompt)
            else:
                perturbed_output = ""

            sim = self.similarity_fn(output, perturbed_output)
            importance = 1.0 - sim  # higher drop = more important
            drop_scores.append(importance)

        scores = normalize_scores(drop_scores)

        sentence_importances = [
            SentenceImportance(text=seg, index=i, score=scores[i])
            for i, seg in enumerate(sentences)
        ]

        return Explanation(
            method="sentence_ablation",
            sentence_importances=sentence_importances,
            summary=self._build_summary(sentence_importances),
            metadata={"num_sentences": len(sentences)},
        )

    def _build_summary(self, importances: List[SentenceImportance]) -> str:
        sorted_imp = sorted(importances, key=lambda s: abs(s.score), reverse=True)
        top = sorted_imp[:3]
        lines = ["Most important sentences (ablation):"]
        for i, s in enumerate(top, 1):
            lines.append(f"  {i}. [{s.score:+.3f}] \"{s.text[:80]}\"")
        return "\n".join(lines)
