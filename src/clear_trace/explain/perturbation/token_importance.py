"""Token-level importance explainer.

Measures the importance of individual tokens by masking each one
and observing the impact on the LLM output.

This is a leave-one-out (LOO) approach: for each token, remove it,
query the LLM, and measure how much the output changes.
"""

from __future__ import annotations

from typing import Any, List, Optional

from clear_trace.explain.core.base import BaseExplainer, LLMClient
from clear_trace.explain.core.types import Explanation, TokenImportance
from clear_trace.explain.core.utils import (
    cosine_similarity_text,
    normalize_scores,
    tokenize_simple,
    detokenize_simple,
)


class TokenExplainer(BaseExplainer):
    """Leave-one-out token importance explainer.

    For each token in the prompt, removes it, queries the LLM,
    and scores how much the output diverges from the original.

    Best for short prompts (< 100 tokens). For longer prompts,
    use PromptLIME with sentence-level segmentation.

    Args:
        llm: LLMClient for querying the model.
        mask_token: Replacement for the removed token.
        similarity_fn: Function to compare outputs.
        batch_size: Number of tokens to process per batch (for parallelism later).

    Example:
        >>> explainer = TokenExplainer(llm=llm)
        >>> explanation = explainer.explain("What is machine learning?", output="...")
        >>> for t in explanation.top_tokens(5):
        ...     print(f"{t.score:+.3f}  '{t.token}'")
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        mask_token: str = "[MASK]",
        similarity_fn: Any = None,
        batch_size: int = 10,
        **config: Any,
    ):
        super().__init__(llm=llm, **config)
        self.mask_token = mask_token
        self.similarity_fn = similarity_fn or cosine_similarity_text
        self.batch_size = batch_size

    def _explain_impl(self, prompt: str, output: str, **kwargs: Any) -> Explanation:
        tokens = tokenize_simple(prompt)
        if not tokens:
            return Explanation(method="token_loo", summary="Empty prompt.")

        base_similarity = 1.0  # reference: original output compared to itself
        drop_scores: List[float] = []

        for i in range(len(tokens)):
            # Remove token i
            perturbed_tokens = tokens[:i] + [self.mask_token] + tokens[i + 1 :]
            perturbed_prompt = detokenize_simple(perturbed_tokens)

            if self.llm:
                perturbed_output = self.llm(perturbed_prompt)
            else:
                perturbed_output = ""

            sim = self.similarity_fn(output, perturbed_output)
            # Importance = how much similarity drops when this token is removed
            importance = base_similarity - sim
            drop_scores.append(importance)

        scores = normalize_scores(drop_scores)

        token_importances = [
            TokenImportance(token=tok, position=i, score=scores[i])
            for i, tok in enumerate(tokens)
        ]

        return Explanation(
            method="token_loo",
            token_importances=token_importances,
            summary=self._build_summary(token_importances),
            metadata={"num_tokens": len(tokens), "mask_token": self.mask_token},
        )

    def _build_summary(self, importances: List[TokenImportance]) -> str:
        sorted_imp = sorted(importances, key=lambda t: abs(t.score), reverse=True)
        top = sorted_imp[:5]
        lines = ["Most important tokens (leave-one-out):"]
        for i, t in enumerate(top, 1):
            lines.append(f"  {i}. [{t.score:+.3f}] '{t.token}' (pos {t.position})")
        return "\n".join(lines)
