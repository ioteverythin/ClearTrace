"""Prompt LIME — Local Interpretable Model-agnostic Explanations for LLM prompts.

This module applies the LIME approach to LLM prompts: perturb segments of the
input prompt, observe how the output changes, and fit a simple interpretable
model to attribute importance to each segment.

Unlike classical LIME (which perturbs tabular features or superpixels), Prism's
PromptLIME perturbs at the **token** or **sentence** level and measures output
divergence via text similarity.

Algorithm:
    1. Segment the prompt into interpretable units (tokens or sentences).
    2. Generate N perturbed prompts by randomly masking/removing segments.
    3. Query the LLM for each perturbed prompt.
    4. Measure the similarity between each perturbed output and the original.
    5. Fit a weighted linear model: segment_presence → output_similarity.
    6. The linear coefficients are the importance scores.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional

import numpy as np

from clear_trace.explain.core.base import BaseExplainer, LLMClient
from clear_trace.explain.core.types import (
    Explanation,
    SentenceImportance,
    TokenImportance,
)
from clear_trace.explain.core.utils import (
    cosine_similarity_text,
    normalize_scores,
    segment_sentences,
    tokenize_simple,
    detokenize_simple,
)


class PromptLIME(BaseExplainer):
    """LIME-style explainer for LLM prompts.

    Perturbs the prompt at the sentence level and measures the impact
    on the LLM's output to determine which parts of the prompt matter most.

    Args:
        llm: An LLMClient instance for querying the model.
        num_perturbations: Number of perturbed samples to generate.
        mask_token: Token used to replace masked segments (default: "[MASKED]").
        seed: Random seed for reproducibility.
        similarity_fn: Optional custom function(str, str) → float for measuring
            output similarity. Defaults to bag-of-words cosine similarity.

    Example:
        >>> from clear_trace import PromptLIME
        >>> from clear_trace.explain.core import LLMClient
        >>> llm = LLMClient.from_openai(client, model="gpt-4o")
        >>> explainer = PromptLIME(llm=llm, num_perturbations=50)
        >>> explanation = explainer.explain(
        ...     prompt="You are a Python expert. Explain decorators in simple terms.",
        ...     output="Decorators are functions that modify other functions..."
        ... )
        >>> for s in explanation.top_sentences(3):
        ...     print(f"{s.score:+.3f}  {s.text}")
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        num_perturbations: int = 50,
        mask_token: str = "[MASKED]",
        seed: int = 42,
        similarity_fn: Any = None,
        **config: Any,
    ):
        super().__init__(llm=llm, **config)
        self.num_perturbations = num_perturbations
        self.mask_token = mask_token
        self.seed = seed
        self.similarity_fn = similarity_fn or cosine_similarity_text

    def _explain_impl(self, prompt: str, output: str, **kwargs: Any) -> Explanation:
        """Run LIME perturbation analysis on the prompt."""
        sentences = segment_sentences(prompt)
        if not sentences:
            return Explanation(method="prompt_lime", summary="Empty prompt — nothing to explain.")

        n_segments = len(sentences)
        rng = random.Random(self.seed)

        # Generate binary perturbation masks: 1 = keep, 0 = mask
        masks: List[List[int]] = []
        perturbed_outputs: List[str] = []
        similarities: List[float] = []

        for _ in range(self.num_perturbations):
            # Each segment has ~50% chance of being kept
            mask = [rng.randint(0, 1) for _ in range(n_segments)]
            # Ensure at least one segment is kept
            if sum(mask) == 0:
                mask[rng.randint(0, n_segments - 1)] = 1
            masks.append(mask)

            # Build perturbed prompt
            perturbed = " ".join(
                seg if mask[i] else self.mask_token for i, seg in enumerate(sentences)
            )

            # Query LLM
            if self.llm:
                perturbed_output = self.llm(perturbed)
            else:
                perturbed_output = ""

            perturbed_outputs.append(perturbed_output)
            similarity = self.similarity_fn(output, perturbed_output)
            similarities.append(similarity)

        # Fit a linear model: mask_vectors → similarity
        X = np.array(masks, dtype=np.float64)  # (N, n_segments)
        y = np.array(similarities, dtype=np.float64)  # (N,)

        # Weighted least squares (samples closer to original get higher weight)
        distances = np.sum(1 - X, axis=1)  # how many segments were masked
        kernel_width = max(n_segments * 0.25, 1.0)
        weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

        # Solve weighted linear regression: (X^T W X)^-1 X^T W y
        W = np.diag(weights)
        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            # Add small regularization for stability
            coefficients = np.linalg.solve(
                XtWX + 1e-6 * np.eye(n_segments), XtWy
            )
        except np.linalg.LinAlgError:
            # Fallback: simple correlation
            coefficients = np.array([
                np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0.0
                for i in range(n_segments)
            ])

        # Normalize scores
        scores = normalize_scores(coefficients.tolist())

        sentence_importances = [
            SentenceImportance(text=seg, index=i, score=scores[i])
            for i, seg in enumerate(sentences)
        ]

        return Explanation(
            method="prompt_lime",
            sentence_importances=sentence_importances,
            summary=self._build_summary(sentence_importances),
            metadata={
                "num_perturbations": self.num_perturbations,
                "num_segments": n_segments,
                "mask_token": self.mask_token,
            },
        )

    def _build_summary(self, importances: List[SentenceImportance]) -> str:
        """Build a human-readable summary of the explanation."""
        sorted_imp = sorted(importances, key=lambda s: abs(s.score), reverse=True)
        top = sorted_imp[:3]
        lines = ["Most influential parts of the prompt:"]
        for i, s in enumerate(top, 1):
            direction = "+" if s.score > 0 else "-"
            lines.append(f"  {i}. [{direction}{abs(s.score):.2f}] \"{s.text[:80]}\"")
        return "\n".join(lines)
