"""Counterfactual explanation generator for LLM prompts.

Finds the **minimal change** to a prompt that causes a meaningfully
different LLM output. Inspired by counterfactual explanation methods
in traditional ML, adapted for natural language.

Strategies:
    1. **Token substitution**: Replace individual tokens with synonyms/antonyms.
    2. **Sentence removal**: Drop sentences to find which one is essential.
    3. **Instruction flip**: Negate or alter key instructions.
    4. **Guided search**: Use an LLM to propose minimal edits.

For each candidate, measure (edit_distance, semantic_distance) and
return the Pareto-optimal set: smallest change → biggest output flip.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from clear_trace.explain.core.base import BaseExplainer, LLMClient
from clear_trace.explain.core.types import CounterfactualResult, Explanation
from clear_trace.explain.core.utils import (
    cosine_similarity_text,
    edit_distance,
    segment_sentences,
    tokenize_simple,
    detokenize_simple,
    diff_tokens,
)


# Common instruction negation pairs
_NEGATION_MAP = {
    "do": "don't",
    "don't": "do",
    "always": "never",
    "never": "always",
    "include": "exclude",
    "exclude": "include",
    "yes": "no",
    "no": "yes",
    "true": "false",
    "false": "true",
    "positive": "negative",
    "negative": "positive",
    "allow": "forbid",
    "forbid": "allow",
    "concise": "verbose",
    "verbose": "concise",
    "formal": "casual",
    "casual": "formal",
    "detailed": "brief",
    "brief": "detailed",
    "simple": "complex",
    "complex": "simple",
}


class CounterfactualGenerator(BaseExplainer):
    """Generate counterfactual explanations for LLM prompts.

    Finds minimal prompt edits that cause the LLM to produce
    meaningfully different outputs.

    Args:
        llm: LLMClient for re-querying the model.
        strategies: List of strategies to use. Default: all.
            Options: "token_sub", "sentence_drop", "instruction_flip", "llm_guided"
        max_candidates: Maximum number of counterfactual candidates to generate.
        flip_threshold: Semantic distance above which a change counts as a "flip".
        seed: Random seed.

    Example:
        >>> gen = CounterfactualGenerator(llm=llm, max_candidates=20)
        >>> explanation = gen.explain(
        ...     "You are a Python expert. Explain decorators simply.",
        ...     output="Decorators are..."
        ... )
        >>> for cf in explanation.flipped_counterfactuals():
        ...     print(f"Change: {cf.change_description}")
        ...     print(f"  Edit distance: {cf.edit_distance}")
        ...     print(f"  New output: {cf.modified_output[:100]}")
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        strategies: Optional[List[str]] = None,
        max_candidates: int = 20,
        flip_threshold: float = 0.5,
        seed: int = 42,
        similarity_fn: Optional[Callable] = None,
        **config: Any,
    ):
        super().__init__(llm=llm, **config)
        self.strategies = strategies or ["token_sub", "sentence_drop", "instruction_flip"]
        self.max_candidates = max_candidates
        self.flip_threshold = flip_threshold
        self.seed = seed
        self.similarity_fn = similarity_fn or cosine_similarity_text

    def _explain_impl(self, prompt: str, output: str, **kwargs: Any) -> Explanation:
        rng = random.Random(self.seed)
        candidates: List[Tuple[str, str]] = []  # (modified_prompt, change_description)

        if "sentence_drop" in self.strategies:
            candidates.extend(self._sentence_drop_candidates(prompt))

        if "instruction_flip" in self.strategies:
            candidates.extend(self._instruction_flip_candidates(prompt))

        if "token_sub" in self.strategies:
            candidates.extend(self._token_sub_candidates(prompt, rng))

        if "llm_guided" in self.strategies and self.llm:
            candidates.extend(self._llm_guided_candidates(prompt, output))

        # Limit candidates
        if len(candidates) > self.max_candidates:
            rng.shuffle(candidates)
            candidates = candidates[: self.max_candidates]

        # Evaluate each candidate
        results: List[CounterfactualResult] = []
        orig_tokens = tokenize_simple(prompt)

        for modified_prompt, change_desc in candidates:
            if self.llm:
                modified_output = self.llm(modified_prompt)
            else:
                modified_output = ""

            sim = self.similarity_fn(output, modified_output)
            semantic_dist = 1.0 - sim
            ed = edit_distance(prompt, modified_prompt)
            mod_tokens = tokenize_simple(modified_prompt)
            changed = diff_tokens(orig_tokens, mod_tokens)

            results.append(
                CounterfactualResult(
                    original_prompt=prompt,
                    modified_prompt=modified_prompt,
                    original_output=output,
                    modified_output=modified_output,
                    change_description=change_desc,
                    edit_distance=ed,
                    semantic_distance=semantic_dist,
                    changed_tokens=changed,
                )
            )

        # Sort by efficiency: highest semantic distance per edit distance
        results.sort(
            key=lambda r: r.semantic_distance / max(r.edit_distance, 1),
            reverse=True,
        )

        return Explanation(
            method="counterfactual",
            counterfactuals=results,
            summary=self._build_summary(results),
            metadata={
                "num_candidates": len(candidates),
                "num_flips": sum(1 for r in results if r.is_flip),
                "strategies": self.strategies,
            },
        )

    def _sentence_drop_candidates(self, prompt: str) -> List[Tuple[str, str]]:
        """Generate candidates by dropping each sentence."""
        sentences = segment_sentences(prompt)
        candidates = []
        for i, sent in enumerate(sentences):
            remaining = [s for j, s in enumerate(sentences) if j != i]
            if remaining:
                modified = " ".join(remaining)
                candidates.append((modified, f"Removed sentence {i + 1}: \"{sent[:60]}\""))
        return candidates

    def _instruction_flip_candidates(self, prompt: str) -> List[Tuple[str, str]]:
        """Generate candidates by flipping instruction keywords."""
        tokens = tokenize_simple(prompt)
        candidates = []
        for i, tok in enumerate(tokens):
            low = tok.lower()
            if low in _NEGATION_MAP:
                replacement = _NEGATION_MAP[low]
                # Preserve capitalization
                if tok[0].isupper():
                    replacement = replacement.capitalize()
                new_tokens = tokens[:i] + [replacement] + tokens[i + 1 :]
                modified = detokenize_simple(new_tokens)
                candidates.append(
                    (modified, f"Flipped '{tok}' → '{replacement}' at position {i}")
                )
        return candidates

    def _token_sub_candidates(
        self, prompt: str, rng: random.Random
    ) -> List[Tuple[str, str]]:
        """Generate candidates by substituting random tokens with [REDACTED]."""
        tokens = tokenize_simple(prompt)
        candidates = []
        # Pick up to 10 content tokens to redact
        content_indices = [
            i for i, t in enumerate(tokens) if len(t) > 2 and t.isalpha()
        ]
        selected = rng.sample(content_indices, min(10, len(content_indices)))
        for i in selected:
            new_tokens = tokens[:i] + ["[REDACTED]"] + tokens[i + 1 :]
            modified = detokenize_simple(new_tokens)
            candidates.append(
                (modified, f"Redacted token '{tokens[i]}' at position {i}")
            )
        return candidates

    def _llm_guided_candidates(
        self, prompt: str, output: str
    ) -> List[Tuple[str, str]]:
        """Use the LLM itself to propose minimal prompt edits."""
        if not self.llm:
            return []

        meta_prompt = (
            "Given this LLM prompt and its output, suggest 3 minimal changes to the "
            "prompt that would cause a significantly different output. "
            "Return ONLY the modified prompts, one per line, prefixed with '>>> '.\n\n"
            f"PROMPT: {prompt}\n\n"
            f"OUTPUT: {output[:500]}\n\n"
            "MODIFIED PROMPTS:"
        )

        try:
            response = self.llm(meta_prompt)
            candidates = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith(">>> "):
                    modified = line[4:].strip()
                    if modified and modified != prompt:
                        candidates.append((modified, "LLM-suggested edit"))
            return candidates[:3]
        except Exception:
            return []

    def _build_summary(self, results: List[CounterfactualResult]) -> str:
        flips = [r for r in results if r.is_flip]
        lines = [
            f"Generated {len(results)} counterfactual candidates, {len(flips)} caused output flips."
        ]
        if flips:
            best = flips[0]  # already sorted by efficiency
            lines.append(f"Most efficient flip: {best.change_description}")
            lines.append(
                f"  Edit distance: {best.edit_distance}, "
                f"Semantic distance: {best.semantic_distance:.3f}"
            )
        return "\n".join(lines)
