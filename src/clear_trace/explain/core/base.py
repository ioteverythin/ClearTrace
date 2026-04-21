"""Base explainer class that all Prism explanation methods inherit from."""

from __future__ import annotations

import abc
import time
from typing import Any, Callable, Dict, List, Optional, Union

from clear_trace.explain.core.types import Explanation


class LLMClient:
    """Thin wrapper around an LLM provider for running perturbation queries.

    Prism needs to call the LLM multiple times during explanation
    (e.g. perturbed prompts, counterfactual candidates). This wrapper
    provides a uniform interface for OpenAI, Anthropic, or any callable.

    Args:
        fn: Callable that takes a prompt string and returns a response string.
        model: Model identifier (for logging).
        provider: Provider name (for logging).
        deterministic: When True, injects ``temperature=0`` into every call
            and caches responses so identical prompts always return the same
            result.  This makes Prism's trace/comparison pipeline fully
            reproducible.

    Usage::

        # From OpenAI client
        llm = LLMClient.from_openai(client, model="gpt-4o")

        # From Anthropic client
        llm = LLMClient.from_anthropic(client, model="claude-sonnet-4-20250514")

        # From any callable (deterministic for SE traceability)
        llm = LLMClient(fn=my_function, deterministic=True)
    """

    def __init__(
        self,
        fn: Optional[Callable[..., str]] = None,
        model: str = "",
        provider: str = "custom",
        deterministic: bool = False,
    ):
        self._fn = fn
        self.model = model
        self.provider = provider
        self.deterministic = deterministic
        self._call_count = 0
        self._total_tokens = 0
        self._cache: dict[str, str] = {}  # prompt-hash → response

    @classmethod
    def from_openai(cls, client: Any, model: str = "gpt-4o", **kwargs: Any) -> "LLMClient":
        """Create an LLMClient from an OpenAI client instance."""

        def _call(prompt: str, system: str = "", **kw: Any) -> str:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            merged = {**kwargs, **kw}
            resp = client.chat.completions.create(model=model, messages=messages, **merged)
            return resp.choices[0].message.content or ""

        instance = cls(fn=_call, model=model, provider="openai")
        instance._raw_client = client
        return instance

    @classmethod
    def from_anthropic(cls, client: Any, model: str = "claude-sonnet-4-20250514", **kwargs: Any) -> "LLMClient":
        """Create an LLMClient from an Anthropic client instance."""

        def _call(prompt: str, system: str = "", **kw: Any) -> str:
            merged = {**kwargs, **kw}
            resp = client.messages.create(
                model=model,
                max_tokens=merged.pop("max_tokens", 1024),
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
                **merged,
            )
            return resp.content[0].text

        instance = cls(fn=_call, model=model, provider="anthropic")
        instance._raw_client = client
        return instance

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Call the underlying LLM with a prompt and return the text response.

        In deterministic mode, ``temperature=0`` is injected and responses
        are cached by prompt text so repeated identical calls return the
        same result.
        """
        if self._fn is None:
            raise RuntimeError(
                "No LLM function configured. "
                "Use from_openai(), from_anthropic(), or pass fn=..."
            )
        self._call_count += 1

        if self.deterministic:
            # Force temperature=0 for reproducibility
            kwargs.setdefault("temperature", 0)

            # Check cache
            import hashlib
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]

            result = self._fn(prompt, **kwargs)
            self._cache[cache_key] = result
            return result

        return self._fn(prompt, **kwargs)

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def cache_size(self) -> int:
        """Number of cached responses (deterministic mode only)."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    def reset_stats(self) -> None:
        self._call_count = 0
        self._total_tokens = 0


class BaseExplainer(abc.ABC):
    """Abstract base class for all Prism explanation methods.

    Every explainer follows the same lifecycle:
        1. Initialize with an LLMClient
        2. Call .explain(prompt, output, ...) → Explanation
        3. Optionally visualize with Explanation → HTMLReport / ConsoleReport

    Subclasses must implement _explain_impl().
    """

    def __init__(self, llm: Optional[LLMClient] = None, **config: Any):
        self.llm = llm
        self.config = config
        self._last_explanation: Optional[Explanation] = None

    @abc.abstractmethod
    def _explain_impl(
        self,
        prompt: str,
        output: str,
        **kwargs: Any,
    ) -> Explanation:
        """Internal implementation of the explanation logic.

        Args:
            prompt: The input prompt sent to the LLM.
            output: The LLM's response.
            **kwargs: Method-specific parameters.

        Returns:
            An Explanation object with the relevant fields populated.
        """
        ...

    def explain(
        self,
        prompt: str,
        output: str = "",
        generate_reasons: bool = False,
        **kwargs: Any,
    ) -> Explanation:
        """Generate an explanation for the given prompt/output pair.

        If output is not provided and an LLM client is configured,
        the LLM will be called to generate the output first.

        Args:
            prompt: The input prompt.
            output: The LLM's response (optional if llm is set).
            generate_reasons: If True and an LLM is available, generate
                natural language 'why' explanations for each finding.
            **kwargs: Passed to _explain_impl.

        Returns:
            A populated Explanation object.
        """
        if not output and self.llm:
            output = self.llm(prompt)

        start = time.time()
        explanation = self._explain_impl(prompt, output, **kwargs)
        elapsed = time.time() - start

        explanation.prompt = prompt
        explanation.output = output
        explanation.metadata["explain_time_seconds"] = round(elapsed, 3)
        if self.llm:
            explanation.model_name = self.llm.model
            explanation.metadata["llm_calls"] = self.llm.call_count

        # Generate causal 'why' reasoning if requested
        if generate_reasons and self.llm:
            from clear_trace.explain.reasoning.engine import ReasoningEngine
            reasoner = ReasoningEngine(self.llm)
            reasoner.add_reasons(explanation)

        self._last_explanation = explanation
        return explanation

    @property
    def last_explanation(self) -> Optional[Explanation]:
        return self._last_explanation
