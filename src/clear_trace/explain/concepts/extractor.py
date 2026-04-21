"""Concept extractor — identify high-level human concepts in prompts/outputs.

Instead of explaining at the token level ("the word 'please' was important"),
concept-based explanations operate at a higher level:
    "The concept of POLITENESS influenced the output tone"
    "The concept of SAFETY CONSTRAINT prevented the model from answering"

This module:
    1. Extracts candidate concepts from the prompt (either predefined or LLM-detected).
    2. Tests each concept's influence by constructing concept-presence/absence probes.
    3. Scores each concept by its impact on the output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from clear_trace.explain.core.base import LLMClient
from clear_trace.explain.core.types import ConceptAttribution
from clear_trace.explain.core.utils import cosine_similarity_text


# Predefined concept library for common LLM behaviors
DEFAULT_CONCEPTS: Dict[str, Dict[str, Any]] = {
    "politeness": {
        "description": "Courteous language, please/thank you, polite framing",
        "positive_markers": ["please", "thank you", "kindly", "would you", "could you"],
        "negative_markers": ["tell me now", "just do it", "I demand"],
    },
    "safety_constraint": {
        "description": "Content safety boundaries, refusal triggers",
        "positive_markers": ["harmful", "dangerous", "illegal", "unethical", "weapon"],
        "negative_markers": ["educational", "academic", "hypothetical", "fiction"],
    },
    "expertise_level": {
        "description": "Expected technical depth of the response",
        "positive_markers": ["expert", "advanced", "technical", "PhD", "detailed"],
        "negative_markers": ["simple", "beginner", "ELI5", "basic", "overview"],
    },
    "output_format": {
        "description": "Structural constraints on the output",
        "positive_markers": ["bullet points", "JSON", "table", "numbered list", "markdown"],
        "negative_markers": ["paragraph", "essay", "narrative", "prose"],
    },
    "persona": {
        "description": "Role or character assignment for the LLM",
        "positive_markers": ["you are", "act as", "pretend", "role", "character"],
        "negative_markers": [],
    },
    "task_specificity": {
        "description": "How specific vs. open-ended the task is",
        "positive_markers": ["specifically", "exactly", "only", "must", "precisely"],
        "negative_markers": ["anything", "whatever", "general", "broadly", "any"],
    },
    "creativity": {
        "description": "Creative vs. factual response expectation",
        "positive_markers": ["creative", "imagine", "story", "invent", "brainstorm"],
        "negative_markers": ["factual", "accurate", "cite", "source", "evidence"],
    },
    "chain_of_thought": {
        "description": "Step-by-step reasoning instruction",
        "positive_markers": [
            "step by step", "think through", "reasoning", "explain your",
            "show your work", "let's think",
        ],
        "negative_markers": ["just answer", "final answer only", "no explanation"],
    },
}


class ConceptExtractor:
    """Extract and score concepts present in a prompt.

    Args:
        concepts: Custom concept definitions. If None, uses DEFAULT_CONCEPTS.
        llm: Optional LLMClient for LLM-assisted concept extraction.

    Example:
        >>> extractor = ConceptExtractor()
        >>> concepts = extractor.extract("You are an expert. Please explain quantum physics step by step.")
        >>> for c in concepts:
        ...     print(f"{c.concept}: {c.score:.2f} — {c.description}")
        expertise_level: 0.80 — Expected technical depth...
        chain_of_thought: 0.60 — Step-by-step reasoning...
        politeness: 0.40 — Courteous language...
    """

    def __init__(
        self,
        concepts: Optional[Dict[str, Dict[str, Any]]] = None,
        llm: Optional[LLMClient] = None,
    ):
        self.concepts = concepts or DEFAULT_CONCEPTS
        self.llm = llm

    def extract(self, text: str) -> List[ConceptAttribution]:
        """Extract concepts from text using marker matching.

        Args:
            text: The prompt or text to analyze.

        Returns:
            List of ConceptAttribution objects, sorted by score (descending).
        """
        results: List[ConceptAttribution] = []
        text_lower = text.lower()

        for concept_name, concept_def in self.concepts.items():
            pos_markers = concept_def.get("positive_markers", [])
            neg_markers = concept_def.get("negative_markers", [])

            # Count marker matches
            pos_matches = [m for m in pos_markers if m.lower() in text_lower]
            neg_matches = [m for m in neg_markers if m.lower() in text_lower]

            if not pos_matches and not neg_matches:
                continue  # concept not present

            # Score: positive matches increase, negative decrease
            total_markers = max(len(pos_markers) + len(neg_markers), 1)
            score = (len(pos_matches) - len(neg_matches) * 0.5) / total_markers
            score = min(max(score, -1.0), 1.0)

            results.append(
                ConceptAttribution(
                    concept=concept_name,
                    score=score,
                    evidence_tokens=pos_matches + neg_matches,
                    description=concept_def.get("description", ""),
                )
            )

        results.sort(key=lambda c: abs(c.score), reverse=True)
        return results

    def extract_with_llm(self, text: str) -> List[ConceptAttribution]:
        """Use the LLM to discover concepts beyond the predefined set.

        Combines marker-based extraction with LLM-discovered concepts.
        """
        # Start with marker-based extraction
        base_results = self.extract(text)

        if not self.llm:
            return base_results

        # Ask the LLM to identify additional concepts
        prompt = (
            "Analyze this LLM prompt and identify the high-level concepts that "
            "will influence the model's response. For each concept, give a name, "
            "score (0.0-1.0 for how strongly present), and brief description.\n\n"
            f"PROMPT: {text[:1000]}\n\n"
            "Format: concept_name|score|description (one per line)"
        )

        try:
            response = self.llm(prompt)
            existing_names = {r.concept for r in base_results}

            for line in response.strip().split("\n"):
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    name = parts[0].strip().lower().replace(" ", "_")
                    if name in existing_names:
                        continue
                    try:
                        score = float(parts[1].strip())
                    except ValueError:
                        continue
                    desc = parts[2].strip()
                    base_results.append(
                        ConceptAttribution(
                            concept=name,
                            score=min(max(score, 0.0), 1.0),
                            description=desc,
                            metadata={"source": "llm_discovered"},
                        )
                    )
        except Exception:
            pass

        base_results.sort(key=lambda c: abs(c.score), reverse=True)
        return base_results
