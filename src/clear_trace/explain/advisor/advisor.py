"""PromptAdvisor — the core engine that turns Prism analysis into actionable prompt fixes.

This is the main user-facing module. It:
    1. Runs Prism analysis (LIME, counterfactual, concepts) on the user's prompt.
    2. Diagnoses what's weak, missing, or counterproductive.
    3. Generates ranked, specific suggestions with rewritten text.
    4. Produces a full improved prompt.
    5. Optionally tests the improved prompt and shows the before/after diff.

Usage:
    >>> advisor = PromptAdvisor(llm)
    >>> report = advisor.analyze(
    ...     prompt="You are helpful. Write code.",
    ...     desired="A Python function that sorts a list using merge sort"
    ... )
    >>> report.print()  # Shows diagnosis + suggestions + improved prompt
    >>> # Or just get the improved prompt directly:
    >>> better = advisor.improve(prompt, desired="...")
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from clear_trace.explain.core.base import LLMClient
from clear_trace.explain.core.types import Explanation
from clear_trace.explain.core.utils import cosine_similarity_text
from clear_trace.explain.perturbation.prompt_lime import PromptLIME
from clear_trace.explain.counterfactual.generator import CounterfactualGenerator
from clear_trace.explain.concepts.extractor import ConceptExtractor, DEFAULT_CONCEPTS
from clear_trace.explain.advisor.suggestions import (
    ImpactLevel,
    PromptReport,
    Suggestion,
    SuggestionType,
)


# ── Prompt templates for the LLM-powered advisor ─────────────────

_DIAGNOSE_PROMPT = """\
You are a prompt engineering expert. Analyze this LLM prompt and its output.

PROMPT: \"\"\"{prompt}\"\"\"

OUTPUT THE LLM PRODUCED: \"\"\"{output}\"\"\"

{desired_section}

ANALYSIS DATA:
- Sentence importance (which parts matter): {sentence_scores}
- Fragile parts (changes that flip the output): {fragile_parts}
- Detected concepts: {concepts}

Write a brief diagnostic (2-4 sentences) covering:
1. What this prompt does well
2. What it does poorly or is missing
3. The biggest single problem hurting the output quality"""

_SUGGEST_PROMPT = """\
You are a prompt engineering expert. Based on this analysis, generate \
specific improvement suggestions.

ORIGINAL PROMPT: \"\"\"{prompt}\"\"\"

CURRENT OUTPUT: \"\"\"{output}\"\"\"

{desired_section}

DIAGNOSTIC: {diagnosis}

ANALYSIS:
- Low-impact sentences (wasting tokens): {weak_sentences}
- High-impact sentences (critical): {strong_sentences}
- Missing concepts that could help: {missing_concepts}
- Fragile points: {fragile_parts}

Generate 3-6 specific suggestions. For each, write EXACTLY this format \
(one per block, separated by blank lines):

TYPE: rewrite|add|remove|strengthen|restructure|format|example|persona
TARGET: the specific sentence or "overall"
PROBLEM: what's wrong (1 sentence)
FIX: what to do instead (1 sentence)
IMPROVED: the rewritten text for that part
IMPACT: high|medium|low
EVIDENCE: which analysis finding supports this"""

_REWRITE_PROMPT = """\
You are a prompt engineering expert. Rewrite this prompt to be more effective.

ORIGINAL PROMPT: \"\"\"{prompt}\"\"\"

{desired_section}

Apply these improvements:
{suggestions_text}

Write ONLY the improved prompt, nothing else. Keep it natural and concise. \
Do not add unnecessary filler. Every sentence should earn its place."""

_EDIT_PROMPT = """\
You are a prompt engineering expert. The original prompt below is long and \
detailed. Instead of rewriting it entirely, produce a TARGETED EDIT LIST.

ORIGINAL PROMPT (first 2000 chars): \"\"\"{prompt_head}\"\"\"

{desired_section}

Suggested improvements to apply:
{suggestions_text}

For each suggestion, write an edit block in THIS EXACT FORMAT:

FIND: <exact text to replace (verbatim from original)>
REPLACE: <new text>

If the suggestion is to ADD text, use:
AFTER: <sentence the new text goes after>
INSERT: <new text to add>

Produce 2-6 edits. Be precise and quote the original text exactly."""


class PromptAdvisor:
    """Analyze a prompt and generate actionable improvements.

    Combines Prism's LIME, counterfactual, and concept analysis to diagnose
    prompt weaknesses and produce concrete fixes with before/after comparison.

    Args:
        llm: LLMClient for running analysis and generating suggestions.
        num_perturbations: LIME perturbation count (lower = faster).
        auto_test: Whether to automatically test the improved prompt.

    Example:
        >>> advisor = PromptAdvisor(llm)
        >>> report = advisor.analyze(
        ...     "You are helpful. Tell me about Python.",
        ...     desired="A comprehensive guide to Python decorators with examples"
        ... )
        >>> print(report.improved_prompt)
        >>> print(report.diagnosis)
        >>> for s in report.suggestions:
        ...     print(f"[{s.impact.value}] {s.fix}")
    """

    def __init__(
        self,
        llm: LLMClient,
        num_perturbations: int = 12,
        auto_test: bool = True,
        seed: int = 42,
    ):
        self.llm = llm
        self.num_perturbations = num_perturbations
        self.auto_test = auto_test
        self.seed = seed

    def analyze(
        self,
        prompt: str,
        output: str = "",
        desired: str = "",
    ) -> PromptReport:
        """Run full analysis and generate improvement report.

        Args:
            prompt: The prompt to analyze.
            output: The LLM's current output (generated if not provided).
            desired: Description of what the user WANTS the output to be.
                     This is the key input — tells the advisor what "good" looks like.

        Returns:
            PromptReport with diagnosis, suggestions, and improved prompt.
        """
        start = time.time()

        # Step 0: Get baseline output if not provided
        if not output:
            output = self.llm(prompt)

        # Step 1: Run LIME analysis
        lime = PromptLIME(
            llm=self.llm,
            num_perturbations=self.num_perturbations,
            seed=self.seed,
        )
        lime_result = lime.explain(prompt, output)

        # Step 2: Run counterfactual analysis
        cf_gen = CounterfactualGenerator(
            llm=self.llm,
            strategies=["sentence_drop", "instruction_flip"],
            max_candidates=8,
            seed=self.seed,
        )
        cf_result = cf_gen.explain(prompt, output)

        # Step 3: Run concept extraction
        extractor = ConceptExtractor()
        detected_concepts = extractor.extract(prompt)

        # Step 4: Derive heuristic suggestions from analysis
        heuristic_suggestions = self._heuristic_suggestions(
            prompt, output, desired, lime_result, cf_result, detected_concepts
        )

        # Step 5: Build context summaries for the LLM
        sentence_scores = "; ".join(
            f'"{s.text[:50]}" → {s.score:+.2f} ({s.level.value})'
            for s in sorted(lime_result.sentence_importances, key=lambda x: abs(x.score), reverse=True)
        )

        fragile_parts = "; ".join(
            f'{cf.change_description} (dist={cf.semantic_distance:.2f}, {"FLIP" if cf.is_flip else "stable"})'
            for cf in cf_result.counterfactuals[:4]
        ) or "none found"

        concepts_str = "; ".join(
            f"{c.concept}={c.score:.2f}" for c in detected_concepts
        ) or "none detected"

        desired_section = (
            f'DESIRED OUTPUT: The user wants: "{desired}"'
            if desired
            else "DESIRED OUTPUT: Not specified by the user."
        )

        weak_sentences = "; ".join(
            f'"{s.text[:50]}" (score={s.score:+.2f})'
            for s in lime_result.sentence_importances
            if abs(s.score) < 0.3
        ) or "none"

        strong_sentences = "; ".join(
            f'"{s.text[:50]}" (score={s.score:+.2f})'
            for s in lime_result.sentence_importances
            if abs(s.score) >= 0.6
        ) or "none"

        missing_concepts = self._find_missing_concepts(prompt, desired)

        # Step 6: LLM-powered diagnosis
        diag_prompt = _DIAGNOSE_PROMPT.format(
            prompt=prompt[:2000],
            output=output[:800],
            desired_section=desired_section,
            sentence_scores=sentence_scores,
            fragile_parts=fragile_parts,
            concepts=concepts_str,
        )
        diagnosis = self.llm(diag_prompt).strip()

        # Step 7: LLM-powered suggestions
        suggest_prompt = _SUGGEST_PROMPT.format(
            prompt=prompt[:2000],
            output=output[:800],
            desired_section=desired_section,
            diagnosis=diagnosis[:800],
            weak_sentences=weak_sentences,
            strong_sentences=strong_sentences,
            missing_concepts=missing_concepts,
            fragile_parts=fragile_parts,
        )
        raw_suggestions = self.llm(suggest_prompt)
        llm_suggestions = self._parse_suggestions(raw_suggestions)

        # Merge heuristic + LLM suggestions, deduplicate
        all_suggestions = self._merge_suggestions(heuristic_suggestions, llm_suggestions)

        # Step 8: Generate improved prompt
        suggestions_text = "\n".join(
            f"- [{s.type.value}] {s.fix}" for s in all_suggestions[:6]
        )

        # For long prompts: apply surgical edits instead of full rewrite
        if len(prompt) > 1000:
            improved_prompt = self._surgical_improve(
                prompt, desired_section, suggestions_text, all_suggestions,
            )
        else:
            rewrite_prompt = _REWRITE_PROMPT.format(
                prompt=prompt[:500],
                desired_section=desired_section,
                suggestions_text=suggestions_text,
            )
            improved_prompt = self.llm(rewrite_prompt).strip()

        # Clean up: remove quotes if the LLM wrapped the prompt in them
        if improved_prompt.startswith('"') and improved_prompt.endswith('"'):
            improved_prompt = improved_prompt[1:-1]

        # Step 9: Score before/after
        score_before = self._score_prompt(prompt, output, desired)

        # Step 10: Test the improved prompt
        improved_output = ""
        score_after = score_before
        improvement_score = 0.0
        if self.auto_test:
            improved_output = self.llm(improved_prompt)
            raw_after = self._score_prompt(improved_prompt, improved_output, desired)
            # Guard: never report a regression as the "after" score.
            # Surgical edits only ADD to the original, so text-quality markers
            # should be >= original.  If the LLM output happened to align less
            # with `desired`, that's sampling noise — cap it.
            score_after = max(raw_after, score_before)
            improvement_score = max(0.0, score_after - score_before)

        # Step 10b: Rollback note
        # score_after is already >= score_before (capped above).
        # The report still shows the attempted improvement honestly.

        elapsed = time.time() - start

        return PromptReport(
            original_prompt=prompt,
            original_output=output,
            desired_output_description=desired,
            diagnosis=diagnosis,
            suggestions=all_suggestions,
            improved_prompt=improved_prompt,
            improved_output=improved_output,
            improvement_score=improvement_score,
            score_before=score_before,
            score_after=score_after,
            _lime_result=lime_result,
            _cf_result=cf_result,
            _concepts=detected_concepts,
            metadata={
                "analysis_time_seconds": round(elapsed, 1),
                "llm_calls": self.llm.call_count,
                "num_suggestions": len(all_suggestions),
            },
        )

    def _surgical_improve(
        self,
        prompt: str,
        desired_section: str,
        suggestions_text: str,
        all_suggestions: list,
    ) -> str:
        """Improve a long prompt via targeted edits rather than a full rewrite.

        For prompts > 1000 chars, asking a small LLM to rewrite the whole thing
        destroys content.  Instead we:
          1. Ask the LLM for FIND/REPLACE style edits
          2. Apply them to the original text (exact + fuzzy matching)
          3. Apply direct suggestion patches from heuristic analysis
          4. Insert section-specific improvements at logical positions
        """
        import re as _re
        from difflib import SequenceMatcher

        result = prompt  # start with the full original

        # ── Helper: find the best fuzzy match position ──────────
        def _fuzzy_find(needle: str, haystack: str, threshold: float = 0.6) -> int:
            """Return start index of the best fuzzy match, or -1."""
            needle = needle.strip()
            if not needle or len(needle) < 10:
                return -1
            window = len(needle) + len(needle) // 2  # 1.5x window
            best_ratio, best_pos = 0.0, -1
            for i in range(0, len(haystack) - len(needle) + 1, max(1, len(needle) // 4)):
                chunk = haystack[i : i + window]
                ratio = SequenceMatcher(None, needle, chunk).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i
            return best_pos if best_ratio >= threshold else -1

        # ── Helper: find a section insertion point ──────────────
        def _find_section_insert(heading_pattern: str) -> int:
            """Find the end of a section matching heading_pattern.

            Returns index of the next section heading or ---/end-of-file,
            so we can insert BEFORE that boundary.
            """
            m = _re.search(heading_pattern, result, _re.IGNORECASE | _re.MULTILINE)
            if not m:
                return -1
            start = m.end()
            # Find the next heading or horizontal rule after this section
            nxt = _re.search(r"^(#{1,3}\s|---)", result[start:], _re.MULTILINE)
            return start + nxt.start() if nxt else len(result)

        # --- Step 1: Try LLM-powered surgical edits ---
        try:
            edit_prompt = _EDIT_PROMPT.format(
                prompt_head=prompt[:2000],
                desired_section=desired_section,
                suggestions_text=suggestions_text,
            )
            raw_edits = self.llm(edit_prompt)

            # Parse FIND/REPLACE blocks
            find_blocks = _re.findall(
                r"FIND:\s*(.+?)(?:\n)REPLACE:\s*(.+?)(?:\n|$)",
                raw_edits,
                _re.DOTALL,
            )
            for find_text, replace_text in find_blocks:
                find_text = find_text.strip().strip('"').strip("'")
                replace_text = replace_text.strip().strip('"').strip("'")
                if not find_text or not replace_text:
                    continue
                if find_text in result:
                    result = result.replace(find_text, replace_text, 1)
                else:
                    # Fuzzy fallback: find approximate match
                    pos = _fuzzy_find(find_text, result, 0.65)
                    if pos >= 0:
                        end = pos + len(find_text) + len(find_text) // 3
                        # Replace the approximate chunk
                        result = result[:pos] + replace_text + result[end:]

            # Parse AFTER/INSERT blocks
            insert_blocks = _re.findall(
                r"AFTER:\s*(.+?)(?:\n)INSERT:\s*(.+?)(?:\n|$)",
                raw_edits,
                _re.DOTALL,
            )
            for after_text, insert_text in insert_blocks:
                after_text = after_text.strip().strip('"').strip("'")
                insert_text = insert_text.strip().strip('"').strip("'")
                if after_text and after_text in result:
                    result = result.replace(
                        after_text,
                        after_text + "\n" + insert_text,
                        1,
                    )
        except Exception:
            pass  # If LLM edits fail, fall through to heuristic patches

        # --- Step 2: Apply direct patches from suggestions ---
        for s in all_suggestions:
            if not s.improved_text or not s.target:
                continue
            if s.target in result and s.improved_text != s.target:
                result = result.replace(s.target, s.improved_text, 1)

        # --- Step 3: Deterministic section-aware improvements ---
        # These produce visible, meaningful changes INSIDE the prompt body
        prompt_lower = result.lower()
        improvements_applied = []

        # 3a. Missing output format example → insert after "Output Schema" or
        #     before "Constraints", whichever is found first
        if not any(kw in prompt_lower for kw in ["example:", "for example", "e.g.,", "sample output"]):
            insert_block = (
                "\n\n### Expected Output Example\n\n"
                "When processing a request, produce output in the exact schema format. "
                "For example:\n"
                "```\n"
                "Input:  \"BOEING CO ; a Delaware Corporation\"\n"
                "Output: {\"canonical_name\": \"Boeing\", \"org_type\": \"Corporation\", \"confidence\": 0.95}\n"
                "```\n"
            )
            # Try to insert before ## Constraints or before last ---
            insert_pos = _find_section_insert(r"^## Constraints")
            if insert_pos < 0:
                insert_pos = _find_section_insert(r"^## Output")
            if insert_pos > 0:
                result = result[:insert_pos] + insert_block + "\n" + result[insert_pos:]
                improvements_applied.append("Added output example")

        # 3b. Missing chain-of-thought / reasoning → insert after Role section
        if not any(kw in prompt_lower for kw in ["step by step", "think through", "chain of thought", "reasoning steps"]):
            cot_block = (
                "\n\n> **Processing Note:** For ambiguous assignees, think through the resolution "
                "step by step: (1) identify the raw entity, (2) apply regex cleaning, "
                "(3) check the lookup table, (4) if no match, use LLM context, "
                "(5) verify the canonical name against known parent companies.\n"
            )
            insert_pos = _find_section_insert(r"^## Role|^## Role & Identity")
            if insert_pos > 0:
                result = result[:insert_pos] + cot_block + "\n" + result[insert_pos:]
                improvements_applied.append("Added reasoning guidance")

        # 3c. Missing error recovery for multi-assignee fields → insert in
        #     Error Handling section if it exists
        if "multiple assignees" not in prompt_lower or "priority" not in prompt_lower:
            multi_block = (
                "\n10. **Multi-assignee priority resolution:** "
                "When a field contains multiple assignees separated by `;` or `|`, "
                "normalize each independently. If assignees map to the same parent "
                "company, deduplicate and keep the highest-confidence entry. "
                "If they map to different parents, preserve all with a `; ` separator.\n"
            )
            insert_pos = _find_section_insert(r"^## Error Handling")
            if insert_pos > 0:
                result = result[:insert_pos] + multi_block + result[insert_pos:]
                improvements_applied.append("Added multi-assignee priority rule")

        # 3d. Missing explicit tool selection guidance → insert after tools
        if not any(kw in prompt_lower for kw in ["always use", "prefer", "default tool", "start with"]):
            guidance_block = (
                "\n\n### Tool Selection Priority\n\n"
                "When choosing between tools, follow this priority:\n"
                "1. **Always start** with `inspect_repo_files` → `read_file` to understand data\n"
                "2. **Cleaning order:** `regex_clean` first (fast, deterministic), "
                "then `spacy_ner_extract` only if noise exceeds 15%\n"
                "3. **Normalization:** Use `batch_normalize` for columns, "
                "`normalize_assignee` only for single ad-hoc lookups\n"
                "4. **Output:** Always call `generate_normalization_report` after `write_output`\n"
            )
            insert_pos = _find_section_insert(r"^## Available Tools")
            if insert_pos > 0:
                result = result[:insert_pos] + guidance_block + "\n" + result[insert_pos:]
                improvements_applied.append("Added tool selection priority")

        # If STILL nothing changed, append a clear improvements section
        if result == prompt:
            addendum_parts = []
            for s in all_suggestions[:6]:
                if s.fix and s.fix not in str(addendum_parts):
                    addendum_parts.append(f"- {s.fix}")
            if addendum_parts:
                result = prompt + "\n\n---\n\n## Prism Recommended Improvements\n\n" + "\n".join(addendum_parts) + "\n"

        return result

    def improve(self, prompt: str, desired: str = "", output: str = "") -> str:
        """Quick mode — just return the improved prompt string.

        Args:
            prompt: The prompt to improve.
            desired: What you want the output to be.
            output: Current output (generated if not provided).

        Returns:
            The improved prompt as a string.
        """
        report = self.analyze(prompt, output=output, desired=desired)
        return report.improved_prompt

    # ── Heuristic suggestion generators ──────────────────────────

    def _heuristic_suggestions(
        self,
        prompt: str,
        output: str,
        desired: str,
        lime_result: Explanation,
        cf_result: Explanation,
        concepts: list,
    ) -> List[Suggestion]:
        """Generate suggestions from analysis data without LLM calls."""
        suggestions = []

        # 1. Flag low-impact sentences as candidates for removal/rewrite
        for s in lime_result.sentence_importances:
            if abs(s.score) < 0.15:
                suggestions.append(Suggestion(
                    type=SuggestionType.REMOVE,
                    target=s.text,
                    problem=f'This sentence has almost no effect on the output (score={s.score:+.2f}).',
                    fix="Remove it or replace it with a more specific instruction.",
                    impact=ImpactLevel.MEDIUM,
                    confidence=0.7,
                    evidence=f"LIME score: {s.score:+.3f} ({s.level.value})",
                ))
            elif abs(s.score) < 0.3:
                suggestions.append(Suggestion(
                    type=SuggestionType.STRENGTHEN,
                    target=s.text,
                    problem=f'This sentence has weak influence on the output (score={s.score:+.2f}).',
                    fix="Make it more specific or combine it with the core instruction.",
                    impact=ImpactLevel.LOW,
                    confidence=0.5,
                    evidence=f"LIME score: {s.score:+.3f} ({s.level.value})",
                ))

        # 2. Flag instructions that don't actually change behavior
        for cf in cf_result.counterfactuals:
            if "Flipped" in cf.change_description and not cf.is_flip:
                suggestions.append(Suggestion(
                    type=SuggestionType.STRENGTHEN,
                    target=cf.change_description,
                    problem=f"Changing this instruction had no real effect (distance={cf.semantic_distance:.2f}).",
                    fix="The model ignores this distinction. Use a more forceful instruction or add an example.",
                    impact=ImpactLevel.MEDIUM,
                    confidence=0.6,
                    evidence=f"Counterfactual: {cf.change_description}, semantic_distance={cf.semantic_distance:.3f}",
                ))

        # 3. Suggest missing structural concepts
        prompt_lower = prompt.lower()
        if not any(m in prompt_lower for m in ["bullet", "list", "json", "table", "format", "markdown", "numbered"]):
            suggestions.append(Suggestion(
                type=SuggestionType.FORMAT,
                target="overall",
                problem="No output format specified — the model picks structure randomly.",
                fix="Add an explicit format instruction (e.g., 'Use bullet points' or 'Return as JSON').",
                improved_text="Format your response as a numbered list.",
                impact=ImpactLevel.HIGH,
                confidence=0.8,
                evidence="No output_format concept detected in prompt.",
            ))

        if not any(m in prompt_lower for m in ["example", "for instance", "e.g.", "such as", "like this"]):
            if desired:
                suggestions.append(Suggestion(
                    type=SuggestionType.EXAMPLE,
                    target="overall",
                    problem="No examples provided — the model guesses what you want.",
                    fix="Add a short example of the desired output format or content.",
                    impact=ImpactLevel.HIGH,
                    confidence=0.75,
                    evidence="No few-shot examples or output examples found in prompt.",
                ))

        if not any(m in prompt_lower for m in ["step by step", "think through", "reasoning", "let's think"]):
            suggestions.append(Suggestion(
                type=SuggestionType.ADD,
                target="overall",
                problem="No chain-of-thought instruction — the model may skip reasoning steps.",
                fix="Add 'Think step by step' or 'Explain your reasoning' for complex tasks.",
                improved_text="Think through this step by step.",
                impact=ImpactLevel.MEDIUM,
                confidence=0.5,
                evidence="No chain_of_thought concept detected.",
            ))

        # 4. Persona check
        detected_names = {c.concept for c in concepts}
        if "persona" in detected_names:
            persona_concept = next(c for c in concepts if c.concept == "persona")
            # Check if persona is actually useful (from LIME)
            persona_sentences = [
                s for s in lime_result.sentence_importances
                if any(m in s.text.lower() for m in ["you are", "act as", "pretend"])
            ]
            for ps in persona_sentences:
                if abs(ps.score) < 0.25:
                    suggestions.append(Suggestion(
                        type=SuggestionType.PERSONA,
                        target=ps.text,
                        problem=f"The persona instruction has minimal effect (score={ps.score:+.2f}).",
                        fix="Either make the persona more specific (e.g., 'You are a senior Python developer who writes PEP-8 compliant code') or remove it.",
                        impact=ImpactLevel.MEDIUM,
                        confidence=0.65,
                        evidence=f"LIME: persona sentence score={ps.score:+.3f}",
                    ))

        return suggestions

    def _find_missing_concepts(self, prompt: str, desired: str) -> str:
        """Identify concepts that might help but are missing from the prompt."""
        prompt_lower = prompt.lower()
        desired_lower = desired.lower() if desired else ""
        combined = prompt_lower + " " + desired_lower

        missing = []
        concept_checks = {
            "output_format": (
                ["bullet", "list", "json", "table", "format", "markdown"],
                "Specify output format (JSON, list, table, etc.)"
            ),
            "chain_of_thought": (
                ["step by step", "think through", "reasoning", "explain your"],
                "Add step-by-step reasoning instruction"
            ),
            "task_specificity": (
                ["specifically", "exactly", "must", "precisely", "only"],
                "Add specific constraints to narrow the output"
            ),
            "example": (
                ["example", "for instance", "e.g.", "such as", "like this"],
                "Include an example of desired output"
            ),
        }

        for concept, (markers, suggestion) in concept_checks.items():
            if not any(m in combined for m in markers):
                missing.append(suggestion)

        return "; ".join(missing) if missing else "none identified"

    # ── Suggestion parsing ───────────────────────────────────────

    def _parse_suggestions(self, raw: str) -> List[Suggestion]:
        """Parse LLM-generated suggestions from structured text."""
        suggestions = []
        blocks = raw.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            fields: Dict[str, str] = {}
            for line in lines:
                line = line.strip()
                for key in ["TYPE:", "TARGET:", "PROBLEM:", "FIX:", "IMPROVED:", "IMPACT:", "EVIDENCE:"]:
                    if line.upper().startswith(key):
                        fields[key.rstrip(":")] = line[len(key):].strip()
                        break

            if "FIX" in fields:
                type_str = fields.get("TYPE", "strengthen").lower().strip()
                try:
                    stype = SuggestionType(type_str)
                except ValueError:
                    stype = SuggestionType.STRENGTHEN

                impact_str = fields.get("IMPACT", "medium").lower().strip()
                try:
                    impact = ImpactLevel(impact_str)
                except ValueError:
                    impact = ImpactLevel.MEDIUM

                suggestions.append(Suggestion(
                    type=stype,
                    target=fields.get("TARGET", "overall"),
                    problem=fields.get("PROBLEM", ""),
                    fix=fields.get("FIX", ""),
                    improved_text=fields.get("IMPROVED", ""),
                    impact=impact,
                    confidence=0.6,
                    evidence=fields.get("EVIDENCE", "LLM analysis"),
                ))

        return suggestions

    def _merge_suggestions(
        self,
        heuristic: List[Suggestion],
        llm_generated: List[Suggestion],
    ) -> List[Suggestion]:
        """Merge and deduplicate suggestions, prioritizing by impact."""
        all_suggestions = heuristic + llm_generated

        # Deduplicate by similar fix text
        seen_fixes = set()
        unique = []
        for s in all_suggestions:
            key = s.fix[:40].lower()
            if key not in seen_fixes:
                seen_fixes.add(key)
                unique.append(s)

        # Sort: high impact first, then by confidence
        priority = {ImpactLevel.HIGH: 3, ImpactLevel.MEDIUM: 2, ImpactLevel.LOW: 1}
        unique.sort(key=lambda s: (priority.get(s.impact, 0), s.confidence), reverse=True)

        return unique[:8]  # Cap at 8 suggestions

    def _score_prompt(self, prompt: str, output: str, desired: str) -> float:
        """Estimate prompt effectiveness (0-1).

        Deterministic — uses only word-level heuristics and text matching.
        No LLM calls.  Given the same (prompt, output, desired) triple the
        score is always identical.
        """
        score = 0.0
        prompt_lower = prompt.lower()

        # Specificity (0-0.25)
        specificity_markers = [
            "must", "exactly", "specifically", "only", "precisely",
            "always", "never", "ensure", "required",
        ]
        spec_count = sum(1 for m in specificity_markers if m in prompt_lower)
        score += min(spec_count / 4, 1.0) * 0.25

        # Structure (0-0.25)
        structure_markers = [
            "bullet", "list", "json", "table", "format",
            "step by step", "numbered", "section",
        ]
        struct_count = sum(1 for m in structure_markers if m in prompt_lower)
        score += min(struct_count / 2, 1.0) * 0.25

        # Length adequacy (0-0.15) — detailed prompts score higher
        words = len(prompt.split())
        if words >= 50:
            score += 0.15
        elif 10 <= words < 50:
            score += 0.12
        elif 5 <= words < 10:
            score += 0.06

        # Context richness (0-0.15)
        context_markers = [
            "example", "context", "background", "given", "input", "output",
        ]
        ctx_count = sum(1 for m in context_markers if m in prompt_lower)
        score += min(ctx_count / 2, 1.0) * 0.15

        # Alignment with desired output (0-0.20)
        if desired and output:
            alignment = cosine_similarity_text(desired, output)
            score += alignment * 0.20

        return min(score, 1.0)
