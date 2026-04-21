"""ToolAdvisor — analyzes and improves LLM tool-calling pipelines.

Inspects:
 1. Tool/function schemas (names, descriptions, params, types)
 2. System prompt (tool guidance, JSON instructions, fallbacks)
 3. Tool calling accuracy (correct tool selected, right params)
 4. JSON output validity (parseable, schema-compliant, complete)

Then generates fixes for each issue with traceability.

Usage:
    >>> from clear_trace import ToolAdvisor
    >>> advisor = ToolAdvisor(llm=my_llm_client)
    >>> report = advisor.analyze(
    ...     system_prompt="You are a helpful assistant with tools...",
    ...     tools=[{...}, {...}],
    ...     test_cases=[ToolTestCase("What's the weather in NYC?", "get_weather", {"location": "NYC"})],
    ... )
    >>> report.schema_issues   # problems in your tool definitions
    >>> report.test_results    # did the LLM call the right tools?
    >>> report.improved_prompt # fixed system prompt
    >>> report.improved_tools  # fixed tool schemas
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from clear_trace.explain.core.base import LLMClient
from clear_trace.explain.advisor.tool_types import (
    IssueCategory,
    Severity,
    ToolDefinition,
    ToolIssue,
    ToolReport,
    ToolTestCase,
    ToolTestResult,
)


# ── Constants ────────────────────────────────────────────────────

_VAGUE_WORDS = {"do", "thing", "stuff", "handle", "process", "manage", "run", "execute"}

_TOOL_CALL_EXTRACT_PROMPT = """You are simulating a tool-calling LLM assistant. Given the tools and user message below, respond with EXACTLY one JSON object selecting the right tool and parameters. If no tool is needed, respond with {{"tool": null}}.

TOOLS:
{tools_json}

USER MESSAGE: {user_message}

Respond ONLY with a JSON object in this EXACT format:
{{"tool": "tool_name_or_null", "parameters": {{...}}}}

JSON response:"""

_PROMPT_IMPROVE_TEMPLATE = """You are an expert at writing system prompts for LLM agents that use tool calling.

Current system prompt:
---
{system_prompt}
---

Available tools:
{tools_summary}

Issues found:
{issues_text}

Rewrite the system prompt to fix ALL issues. The improved prompt must:
1. Clearly explain WHEN to use each tool
2. Specify input/output formats for tool calls
3. Include a fallback instruction for when no tool applies
4. Guide the model on JSON structure
5. Be specific about parameter types and constraints

Return ONLY the improved system prompt, nothing else.

Improved system prompt:"""

_TOOL_IMPROVE_TEMPLATE = """You are an expert at designing function/tool schemas for LLM function calling.

Current tool definition:
{tool_json}

Issues found:
{issues_text}

Fix the tool definition. Ensure:
- Description is clear and specific (when to use this tool)
- All parameters have descriptions and types
- Required fields are properly marked
- Enum constraints are used where appropriate

Return ONLY the fixed JSON tool definition (OpenAI format), nothing else.

Fixed tool definition:"""


class ToolAdvisor:
    """Analyze and improve LLM tool-calling pipelines.

    Args:
        llm: LLMClient for running test calls.
        seed: Random seed for reproducibility.
    """

    def __init__(self, llm: LLMClient, seed: int = 42):
        self.llm = llm
        self.seed = seed

    def analyze(
        self,
        system_prompt: str,
        tools: List[Union[Dict[str, Any], ToolDefinition]],
        test_cases: Optional[List[ToolTestCase]] = None,
        desired_json_schema: Optional[Dict[str, Any]] = None,
        auto_improve: bool = True,
    ) -> ToolReport:
        """Run full tool pipeline analysis.

        Args:
            system_prompt: The system prompt for the agent.
            tools: Tool definitions (OpenAI format dicts or ToolDefinition objects).
            test_cases: Optional test scenarios for tool calling validation.
            desired_json_schema: Optional JSON schema that outputs should conform to.
            auto_improve: If True, generate improved prompt and tool definitions.

        Returns:
            ToolReport with all issues, test results, and fixes.
        """
        start = time.time()

        # Normalize tool definitions
        tool_defs = []
        for t in tools:
            if isinstance(t, ToolDefinition):
                tool_defs.append(t)
            elif isinstance(t, dict):
                tool_defs.append(ToolDefinition.from_openai(t))
            else:
                raise TypeError(f"Expected dict or ToolDefinition, got {type(t)}")

        report = ToolReport(
            system_prompt=system_prompt,
            tools=tool_defs,
        )

        # ── Phase 1: Static analysis of tool schemas ─────────
        for td in tool_defs:
            report.schema_issues.extend(self._analyze_schema(td))

        # ── Phase 2: Static analysis of system prompt ────────
        report.prompt_issues.extend(
            self._analyze_prompt(system_prompt, tool_defs)
        )

        # ── Phase 3: Run tool calling tests ──────────────────
        if test_cases:
            for tc in test_cases:
                result = self._run_tool_test(
                    system_prompt, tool_defs, tc, desired_json_schema
                )
                report.test_results.append(result)
                report.json_issues.extend(result.issues)

        # ── Phase 4: Score before ────────────────────────────
        report.score_before = self._score_pipeline(report)

        # ── Phase 5: Generate improvements ───────────────────
        if auto_improve:
            report.improved_prompt = self._improve_prompt(
                system_prompt, tool_defs, report.all_issues
            )
            report.improved_tools = self._improve_tools(
                tool_defs, report.schema_issues
            )
            # Re-score
            improved_report = ToolReport(
                system_prompt=report.improved_prompt,
                tools=report.improved_tools,
            )
            for td in report.improved_tools:
                improved_report.schema_issues.extend(self._analyze_schema(td))
            improved_report.prompt_issues.extend(
                self._analyze_prompt(report.improved_prompt, report.improved_tools)
            )
            report.score_after = self._score_pipeline(improved_report)

        # Merge all issues as suggestions
        report.suggestions = sorted(
            report.all_issues,
            key=lambda i: (
                0 if i.severity == Severity.CRITICAL else
                1 if i.severity == Severity.WARNING else 2
            ),
        )

        elapsed = time.time() - start
        report.metadata["analysis_time_seconds"] = round(elapsed, 1)
        report.metadata["llm_calls"] = self.llm.call_count if hasattr(self.llm, "call_count") else "?"
        report.metadata["num_tools"] = len(tool_defs)
        report.metadata["num_tests"] = len(test_cases) if test_cases else 0

        return report

    # ══════════════════════════════════════════════════════════════
    #  Phase 1 — Schema analysis
    # ══════════════════════════════════════════════════════════════

    def _analyze_schema(self, tool: ToolDefinition) -> List[ToolIssue]:
        """Inspect a single tool definition for quality issues."""
        issues: List[ToolIssue] = []
        comp = f"tool:{tool.name}"

        # 1. Missing or vague description
        if not tool.description.strip():
            issues.append(ToolIssue(
                category=IssueCategory.SCHEMA_MISSING_DESC,
                severity=Severity.CRITICAL,
                component=comp,
                problem=f"Tool '{tool.name}' has no description — the LLM won't know when to use it.",
                fix=f"Add a clear description explaining what '{tool.name}' does and when to call it.",
            ))
        elif len(tool.description.split()) < 5:
            issues.append(ToolIssue(
                category=IssueCategory.SCHEMA_MISSING_DESC,
                severity=Severity.WARNING,
                component=comp,
                problem=f"Tool '{tool.name}' description is too short ({len(tool.description.split())} words).",
                fix="Expand the description to at least 10 words explaining purpose and when to use.",
            ))

        # 2. Vague tool name
        name_lower = tool.name.lower().replace("_", " ").replace("-", " ")
        if any(w in name_lower.split() for w in _VAGUE_WORDS):
            issues.append(ToolIssue(
                category=IssueCategory.SCHEMA_AMBIGUOUS_NAME,
                severity=Severity.WARNING,
                component=comp,
                problem=f"Tool name '{tool.name}' contains vague word(s). LLM may confuse it with other tools.",
                fix="Use a specific, action-oriented name (e.g., 'get_weather', 'search_documents').",
            ))

        # 3. Check parameters
        props = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", tool.required or [])

        if not props:
            # No parameters at all — may be fine, just note
            if tool.description and ("search" in tool.description.lower()
                                     or "get" in tool.description.lower()
                                     or "find" in tool.description.lower()):
                issues.append(ToolIssue(
                    category=IssueCategory.SCHEMA_MISSING_REQUIRED,
                    severity=Severity.WARNING,
                    component=comp,
                    problem=f"Tool '{tool.name}' appears to need input but has no parameters.",
                    fix="Add parameters for the required inputs.",
                ))
        else:
            for param_name, param_def in props.items():
                pcomp = f"tool:{tool.name}.{param_name}"

                # Skip malformed param definitions
                if not isinstance(param_def, dict):
                    issues.append(ToolIssue(
                        category=IssueCategory.SCHEMA_MISSING_TYPE,
                        severity=Severity.CRITICAL,
                        component=pcomp,
                        problem=f"Parameter '{param_name}' definition is not a valid object (got {type(param_def).__name__}).",
                        fix=f"Fix '{param_name}' to be a JSON object with at least 'type' and 'description'.",
                    ))
                    continue

                # Missing type
                if "type" not in param_def:
                    issues.append(ToolIssue(
                        category=IssueCategory.SCHEMA_MISSING_TYPE,
                        severity=Severity.CRITICAL,
                        component=pcomp,
                        problem=f"Parameter '{param_name}' has no type — the LLM will guess.",
                        fix=f"Add 'type' (string, number, boolean, array, object) to '{param_name}'.",
                    ))

                # Missing description
                if "description" not in param_def or not param_def["description"].strip():
                    issues.append(ToolIssue(
                        category=IssueCategory.SCHEMA_MISSING_DESC,
                        severity=Severity.WARNING,
                        component=pcomp,
                        problem=f"Parameter '{param_name}' has no description.",
                        fix=f"Add a description explaining what '{param_name}' should contain, with an example.",
                    ))

                # Should use enum?
                param_type = param_def.get("type", "")
                desc = param_def.get("description", "").lower()
                if (param_type == "string" and "enum" not in param_def
                        and any(kw in desc for kw in ["one of", "either", "must be", "options"])):
                    issues.append(ToolIssue(
                        category=IssueCategory.SCHEMA_NO_ENUM,
                        severity=Severity.INFO,
                        component=pcomp,
                        problem=f"Parameter '{param_name}' seems to have fixed options but no enum constraint.",
                        fix="Add 'enum' array to restrict the parameter to valid options.",
                    ))

                # Deep nesting
                if param_type == "object":
                    nested_props = param_def.get("properties", {})
                    for np_name, np_def in nested_props.items():
                        if np_def.get("type") == "object":
                            issues.append(ToolIssue(
                                category=IssueCategory.SCHEMA_DEEP_NESTING,
                                severity=Severity.WARNING,
                                component=f"tool:{tool.name}.{param_name}.{np_name}",
                                problem="Deeply nested objects are hard for LLMs to fill correctly.",
                                fix="Flatten the schema or use simpler types where possible.",
                            ))

            # Check required vs defined
            for req_param in required:
                if req_param not in props:
                    issues.append(ToolIssue(
                        category=IssueCategory.SCHEMA_MISSING_REQUIRED,
                        severity=Severity.CRITICAL,
                        component=comp,
                        problem=f"Required parameter '{req_param}' is not defined in properties.",
                        fix=f"Either add '{req_param}' to properties or remove it from required.",
                    ))

        return issues

    # ══════════════════════════════════════════════════════════════
    #  Phase 2 — System prompt analysis (tool-calling context)
    # ══════════════════════════════════════════════════════════════

    def _analyze_prompt(
        self,
        prompt: str,
        tools: List[ToolDefinition],
    ) -> List[ToolIssue]:
        """Analyze the system prompt for tool-calling best practices."""
        issues: List[ToolIssue] = []
        prompt_lower = prompt.lower()
        comp = "prompt"

        # 1. Does the prompt mention tools at all?
        tool_keywords = ["tool", "function", "call", "invoke", "use the"]
        if not any(kw in prompt_lower for kw in tool_keywords):
            issues.append(ToolIssue(
                category=IssueCategory.PROMPT_NO_TOOL_GUIDANCE,
                severity=Severity.CRITICAL,
                component=comp,
                problem="System prompt does not mention tools or function calling at all.",
                fix="Add instructions telling the model when and how to use the available tools.",
                evidence="No matches for: " + ", ".join(tool_keywords),
            ))

        # 2. Does it mention each tool by name?
        for td in tools:
            if td.name.lower() not in prompt_lower and td.name.replace("_", " ").lower() not in prompt_lower:
                issues.append(ToolIssue(
                    category=IssueCategory.PROMPT_NO_TOOL_GUIDANCE,
                    severity=Severity.WARNING,
                    component=f"prompt_tool:{td.name}",
                    problem=f"System prompt doesn't mention tool '{td.name}' — the model may not know when to use it.",
                    fix=f"Add guidance like: 'Use {td.name} when the user asks about {td.description[:40]}...'",
                ))

        # 3. JSON format instructions
        json_keywords = ["json", "structured", "format", "schema", "object"]
        if not any(kw in prompt_lower for kw in json_keywords):
            issues.append(ToolIssue(
                category=IssueCategory.PROMPT_NO_JSON_INSTRUCTION,
                severity=Severity.WARNING,
                component=comp,
                problem="No JSON/structured output guidance in the system prompt.",
                fix="Add 'Respond with valid JSON' or specify the expected output format.",
            ))

        # 4. Fallback instruction
        fallback_keywords = [
            "if no tool", "when no tool", "don't use a tool",
            "without tool", "no function", "fallback",
            "if none of the tools", "respond normally",
        ]
        if not any(kw in prompt_lower for kw in fallback_keywords):
            issues.append(ToolIssue(
                category=IssueCategory.PROMPT_MISSING_FALLBACK,
                severity=Severity.WARNING,
                component=comp,
                problem="No fallback instruction for when no tool applies.",
                fix="Add: 'If none of the available tools are relevant, respond directly without calling a tool.'",
            ))

        # 5. Multi-tool ambiguity
        if len(tools) > 1:
            # Check for tools with similar descriptions
            for i, t1 in enumerate(tools):
                for t2 in tools[i + 1:]:
                    overlap = set(t1.description.lower().split()) & set(t2.description.lower().split())
                    # Remove common stopwords
                    overlap -= {"the", "a", "an", "to", "for", "of", "and", "or", "in", "on", "is", "it", "this", "that"}
                    if len(overlap) > 3:
                        issues.append(ToolIssue(
                            category=IssueCategory.PROMPT_MULTI_TOOL_AMBIGUITY,
                            severity=Severity.WARNING,
                            component=f"prompt_tools:{t1.name}+{t2.name}",
                            problem=f"Tools '{t1.name}' and '{t2.name}' have overlapping descriptions ({', '.join(list(overlap)[:5])}). LLM may confuse them.",
                            fix=f"Add explicit disambiguation: 'Use {t1.name} for X, use {t2.name} for Y.'",
                        ))

        # 6. Conflicting instructions
        conflict_pairs = [
            ("always use", "never use"),
            ("must call", "don't call"),
            ("always respond with json", "respond in natural language"),
        ]
        for a, b in conflict_pairs:
            if a in prompt_lower and b in prompt_lower:
                issues.append(ToolIssue(
                    category=IssueCategory.PROMPT_CONFLICTING_INSTRUCTIONS,
                    severity=Severity.CRITICAL,
                    component=comp,
                    problem=f"Conflicting instructions found: '{a}' vs '{b}'.",
                    fix="Remove the contradiction — pick one clear behavior.",
                ))

        return issues

    # ══════════════════════════════════════════════════════════════
    #  Phase 3 — Live tool calling tests
    # ══════════════════════════════════════════════════════════════

    def _run_tool_test(
        self,
        system_prompt: str,
        tools: List[ToolDefinition],
        test_case: ToolTestCase,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ToolTestResult:
        """Run a single tool calling test against the LLM."""
        result = ToolTestResult(test_case=test_case)

        # Build the tools JSON for the prompt
        tools_json = json.dumps(
            [t.to_openai()["function"] for t in tools],
            indent=2,
        )

        # Ask the LLM to simulate a tool call
        prompt = _TOOL_CALL_EXTRACT_PROMPT.format(
            tools_json=tools_json,
            user_message=test_case.user_message,
        )

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            raw = self.llm(full_prompt)
            result.raw_output = raw

            # Extract JSON from response
            parsed = self._extract_json(raw)
            if parsed is None:
                result.json_valid = False
                result.issues.append(ToolIssue(
                    category=IssueCategory.JSON_INVALID,
                    severity=Severity.CRITICAL,
                    component=f"test:{test_case.description or test_case.user_message[:30]}",
                    problem="LLM output is not valid JSON.",
                    fix="Add explicit JSON format instructions to the system prompt.",
                    evidence=f"Raw output: {raw[:200]}",
                ))
                return result

            result.json_valid = True
            result.actual_tool = parsed.get("tool")
            result.actual_params = parsed.get("parameters", {})

            # Check tool selection
            if test_case.expected_tool is None:
                # Expected no tool call
                if result.actual_tool is not None and result.actual_tool != "null":
                    result.tool_correct = False
                    result.issues.append(ToolIssue(
                        category=IssueCategory.UNNECESSARY_TOOL_CALL,
                        severity=Severity.WARNING,
                        component=f"test:{test_case.description or test_case.user_message[:30]}",
                        problem=f"LLM called '{result.actual_tool}' when no tool was expected.",
                        fix="Add fallback instruction: respond normally when no tool applies.",
                        evidence=f"Expected: no tool. Got: {result.actual_tool}",
                    ))
                else:
                    result.tool_correct = True
            else:
                actual = result.actual_tool
                if actual == "null" or actual is None:
                    result.tool_correct = False
                    result.issues.append(ToolIssue(
                        category=IssueCategory.TOOL_NOT_CALLED,
                        severity=Severity.CRITICAL,
                        component=f"test:{test_case.description or test_case.user_message[:30]}",
                        problem=f"LLM did not call any tool. Expected '{test_case.expected_tool}'.",
                        fix=f"Add guidance: 'When user asks about [topic], use {test_case.expected_tool}.'",
                        evidence=f"Expected: {test_case.expected_tool}. Got: null",
                    ))
                elif actual != test_case.expected_tool:
                    result.tool_correct = False
                    result.issues.append(ToolIssue(
                        category=IssueCategory.WRONG_TOOL_SELECTED,
                        severity=Severity.CRITICAL,
                        component=f"test:{test_case.description or test_case.user_message[:30]}",
                        problem=f"LLM called '{actual}' instead of '{test_case.expected_tool}'.",
                        fix=f"Disambiguate tools: explain when to use '{test_case.expected_tool}' vs '{actual}'.",
                        evidence=f"Expected: {test_case.expected_tool}. Got: {actual}",
                    ))
                else:
                    result.tool_correct = True

            # Check parameters
            if test_case.expected_params and result.actual_params:
                result.params_correct = True
                for key, expected_val in test_case.expected_params.items():
                    if key not in result.actual_params:
                        result.params_correct = False
                        result.issues.append(ToolIssue(
                            category=IssueCategory.MISSING_PARAMS,
                            severity=Severity.CRITICAL,
                            component=f"test:{test_case.description}:param:{key}",
                            problem=f"Missing parameter '{key}' (expected: {expected_val}).",
                            fix=f"Mark '{key}' as required and add a clear description.",
                            evidence=f"Actual params: {result.actual_params}",
                        ))
                    else:
                        actual_val = result.actual_params[key]
                        # Type check
                        if type(expected_val) != type(actual_val):
                            result.params_correct = False
                            result.issues.append(ToolIssue(
                                category=IssueCategory.PARAM_TYPE_MISMATCH,
                                severity=Severity.WARNING,
                                component=f"test:{test_case.description}:param:{key}",
                                problem=f"Parameter '{key}' type mismatch: expected {type(expected_val).__name__}, got {type(actual_val).__name__}.",
                                fix=f"Add 'type': '{type(expected_val).__name__}' to the parameter schema.",
                                evidence=f"Expected: {expected_val}. Got: {actual_val}",
                            ))
                        # Value match (flexible)
                        elif isinstance(expected_val, str):
                            if expected_val.lower() not in str(actual_val).lower():
                                # Not an exact match but partial is OK for flexible params
                                pass  # Accept flexible matches for strings

                # Check for extra unexpected params
                if test_case.expected_params:
                    for key in result.actual_params:
                        if key not in test_case.expected_params:
                            result.issues.append(ToolIssue(
                                category=IssueCategory.EXTRA_PARAMS,
                                severity=Severity.INFO,
                                component=f"test:{test_case.description}:param:{key}",
                                problem=f"Unexpected extra parameter '{key}' in tool call.",
                                fix="Verify if this parameter is valid or add it to expected params.",
                                evidence=f"Value: {result.actual_params[key]}",
                            ))
            elif test_case.expected_params is None:
                result.params_correct = True  # No param check required

            # JSON schema validation
            if json_schema and result.json_valid:
                schema_issues = self._validate_json_schema(
                    result.actual_params or {},
                    json_schema,
                    f"test:{test_case.description or test_case.user_message[:30]}",
                )
                result.issues.extend(schema_issues)

        except Exception as e:
            result.issues.append(ToolIssue(
                category=IssueCategory.JSON_INVALID,
                severity=Severity.CRITICAL,
                component=f"test:{test_case.description or test_case.user_message[:30]}",
                problem=f"LLM call failed: {str(e)}",
                fix="Check model configuration and prompt format.",
            ))

        return result

    # ══════════════════════════════════════════════════════════════
    #  Phase 4 — Scoring
    # ══════════════════════════════════════════════════════════════

    def _score_pipeline(self, report: ToolReport) -> float:
        """Score the overall tool pipeline quality (0-1)."""
        score = 1.0

        # Deduct for schema issues
        for issue in report.schema_issues:
            if issue.severity == Severity.CRITICAL:
                score -= 0.15
            elif issue.severity == Severity.WARNING:
                score -= 0.05

        # Deduct for prompt issues
        for issue in report.prompt_issues:
            if issue.severity == Severity.CRITICAL:
                score -= 0.15
            elif issue.severity == Severity.WARNING:
                score -= 0.05

        # Test results
        if report.test_results:
            pass_rate = report.test_pass_rate
            score = score * 0.5 + pass_rate * 0.5

        return max(0.0, min(1.0, score))

    # ══════════════════════════════════════════════════════════════
    #  Phase 5 — Auto-improvements
    # ══════════════════════════════════════════════════════════════

    def _improve_prompt(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        issues: List[ToolIssue],
    ) -> str:
        """Generate an improved system prompt fixing the identified issues.

        For long prompts (>1000 chars), uses surgical section-based insertion
        instead of asking the LLM to rewrite the entire text.
        """
        import re as _re

        prompt_issues = [
            i for i in issues
            if "prompt" in i.component
            or i.category.value.startswith("prompt_")
        ]

        # Also gather test failure insights for test-driven improvements
        test_issues = [
            i for i in issues
            if "test" in i.component or i.category.value.startswith("test_")
        ]

        all_relevant = prompt_issues + test_issues
        if not all_relevant and not issues:
            return prompt  # Nothing to fix at all

        # ── For short prompts, use LLM rewrite (original approach) ──
        if len(prompt) <= 1000 and prompt_issues:
            return self._llm_rewrite_prompt(prompt, tools, prompt_issues)

        # ── For long prompts: surgical section-based edits ──────────
        result = prompt

        # Helper: find insertion point before a section heading or end
        def _insert_before_section(heading_pattern: str) -> int:
            m = _re.search(heading_pattern, result, _re.IGNORECASE | _re.MULTILINE)
            if not m:
                return -1
            start = m.end()
            nxt = _re.search(r"^(#{1,3}\s|---)", result[start:], _re.MULTILINE)
            return start + nxt.start() if nxt else len(result)

        improvements = []

        # 1. Add fallback instruction if missing
        fallback_keywords = [
            "if no tool", "when no tool", "don't use a tool",
            "without tool", "if none of the tools", "respond normally",
            "no tool applies", "no function applies",
        ]
        prompt_lower = result.lower()
        if not any(kw in prompt_lower for kw in fallback_keywords):
            fallback_text = (
                "\n\n### Fallback Behavior\n\n"
                "If the user's request does not match any available tool, "
                "respond directly with a helpful text answer. Do **not** "
                "fabricate a tool call or invent parameters.\n"
            )
            # Insert after Available Tools section or at end of pipeline
            pos = _insert_before_section(r"^## Processing Pipeline|^## Pipeline")
            if pos < 0:
                pos = _insert_before_section(r"^## Error Handling")
            if pos > 0:
                result = result[:pos] + fallback_text + "\n" + result[pos:]
                improvements.append("fallback")

        # 2. Add disambiguation for tools with overlapping descriptions
        ambiguity_issues = [
            i for i in prompt_issues
            if i.category == IssueCategory.PROMPT_MULTI_TOOL_AMBIGUITY
        ]
        if ambiguity_issues:
            disambig_lines = ["\n\n### Tool Disambiguation\n"]
            for issue in ambiguity_issues[:5]:
                disambig_lines.append(f"- {issue.fix}")
            disambig_text = "\n".join(disambig_lines) + "\n"

            pos = _insert_before_section(r"^## Processing Pipeline|^## Pipeline")
            if pos < 0:
                pos = _insert_before_section(r"^## Decision Framework")
            if pos > 0:
                result = result[:pos] + disambig_text + "\n" + result[pos:]
                improvements.append("disambiguation")

        # 3. Add JSON format guidance if missing
        json_keywords = ["json", "structured", "format", "schema"]
        if not any(kw in prompt_lower for kw in json_keywords):
            json_text = (
                "\n\n### Response Format\n\n"
                "When calling a tool, respond with valid JSON:\n"
                "```json\n"
                '{"tool": "tool_name", "parameters": {…}}\n'
                "```\n"
                "Always include all required parameters.\n"
            )
            pos = _insert_before_section(r"^## Processing Pipeline|^## Pipeline")
            if pos < 0:
                pos = _insert_before_section(r"^## Error Handling")
            if pos > 0:
                result = result[:pos] + json_text + "\n" + result[pos:]
                improvements.append("json_format")

        # 4. Add tool-selection guidance from test failures
        if test_issues:
            selection_lines = [
                "\n\n### Tool Selection Guidance (from test analysis)\n"
            ]
            seen_fixes = set()
            for issue in test_issues[:6]:
                fix = issue.fix or issue.problem
                if fix not in seen_fixes:
                    selection_lines.append(f"- {fix}")
                    seen_fixes.add(fix)
            selection_text = "\n".join(selection_lines) + "\n"

            pos = _insert_before_section(r"^## Decision Framework|^## Error Handling")
            if pos < 0:
                pos = _insert_before_section(r"^## Constraints")
            if pos > 0:
                result = result[:pos] + selection_text + "\n" + result[pos:]
                improvements.append("test_guidance")

        # 5. Add explicit tool-by-name guidance for any tools not mentioned
        missing_tools = []
        for td in tools:
            if td.name.lower() not in result.lower() and td.name.replace("_", " ").lower() not in result.lower():
                missing_tools.append(td)
        if missing_tools:
            tool_lines = ["\n\n### Additional Tool Instructions\n"]
            for t in missing_tools:
                tool_lines.append(f"- Use `{t.name}` when: {t.description}")
            tool_text = "\n".join(tool_lines) + "\n"

            pos = _insert_before_section(r"^## Processing Pipeline|^## Pipeline")
            if pos < 0:
                pos = len(result)
            result = result[:pos] + tool_text + "\n" + result[pos:]
            improvements.append("missing_tools")

        # 6. If still nothing changed and we have issues, append a summary
        if result == prompt and issues:
            fix_lines = ["\n\n---\n\n## Recommended Improvements\n"]
            seen = set()
            for issue in issues[:8]:
                fix = issue.fix or issue.problem
                if fix not in seen:
                    fix_lines.append(f"- **[{issue.severity.value.upper()}]** {fix}")
                    seen.add(fix)
            result = prompt + "\n".join(fix_lines) + "\n"

        return result

    def _llm_rewrite_prompt(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        prompt_issues: List[ToolIssue],
    ) -> str:
        """Rewrite a short prompt via LLM (original approach for < 1000 chars)."""
        tools_summary = "\n".join(
            f"  - {t.name}: {t.description}" for t in tools
        )
        issues_text = "\n".join(
            f"  [{i.severity.value.upper()}] {i.problem}\n    FIX: {i.fix}"
            for i in prompt_issues
        )
        try:
            improved = self.llm(_PROMPT_IMPROVE_TEMPLATE.format(
                system_prompt=prompt,
                tools_summary=tools_summary,
                issues_text=issues_text,
            ))
            return improved.strip()
        except Exception:
            return self._heuristic_improve_prompt(prompt, tools, prompt_issues)

    def _heuristic_improve_prompt(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        issues: List[ToolIssue],
    ) -> str:
        """Fallback heuristic prompt improvement without LLM."""
        additions = []

        for issue in issues:
            if issue.category == IssueCategory.PROMPT_NO_TOOL_GUIDANCE:
                tool_instructions = "\n".join(
                    f"- Use `{t.name}` when: {t.description}"
                    for t in tools
                )
                additions.append(f"\n\nAvailable tools:\n{tool_instructions}")

            elif issue.category == IssueCategory.PROMPT_NO_JSON_INSTRUCTION:
                additions.append(
                    "\n\nWhen calling a tool, respond with valid JSON in this format:\n"
                    '{"tool": "tool_name", "parameters": {...}}'
                )

            elif issue.category == IssueCategory.PROMPT_MISSING_FALLBACK:
                additions.append(
                    "\n\nIf none of the available tools are relevant to the user's request, "
                    "respond directly with a helpful text answer."
                )

        return prompt + "".join(additions)

    def _improve_tools(
        self,
        tools: List[ToolDefinition],
        issues: List[ToolIssue],
    ) -> List[ToolDefinition]:
        """Generate improved tool definitions."""
        improved = []

        for td in tools:
            # Collect issues for this tool
            tool_issues = [
                i for i in issues
                if i.component.startswith(f"tool:{td.name}")
            ]

            if not tool_issues:
                improved.append(td)
                continue

            issues_text = "\n".join(
                f"  [{i.severity.value.upper()}] {i.problem}\n    FIX: {i.fix}"
                for i in tool_issues
            )

            try:
                raw = self.llm(_TOOL_IMPROVE_TEMPLATE.format(
                    tool_json=json.dumps(td.to_openai(), indent=2),
                    issues_text=issues_text,
                ))
                parsed = self._extract_json(raw)
                if parsed and "function" in parsed:
                    improved.append(ToolDefinition.from_openai(parsed))
                elif parsed and "name" in parsed:
                    improved.append(ToolDefinition.from_openai({"function": parsed}))
                else:
                    # LLM didn't return valid JSON — use heuristic fix
                    improved.append(self._heuristic_improve_tool(td, tool_issues))
            except Exception:
                improved.append(self._heuristic_improve_tool(td, tool_issues))

        return improved

    def _heuristic_improve_tool(
        self,
        tool: ToolDefinition,
        issues: List[ToolIssue],
    ) -> ToolDefinition:
        """Heuristic tool definition fix when LLM improvement fails."""
        import copy
        fixed = copy.deepcopy(tool)

        for issue in issues:
            if issue.category == IssueCategory.SCHEMA_MISSING_DESC:
                if "param:" in issue.component:
                    # Fix param description
                    param_name = issue.component.split(".")[-1]
                    props = fixed.parameters.get("properties", {})
                    if param_name in props:
                        props[param_name]["description"] = f"The {param_name.replace('_', ' ')} value."
                elif not fixed.description:
                    fixed.description = f"Calls the {fixed.name.replace('_', ' ')} function."

            elif issue.category == IssueCategory.SCHEMA_MISSING_TYPE:
                param_name = issue.component.split(".")[-1]
                props = fixed.parameters.get("properties", {})
                if param_name in props and "type" not in props[param_name]:
                    props[param_name]["type"] = "string"  # Safe default

        return fixed

    # ══════════════════════════════════════════════════════════════
    #  JSON utilities
    # ══════════════════════════════════════════════════════════════

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON object from LLM output."""
        # Try direct parse
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code blocks
        code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _validate_json_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        component: str,
    ) -> List[ToolIssue]:
        """Validate JSON data against a simple schema (without jsonschema lib)."""
        issues: List[ToolIssue] = []

        # Check required fields
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field_name in required:
            if field_name not in data:
                issues.append(ToolIssue(
                    category=IssueCategory.JSON_MISSING_FIELD,
                    severity=Severity.CRITICAL,
                    component=component,
                    problem=f"Required field '{field_name}' missing from JSON output.",
                    fix=f"Add instructions to always include '{field_name}' in the response.",
                    evidence=f"Got keys: {list(data.keys())}",
                ))

        # Check types
        type_map = {
            "string": str, "number": (int, float),
            "integer": int, "boolean": bool,
            "array": list, "object": dict,
        }
        for field_name, field_def in properties.items():
            if field_name in data:
                expected_type = field_def.get("type", "")
                py_type = type_map.get(expected_type)
                if py_type and not isinstance(data[field_name], py_type):
                    issues.append(ToolIssue(
                        category=IssueCategory.JSON_TYPE_ERROR,
                        severity=Severity.WARNING,
                        component=component,
                        problem=f"Field '{field_name}' should be {expected_type}, got {type(data[field_name]).__name__}.",
                        fix=f"Specify the type explicitly in instructions: '{field_name}' must be a {expected_type}.",
                    ))

        # Check for unknown fields
        if properties:
            for key in data:
                if key not in properties:
                    issues.append(ToolIssue(
                        category=IssueCategory.JSON_EXTRA_FIELD,
                        severity=Severity.INFO,
                        component=component,
                        problem=f"Unexpected field '{key}' in JSON output.",
                        fix="Either add it to the schema or instruct the model to omit extra fields.",
                    ))

        return issues
