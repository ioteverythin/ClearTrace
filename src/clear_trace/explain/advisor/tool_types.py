"""Data types for tool / agentic pipeline analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Issue severity ───────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "critical"   # Will likely cause tool call failures
    WARNING = "warning"     # May cause incorrect behavior
    INFO = "info"           # Suggestion for improvement


class IssueCategory(str, Enum):
    # Tool definition issues
    SCHEMA_MISSING_DESC = "schema_missing_description"
    SCHEMA_MISSING_TYPE = "schema_missing_type"
    SCHEMA_MISSING_REQUIRED = "schema_missing_required"
    SCHEMA_AMBIGUOUS_NAME = "schema_ambiguous_name"
    SCHEMA_NO_ENUM = "schema_no_enum"         # Should use enum but doesn't
    SCHEMA_MISSING_EXAMPLE = "schema_missing_example"
    SCHEMA_DEEP_NESTING = "schema_deep_nesting"

    # Tool calling issues
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    TOOL_NOT_CALLED = "tool_not_called"
    UNNECESSARY_TOOL_CALL = "unnecessary_tool_call"
    WRONG_PARAMS = "wrong_params"
    MISSING_PARAMS = "missing_params"
    EXTRA_PARAMS = "extra_params"
    PARAM_TYPE_MISMATCH = "param_type_mismatch"

    # JSON issues
    JSON_INVALID = "json_invalid"
    JSON_SCHEMA_MISMATCH = "json_schema_mismatch"
    JSON_MISSING_FIELD = "json_missing_field"
    JSON_EXTRA_FIELD = "json_extra_field"
    JSON_TYPE_ERROR = "json_type_error"
    JSON_TRUNCATED = "json_truncated"

    # Prompt issues (tool-calling specific)
    PROMPT_NO_TOOL_GUIDANCE = "prompt_no_tool_guidance"
    PROMPT_CONFLICTING_INSTRUCTIONS = "prompt_conflicting_instructions"
    PROMPT_MISSING_FALLBACK = "prompt_missing_fallback"
    PROMPT_NO_JSON_INSTRUCTION = "prompt_no_json_instruction"
    PROMPT_MULTI_TOOL_AMBIGUITY = "prompt_multi_tool_ambiguity"


# ── Individual finding ───────────────────────────────────────────

@dataclass
class ToolIssue:
    """A single issue found in the tool pipeline.

    Attributes:
        category: Type of issue.
        severity: CRITICAL / WARNING / INFO.
        component: What it affects ('tool:get_weather', 'param:location', 'json_output', 'prompt').
        problem: Human-readable description of the issue.
        fix: Recommended corrective action.
        evidence: Data supporting the finding.
    """
    category: IssueCategory
    severity: Severity
    component: str
    problem: str
    fix: str
    evidence: str = ""


# ── Tool definition types ───────────────────────────────────────

@dataclass
class ToolDefinition:
    """A tool/function definition as provided to the LLM.

    Mirrors the OpenAI / Anthropic function-calling schema.
    """
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    # parameters follows JSON Schema: {type: "object", properties: {...}, required: [...]}
    required: List[str] = field(default_factory=list)

    @classmethod
    def from_openai(cls, tool_dict: Dict[str, Any]) -> "ToolDefinition":
        """Parse from OpenAI tools format: {type:'function', function:{name,description,parameters}}."""
        fn = tool_dict.get("function", tool_dict)
        params = fn.get("parameters", {})
        return cls(
            name=fn.get("name", ""),
            description=fn.get("description", ""),
            parameters=params,
            required=params.get("required", []),
        )

    def to_openai(self) -> Dict[str, Any]:
        """Export to OpenAI tools list format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ── Test case for tool calling ──────────────────────────────────

@dataclass
class ToolTestCase:
    """A test scenario for tool calling.

    Attributes:
        user_message: What the user says.
        expected_tool: Which tool should be called (None = no tool expected).
        expected_params: Expected parameter values.
        description: Human-readable test description.
    """
    user_message: str
    expected_tool: Optional[str] = None
    expected_params: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass
class ToolTestResult:
    """Result of running a single tool test case."""
    test_case: ToolTestCase
    actual_tool: Optional[str] = None
    actual_params: Optional[Dict[str, Any]] = None
    raw_output: str = ""
    tool_correct: bool = False
    params_correct: bool = False
    json_valid: bool = False
    issues: List[ToolIssue] = field(default_factory=list)


# ── Full report ─────────────────────────────────────────────────

@dataclass
class ToolReport:
    """Full tool pipeline assessment report.

    Attributes:
        system_prompt: The system prompt being analyzed.
        tools: The tool definitions being analyzed.
        schema_issues: Issues found in tool definitions.
        prompt_issues: Issues found in the system prompt (tool-calling context).
        test_results: Results of running test cases.
        improved_prompt: Improved system prompt.
        improved_tools: Improved tool definitions.
        score_before: Overall pipeline quality score (0-1) before.
        score_after: Score after improvements.
        suggestions: Ordered list of all fixes.
    """
    system_prompt: str
    tools: List[ToolDefinition] = field(default_factory=list)
    schema_issues: List[ToolIssue] = field(default_factory=list)
    prompt_issues: List[ToolIssue] = field(default_factory=list)
    test_results: List[ToolTestResult] = field(default_factory=list)
    json_issues: List[ToolIssue] = field(default_factory=list)
    improved_prompt: str = ""
    improved_tools: List[ToolDefinition] = field(default_factory=list)
    score_before: float = 0.0
    score_after: float = 0.0
    suggestions: List[ToolIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_issues(self) -> List[ToolIssue]:
        return self.schema_issues + self.prompt_issues + self.json_issues

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == Severity.CRITICAL)

    @property
    def test_pass_rate(self) -> float:
        if not self.test_results:
            return 0.0
        passed = sum(1 for t in self.test_results if t.tool_correct and t.params_correct)
        return passed / len(self.test_results)
