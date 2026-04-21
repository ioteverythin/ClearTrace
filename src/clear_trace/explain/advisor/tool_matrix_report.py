"""SE matrix reports for tool pipeline analysis.

Generates 5 assessment matrices specific to tool calling:
  - T1: Tool Schema Quality Matrix (TSQM)
  - T2: Prompt-Tool Alignment Matrix (PTAM)
  - T3: Tool Call Accuracy Matrix (TCAM)
  - T4: JSON Compliance Matrix (JCM)
  - T5: Corrective Action Register (CAR)

Each matrix is available as Rich table, CSV, dict, and matplotlib plot.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clear_trace.explain.advisor.tool_types import (
    Severity,
    ToolDefinition,
    ToolIssue,
    ToolReport,
    ToolTestResult,
)
from clear_trace.explain.advisor.matrix_report import Matrix, MatrixRow


def _sev_icon(sev: str) -> str:
    return {
        "CRITICAL": "[red bold]CRITICAL[/red bold]",
        "WARNING": "[yellow]WARNING[/yellow]",
        "INFO": "[dim]INFO[/dim]",
    }.get(sev, sev)


def _pass_icon(passed: bool) -> str:
    return "[green]\u2713 PASS[/green]" if passed else "[red]\u2717 FAIL[/red]"


class ToolMatrixReport:
    """Generate SE-style matrices from ToolAdvisor analysis.

    Args:
        report: The ToolReport from ToolAdvisor.analyze().
    """

    def __init__(self, report: ToolReport):
        self.report = report

        self.t1 = self._build_tsqm()
        self.t2 = self._build_ptam()
        self.t3 = self._build_tcam()
        self.t4 = self._build_jcm()
        self.t5 = self._build_car()

        self.matrices = [self.t1, self.t2, self.t3, self.t4, self.t5]

    # ── T1: Tool Schema Quality Matrix ───────────────────────────

    def _build_tsqm(self) -> Matrix:
        """Rows = tool parameters, Cols = quality checks."""
        m = Matrix(
            id="T1",
            title="Tool Schema Quality Matrix (TSQM)",
            description="Assesses each tool and parameter against schema best practices.",
            headers=[
                "ID", "Tool", "Parameter", "Has Type", "Has Desc",
                "Has Enum", "Nesting", "Issues", "Status",
            ],
        )

        idx = 0
        for td in self.report.tools:
            props = td.parameters.get("properties", {})
            required = td.parameters.get("required", td.required or [])

            if not props:
                # Tool-level row (no params)
                tool_issues = [
                    i for i in self.report.schema_issues
                    if i.component == f"tool:{td.name}"
                ]
                m.rows.append(MatrixRow(cells={
                    "ID": f"T{idx}",
                    "Tool": td.name,
                    "Parameter": "(no params)",
                    "Has Type": "—",
                    "Has Desc": "YES" if td.description else "NO",
                    "Has Enum": "—",
                    "Nesting": "—",
                    "Issues": str(len(tool_issues)),
                    "Status": "FAIL" if any(i.severity == Severity.CRITICAL for i in tool_issues) else "PASS",
                }))
                idx += 1
            else:
                for param_name, param_def in props.items():
                    has_type = "YES" if "type" in param_def else "NO"
                    has_desc = "YES" if param_def.get("description") else "NO"
                    has_enum = "YES" if "enum" in param_def else "—"
                    is_nested = "YES" if param_def.get("type") == "object" else "—"
                    is_required = param_name in required

                    param_issues = [
                        i for i in self.report.schema_issues
                        if f"tool:{td.name}.{param_name}" in i.component
                    ]

                    status = "PASS"
                    if any(i.severity == Severity.CRITICAL for i in param_issues):
                        status = "FAIL"
                    elif any(i.severity == Severity.WARNING for i in param_issues):
                        status = "WARN"

                    m.rows.append(MatrixRow(cells={
                        "ID": f"T{idx}",
                        "Tool": td.name,
                        "Parameter": f"{'*' if is_required else ''}{param_name}",
                        "Has Type": has_type,
                        "Has Desc": has_desc,
                        "Has Enum": has_enum,
                        "Nesting": is_nested,
                        "Issues": str(len(param_issues)),
                        "Status": status,
                    }))
                    idx += 1

        return m

    # ── T2: Prompt-Tool Alignment Matrix ─────────────────────────

    def _build_ptam(self) -> Matrix:
        """Rows = tools, Cols = prompt alignment checks."""
        m = Matrix(
            id="T2",
            title="Prompt-Tool Alignment Matrix (PTAM)",
            description="Checks if the system prompt properly guides tool selection.",
            headers=[
                "ID", "Tool", "Mentioned in Prompt", "Usage Guidance",
                "JSON Format", "Fallback", "Disambiguation", "Status",
            ],
        )

        prompt_lower = self.report.system_prompt.lower()
        prompt_issues = self.report.prompt_issues

        for i, td in enumerate(self.report.tools):
            mentioned = td.name.lower() in prompt_lower or td.name.replace("_", " ").lower() in prompt_lower

            # Check for usage guidance (e.g., "use X when...")
            use_pattern = f"use {td.name.lower()}" in prompt_lower or f"use the {td.name.replace('_', ' ').lower()}" in prompt_lower
            has_usage = "YES" if use_pattern else ("PARTIAL" if mentioned else "NO")

            # JSON format
            has_json = "YES" if any(kw in prompt_lower for kw in ["json", "structured", "format"]) else "NO"

            # Fallback
            has_fallback = "YES" if any(kw in prompt_lower for kw in [
                "if no tool", "fallback", "respond normally", "none of the tools"
            ]) else "NO"

            # Disambiguation (relevant if >1 tool)
            has_disambig = "—"
            if len(self.report.tools) > 1:
                disamb_issues = [
                    pi for pi in prompt_issues
                    if pi.category.value == "prompt_multi_tool_ambiguity"
                    and td.name in pi.component
                ]
                has_disambig = "NO" if disamb_issues else "YES"

            # Overall status
            checks = [mentioned, has_usage != "NO", has_json == "YES", has_fallback == "YES"]
            pass_count = sum(checks)
            status = "PASS" if pass_count >= 3 else "WARN" if pass_count >= 2 else "FAIL"

            m.rows.append(MatrixRow(cells={
                "ID": f"P{i}",
                "Tool": td.name,
                "Mentioned in Prompt": "YES" if mentioned else "NO",
                "Usage Guidance": has_usage,
                "JSON Format": has_json,
                "Fallback": has_fallback,
                "Disambiguation": has_disambig,
                "Status": status,
            }))

        return m

    # ── T3: Tool Call Accuracy Matrix ────────────────────────────

    def _build_tcam(self) -> Matrix:
        """Rows = test cases, Cols = accuracy metrics."""
        m = Matrix(
            id="T3",
            title="Tool Call Accuracy Matrix (TCAM)",
            description="Results of testing tool selection and parameter extraction against the LLM.",
            headers=[
                "ID", "Test Scenario", "Expected Tool",
                "Actual Tool", "Tool Correct", "Params Correct",
                "JSON Valid", "Issues", "Status",
            ],
        )

        for i, tr in enumerate(self.report.test_results):
            tc = tr.test_case
            status = "PASS" if (tr.tool_correct and tr.params_correct and tr.json_valid) else "FAIL"

            m.rows.append(MatrixRow(cells={
                "ID": f"TC{i}",
                "Test Scenario": tc.description or tc.user_message[:40],
                "Expected Tool": tc.expected_tool or "(none)",
                "Actual Tool": str(tr.actual_tool) if tr.actual_tool else "(none)",
                "Tool Correct": "YES" if tr.tool_correct else "NO",
                "Params Correct": "YES" if tr.params_correct else "NO",
                "JSON Valid": "YES" if tr.json_valid else "NO",
                "Issues": str(len(tr.issues)),
                "Status": status,
            }))

        return m

    # ── T4: JSON Compliance Matrix ───────────────────────────────

    def _build_jcm(self) -> Matrix:
        """Rows = JSON issues, Cols = details."""
        m = Matrix(
            id="T4",
            title="JSON Compliance Matrix (JCM)",
            description="JSON validity and schema compliance across all test outputs.",
            headers=[
                "ID", "Source", "Issue Type", "Severity",
                "Problem", "Fix",
            ],
        )

        json_issues = [
            i for i in (self.report.json_issues + self.report.schema_issues)
            if i.category.value.startswith("json_") or i.category.value.startswith("param_type")
        ]

        for i, issue in enumerate(json_issues):
            m.rows.append(MatrixRow(cells={
                "ID": f"J{i}",
                "Source": issue.component[:35],
                "Issue Type": issue.category.value.replace("json_", "").upper(),
                "Severity": issue.severity.value.upper(),
                "Problem": issue.problem[:60],
                "Fix": issue.fix[:60],
            }))

        return m

    # ── T5: Corrective Action Register ───────────────────────────

    def _build_car(self) -> Matrix:
        """All issues ranked by severity."""
        m = Matrix(
            id="T5",
            title="Tool Pipeline Corrective Action Register (CAR)",
            description="All findings prioritized by severity with fixes and traceability.",
            headers=[
                "CA-ID", "Component", "Category", "Severity",
                "Finding", "Corrective Action", "Evidence",
            ],
        )

        for i, issue in enumerate(self.report.suggestions):
            m.rows.append(MatrixRow(cells={
                "CA-ID": f"TCA{i+1:02d}",
                "Component": issue.component[:30],
                "Category": issue.category.value[:25],
                "Severity": issue.severity.value.upper(),
                "Finding": issue.problem[:55],
                "Corrective Action": issue.fix[:55],
                "Evidence": issue.evidence[:40] if issue.evidence else "—",
            }))

        return m

    # ── Output methods ───────────────────────────────────────────

    def print_matrices(self) -> None:
        """Print all matrices as Rich tables."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console()

            # Header
            r = self.report
            console.print(Panel(
                "[bold]Prism Tool Pipeline Assessment[/bold]\n"
                f"Tools: {len(r.tools)}  |  Tests: {len(r.test_results)}  |  "
                f"Issues: {len(r.all_issues)}\n"
                f"Score: {r.score_before:.0%} → {r.score_after:.0%}",
                title="\U0001f527 Tool Assessment",
                border_style="blue",
            ))

            for mx in self.matrices:
                if not mx.rows:
                    continue

                console.print()
                table = Table(
                    title=f"{mx.id}: {mx.title}",
                    caption=mx.description,
                    show_lines=True,
                    title_style="bold blue",
                    caption_style="dim italic",
                )

                for h in mx.headers:
                    table.add_column(h)

                for row in mx.rows:
                    cells = []
                    for h in mx.headers:
                        val = row.cells.get(h, "")
                        # Color status cells
                        if h in ("Status", "Severity"):
                            if val in ("PASS", "YES"):
                                val = f"[green]{val}[/green]"
                            elif val == "WARN":
                                val = f"[yellow]{val}[/yellow]"
                            elif val in ("FAIL", "NO", "CRITICAL"):
                                val = f"[red bold]{val}[/red bold]"
                            elif val == "WARNING":
                                val = f"[yellow]{val}[/yellow]"
                            elif val == "INFO":
                                val = f"[dim]{val}[/dim]"
                        elif h in ("Tool Correct", "Params Correct", "JSON Valid"):
                            if val == "YES":
                                val = "[green]YES[/green]"
                            elif val == "NO":
                                val = "[red]NO[/red]"
                        cells.append(val)
                    table.add_row(*cells)

                console.print(table)

            # Summary
            crit = sum(1 for i in r.all_issues if i.severity == Severity.CRITICAL)
            warn = sum(1 for i in r.all_issues if i.severity == Severity.WARNING)
            console.print(Panel(
                f"[bold]Pipeline Summary[/bold]\n"
                f"  Critical Issues:  [red]{crit}[/red]\n"
                f"  Warnings:         [yellow]{warn}[/yellow]\n"
                f"  Test Pass Rate:   {r.test_pass_rate:.0%}\n"
                f"  Score:            {r.score_before:.0%} → {r.score_after:.0%}",
                title="Summary",
                border_style="green",
            ))

        except ImportError:
            print(self.to_text())

    def to_text(self) -> str:
        """Plain text rendering."""
        lines = [
            "=" * 80,
            "PRISM TOOL PIPELINE ASSESSMENT",
            "=" * 80,
        ]
        for mx in self.matrices:
            if not mx.rows:
                continue
            lines.append(f"\n{'─' * 80}")
            lines.append(f"{mx.id}: {mx.title}")
            lines.append(f"{'─' * 80}")

            widths = {h: len(h) for h in mx.headers}
            for row in mx.rows:
                for h in mx.headers:
                    widths[h] = max(widths[h], len(row.cells.get(h, "")))

            header_line = " | ".join(h.ljust(widths[h]) for h in mx.headers)
            lines.append(header_line)
            lines.append("-+-".join("-" * widths[h] for h in mx.headers))
            for row in mx.rows:
                lines.append(" | ".join(row.cells.get(h, "").ljust(widths[h]) for h in mx.headers))

        return "\n".join(lines)

    def to_csv(self, output_dir: str = ".") -> List[str]:
        """Export all matrices as CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        names = ["tsqm", "ptam", "tcam", "jcm", "car"]
        for mx, name in zip(self.matrices, names):
            if not mx.rows:
                continue
            filepath = os.path.join(output_dir, f"prism_tool_{mx.id.lower()}_{name}.csv")
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=mx.headers)
                writer.writeheader()
                for row in mx.rows:
                    writer.writerow({h: row.cells.get(h, "") for h in mx.headers})
            paths.append(filepath)
        return paths

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {mx.id: mx.to_list_of_dicts() for mx in self.matrices}
