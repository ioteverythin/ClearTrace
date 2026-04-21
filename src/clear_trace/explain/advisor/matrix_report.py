"""Systems Engineering matrix reports for Prism Prompt Advisor.

Generates structured assessment matrices in the style of SE artifacts:
  - M1: Prompt Component Assessment Matrix (PCAM)
  - M2: Sensitivity & Robustness Matrix (SRM)
  - M3: Requirements Gap Analysis Matrix (RGAM)
  - M4: Corrective Action Register (CAR)
  - M5: Before/After Verification Matrix (BAVM)

Each matrix is available as:
  - Rich console table  (print_matrices())
  - CSV string          (to_csv())
  - Dict of lists       (to_dict())

Usage:
    >>> from clear_trace import PromptAdvisor
    >>> from clear_trace.explain.advisor.matrix_report import MatrixReport
    >>> report = advisor.analyze(prompt, desired="...")
    >>> mx = MatrixReport(report, lime_result, cf_result, concepts)
    >>> mx.print_matrices()    # Rich tables to terminal
    >>> mx.to_csv("output/")   # Export all matrices as CSV
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from clear_trace.explain.core.types import (
    ConceptAttribution,
    Explanation,
    SentenceImportance,
)
from clear_trace.explain.advisor.suggestions import (
    ImpactLevel,
    PromptReport,
    Suggestion,
    SuggestionType,
)


# ── Matrix cell helpers ──────────────────────────────────────────

def _rating(score: float) -> str:
    """Convert a 0-1 score to a 5-level SE rating."""
    if score >= 0.8:
        return "PASS"
    elif score >= 0.6:
        return "ACCEPTABLE"
    elif score >= 0.3:
        return "MARGINAL"
    elif score >= 0.1:
        return "WEAK"
    else:
        return "FAIL"


def _risk(impact: str, likelihood: float) -> str:
    """Simple risk = impact × likelihood classification."""
    impact_val = {"high": 3, "medium": 2, "low": 1}.get(impact, 1)
    risk_score = impact_val * likelihood
    if risk_score >= 2.0:
        return "HIGH"
    elif risk_score >= 1.0:
        return "MEDIUM"
    else:
        return "LOW"


def _status_icon(status: str) -> str:
    icons = {
        "PASS": "[green]\u2713[/green]",
        "ACCEPTABLE": "[green]\u2713[/green]",
        "MARGINAL": "[yellow]~[/yellow]",
        "WEAK": "[red]\u2717[/red]",
        "FAIL": "[red]\u2717[/red]",
        "COVERED": "[green]\u2713[/green]",
        "PARTIAL": "[yellow]~[/yellow]",
        "GAP": "[red]\u2717[/red]",
        "HIGH": "[red]HIGH[/red]",
        "MEDIUM": "[yellow]MED[/yellow]",
        "LOW": "[dim]LOW[/dim]",
    }
    return icons.get(status, status)


def _plain_icon(status: str) -> str:
    icons = {
        "PASS": "[OK]",
        "ACCEPTABLE": "[OK]",
        "MARGINAL": "[~]",
        "WEAK": "[X]",
        "FAIL": "[X]",
        "COVERED": "[OK]",
        "PARTIAL": "[~]",
        "GAP": "[X]",
    }
    return icons.get(status, status)


# ── Matrix data structures ───────────────────────────────────────

@dataclass
class MatrixRow:
    """A single row in a matrix."""
    cells: Dict[str, str] = field(default_factory=dict)


@dataclass
class Matrix:
    """A named matrix with headers and rows."""
    id: str          # e.g. "M1", "M2"
    title: str
    description: str
    headers: List[str]
    rows: List[MatrixRow] = field(default_factory=list)

    def to_list_of_dicts(self) -> List[Dict[str, str]]:
        return [row.cells for row in self.rows]

    def to_csv_string(self) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.headers)
        writer.writeheader()
        for row in self.rows:
            writer.writerow({h: row.cells.get(h, "") for h in self.headers})
        return buf.getvalue()


class MatrixReport:
    """Generate SE-style assessment matrices from Prism analysis results.

    Args:
        report: The PromptReport from PromptAdvisor.analyze().
        lime_result: LIME Explanation (sentence importances).
        cf_result: Counterfactual Explanation.
        concepts: List of detected ConceptAttributions.
    """

    def __init__(
        self,
        report: PromptReport,
        lime_result: Optional[Explanation] = None,
        cf_result: Optional[Explanation] = None,
        concepts: Optional[List[ConceptAttribution]] = None,
    ):
        self.report = report
        self.lime = lime_result
        self.cf = cf_result
        self.concepts = concepts or []

        # Build all matrices
        self.m1 = self._build_pcam()
        self.m2 = self._build_srm()
        self.m3 = self._build_rgam()
        self.m4 = self._build_car()
        self.m5 = self._build_bavm()

        self.matrices = [self.m1, self.m2, self.m3, self.m4, self.m5]

    # ── M1: Prompt Component Assessment Matrix ───────────────────

    def _build_pcam(self) -> Matrix:
        m = Matrix(
            id="M1",
            title="Prompt Component Assessment Matrix (PCAM)",
            description="Evaluates each sentence of the prompt against SE quality criteria.",
            headers=[
                "ID", "Component (Sentence)", "Importance",
                "Score", "Effectiveness", "Fragility",
                "Concept Link", "Assessment",
            ],
        )

        if not self.lime:
            return m

        sentences = sorted(
            self.lime.sentence_importances,
            key=lambda s: s.index,
        )

        for s in sentences:
            # Determine fragility from counterfactuals
            fragility = "ROBUST"
            if self.cf:
                for cf in self.cf.counterfactuals:
                    if s.text[:20] in cf.change_description:
                        if cf.is_flip:
                            fragility = "FRAGILE"
                        elif cf.semantic_distance > 0.3:
                            fragility = "SENSITIVE"
                        break

            # Determine linked concepts
            concept_links = []
            text_lower = s.text.lower()
            for c in self.concepts:
                for tok in c.evidence_tokens:
                    if tok.lower() in text_lower:
                        concept_links.append(c.concept)
                        break

            m.rows.append(MatrixRow(cells={
                "ID": f"S{s.index}",
                "Component (Sentence)": s.text[:60],
                "Importance": s.level.value.upper(),
                "Score": f"{s.score:+.3f}",
                "Effectiveness": _rating(abs(s.score)),
                "Fragility": fragility,
                "Concept Link": ", ".join(concept_links) or "—",
                "Assessment": _rating(abs(s.score)),
            }))

        return m

    # ── M2: Sensitivity & Robustness Matrix ──────────────────────

    def _build_srm(self) -> Matrix:
        m = Matrix(
            id="M2",
            title="Sensitivity & Robustness Matrix (SRM)",
            description="Maps each perturbation to its output impact — identifies what breaks and what holds.",
            headers=[
                "ID", "Perturbation", "Type",
                "Edit Distance", "Semantic Distance",
                "Output Status", "Risk Level",
            ],
        )

        if not self.cf:
            return m

        for i, cf in enumerate(self.cf.counterfactuals):
            # Classify perturbation type
            if "Removed sentence" in cf.change_description:
                ptype = "ABLATION"
            elif "Flipped" in cf.change_description:
                ptype = "NEGATION"
            elif "Redacted" in cf.change_description:
                ptype = "REDACTION"
            else:
                ptype = "SUBSTITUTION"

            status = "FLIP" if cf.is_flip else "STABLE"
            risk = _risk(
                "high" if cf.is_flip else "medium" if cf.semantic_distance > 0.3 else "low",
                cf.semantic_distance,
            )

            m.rows.append(MatrixRow(cells={
                "ID": f"P{i}",
                "Perturbation": cf.change_description[:55],
                "Type": ptype,
                "Edit Distance": str(cf.edit_distance),
                "Semantic Distance": f"{cf.semantic_distance:.3f}",
                "Output Status": status,
                "Risk Level": risk,
            }))

        return m

    # ── M3: Requirements Gap Analysis Matrix ─────────────────────

    def _build_rgam(self) -> Matrix:
        m = Matrix(
            id="M3",
            title="Requirements Gap Analysis Matrix (RGAM)",
            description="Maps desired output requirements to prompt coverage — identifies gaps.",
            headers=[
                "REQ-ID", "Requirement", "Category",
                "Prompt Coverage", "Status",
                "Corrective Action",
            ],
        )

        # Build requirements from desired description + SE best practices
        requirements = self._derive_requirements()
        prompt_lower = self.report.original_prompt.lower()

        for i, (req_name, category, markers, action) in enumerate(requirements):
            covered = any(m in prompt_lower for m in markers)

            # Partial: the concept exists but is weak
            partial = False
            if not covered and self.lime:
                for s in self.lime.sentence_importances:
                    if any(m in s.text.lower() for m in markers) and abs(s.score) < 0.3:
                        partial = True
                        break

            if covered:
                status = "COVERED"
                cov_text = "Addressed"
            elif partial:
                status = "PARTIAL"
                cov_text = "Weak/Partial"
            else:
                status = "GAP"
                cov_text = "Missing"

            m.rows.append(MatrixRow(cells={
                "REQ-ID": f"R{i+1:02d}",
                "Requirement": req_name,
                "Category": category,
                "Prompt Coverage": cov_text,
                "Status": status,
                "Corrective Action": action if status != "COVERED" else "—",
            }))

        return m

    def _derive_requirements(self) -> List[Tuple[str, str, List[str], str]]:
        """Derive implicit requirements from desired output + best practices.

        Returns list of (requirement_name, category, detection_markers, corrective_action).
        """
        reqs = []
        desired = self.report.desired_output_description.lower() if self.report.desired_output_description else ""

        # Universal SE requirements for any prompt
        reqs.append((
            "Task specificity",
            "CLARITY",
            ["specifically", "exactly", "must", "precisely", "only",
             "ensure", "required", "always", "never"],
            "Add explicit constraints (must, exactly, specifically)",
        ))
        reqs.append((
            "Output format specification",
            "STRUCTURE",
            ["bullet", "list", "json", "table", "format", "markdown",
             "numbered", "section", "heading"],
            "Specify output format (list/table/JSON/sections)",
        ))
        reqs.append((
            "Chain-of-thought guidance",
            "REASONING",
            ["step by step", "think through", "reasoning",
             "explain your", "let's think", "show your work"],
            "Add 'Think step by step' or 'Show your reasoning'",
        ))
        reqs.append((
            "Example / few-shot reference",
            "GROUNDING",
            ["example", "for instance", "e.g.", "such as",
             "like this", "sample", "here is"],
            "Add an example of desired output",
        ))
        reqs.append((
            "Scope boundaries",
            "SCOPE",
            ["only", "do not", "avoid", "limit", "focus on",
             "scope", "exclude", "don't"],
            "Define what to include AND exclude",
        ))
        reqs.append((
            "Success criteria clarity",
            "VERIFICATION",
            ["correct", "accurate", "complete", "verify",
             "check", "validate", "ensure"],
            "State what 'good' looks like so output can be verified",
        ))
        reqs.append((
            "Length / depth guidance",
            "CONSTRAINT",
            ["brief", "detailed", "concise", "comprehensive",
             "short", "paragraph", "sentence", "words"],
            "Specify expected length/depth (brief, detailed, ~N words)",
        ))

        # Derived from desired output keywords
        if desired:
            if any(w in desired for w in ["code", "function", "implement", "class", "script"]):
                reqs.append((
                    "Code output expected",
                    "OUTPUT-TYPE",
                    ["code", "function", "implement", "class", "def ",
                     "script", "program", "```"],
                    "Explicitly request code with language and constraints",
                ))
            if any(w in desired for w in ["compare", "difference", "versus", "vs", "contrast"]):
                reqs.append((
                    "Comparison structure",
                    "OUTPUT-TYPE",
                    ["compare", "contrast", "difference", "versus",
                     "vs", "table", "side by side"],
                    "Request a structured comparison (table/side-by-side)",
                ))
            if any(w in desired for w in ["explain", "understand", "learn", "teach"]):
                reqs.append((
                    "Educational clarity",
                    "OUTPUT-TYPE",
                    ["explain", "simple", "beginner", "learn",
                     "understand", "teach", "tutorial"],
                    "Specify audience level and pedagogical approach",
                ))

        return reqs

    # ── M4: Corrective Action Register ───────────────────────────

    def _build_car(self) -> Matrix:
        m = Matrix(
            id="M4",
            title="Corrective Action Register (CAR)",
            description="Prioritized list of prompt improvements with traceability to findings.",
            headers=[
                "CA-ID", "Finding", "Action Type",
                "Corrective Action", "Priority",
                "Impact", "Confidence", "Trace-To",
            ],
        )

        for i, s in enumerate(self.report.suggestions):
            priority = {"high": "P1-CRITICAL", "medium": "P2-MAJOR", "low": "P3-MINOR"}.get(
                s.impact.value, "P3-MINOR"
            )

            m.rows.append(MatrixRow(cells={
                "CA-ID": f"CA{i+1:02d}",
                "Finding": s.problem[:55],
                "Action Type": s.type.value.upper(),
                "Corrective Action": s.fix[:60],
                "Priority": priority,
                "Impact": s.impact.value.upper(),
                "Confidence": f"{s.confidence:.0%}",
                "Trace-To": s.evidence[:40] if s.evidence else "—",
            }))

        return m

    # ── M5: Before/After Verification Matrix ─────────────────────

    def _build_bavm(self) -> Matrix:
        m = Matrix(
            id="M5",
            title="Before/After Verification Matrix (BAVM)",
            description="Side-by-side comparison of prompt and output quality before and after improvements.",
            headers=[
                "Metric", "Before", "After", "Delta", "Status",
            ],
        )

        r = self.report

        # Overall scores
        delta_score = r.score_after - r.score_before
        m.rows.append(MatrixRow(cells={
            "Metric": "Effectiveness Score",
            "Before": f"{r.score_before:.0%}",
            "After": f"{r.score_after:.0%}",
            "Delta": f"{delta_score:+.0%}",
            "Status": "IMPROVED" if delta_score > 0 else "UNCHANGED" if delta_score == 0 else "DEGRADED",
        }))

        # Word count
        before_words = len(r.original_prompt.split())
        after_words = len(r.improved_prompt.split()) if r.improved_prompt else 0
        m.rows.append(MatrixRow(cells={
            "Metric": "Prompt Length (words)",
            "Before": str(before_words),
            "After": str(after_words),
            "Delta": f"{after_words - before_words:+d}",
            "Status": "OK" if 10 <= after_words <= 100 else "REVIEW",
        }))

        # Output length
        before_out = len(r.original_output.split()) if r.original_output else 0
        after_out = len(r.improved_output.split()) if r.improved_output else 0
        m.rows.append(MatrixRow(cells={
            "Metric": "Output Length (words)",
            "Before": str(before_out),
            "After": str(after_out),
            "Delta": f"{after_out - before_out:+d}",
            "Status": "OK",
        }))

        # Specificity
        def count_specificity(text: str) -> int:
            markers = ["must", "exactly", "specifically", "only", "precisely",
                        "always", "never", "ensure", "required"]
            return sum(1 for m in markers if m in text.lower())

        spec_b = count_specificity(r.original_prompt)
        spec_a = count_specificity(r.improved_prompt) if r.improved_prompt else 0
        m.rows.append(MatrixRow(cells={
            "Metric": "Specificity Markers",
            "Before": str(spec_b),
            "After": str(spec_a),
            "Delta": f"{spec_a - spec_b:+d}",
            "Status": "PASS" if spec_a >= 2 else "MARGINAL" if spec_a >= 1 else "FAIL",
        }))

        # Structure markers
        def count_structure(text: str) -> int:
            markers = ["bullet", "list", "json", "table", "format",
                        "step by step", "numbered", "section"]
            return sum(1 for m in markers if m in text.lower())

        str_b = count_structure(r.original_prompt)
        str_a = count_structure(r.improved_prompt) if r.improved_prompt else 0
        m.rows.append(MatrixRow(cells={
            "Metric": "Structure Markers",
            "Before": str(str_b),
            "After": str(str_a),
            "Delta": f"{str_a - str_b:+d}",
            "Status": "PASS" if str_a >= 1 else "FAIL",
        }))

        # Number of suggestions applied
        m.rows.append(MatrixRow(cells={
            "Metric": "Corrective Actions Applied",
            "Before": "0",
            "After": str(r.num_suggestions),
            "Delta": f"+{r.num_suggestions}",
            "Status": "APPLIED",
        }))

        return m

    # ── Output methods ───────────────────────────────────────────

    def print_matrices(self) -> None:
        """Print all matrices as Rich tables to the terminal."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console()

            # Header
            console.print(Panel(
                "[bold]Prism Systems Engineering Assessment[/bold]\n"
                f"Prompt: \"{self.report.original_prompt[:60]}...\"\n"
                f"Matrices: {len(self.matrices)}",
                title="\u2630 SE Matrix Report",
                border_style="blue",
            ))

            for mx in self.matrices:
                if not mx.rows:
                    continue

                console.print(f"\n")
                table = Table(
                    title=f"{mx.id}: {mx.title}",
                    caption=mx.description,
                    show_lines=True,
                    title_style="bold blue",
                    caption_style="dim italic",
                )

                for h in mx.headers:
                    justify = "right" if h in ("Score", "Edit Distance", "Semantic Distance",
                                                 "Delta", "Confidence") else "left"
                    width = 8 if h in ("ID", "REQ-ID", "CA-ID", "Status", "Score") else None
                    table.add_column(h, justify=justify, max_width=width)

                for row in mx.rows:
                    cells = []
                    for h in mx.headers:
                        val = row.cells.get(h, "")
                        # Apply status coloring
                        if h in ("Assessment", "Status", "Output Status", "Risk Level", "Priority"):
                            val = _status_icon(val) if val in (
                                "PASS", "ACCEPTABLE", "MARGINAL", "WEAK", "FAIL",
                                "COVERED", "PARTIAL", "GAP", "HIGH", "MEDIUM", "LOW",
                            ) else val
                            # Priority coloring
                            if "P1" in val:
                                val = f"[red bold]{val}[/red bold]"
                            elif "P2" in val:
                                val = f"[yellow]{val}[/yellow]"
                            elif "P3" in val:
                                val = f"[dim]{val}[/dim]"
                            # Other status coloring
                            if val in ("FLIP", "FRAGILE"):
                                val = f"[red]{val}[/red]"
                            elif val in ("STABLE", "ROBUST"):
                                val = f"[green]{val}[/green]"
                            elif val in ("SENSITIVE",):
                                val = f"[yellow]{val}[/yellow]"
                            elif val in ("IMPROVED", "APPLIED"):
                                val = f"[green]{val}[/green]"
                            elif val in ("DEGRADED",):
                                val = f"[red]{val}[/red]"
                        cells.append(val)
                    table.add_row(*cells)

                console.print(table)

            # Summary footer
            gaps = sum(1 for row in self.m3.rows if row.cells.get("Status") == "GAP")
            criticals = sum(1 for row in self.m4.rows if "P1" in row.cells.get("Priority", ""))
            fragiles = sum(1 for row in self.m2.rows if row.cells.get("Output Status") == "FLIP")

            console.print(Panel(
                f"[bold]Assessment Summary[/bold]\n"
                f"  Requirements Gaps:   [red]{gaps}[/red]\n"
                f"  Critical Actions:    [red]{criticals}[/red]\n"
                f"  Fragile Points:      [red]{fragiles}[/red]\n"
                f"  Score Delta:         {self.report.score_after - self.report.score_before:+.0%}",
                title="Summary",
                border_style="green",
            ))

        except ImportError:
            print(self.to_text())

    def to_text(self) -> str:
        """Plain text rendering of all matrices."""
        lines = [
            "=" * 80,
            "PRISM SYSTEMS ENGINEERING ASSESSMENT",
            "=" * 80,
        ]

        for mx in self.matrices:
            if not mx.rows:
                continue
            lines.append(f"\n{'─' * 80}")
            lines.append(f"{mx.id}: {mx.title}")
            lines.append(f"{'─' * 80}")

            # Calculate column widths
            widths = {h: len(h) for h in mx.headers}
            for row in mx.rows:
                for h in mx.headers:
                    widths[h] = max(widths[h], len(row.cells.get(h, "")))

            # Header row
            header_line = " | ".join(h.ljust(widths[h]) for h in mx.headers)
            lines.append(header_line)
            lines.append("-+-".join("-" * widths[h] for h in mx.headers))

            # Data rows
            for row in mx.rows:
                data_line = " | ".join(
                    row.cells.get(h, "").ljust(widths[h]) for h in mx.headers
                )
                lines.append(data_line)

        return "\n".join(lines)

    def to_csv(self, output_dir: str = ".") -> List[str]:
        """Export all matrices as CSV files.

        Args:
            output_dir: Directory to write CSV files to.

        Returns:
            List of file paths written.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        for mx in self.matrices:
            if not mx.rows:
                continue
            filename = f"prism_{mx.id.lower()}_{mx.title.split('(')[1].split(')')[0].lower()}.csv"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=mx.headers)
                writer.writeheader()
                for row in mx.rows:
                    writer.writerow({h: row.cells.get(h, "") for h in mx.headers})
            paths.append(filepath)

        return paths

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        """Return all matrices as a dict of {matrix_id: [row_dicts]}."""
        return {mx.id: mx.to_list_of_dicts() for mx in self.matrices}
