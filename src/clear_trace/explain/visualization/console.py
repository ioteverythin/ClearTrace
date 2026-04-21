"""Console-based visualization using Rich.

Renders Prism explanations as rich terminal output with
color-coded importance, tables, and trees.
"""

from __future__ import annotations

from typing import Optional

from clear_trace.explain.core.types import (
    Explanation,
    ImportanceLevel,
    TokenImportance,
    SentenceImportance,
    ConceptAttribution,
)

# Color mapping for importance levels
_LEVEL_COLORS = {
    ImportanceLevel.CRITICAL: "bold red",
    ImportanceLevel.HIGH: "red",
    ImportanceLevel.MEDIUM: "yellow",
    ImportanceLevel.LOW: "dim",
    ImportanceLevel.NEGLIGIBLE: "dim italic",
}

_SCORE_BAR_CHARS = "░▒▓█"


def _score_bar(score: float, width: int = 20) -> str:
    """Create a text-based bar for a score value."""
    abs_score = min(abs(score), 1.0)
    filled = int(abs_score * width)
    sign = "+" if score >= 0 else "-"
    bar = "█" * filled + "░" * (width - filled)
    return f"[{sign}] {bar} {score:+.3f}"


class ConsoleReport:
    """Render Prism explanations to the terminal using Rich.

    Example:
        >>> report = ConsoleReport(explanation)
        >>> report.print()  # Rich-formatted output
        >>> text = report.to_text()  # Plain text fallback
    """

    def __init__(self, explanation: Explanation):
        self.explanation = explanation

    def print(self) -> None:
        """Print the explanation using Rich formatting."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.tree import Tree
            from rich.text import Text

            console = Console()
            exp = self.explanation

            # Header
            console.print(
                Panel(
                    f"[bold]Prism Explanation[/bold]\n"
                    f"Method: {exp.method}\n"
                    f"Model: {exp.model_name or 'N/A'}",
                    title="🔍 Prism",
                    border_style="blue",
                )
            )

            # Summary
            if exp.summary:
                console.print(f"\n[bold]Summary:[/bold]\n{exp.summary}\n")

            # Token importances
            if exp.token_importances:
                table = Table(title="Token Importance", show_lines=True)
                table.add_column("Token", style="bold")
                table.add_column("Score", justify="right")
                table.add_column("Level")
                table.add_column("Bar")

                for t in sorted(
                    exp.token_importances, key=lambda x: abs(x.score), reverse=True
                )[:15]:
                    color = _LEVEL_COLORS.get(t.level, "white")
                    table.add_row(
                        t.token,
                        f"{t.score:+.3f}",
                        f"[{color}]{t.level.value}[/{color}]",
                        _score_bar(t.score, 15),
                    )
                console.print(table)

            # Sentence importances
            if exp.sentence_importances:
                table = Table(title="Sentence Importance", show_lines=True)
                table.add_column("#", justify="right", width=3)
                table.add_column("Sentence")
                table.add_column("Score", justify="right")
                table.add_column("Bar")

                for s in sorted(
                    exp.sentence_importances, key=lambda x: abs(x.score), reverse=True
                ):
                    color = _LEVEL_COLORS.get(s.level, "white")
                    table.add_row(
                        str(s.index),
                        f"[{color}]{s.text[:80]}[/{color}]",
                        f"{s.score:+.3f}",
                        _score_bar(s.score, 15),
                    )
                console.print(table)

                # Print reasons if available
                has_reasons = any(s.reason for s in exp.sentence_importances)
                if has_reasons:
                    console.print("\n[bold]Why these matter:[/bold]")
                    for s in sorted(
                        exp.sentence_importances, key=lambda x: abs(x.score), reverse=True
                    ):
                        if s.reason:
                            color = _LEVEL_COLORS.get(s.level, "white")
                            console.print(
                                f"  [{color}]●[/{color}] \"{s.text[:50]}\" → {s.reason}"
                            )

            # Counterfactuals
            if exp.counterfactuals:
                console.print(f"\n[bold]Counterfactual Explanations[/bold] ({len(exp.counterfactuals)} candidates)")
                for i, cf in enumerate(exp.counterfactuals[:5], 1):
                    flip_tag = "[red]FLIP[/red]" if cf.is_flip else "[green]STABLE[/green]"
                    console.print(
                        f"  {i}. {flip_tag} {cf.change_description}\n"
                        f"     Edit dist: {cf.edit_distance}, "
                        f"Semantic dist: {cf.semantic_distance:.3f}"
                    )
                    if cf.reason:
                        console.print(f"     [italic]Why: {cf.reason}[/italic]")

            # Trajectory
            if exp.trajectory and exp.trajectory.decisions:
                tree = Tree("🤖 Agent Trajectory")
                for d in exp.trajectory.decisions:
                    is_critical = d.step in exp.trajectory.critical_decision_indices
                    prefix = "⚡ " if is_critical else "  "
                    node = tree.add(
                        f"{prefix}Step {d.step}: [bold]{d.tool_name}[/bold] "
                        f"(confidence: {d.confidence:.2f})"
                    )
                    if d.reason:
                        node.add(f"[italic]Why: {d.reason}[/italic]")
                    if d.attribution_scores:
                        top_attrs = sorted(
                            d.attribution_scores.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:3]
                        for k, v in top_attrs:
                            node.add(f"{k}: {v:.2f}")
                console.print(tree)

            # Concepts
            if exp.concept_attributions:
                table = Table(title="Concept Attribution", show_lines=True)
                table.add_column("Concept", style="bold")
                table.add_column("Score", justify="right")
                table.add_column("Evidence")
                table.add_column("Description")

                for c in exp.concept_attributions[:10]:
                    table.add_row(
                        c.concept,
                        f"{c.score:+.3f}",
                        ", ".join(c.evidence_tokens[:3]),
                        c.description[:60],
                    )
                console.print(table)

                # Print reasons if available
                has_reasons = any(c.reason for c in exp.concept_attributions)
                if has_reasons:
                    console.print("\n[bold]Why these concepts matter:[/bold]")
                    for c in exp.concept_attributions:
                        if c.reason:
                            console.print(f"  ● {c.concept} → {c.reason}")

            # Metadata
            if exp.metadata:
                console.print(f"\n[dim]Metadata: {exp.metadata}[/dim]")

        except ImportError:
            # Fallback to plain text if Rich is not available
            print(self.to_text())

    def to_text(self) -> str:
        """Generate a plain-text representation of the explanation."""
        exp = self.explanation
        lines = [
            "=" * 60,
            "PRISM EXPLANATION",
            f"Method: {exp.method}",
            f"Model: {exp.model_name or 'N/A'}",
            "=" * 60,
        ]

        if exp.summary:
            lines.append(f"\n{exp.summary}")

        if exp.token_importances:
            lines.append("\nToken Importance:")
            for t in sorted(exp.token_importances, key=lambda x: abs(x.score), reverse=True)[:10]:
                lines.append(f"  {t.score:+.3f}  '{t.token}' [{t.level.value}]")

        if exp.sentence_importances:
            lines.append("\nSentence Importance:")
            for s in sorted(exp.sentence_importances, key=lambda x: abs(x.score), reverse=True):
                lines.append(f"  {s.score:+.3f}  \"{s.text[:80]}\"")

        if exp.counterfactuals:
            lines.append(f"\nCounterfactuals ({len(exp.counterfactuals)}):")
            for i, cf in enumerate(exp.counterfactuals[:5], 1):
                tag = "FLIP" if cf.is_flip else "STABLE"
                lines.append(f"  {i}. [{tag}] {cf.change_description}")

        if exp.trajectory and exp.trajectory.decisions:
            lines.append("\nTrajectory:")
            for d in exp.trajectory.decisions:
                lines.append(f"  Step {d.step}: {d.tool_name} (conf: {d.confidence:.2f})")

        if exp.concept_attributions:
            lines.append("\nConcepts:")
            for c in exp.concept_attributions:
                lines.append(f"  {c.score:+.3f}  {c.concept}: {c.description[:60]}")

        return "\n".join(lines)


class AdvisorReport:
    """Render a PromptReport (from PromptAdvisor) to the terminal using Rich.

    Example:
        >>> report = advisor.analyze(prompt, desired="...")
        >>> AdvisorReport(report).print()
    """

    def __init__(self, report: Any):
        self.report = report

    def print(self) -> None:
        """Print the advisor report with Rich formatting."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.text import Text

            console = Console()
            r = self.report

            # Header
            console.print(Panel(
                "[bold]Prism Prompt Advisor[/bold]\n"
                "Analyzing your prompt and suggesting improvements",
                title="🔧 Prompt Advisor",
                border_style="green",
            ))

            # Original prompt
            console.print(f"\n[bold]Your Prompt:[/bold]")
            console.print(Panel(r.original_prompt, border_style="red", title="Before"))

            if r.desired_output_description:
                console.print(f"[bold]What you want:[/bold] {r.desired_output_description}\n")

            # Diagnosis
            if r.diagnosis:
                console.print(Panel(r.diagnosis, title="📋 Diagnosis", border_style="yellow"))

            # Suggestions table
            if r.suggestions:
                console.print(f"\n[bold]Suggestions ({len(r.suggestions)}):[/bold]\n")

                for i, s in enumerate(r.suggestions, 1):
                    impact_color = {
                        "high": "red bold",
                        "medium": "yellow",
                        "low": "dim",
                    }.get(s.impact.value, "white")

                    console.print(
                        f"  [{impact_color}]{i}. [{s.impact.value.upper()}][/{impact_color}]"
                        f" [{s.type.value}]"
                    )
                    console.print(f"     [red]Problem:[/red] {s.problem}")
                    console.print(f"     [green]Fix:[/green] {s.fix}")
                    if s.improved_text:
                        console.print(f"     [blue]→[/blue] {s.improved_text}")
                    if s.evidence:
                        console.print(f"     [dim]Evidence: {s.evidence}[/dim]")
                    console.print()

            # Improved prompt
            if r.improved_prompt:
                console.print(Panel(r.improved_prompt, border_style="green", title="✅ Improved Prompt"))

            # Before/After comparison
            if r.improved_output:
                console.print(f"\n[bold]Output Comparison:[/bold]\n")

                table = Table(show_lines=True)
                table.add_column("Before", style="red", max_width=50)
                table.add_column("After", style="green", max_width=50)
                table.add_row(
                    r.original_output[:300] + ("..." if len(r.original_output) > 300 else ""),
                    r.improved_output[:300] + ("..." if len(r.improved_output) > 300 else ""),
                )
                console.print(table)

            # Score
            bar_before = "█" * int(r.score_before * 20) + "░" * (20 - int(r.score_before * 20))
            bar_after = "█" * int(r.score_after * 20) + "░" * (20 - int(r.score_after * 20))
            change = r.score_after - r.score_before
            change_str = f"+{change:.0%}" if change >= 0 else f"{change:.0%}"

            console.print(f"\n[bold]Effectiveness Score:[/bold]")
            console.print(f"  Before: [red]{bar_before}[/red] {r.score_before:.0%}")
            console.print(f"  After:  [green]{bar_after}[/green] {r.score_after:.0%} ({change_str})")

            if r.metadata:
                console.print(f"\n[dim]{r.metadata}[/dim]")

        except ImportError:
            print(self.to_text())

    def to_text(self) -> str:
        """Plain text fallback."""
        r = self.report
        lines = [
            "=" * 60,
            "PRISM PROMPT ADVISOR",
            "=" * 60,
            f"\nOriginal:  {r.original_prompt}",
        ]
        if r.desired_output_description:
            lines.append(f"Desired:   {r.desired_output_description}")
        if r.diagnosis:
            lines.append(f"\nDiagnosis: {r.diagnosis}")
        for i, s in enumerate(r.suggestions, 1):
            lines.append(f"\n{i}. [{s.impact.value}] {s.fix}")
            if s.problem:
                lines.append(f"   Problem: {s.problem}")
        if r.improved_prompt:
            lines.append(f"\nImproved:  {r.improved_prompt}")
        lines.append(f"\nScore: {r.score_before:.0%} → {r.score_after:.0%}")
        return "\n".join(lines)
