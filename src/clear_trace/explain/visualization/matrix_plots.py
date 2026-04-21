"""Matplotlib-based 2D matrix visualizations for Prism SE matrices.

Generates heatmap-style plots with labeled X and Y axes:
  - M1: PCAM heatmap — Sentences × Quality Attributes
  - M2: SRM heatmap — Perturbations × Impact Metrics
  - M3: RGAM heatmap — Requirements × Coverage Assessment
  - M4: CAR heatmap — Corrective Actions × Priority/Impact/Confidence
  - M5: BAVM grouped bar chart — Metrics × Before/After

Usage:
    >>> from clear_trace.explain.visualization.matrix_plots import MatrixPlotter
    >>> plotter = MatrixPlotter(matrix_report)
    >>> plotter.plot_all()               # Show all 5 figures
    >>> plotter.save_all("output/plots") # Save as PNG files
    >>> fig = plotter.plot_pcam()        # Get individual figure
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for file output
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from clear_trace.explain.advisor.matrix_report import MatrixReport


# ── Color maps ───────────────────────────────────────────────────

# Custom colormaps for SE assessment levels
_SE_COLORS = {
    "PASS":       "#2ecc71",  # green
    "ACCEPTABLE": "#82e0aa",  # light green
    "MARGINAL":   "#f9e79f",  # yellow
    "WEAK":       "#f5b041",  # orange
    "FAIL":       "#e74c3c",  # red
    "COVERED":    "#2ecc71",
    "PARTIAL":    "#f9e79f",
    "GAP":        "#e74c3c",
    "HIGH":       "#e74c3c",
    "MEDIUM":     "#f5b041",
    "LOW":        "#82e0aa",
    "FLIP":       "#e74c3c",
    "STABLE":     "#2ecc71",
    "SENSITIVE":  "#f5b041",
    "ROBUST":     "#2ecc71",
    "FRAGILE":    "#e74c3c",
    "IMPROVED":   "#2ecc71",
    "DEGRADED":   "#e74c3c",
    "UNCHANGED":  "#aab7b8",
    "OK":         "#2ecc71",
    "REVIEW":     "#f5b041",
    "APPLIED":    "#2ecc71",
    "P1-CRITICAL":"#e74c3c",
    "P2-MAJOR":   "#f5b041",
    "P3-MINOR":   "#82e0aa",
}

_BG_DARK = "#1a1a2e"
_BG_CELL = "#16213e"
_TEXT_COLOR = "#e0e0e0"
_GRID_COLOR = "#2c3e6b"
_TITLE_COLOR = "#00d4ff"


def _status_to_numeric(status: str) -> float:
    """Map status strings to numeric values for heatmap coloring."""
    mapping = {
        "PASS": 1.0, "ACCEPTABLE": 0.8, "MARGINAL": 0.5,
        "WEAK": 0.25, "FAIL": 0.0,
        "COVERED": 1.0, "PARTIAL": 0.5, "GAP": 0.0,
        "HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0,  # Risk: low is good
        "FLIP": 0.0, "STABLE": 1.0, "SENSITIVE": 0.5,
        "ROBUST": 1.0, "FRAGILE": 0.0,
        "IMPROVED": 1.0, "DEGRADED": 0.0, "UNCHANGED": 0.5,
        "OK": 1.0, "REVIEW": 0.5,
        "APPLIED": 1.0,
        "P1-CRITICAL": 0.0, "P2-MAJOR": 0.5, "P3-MINOR": 1.0,
    }
    return mapping.get(status, 0.5)


def _try_float(val: str) -> Optional[float]:
    """Try to parse a string as float, return None if not possible."""
    try:
        return float(val.replace("%", "").replace("+", ""))
    except (ValueError, AttributeError):
        return None


def _make_se_cmap():
    """Create a red-yellow-green colormap for SE assessments."""
    colors = ["#e74c3c", "#f5b041", "#f9e79f", "#82e0aa", "#2ecc71"]
    return mcolors.LinearSegmentedColormap.from_list("se_quality", colors, N=256)


class MatrixPlotter:
    """Generate matplotlib heatmap/matrix plots from MatrixReport data.

    Args:
        matrix_report: A MatrixReport instance with computed matrices.
        style: 'dark' for dark theme (default) or 'light' for light theme.
    """

    def __init__(self, matrix_report: MatrixReport, style: str = "dark"):
        if not HAS_MPL:
            raise ImportError("matplotlib is required: pip install matplotlib")

        self.mr = matrix_report
        self.style = style
        self._se_cmap = _make_se_cmap()

        # Theme colors
        if style == "dark":
            self.bg = _BG_DARK
            self.cell_bg = _BG_CELL
            self.text = _TEXT_COLOR
            self.grid = _GRID_COLOR
            self.title_color = _TITLE_COLOR
        else:
            self.bg = "#ffffff"
            self.cell_bg = "#f5f5f5"
            self.text = "#222222"
            self.grid = "#cccccc"
            self.title_color = "#0066cc"

    def _apply_style(self, fig, ax):
        """Apply consistent styling to a figure."""
        fig.patch.set_facecolor(self.bg)
        ax.set_facecolor(self.bg)
        ax.tick_params(colors=self.text, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(self.grid)

    # ── M1: PCAM Heatmap ────────────────────────────────────────

    def plot_pcam(self) -> plt.Figure:
        """M1: Prompt Component Assessment Matrix as a heatmap.

        Y-axis: Sentences (prompt components)
        X-axis: Quality attributes (Importance, Effectiveness, Fragility, Assessment)
        Cell color: SE quality level (red→yellow→green)
        Cell text: The actual value
        """
        mx = self.mr.m1
        if not mx.rows:
            return self._empty_figure("M1: PCAM", "No LIME data available")

        # Build matrix data
        y_labels = []
        attrs = ["Importance", "Score", "Effectiveness", "Fragility", "Concept Link", "Assessment"]
        n_rows = len(mx.rows)
        n_cols = len(attrs)
        data = np.full((n_rows, n_cols), 0.5)
        cell_text = []

        for i, row in enumerate(mx.rows):
            sid = row.cells.get("ID", f"S{i}")
            sentence = row.cells.get("Component (Sentence)", "")
            # Truncate for y-label
            label = f"{sid}: {sentence[:35]}..." if len(sentence) > 35 else f"{sid}: {sentence}"
            y_labels.append(label)

            row_text = []
            for j, attr in enumerate(attrs):
                val = row.cells.get(attr, "")
                row_text.append(val)
                # Convert to numeric for coloring
                numeric = _status_to_numeric(val)
                if numeric is not None and attr in ("Effectiveness", "Fragility", "Assessment", "Importance"):
                    data[i, j] = numeric
                elif attr == "Score":
                    fval = _try_float(val)
                    data[i, j] = min(1.0, abs(fval)) if fval is not None else 0.5
                else:
                    data[i, j] = 0.5  # neutral for text-only
            cell_text.append(row_text)

        fig, ax = plt.subplots(figsize=(14, max(3, 1.5 + n_rows * 0.8)))
        self._apply_style(fig, ax)

        im = ax.imshow(data, cmap=self._se_cmap, aspect="auto", vmin=0, vmax=1)

        # Labels
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(attrs, fontsize=10, color=self.text, fontweight="bold")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_labels, fontsize=9, color=self.text)

        # Cell text
        for i in range(n_rows):
            for j in range(n_cols):
                txt = cell_text[i][j]
                # Choose contrasting text color
                brightness = data[i, j]
                tc = "#1a1a1a" if brightness > 0.5 else "#ffffff"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=tc, fontweight="bold")

        # Grid
        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color=self.grid, linewidth=0.5)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color=self.grid, linewidth=0.5)

        ax.set_title("M1: Prompt Component Assessment Matrix (PCAM)",
                      fontsize=14, color=self.title_color, fontweight="bold", pad=15)
        ax.set_xlabel("Quality Attributes", fontsize=11, color=self.text, labelpad=10)
        ax.set_ylabel("Prompt Components", fontsize=11, color=self.text, labelpad=10)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label("Quality Level", color=self.text, fontsize=9)
        cbar.ax.tick_params(colors=self.text, labelsize=8)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["FAIL", "WEAK", "MARGINAL", "ACCEPTABLE", "PASS"])

        fig.tight_layout()
        return fig

    # ── M2: SRM Heatmap ─────────────────────────────────────────

    def plot_srm(self) -> plt.Figure:
        """M2: Sensitivity & Robustness Matrix as a heatmap.

        Y-axis: Perturbations
        X-axis: Impact metrics (Type, Edit Distance, Semantic Distance, Output Status, Risk Level)
        """
        mx = self.mr.m2
        if not mx.rows:
            return self._empty_figure("M2: SRM", "No counterfactual data available")

        y_labels = []
        attrs = ["Type", "Edit Distance", "Semantic Distance", "Output Status", "Risk Level"]
        n_rows = len(mx.rows)
        n_cols = len(attrs)
        data = np.full((n_rows, n_cols), 0.5)
        cell_text = []

        for i, row in enumerate(mx.rows):
            pid = row.cells.get("ID", f"P{i}")
            desc = row.cells.get("Perturbation", "")
            label = f"{pid}: {desc[:40]}..." if len(desc) > 40 else f"{pid}: {desc}"
            y_labels.append(label)

            row_text = []
            for j, attr in enumerate(attrs):
                val = row.cells.get(attr, "")
                row_text.append(val)

                if attr == "Output Status":
                    data[i, j] = _status_to_numeric(val)
                elif attr == "Risk Level":
                    data[i, j] = _status_to_numeric(val)
                elif attr == "Semantic Distance":
                    fval = _try_float(val)
                    if fval is not None:
                        data[i, j] = 1.0 - min(1.0, fval)  # high distance = bad
                elif attr == "Edit Distance":
                    fval = _try_float(val)
                    if fval is not None:
                        data[i, j] = max(0, 1.0 - fval / 100)  # normalize
                else:
                    data[i, j] = 0.5
            cell_text.append(row_text)

        fig, ax = plt.subplots(figsize=(14, max(3, 1.5 + n_rows * 0.8)))
        self._apply_style(fig, ax)

        im = ax.imshow(data, cmap=self._se_cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(attrs, fontsize=10, color=self.text, fontweight="bold")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_labels, fontsize=9, color=self.text)

        for i in range(n_rows):
            for j in range(n_cols):
                brightness = data[i, j]
                tc = "#1a1a1a" if brightness > 0.5 else "#ffffff"
                ax.text(j, i, cell_text[i][j], ha="center", va="center",
                        fontsize=8, color=tc, fontweight="bold")

        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color=self.grid, linewidth=0.5)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color=self.grid, linewidth=0.5)

        ax.set_title("M2: Sensitivity & Robustness Matrix (SRM)",
                      fontsize=14, color=self.title_color, fontweight="bold", pad=15)
        ax.set_xlabel("Impact Metrics", fontsize=11, color=self.text, labelpad=10)
        ax.set_ylabel("Perturbations", fontsize=11, color=self.text, labelpad=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label("Robustness", color=self.text, fontsize=9)
        cbar.ax.tick_params(colors=self.text, labelsize=8)

        fig.tight_layout()
        return fig

    # ── M3: RGAM Heatmap ────────────────────────────────────────

    def plot_rgam(self) -> plt.Figure:
        """M3: Requirements Gap Analysis Matrix as a heatmap.

        Y-axis: Requirements
        X-axis: Assessment dimensions (Category, Coverage, Status)
        Color encodes: COVERED (green), PARTIAL (yellow), GAP (red)
        """
        mx = self.mr.m3
        if not mx.rows:
            return self._empty_figure("M3: RGAM", "No requirements data available")

        y_labels = []
        attrs = ["Category", "Prompt Coverage", "Status", "Corrective Action"]
        n_rows = len(mx.rows)
        n_cols = len(attrs)
        data = np.full((n_rows, n_cols), 0.5)
        cell_text = []

        for i, row in enumerate(mx.rows):
            rid = row.cells.get("REQ-ID", f"R{i}")
            req = row.cells.get("Requirement", "")
            y_labels.append(f"{rid}: {req}")

            row_text = []
            for j, attr in enumerate(attrs):
                val = row.cells.get(attr, "")
                row_text.append(val[:30] if len(val) > 30 else val)

                if attr == "Status":
                    data[i, j] = _status_to_numeric(val)
                elif attr == "Prompt Coverage":
                    if val == "Addressed":
                        data[i, j] = 1.0
                    elif val == "Weak/Partial":
                        data[i, j] = 0.5
                    else:
                        data[i, j] = 0.0
                else:
                    # Color corrective action column based on status
                    status = row.cells.get("Status", "")
                    data[i, j] = _status_to_numeric(status)
            cell_text.append(row_text)

        fig, ax = plt.subplots(figsize=(16, max(4, 1.5 + n_rows * 0.7)))
        self._apply_style(fig, ax)

        im = ax.imshow(data, cmap=self._se_cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(attrs, fontsize=10, color=self.text, fontweight="bold")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_labels, fontsize=9, color=self.text)

        for i in range(n_rows):
            for j in range(n_cols):
                brightness = data[i, j]
                tc = "#1a1a1a" if brightness > 0.5 else "#ffffff"
                ax.text(j, i, cell_text[i][j], ha="center", va="center",
                        fontsize=7, color=tc, fontweight="bold")

        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color=self.grid, linewidth=0.5)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color=self.grid, linewidth=0.5)

        ax.set_title("M3: Requirements Gap Analysis Matrix (RGAM)",
                      fontsize=14, color=self.title_color, fontweight="bold", pad=15)
        ax.set_xlabel("Assessment Dimensions", fontsize=11, color=self.text, labelpad=10)
        ax.set_ylabel("Requirements", fontsize=11, color=self.text, labelpad=10)

        # Legend instead of colorbar for categorical
        from matplotlib.patches import Patch
        legend_items = [
            Patch(facecolor="#2ecc71", label="COVERED"),
            Patch(facecolor="#f9e79f", label="PARTIAL"),
            Patch(facecolor="#e74c3c", label="GAP"),
        ]
        ax.legend(handles=legend_items, loc="upper right", fontsize=8,
                  facecolor=self.bg, edgecolor=self.grid, labelcolor=self.text)

        fig.tight_layout()
        return fig

    # ── M4: CAR Heatmap ─────────────────────────────────────────

    def plot_car(self) -> plt.Figure:
        """M4: Corrective Action Register as a heatmap.

        Y-axis: Corrective actions
        X-axis: Assessment attributes (Action Type, Priority, Impact, Confidence)
        """
        mx = self.mr.m4
        if not mx.rows:
            return self._empty_figure("M4: CAR", "No suggestions available")

        y_labels = []
        attrs = ["Action Type", "Priority", "Impact", "Confidence"]
        n_rows = len(mx.rows)
        n_cols = len(attrs)
        data = np.full((n_rows, n_cols), 0.5)
        cell_text = []

        for i, row in enumerate(mx.rows):
            caid = row.cells.get("CA-ID", f"CA{i}")
            finding = row.cells.get("Finding", "")
            label = f"{caid}: {finding[:40]}..." if len(finding) > 40 else f"{caid}: {finding}"
            y_labels.append(label)

            row_text = []
            for j, attr in enumerate(attrs):
                val = row.cells.get(attr, "")
                row_text.append(val)

                if attr == "Priority":
                    data[i, j] = _status_to_numeric(val)
                elif attr == "Impact":
                    data[i, j] = {"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0}.get(val, 0.5)
                elif attr == "Confidence":
                    fval = _try_float(val)
                    if fval is not None:
                        data[i, j] = fval / 100 if fval > 1 else fval
                else:
                    data[i, j] = 0.5
            cell_text.append(row_text)

        fig, ax = plt.subplots(figsize=(12, max(3, 1.5 + n_rows * 0.8)))
        self._apply_style(fig, ax)

        # Use reversed cmap for priority (red = urgent = needs action)
        cmap_priority = mcolors.LinearSegmentedColormap.from_list(
            "priority", ["#e74c3c", "#f5b041", "#2ecc71"], N=256
        )
        im = ax.imshow(data, cmap=cmap_priority, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(attrs, fontsize=10, color=self.text, fontweight="bold")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_labels, fontsize=8, color=self.text)

        for i in range(n_rows):
            for j in range(n_cols):
                brightness = data[i, j]
                tc = "#1a1a1a" if brightness > 0.5 else "#ffffff"
                ax.text(j, i, cell_text[i][j], ha="center", va="center",
                        fontsize=8, color=tc, fontweight="bold")

        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color=self.grid, linewidth=0.5)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color=self.grid, linewidth=0.5)

        ax.set_title("M4: Corrective Action Register (CAR)",
                      fontsize=14, color=self.title_color, fontweight="bold", pad=15)
        ax.set_xlabel("Assessment Attributes", fontsize=11, color=self.text, labelpad=10)
        ax.set_ylabel("Corrective Actions", fontsize=11, color=self.text, labelpad=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label("Urgency (red = act now)", color=self.text, fontsize=9)
        cbar.ax.tick_params(colors=self.text, labelsize=8)

        fig.tight_layout()
        return fig

    # ── M5: BAVM Grouped Bar Chart ──────────────────────────────

    def plot_bavm(self) -> plt.Figure:
        """M5: Before/After Verification Matrix as grouped bar chart.

        Y-axis: Metrics
        X-axis: Values (Before vs After, with Delta annotation)
        """
        mx = self.mr.m5
        if not mx.rows:
            return self._empty_figure("M5: BAVM", "No before/after data available")

        metrics = []
        before_vals = []
        after_vals = []
        deltas = []
        statuses = []

        for row in mx.rows:
            metric = row.cells.get("Metric", "")
            before = row.cells.get("Before", "0")
            after = row.cells.get("After", "0")
            delta = row.cells.get("Delta", "0")
            status = row.cells.get("Status", "")

            metrics.append(metric)
            deltas.append(delta)
            statuses.append(status)

            # Parse numeric values
            bv = _try_float(before)
            av = _try_float(after)
            before_vals.append(bv if bv is not None else 0)
            after_vals.append(av if av is not None else 0)

        n = len(metrics)
        y_pos = np.arange(n)
        bar_height = 0.35

        fig, ax = plt.subplots(figsize=(14, max(4, 1.5 + n * 1.0)))
        self._apply_style(fig, ax)

        bars_before = ax.barh(y_pos + bar_height / 2, before_vals, bar_height,
                               label="Before", color="#e74c3c", alpha=0.85,
                               edgecolor=self.grid, linewidth=0.5)
        bars_after = ax.barh(y_pos - bar_height / 2, after_vals, bar_height,
                              label="After", color="#2ecc71", alpha=0.85,
                              edgecolor=self.grid, linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars_before, before_vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:g}", va="center", ha="left", fontsize=8, color="#e74c3c")
        for bar, val in zip(bars_after, after_vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:g}", va="center", ha="left", fontsize=8, color="#2ecc71")

        # Delta annotations on right side
        max_val = max(max(before_vals, default=1), max(after_vals, default=1))
        for i, (delta, status) in enumerate(zip(deltas, statuses)):
            color = _SE_COLORS.get(status, self.text)
            ax.text(max_val * 1.15, i, f"Δ {delta}  [{status}]",
                    va="center", ha="left", fontsize=9, color=color, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=10, color=self.text)
        ax.set_xlabel("Value", fontsize=11, color=self.text, labelpad=10)
        ax.set_title("M5: Before / After Verification Matrix (BAVM)",
                      fontsize=14, color=self.title_color, fontweight="bold", pad=15)

        ax.legend(loc="lower right", fontsize=10,
                  facecolor=self.bg, edgecolor=self.grid, labelcolor=self.text)
        ax.set_xlim(0, max_val * 1.6)
        ax.invert_yaxis()

        fig.tight_layout()
        return fig

    # ── Dashboard: All 5 matrices in one figure ─────────────────

    def plot_dashboard(self) -> plt.Figure:
        """Generate a combined dashboard with all 5 matrices as subplots.

        Returns a single figure with a 3×2 grid layout.
        """
        fig = plt.figure(figsize=(24, 20))
        fig.patch.set_facecolor(self.bg)
        fig.suptitle("PRISM — Systems Engineering Assessment Dashboard",
                     fontsize=18, color=self.title_color, fontweight="bold", y=0.98)

        # We'll create individual plots and copy their axes content
        # into subplots. Simpler: just build inline.
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        # M1: PCAM (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_pcam_on_ax(ax1)

        # M2: SRM (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_srm_on_ax(ax2)

        # M3: RGAM (middle, full width)
        ax3 = fig.add_subplot(gs[1, :])
        self._draw_rgam_on_ax(ax3)

        # M4: CAR (bottom-left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._draw_car_on_ax(ax4)

        # M5: BAVM (bottom-right)
        ax5 = fig.add_subplot(gs[2, 1])
        self._draw_bavm_on_ax(ax5)

        return fig

    # ── Inline subplot renderers ────────────────────────────────

    def _draw_pcam_on_ax(self, ax):
        mx = self.mr.m1
        if not mx.rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.text)
            ax.set_title("M1: PCAM", color=self.title_color, fontsize=11, fontweight="bold")
            return

        attrs = ["Importance", "Score", "Effectiveness", "Fragility", "Assessment"]
        n_rows, n_cols = len(mx.rows), len(attrs)
        data = np.full((n_rows, n_cols), 0.5)
        labels = []
        cell_text = []

        for i, row in enumerate(mx.rows):
            sid = row.cells.get("ID", "")
            sent = row.cells.get("Component (Sentence)", "")[:25]
            labels.append(f"{sid}: {sent}")
            rt = []
            for j, attr in enumerate(attrs):
                val = row.cells.get(attr, "")
                rt.append(val[:10])
                numeric = _status_to_numeric(val)
                if attr == "Score":
                    fval = _try_float(val)
                    data[i, j] = min(1.0, abs(fval)) if fval is not None else 0.5
                elif attr in ("Effectiveness", "Fragility", "Assessment", "Importance"):
                    data[i, j] = numeric
                else:
                    data[i, j] = 0.5
            cell_text.append(rt)

        ax.imshow(data, cmap=self._se_cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(attrs, fontsize=7, color=self.text)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels, fontsize=7, color=self.text)
        for i in range(n_rows):
            for j in range(n_cols):
                b = data[i, j]
                tc = "#1a1a1a" if b > 0.5 else "#ffffff"
                ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=6, color=tc)
        ax.set_title("M1: PCAM", color=self.title_color, fontsize=11, fontweight="bold")

    def _draw_srm_on_ax(self, ax):
        mx = self.mr.m2
        if not mx.rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.text)
            ax.set_title("M2: SRM", color=self.title_color, fontsize=11, fontweight="bold")
            return

        attrs = ["Type", "Edit Distance", "Semantic Distance", "Output Status", "Risk Level"]
        n_rows, n_cols = len(mx.rows), len(attrs)
        data = np.full((n_rows, n_cols), 0.5)
        labels = []
        cell_text = []

        for i, row in enumerate(mx.rows):
            pid = row.cells.get("ID", "")
            desc = row.cells.get("Perturbation", "")[:30]
            labels.append(f"{pid}: {desc}")
            rt = []
            for j, attr in enumerate(attrs):
                val = row.cells.get(attr, "")
                rt.append(val[:10])
                if attr in ("Output Status", "Risk Level"):
                    data[i, j] = _status_to_numeric(val)
                elif attr == "Semantic Distance":
                    fval = _try_float(val)
                    data[i, j] = 1.0 - min(1.0, fval) if fval is not None else 0.5
                else:
                    data[i, j] = 0.5
            cell_text.append(rt)

        ax.imshow(data, cmap=self._se_cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(attrs, fontsize=7, color=self.text)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels, fontsize=7, color=self.text)
        for i in range(n_rows):
            for j in range(n_cols):
                b = data[i, j]
                tc = "#1a1a1a" if b > 0.5 else "#ffffff"
                ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=6, color=tc)
        ax.set_title("M2: SRM", color=self.title_color, fontsize=11, fontweight="bold")

    def _draw_rgam_on_ax(self, ax):
        mx = self.mr.m3
        if not mx.rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.text)
            ax.set_title("M3: RGAM", color=self.title_color, fontsize=11, fontweight="bold")
            return

        attrs = ["Category", "Prompt Coverage", "Status"]
        n_rows, n_cols = len(mx.rows), len(attrs)
        data = np.full((n_rows, n_cols), 0.5)
        labels = []
        cell_text = []

        for i, row in enumerate(mx.rows):
            rid = row.cells.get("REQ-ID", "")
            req = row.cells.get("Requirement", "")
            labels.append(f"{rid}: {req}")
            rt = []
            for j, attr in enumerate(attrs):
                val = row.cells.get(attr, "")
                rt.append(val[:15])
                if attr == "Status":
                    data[i, j] = _status_to_numeric(val)
                elif attr == "Prompt Coverage":
                    data[i, j] = {"Addressed": 1.0, "Weak/Partial": 0.5, "Missing": 0.0}.get(val, 0.5)
                else:
                    status = row.cells.get("Status", "")
                    data[i, j] = _status_to_numeric(status)
            cell_text.append(rt)

        ax.imshow(data, cmap=self._se_cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(attrs, fontsize=7, color=self.text)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels, fontsize=7, color=self.text)
        for i in range(n_rows):
            for j in range(n_cols):
                b = data[i, j]
                tc = "#1a1a1a" if b > 0.5 else "#ffffff"
                ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=6, color=tc)
        ax.set_title("M3: RGAM", color=self.title_color, fontsize=11, fontweight="bold")

    def _draw_car_on_ax(self, ax):
        mx = self.mr.m4
        if not mx.rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.text)
            ax.set_title("M4: CAR", color=self.title_color, fontsize=11, fontweight="bold")
            return

        attrs = ["Action Type", "Priority", "Impact", "Confidence"]
        n_rows, n_cols = len(mx.rows), len(attrs)
        data = np.full((n_rows, n_cols), 0.5)
        labels = []
        cell_text = []

        for i, row in enumerate(mx.rows):
            caid = row.cells.get("CA-ID", "")
            finding = row.cells.get("Finding", "")[:30]
            labels.append(f"{caid}: {finding}")
            rt = []
            for j, attr in enumerate(attrs):
                val = row.cells.get(attr, "")
                rt.append(val[:10])
                if attr == "Priority":
                    data[i, j] = _status_to_numeric(val)
                elif attr == "Impact":
                    data[i, j] = {"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0}.get(val, 0.5)
                elif attr == "Confidence":
                    fval = _try_float(val)
                    data[i, j] = fval / 100 if fval is not None and fval > 1 else (fval or 0.5)
                else:
                    data[i, j] = 0.5
            cell_text.append(rt)

        cmap_p = mcolors.LinearSegmentedColormap.from_list("p", ["#e74c3c", "#f5b041", "#2ecc71"])
        ax.imshow(data, cmap=cmap_p, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(attrs, fontsize=7, color=self.text)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels, fontsize=7, color=self.text)
        for i in range(n_rows):
            for j in range(n_cols):
                b = data[i, j]
                tc = "#1a1a1a" if b > 0.5 else "#ffffff"
                ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=6, color=tc)
        ax.set_title("M4: CAR", color=self.title_color, fontsize=11, fontweight="bold")

    def _draw_bavm_on_ax(self, ax):
        mx = self.mr.m5
        if not mx.rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.text)
            ax.set_title("M5: BAVM", color=self.title_color, fontsize=11, fontweight="bold")
            return

        metrics, before_vals, after_vals, statuses = [], [], [], []
        for row in mx.rows:
            metrics.append(row.cells.get("Metric", ""))
            bv = _try_float(row.cells.get("Before", "0"))
            av = _try_float(row.cells.get("After", "0"))
            before_vals.append(bv or 0)
            after_vals.append(av or 0)
            statuses.append(row.cells.get("Status", ""))

        n = len(metrics)
        y_pos = np.arange(n)
        h = 0.35
        ax.barh(y_pos + h/2, before_vals, h, label="Before", color="#e74c3c", alpha=0.85)
        ax.barh(y_pos - h/2, after_vals, h, label="After", color="#2ecc71", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=7, color=self.text)
        ax.invert_yaxis()
        ax.legend(fontsize=7, facecolor=self.bg, edgecolor=self.grid, labelcolor=self.text)
        ax.set_title("M5: BAVM", color=self.title_color, fontsize=11, fontweight="bold")

    # ── Convenience methods ─────────────────────────────────────

    def plot_all(self, show: bool = True) -> List[plt.Figure]:
        """Generate all 5 individual matrix plots.

        Args:
            show: If True, call plt.show() at the end.

        Returns:
            List of 5 matplotlib Figure objects.
        """
        figs = [
            self.plot_pcam(),
            self.plot_srm(),
            self.plot_rgam(),
            self.plot_car(),
            self.plot_bavm(),
        ]
        if show:
            plt.show()
        return figs

    def save_all(self, output_dir: str = ".", dpi: int = 150,
                 fmt: str = "png") -> List[str]:
        """Save all 5 matrix plots as image files.

        Args:
            output_dir: Directory to save images.
            dpi: Resolution (default 150).
            fmt: Image format ('png', 'svg', 'pdf').

        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        plots = [
            ("m1_pcam", self.plot_pcam),
            ("m2_srm", self.plot_srm),
            ("m3_rgam", self.plot_rgam),
            ("m4_car", self.plot_car),
            ("m5_bavm", self.plot_bavm),
        ]

        paths = []
        for name, plot_fn in plots:
            fig = plot_fn()
            path = os.path.join(output_dir, f"prism_{name}.{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            paths.append(path)

        return paths

    def save_dashboard(self, path: str = "prism_dashboard.png",
                       dpi: int = 150) -> str:
        """Save the combined dashboard as a single image."""
        fig = self.plot_dashboard()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ── Helpers ──────────────────────────────────────────────────

    def _empty_figure(self, title: str, message: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 3))
        self._apply_style(fig, ax)
        ax.text(0.5, 0.5, message, ha="center", va="center",
                fontsize=14, color=self.text, transform=ax.transAxes)
        ax.set_title(title, fontsize=14, color=self.title_color, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        return fig
