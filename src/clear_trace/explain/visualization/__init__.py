"""Reports and visualization — console, HTML, Word, matrix heatmaps."""

from clear_trace.explain.visualization.html_report import HTMLReport
from clear_trace.explain.visualization.console import ConsoleReport, AdvisorReport

# Optional: matplotlib-based matrix heatmaps
try:
    from clear_trace.explain.visualization.matrix_plots import MatrixPlotter
except ImportError:
    pass

# Optional: Word document reports
try:
    from clear_trace.explain.visualization.word_report import WordReport, ToolWordReport
except ImportError:
    pass

__all__ = [
    "HTMLReport",
    "ConsoleReport",
    "AdvisorReport",
    "MatrixPlotter",
    "WordReport",
    "ToolWordReport",
]
