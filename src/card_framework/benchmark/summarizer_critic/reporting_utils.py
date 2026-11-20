"""Shared markdown-table and confidence-interval helpers for benchmark reports.

References:
    [1] Penn State Eberly College of Science, "3.5 - The t-Interval for a
    Population Mean," STAT 200, accessed Mar. 14, 2026.
    Available: https://online.stat.psu.edu/stat200/book/export/html/166
    [2] Penn State Eberly College of Science, "6.3 - Confidence Interval for
    the Mean Response," STAT 415, accessed Mar. 14, 2026.
    Available: https://online.stat.psu.edu/stat415/book/export/html/800
    [3] SciPy Developers, "scipy.stats.t," SciPy API Reference, accessed
    Mar. 14, 2026.
    Available: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from statistics import fmean, stdev

from scipy.stats import t as student_t


def _escape_markdown_cell(value: object) -> str:
    """Escape one markdown-table cell for stable report rendering."""
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def render_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    """Render a standards-compliant GitHub-flavored Markdown table."""
    escaped_headers = [_escape_markdown_cell(header) for header in headers]
    escaped_rows = [
        [_escape_markdown_cell(value) for value in row]
        for row in rows
    ]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in escaped_headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in escaped_rows)
    return "\n".join(lines)


def mean_confidence_interval_95(
    values: Sequence[float],
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float] | None:
    """Return a two-sided 95% Student-t interval for the sample mean.

    The interval is undefined for fewer than two observations and is only an
    approximation when the underlying metric values are not close to normal.
    """
    sample = [float(value) for value in values]
    if len(sample) < 2:
        return None

    mean_value = fmean(sample)
    sample_stdev = stdev(sample)
    standard_error = sample_stdev / math.sqrt(len(sample))
    # IEEE citation: Two-sided Student-t confidence interval for the mean
    # adapted from [1], [2], with SciPy's t quantile from [3].
    critical_value = float(student_t.ppf(0.975, df=len(sample) - 1))
    half_width = critical_value * standard_error
    lower = mean_value - half_width
    upper = mean_value + half_width
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    if upper_bound is not None:
        upper = min(upper_bound, upper)
    return lower, upper


def format_confidence_interval(
    bounds: tuple[float, float] | None,
    *,
    precision: int = 3,
) -> str:
    """Format a confidence interval for markdown table display."""
    if bounds is None:
        return "n/a"
    lower, upper = bounds
    return f"[{lower:.{precision}f}, {upper:.{precision}f}]"
