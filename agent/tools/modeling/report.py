"""Tiny text-report helpers used by the modeling tools.

Each report is built as a list of lines via a :class:`ReportBuilder`,
then joined into the final string returned by the tool. Centralising the
banner / key-value / table primitives keeps the three modeling tools
visually consistent and removes ~3x of manual padding code.
"""
from __future__ import annotations

from typing import Iterable


class ReportBuilder:
    """Accumulates lines of a fixed-width text report."""

    def __init__(self, width: int = 76) -> None:
        self.width = width
        self._lines: list[str] = []

    def line(self, text: str = "") -> "ReportBuilder":
        self._lines.append(text)
        return self

    def rule(self, char: str = "-") -> "ReportBuilder":
        return self.line(char * self.width)

    def banner(self, title: str) -> "ReportBuilder":
        return self.rule("=").line(f"{title:^{self.width}}").rule("=")

    def section(self, title: str) -> "ReportBuilder":
        return self.line().rule("-").line(f"{title:^{self.width}}").rule("-")

    def heading(self, title: str) -> "ReportBuilder":
        """Bracketed `[ Heading ]` followed by a thin rule."""
        return self.line(f"[ {title} ]").rule()

    def kv(self, key: str, value: str, key_width: int = 22) -> "ReportBuilder":
        return self.line(f"{key:<{key_width}} : {value}")

    def table(
        self,
        headers: Iterable[str],
        rows: Iterable[Iterable[str]],
        col_widths: Iterable[int],
        sep: str = " | ",
    ) -> "ReportBuilder":
        widths = list(col_widths)
        header_cells = [f"{h:<{w}}" for h, w in zip(headers, widths)]
        self.line(sep.join(header_cells))
        self.rule()
        for row in rows:
            cells = [f"{c:<{w}}" for c, w in zip(row, widths)]
            self.line(sep.join(cells))
        return self

    def build(self) -> str:
        return "\n".join(self._lines)
