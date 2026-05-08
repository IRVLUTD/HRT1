"""Industry-grade colored logging + progress UI for the vie scripts.

Built on `rich`. Provides:
  - `setup(name)`: returns a configured Logger with RichHandler
  - `section(title, subtitle=...)`: prints a colored section header
  - `step(msg)`: prints a checkmark step line
  - `note(msg)`: dim informational line
  - `warn(msg)`, `error(msg)`: yellow/red prints
  - `success(msg)`: green
  - `progress(...)`: context manager wrapping rich.progress.Progress
  - `summary(rows: dict)`: prints a final table summary with timings/counts

All output goes through a single shared `Console`; tqdm bars from third-party
libs (SAM2, detectron2) still work alongside this — `rich`'s console plays
nicely with tqdm because both use stderr by default.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterable, Mapping, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


_console: Optional[Console] = None


def _get_console() -> Console:
    global _console
    if _console is None:
        # Force colors even when stdout isn't a TTY so logs are pretty in
        # captured runs (e.g., bench scripts). Set NO_COLOR=1 to disable.
        force = os.environ.get("NO_COLOR") is None
        _console = Console(force_terminal=force, highlight=False)
    return _console


def setup(name: str = "vie", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a RichHandler attached. Idempotent — call once."""
    logger = logging.getLogger(name)
    if getattr(logger, "_rich_configured", False):
        return logger
    logger.setLevel(level)
    logger.propagate = False
    handler = RichHandler(
        console=_get_console(),
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        log_time_format="[%H:%M:%S]",
    )
    handler.setLevel(level)
    logger.addHandler(handler)
    logger._rich_configured = True  # type: ignore[attr-defined]
    return logger


def section(title: str, subtitle: Optional[str] = None) -> None:
    """Print a banner-style section header."""
    console = _get_console()
    text = Text(title, style="bold cyan")
    if subtitle:
        text.append("  ", style="")
        text.append(subtitle, style="dim")
    console.print()
    console.rule(text, style="cyan")


def step(msg: str) -> None:
    """Single completed-step line with a green check."""
    _get_console().print(f"[green]✓[/] {msg}")


def note(msg: str) -> None:
    """Dim informational line (e.g., file paths, configuration echoes)."""
    _get_console().print(f"  [dim]{msg}[/]")


def warn(msg: str) -> None:
    _get_console().print(f"[yellow]![/] {msg}")


def error(msg: str) -> None:
    _get_console().print(f"[red]✗[/] {msg}", style="red")


def success(msg: str) -> None:
    _get_console().print(f"[bold green]✓[/] {msg}")


@contextmanager
def progress(description: str = "working", total: Optional[int] = None):
    """Context manager that yields a (Progress, task_id) pair. Use as:

        with progress("Tracking frames", total=N) as (p, task):
            for i in range(N):
                ...
                p.update(task, advance=1)
    """
    p = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_get_console(),
        transient=False,
    )
    p.start()
    try:
        task = p.add_task(description, total=total)
        yield p, task
    finally:
        p.stop()


def summary(rows: Mapping[str, str], title: str = "Summary") -> None:
    """Print a pretty 2-column table panel with the given rows."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("k", style="bold cyan")
    table.add_column("v")
    for k, v in rows.items():
        table.add_row(str(k), str(v))
    panel = Panel(table, title=f"[bold]{title}[/]", border_style="cyan", expand=False)
    _get_console().print()
    _get_console().print(panel)


def fmt_duration(seconds: float) -> str:
    """Format a duration in a friendly way: 23 ms / 1.4 s / 2 m 13 s."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    m, s = divmod(int(seconds), 60)
    return f"{m} m {s} s"


def fmt_rate(count: int, seconds: float) -> str:
    if seconds <= 0:
        return "—"
    return f"{count / seconds:.1f}/s"
