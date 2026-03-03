"""CLI terminal/reporting helpers for FlowCompile."""
from __future__ import annotations

import contextvars
import io
import sys
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

from tqdm import tqdm


FLOWCOMPILE_BANNER = r"""
 ______ _                 ____                      _ __
|  ____| |               / __ \                    (_) /
| |__  | | _____      __| |  | |___  _ __ ___  _ __ _| | ___
|  __| | |/ _ \ \ /\ / /| |  | / _ \| '_ ` _ \| '_ \ | |/ _ \
| |    | | (_) \ V  V / | |__| | (_) | | | | | | |_) | | |  __/
|_|    |_|\___/ \_/\_/   \____/ \___/|_| |_| |_| .__/|_|_|\___|
                                               | |
                                               |_|
"""


@dataclass
class CliOutputConfig:
    verbose: bool = False
    quiet: bool = False
    plain: bool = False
    no_banner: bool = False
    stderr_is_tty: bool = False
    stdout_is_tty: bool = False

    @property
    def live_progress(self) -> bool:
        return self.stderr_is_tty and not self.plain and not self.quiet

    @property
    def show_banner(self) -> bool:
        return self.stderr_is_tty and not self.plain and not self.quiet and not self.no_banner


_default_config = CliOutputConfig()
_reporter_var: contextvars.ContextVar["CliReporter"] = contextvars.ContextVar("flowcompile_reporter")


class ProgressHandle:
    """Thin wrapper around tqdm for manual progress updates."""

    def __init__(
        self,
        reporter: "CliReporter",
        total: Optional[int],
        desc: Optional[str],
        *,
        unit: Optional[str] = None,
        leave: bool = False,
    ) -> None:
        self._reporter = reporter
        self._desc = desc
        self._bar = None
        self._completed = 0
        self._total = total
        if reporter.config.live_progress:
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                leave=leave,
                file=sys.stderr,
            )
        elif desc:
            reporter.step(desc)

    def advance(self, amount: int = 1) -> None:
        self._completed += amount
        if self._bar is not None:
            self._bar.update(amount)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        elif self._desc and self._total is not None and not self._reporter.config.quiet:
            self._reporter.detail(f"{self._desc}: {self._completed}/{self._total}")

    def __enter__(self) -> "ProgressHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class CliReporter:
    """Centralized CLI reporter with warning de-duplication."""

    def __init__(
        self,
        config: Optional[CliOutputConfig] = None,
        *,
        prefix: Optional[str] = None,
        _shared: Optional[Dict[str, object]] = None,
    ) -> None:
        self.config = config or _default_config
        self.prefix = prefix or ""
        self._shared = _shared or {
            "banner_printed": False,
            "warnings_seen": {},
            "warnings_suppressed": {},
        }

    def child(self, prefix: str) -> "CliReporter":
        next_prefix = f"{self.prefix} {prefix}".strip() if self.prefix else prefix
        return CliReporter(self.config, prefix=next_prefix, _shared=self._shared)

    def _format(self, message: str) -> str:
        if not self.prefix:
            return message
        return f"[{self.prefix}] {message}"

    def _write(self, stream: io.TextIOBase, message: str) -> None:
        stream.write(self._format(message).rstrip() + "\n")
        stream.flush()

    def banner(self) -> None:
        if not self.config.show_banner:
            return
        if self._shared["banner_printed"]:
            return
        self._shared["banner_printed"] = True
        self._write(sys.stderr, FLOWCOMPILE_BANNER.rstrip("\n"))
        self._write(sys.stderr, "Pareto-optimal agentic workflow compilation")

    def section(self, title: str) -> None:
        if self.config.quiet:
            return
        self._write(sys.stderr, f"{title}")

    def step(self, message: str) -> None:
        if self.config.quiet:
            return
        self._write(sys.stderr, f"- {message}")

    def detail(self, message: str) -> None:
        if self.config.quiet or not self.config.verbose:
            return
        self._write(sys.stderr, f"  {message}")

    def warn(self, message: str) -> None:
        seen: Dict[str, int] = self._shared["warnings_seen"]  # type: ignore[assignment]
        suppressed: Dict[str, int] = self._shared["warnings_suppressed"]  # type: ignore[assignment]
        normalized = " ".join(str(message).split())
        seen[normalized] = seen.get(normalized, 0) + 1
        if self.config.verbose or seen[normalized] == 1:
            self._write(sys.stderr, f"warning: {message}")
        else:
            suppressed[normalized] = suppressed.get(normalized, 0) + 1

    def error(self, message: str) -> None:
        self._write(sys.stderr, f"error: {message}")

    def success(self, message: str) -> None:
        if self.config.quiet:
            return
        self._write(sys.stderr, f"ok: {message}")

    def summary(self, lines: Iterable[str], *, title: Optional[str] = None) -> None:
        rendered: List[str] = []
        if title:
            rendered.append(title)
        rendered.extend(str(line) for line in lines if str(line).strip())
        if not rendered:
            return
        for line in rendered:
            self._write(sys.stdout, line)

    def flush_warning_summary(self) -> None:
        if self.config.verbose:
            return
        suppressed: Dict[str, int] = self._shared["warnings_suppressed"]  # type: ignore[assignment]
        if not suppressed:
            return
        for message, count in sorted(suppressed.items()):
            self._write(sys.stderr, f"warning: {message} ({count + 1} total)")
        suppressed.clear()

    def progress(
        self,
        iterable: Optional[Iterable] = None,
        *,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: Optional[str] = None,
        leave: bool = False,
    ):
        if iterable is None:
            return ProgressHandle(self, total=total, desc=desc, unit=unit, leave=leave)
        if self.config.live_progress:
            return tqdm(
                iterable,
                total=total,
                desc=desc,
                unit=unit,
                leave=leave,
                file=sys.stderr,
            )
        if desc and not self.config.quiet:
            self.step(desc)
        return iterable

    @contextmanager
    def capture_stdout(self, *, replay_verbose: bool = True) -> Iterator[io.StringIO]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            yield buffer
        output = buffer.getvalue()
        if replay_verbose and self.config.verbose and output.strip():
            for raw_line in output.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if "warning" in line.lower():
                    self.warn(line)
                else:
                    self.detail(line)


def set_reporter(reporter: CliReporter):
    return _reporter_var.set(reporter)


def reset_reporter(token) -> None:
    _reporter_var.reset(token)


def get_reporter() -> CliReporter:
    try:
        return _reporter_var.get()
    except LookupError:
        reporter = CliReporter(_default_config)
        _reporter_var.set(reporter)
        return reporter
