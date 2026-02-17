"""
K2DO Terminal Dashboard — full-screen cockpit mode.

A live-updating terminal UI showing:
- Agent status and active processing
- DeepThink / Refinement progress
- Thinking stream (live chain-of-thought)
- Memory usage stats
- Request history with timing
- Response speed metrics
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Any

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from k2do import __logo__, __name_display__, __version__


# ── Sparkline: tiny ASCII bar chart ────────────────────────────
def _sparkline(values: list[float], width: int = 20) -> str:
    """Render a tiny sparkline bar from numeric values."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    tail = values[-width:] if len(values) > width else values
    return "".join(blocks[min(int((v - mn) / rng * 8), 8)] for v in tail)


class DashboardState:
    """Shared state that the dashboard renders."""

    def __init__(self):
        # Agent status
        self.mode: str = "idle"  # idle, simple, deepthink, refine
        self.current_query: str = ""
        self.active_agents: dict[str, str] = {}  # name -> status

        # Thinking stream
        self.thinking_lines: deque[str] = deque(maxlen=12)

        # Refinement rounds
        self.refine_rounds: list[dict[str, Any]] = []

        # History
        self.history: deque[dict[str, Any]] = deque(maxlen=20)

        # Stats
        self.total_requests: int = 0
        self.response_times_ms: list[float] = []
        self.deepthink_count: int = 0
        self.refine_count: int = 0
        self.start_time: float = time.monotonic()

    # ── Update methods (called from agent/CLI) ──

    def set_mode(self, mode: str, query: str = ""):
        self.mode = mode
        self.current_query = query
        if mode != "idle":
            self.active_agents.clear()

    def update_agent(self, name: str, status: str):
        self.active_agents[name] = status
        self.thinking_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {name}: {status}")

    def add_thinking(self, text: str):
        self.thinking_lines.append(text)

    def add_refine_round(self, round_num: int, role: str, status: str):
        while len(self.refine_rounds) < round_num:
            self.refine_rounds.append({})
        self.refine_rounds[round_num - 1] = {"role": role, "status": status, "round": round_num}
        self.thinking_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round_num} ({role}): {status}")

    def finish_request(self, query: str, response_preview: str, duration_ms: float, mode: str):
        self.total_requests += 1
        self.response_times_ms.append(duration_ms)
        if mode == "deepthink":
            self.deepthink_count += 1
        elif mode == "refine":
            self.refine_count += 1
        self.history.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "query": query[:40],
            "response": response_preview[:40],
            "duration_ms": int(duration_ms),
            "mode": mode,
        })
        self.mode = "idle"
        self.current_query = ""
        self.active_agents.clear()
        self.refine_rounds.clear()


class Dashboard:
    """Full-screen Rich dashboard with live updates."""

    def __init__(self, state: DashboardState | None = None):
        self.state = state or DashboardState()
        self.console = Console()
        self.live: Live | None = None

    def start(self):
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()
            self.live = None

    def refresh(self):
        if self.live:
            self.live.update(self._render())

    def _render(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()

        # Top: header
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(self._render_header())
        layout["footer"].update(self._render_footer())

        # Body: left (agents + refine) | right (thinking + history)
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        # Left: agents panel + refinement progress
        layout["left"].split_column(
            Layout(name="agents", ratio=2),
            Layout(name="refine", ratio=1),
        )

        # Right: thinking stream + history
        layout["right"].split_column(
            Layout(name="thinking", ratio=2),
            Layout(name="history", ratio=1),
        )

        layout["agents"].update(self._render_agents())
        layout["refine"].update(self._render_refine())
        layout["thinking"].update(self._render_thinking())
        layout["history"].update(self._render_history())

        return layout

    def _render_header(self) -> Panel:
        uptime = int(time.monotonic() - self.state.start_time)
        mins, secs = divmod(uptime, 60)
        hrs, mins = divmod(mins, 60)

        mode_colors = {
            "idle": "dim",
            "simple": "green",
            "deepthink": "cyan bold",
            "refine": "magenta bold",
        }
        mode_style = mode_colors.get(self.state.mode, "white")
        mode_label = self.state.mode.upper()

        header = Text.assemble(
            (f" {__logo__} K2DO DASHBOARD ", "bold white on blue"),
            " ",
            (f" {mode_label} ", f"bold white on {mode_style.split()[0] if mode_style != 'dim' else 'black'}"),
            " ",
            (f" v{__version__} ", "dim"),
            "  ",
            (f"uptime {hrs:02d}:{mins:02d}:{secs:02d}", "dim"),
            "  ",
            (f"requests: {self.state.total_requests}", "dim"),
        )
        return Panel(Align.center(header), style="blue", box=box.HEAVY)

    def _render_footer(self) -> Panel:
        # Speed sparkline
        spark = _sparkline(self.state.response_times_ms)
        avg_ms = int(sum(self.state.response_times_ms) / len(self.state.response_times_ms)) if self.state.response_times_ms else 0

        footer = Text.assemble(
            ("Speed: ", "dim"),
            (spark, "green"),
            (f"  avg: {avg_ms}ms", "dim"),
            ("  |  ", "dim"),
            (f"DeepThink: {self.state.deepthink_count}", "cyan"),
            ("  |  ", "dim"),
            (f"Refine: {self.state.refine_count}", "magenta"),
            ("  |  ", "dim"),
            ("Ctrl+C to exit", "dim"),
        )
        return Panel(Align.center(footer), style="blue", box=box.HEAVY)

    def _render_agents(self) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True, header_style="bold")
        table.add_column("Agent", style="bold yellow", ratio=1)
        table.add_column("Status", ratio=2)

        if self.state.mode == "idle":
            table.add_row("[dim]---[/dim]", "[dim]Waiting for query...[/dim]")
        else:
            if self.state.current_query:
                table.add_row(
                    "[bold white]Query[/bold white]",
                    Text(self.state.current_query[:60] + ("..." if len(self.state.current_query) > 60 else ""), style="white"),
                )

            for name, status in self.state.active_agents.items():
                if "Done" in status or "ready" in status:
                    style = "green"
                    icon = "OK"
                elif "Error" in status:
                    style = "red"
                    icon = "!!"
                elif "Thinking" in status or "Generating" in status or "Analyzing" in status or "Creating" in status:
                    style = "cyan"
                    icon = ">>"
                elif "Evaluating" in status:
                    style = "yellow"
                    icon = ">>"
                else:
                    style = "dim"
                    icon = ".."
                table.add_row(
                    f"[{style}]{icon}[/{style}] {name}",
                    Text(status, style=style),
                )

        return Panel(table, title="[bold yellow]Active Agents[/bold yellow]", border_style="yellow", box=box.ROUNDED)

    def _render_refine(self) -> Panel:
        if not self.state.refine_rounds and self.state.mode != "refine":
            content = Text("No active refinement", style="dim", justify="center")
            return Panel(content, title="[bold magenta]Refinement Progress[/bold magenta]", border_style="magenta", box=box.ROUNDED)

        table = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold")
        table.add_column("Round", width=8, style="bold")
        table.add_column("Role", width=10)
        table.add_column("Status", ratio=1)

        round_icons = {1: "R1", 2: "R2", 3: "R3"}
        role_colors = {"initial": "cyan", "critic": "yellow", "refiner": "green"}

        for r in self.state.refine_rounds:
            rnum = r.get("round", 0)
            role = r.get("role", "")
            status = r.get("status", "")
            color = role_colors.get(role, "white")
            done = "Done" in status

            table.add_row(
                f"[bold]{round_icons.get(rnum, '??')}[/bold]",
                Text(role.title(), style=f"bold {color}"),
                Text(status, style="green" if done else color),
            )

        # Fill empty rounds
        for i in range(len(self.state.refine_rounds) + 1, 4):
            table.add_row(
                f"[dim]{round_icons.get(i, '??')}[/dim]",
                Text("---", style="dim"),
                Text("pending", style="dim"),
            )

        return Panel(table, title="[bold magenta]Refinement Rounds[/bold magenta]", border_style="magenta", box=box.ROUNDED)

    def _render_thinking(self) -> Panel:
        lines = list(self.state.thinking_lines)
        if not lines:
            lines = ["[dim]Waiting for agent activity...[/dim]"]

        text_parts = []
        for line in lines:
            if "Done" in line or "ready" in line:
                text_parts.append(Text(line, style="green"))
            elif "Error" in line:
                text_parts.append(Text(line, style="red"))
            elif "Thinking" in line or "Generating" in line:
                text_parts.append(Text(line, style="cyan"))
            elif "Round" in line:
                text_parts.append(Text(line, style="magenta"))
            else:
                text_parts.append(Text(line, style="dim"))

        content = Group(*text_parts)
        return Panel(
            content,
            title="[bold cyan]Live Thinking Stream[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def _render_history(self) -> Panel:
        table = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold dim")
        table.add_column("Time", width=8, style="dim")
        table.add_column("Mode", width=6)
        table.add_column("Query", ratio=2)
        table.add_column("ms", width=6, justify="right")

        if not self.state.history:
            table.add_row("[dim]---[/dim]", "", "[dim]No requests yet[/dim]", "")
        else:
            for entry in list(self.state.history)[:8]:
                mode = entry.get("mode", "?")
                mode_style = {"simple": "green", "deepthink": "cyan", "refine": "magenta"}.get(mode, "dim")
                table.add_row(
                    entry.get("time", ""),
                    Text(mode[:5], style=mode_style),
                    entry.get("query", "")[:35],
                    str(entry.get("duration_ms", "")),
                )

        return Panel(
            table,
            title="[bold green]Request History[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
