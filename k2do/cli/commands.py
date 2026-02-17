"""CLI commands for K2DO — AI Agent with DeepThink."""

import asyncio
import os
import signal
from pathlib import Path
import select
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from k2do import __version__, __logo__, __name_display__
from k2do.config.schema import Config

app = typer.Typer(
    name="k2do",
    help=f"{__logo__} K2DO - AI Agent with DeepThink",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input helpers
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None


def _flush_pending_tty_input() -> None:
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return
    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass
    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass
    history_file = Path.home() / ".k2do" / "history" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,
    )


# ---------------------------------------------------------------------------
# DeepThink live visualization
# ---------------------------------------------------------------------------

class DeepThinkDisplay:
    """Rich live display for DeepThink multi-agent processing."""

    def __init__(self):
        self.agents: dict[str, str] = {}
        self.live: Live | None = None

    def start(self):
        self.live = Live(self._render(), console=console, refresh_per_second=4)
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()
            self.live = None

    def update(self, agent_name: str, status: str):
        self.agents[agent_name] = status
        if self.live:
            self.live.update(self._render())

    def _render(self):
        table = Table(
            title=f"{__logo__} DeepThink — Multi-Agent Reasoning",
            box=box.DOUBLE_EDGE,
            title_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Agent", style="bold yellow", width=15)
        table.add_column("Status", style="white")

        for name, status in self.agents.items():
            if "Done" in status or "ready" in status:
                style = "green"
            elif "Error" in status:
                style = "red"
            elif "Thinking" in status or "Evaluating" in status:
                style = "cyan bold"
            else:
                style = "dim"
            table.add_row(f"  {name}", Text(status, style=style))

        return Panel(table, border_style="blue")


def _print_agent_response(response: str, render_markdown: bool, route: str = "simple", complexity: float = 0.0) -> None:
    """Render assistant response with K2DO styling."""
    content = response or ""

    # Route indicator
    from k2do.agent.router import get_route_label
    route_label = get_route_label(route)
    if route == "deepthink":
        route_color = "cyan"
    else:
        route_color = "green"

    console.print()
    console.print(Panel(
        f"[bold]{__logo__} K2DO[/bold]  [{route_color}]{route_label}[/{route_color}]  "
        f"[dim]complexity: {complexity:.0%}[/dim]",
        border_style=route_color,
        box=box.ROUNDED,
    ))

    body = Markdown(content) if render_markdown else Text(content)
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansicyan'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} K2DO v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True),
):
    """K2DO - AI Agent with DeepThink."""
    pass


# ============================================================================
# Onboard
# ============================================================================

@app.command()
def onboard():
    """Initialize K2DO configuration and workspace."""
    from k2do.config.loader import get_config_path, load_config, save_config

    config_path = get_config_path()
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        if typer.confirm("Overwrite?"):
            save_config(Config())
            console.print(f"[green]OK[/green] Config reset at {config_path}")
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]OK[/green] Config refreshed at {config_path}")
    else:
        save_config(Config())
        console.print(f"[green]OK[/green] Created config at {config_path}")

    from k2do.utils.helpers import get_workspace_path
    workspace = get_workspace_path()
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]OK[/green] Created workspace at {workspace}")

    _create_workspace_templates(workspace)

    console.print(f"\n{__logo__} K2DO is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your K2 API key to [cyan]~/.k2do/config.json[/cyan]")
    console.print("     (providers.k2Think.apiBase should be [cyan]https://build-api.k2think.ai/v1[/cyan])")
    console.print("  2. Chat: [cyan]k2do agent -m \"Hello!\"[/cyan]")
    console.print("  3. DeepThink: [cyan]k2do agent[/cyan] then type [cyan]/deepthink <question>[/cyan]")


def _create_workspace_templates(workspace: Path):
    templates = {
        "AGENTS.md": """# K2DO Agent Instructions

You are K2DO, an AI agent powered by K2 Think with multi-agent DeepThink.

## Guidelines
- Think step-by-step before taking actions
- Use DeepThink for complex multi-faceted problems
- Remember important information in memory/MEMORY.md
- Be concise, accurate, and helpful
""",
        "SOUL.md": """# K2DO Soul

I am K2DO, an intelligent AI agent with DeepThink capabilities.

## Personality
- Sharp and analytical
- Creative problem solver
- Efficient and direct

## Values
- Correctness through multi-perspective reasoning
- Speed for simple tasks, depth for complex ones
- Transparency in decision-making
""",
        "USER.md": """# User

Information about the user goes here.

## Preferences
- Communication style: (casual/formal)
- Timezone: (your timezone)
- Language: (your preferred language)
""",
    }
    for filename, content in templates.items():
        fp = workspace / filename
        if not fp.exists():
            fp.write_text(content)
            console.print(f"  [dim]Created {filename}[/dim]")

    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    mem_file = memory_dir / "MEMORY.md"
    if not mem_file.exists():
        mem_file.write_text("# Long-term Memory\n\n(Important facts persist here)\n")
        console.print("  [dim]Created memory/MEMORY.md[/dim]")
    hist_file = memory_dir / "HISTORY.md"
    if not hist_file.exists():
        hist_file.write_text("")
        console.print("  [dim]Created memory/HISTORY.md[/dim]")
    skills_dir = workspace / "skills"
    skills_dir.mkdir(exist_ok=True)


def _make_provider(config: Config):
    """Create LLM provider from config (K2 Think + K2 Instruct)."""
    from k2do.providers.litellm_provider import LiteLLMProvider

    model = config.agents.defaults.model or "k2-think-v2/LLM360/K2-Think-V2"
    if "k2-" not in model.lower() and "llm360/k2-" not in model.lower():
        model = "k2-think-v2/LLM360/K2-Think-V2"
    provider_name = config.get_provider_name(model) or "k2_think"

    think_cfg = config.providers.k2_think
    instruct_cfg = config.providers.k2_instruct

    think_key = think_cfg.api_key or instruct_cfg.api_key
    instruct_key = instruct_cfg.api_key or think_cfg.api_key
    think_base = think_cfg.api_base or "https://build-api.k2think.ai/v1"
    instruct_base = instruct_cfg.api_base or think_base

    if not think_key and not instruct_key:
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Set providers.k2Think.apiKey (or providers.k2Instruct.apiKey) in ~/.k2do/config.json")
        raise typer.Exit(1)

    if provider_name == "k2_instruct":
        primary_key = instruct_key or think_key
        primary_base = instruct_base
        primary_headers = instruct_cfg.extra_headers
    else:
        primary_key = think_key or instruct_key
        primary_base = think_base
        primary_headers = think_cfg.extra_headers

    return LiteLLMProvider(
        api_key=primary_key,
        api_base=primary_base,
        default_model=model,
        extra_headers=primary_headers,
        provider_name=provider_name,
        provider_configs={
            "k2_think": {
                "api_key": think_key or instruct_key,
                "api_base": think_base,
                "extra_headers": think_cfg.extra_headers or {},
            },
            "k2_instruct": {
                "api_key": instruct_key or think_key,
                "api_base": instruct_base,
                "extra_headers": instruct_cfg.extra_headers or {},
            },
        },
    )


# ============================================================================
# Gateway
# ============================================================================

@app.command()
def gateway(
    port: int | None = typer.Option(None, "--port", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start K2DO gateway with all channels."""
    from k2do.config.loader import load_config, get_data_dir
    from k2do.bus.queue import MessageBus
    from k2do.agent.loop import AgentLoop
    from k2do.channels.manager import ChannelManager
    from k2do.session.manager import SessionManager
    from k2do.cron.service import CronService
    from k2do.cron.types import CronJob
    from k2do.heartbeat.service import HeartbeatService

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    config = load_config()
    host = config.gateway.host
    effective_port = port if port is not None else config.gateway.port

    console.print(Panel(
        f"[bold cyan]{__logo__} K2DO Gateway[/bold cyan] health on {host}:{effective_port}",
        border_style="cyan",
    ))

    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)

    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    agent = AgentLoop(
        bus=bus, provider=provider, workspace=config.workspace_path,
        model=config.agents.defaults.model,
        fallback_model=config.agents.defaults.fallback_model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        context_max_chars=config.agents.defaults.context_max_chars,
        simple_route_max_complexity=config.agents.defaults.simple_route_max_complexity,
        store_reasoning=config.agents.defaults.store_reasoning,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec, cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        deepthink_enabled=config.agents.deepthink.enabled,
        complexity_threshold=config.agents.deepthink.complexity_threshold,
        deepthink_agents=[
            {
                "name": a.name,
                "model": a.model,
                "temperature": a.temperature,
                "role_prompt": a.role_prompt,
            }
            for a in config.agents.deepthink.agents
        ] if config.agents.deepthink.agents else None,
        deepthink_max_agents=config.agents.deepthink.max_agents,
        deepthink_judge_model=config.agents.deepthink.judge_model or None,
        deepthink_thinker_timeout_s=config.agents.deepthink.thinker_timeout_s,
        deepthink_judge_timeout_s=config.agents.deepthink.judge_timeout_s,
    )

    async def on_cron_job(job: CronJob) -> str | None:
        response = await agent.process_direct(
            job.payload.message, session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli", chat_id=job.payload.to or "direct",
        )
        if job.payload.deliver and job.payload.to:
            from k2do.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli", chat_id=job.payload.to, content=response or ""
            ))
        return response
    cron.on_job = on_cron_job

    async def on_heartbeat(prompt: str) -> str:
        return await agent.process_direct(prompt, session_key="heartbeat")

    heartbeat = HeartbeatService(
        workspace=config.workspace_path, on_heartbeat=on_heartbeat,
        interval_s=30 * 60, enabled=True,
    )

    channels = ChannelManager(config, bus)
    if channels.enabled_channels:
        console.print(f"[green]OK[/green] Channels: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]No channels enabled[/yellow]")

    console.print(f"[green]OK[/green] DeepThink: {'enabled' if config.agents.deepthink.enabled else 'disabled'}")
    console.print(f"[green]OK[/green] Heartbeat: every 30m")

    async def _health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            # Read and ignore request bytes; return basic health response.
            await reader.read(2048)
            body = "ok\n"
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/plain; charset=utf-8\r\n"
                f"Content-Length: {len(body.encode('utf-8'))}\r\n"
                "Connection: close\r\n"
                "\r\n"
                f"{body}"
            )
            writer.write(response.encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def run():
        health_server: asyncio.AbstractServer | None = None
        try:
            health_server = await asyncio.start_server(
                _health_handler,
                host=host,
                port=effective_port,
            )
            console.print(f"[green]OK[/green] Health endpoint: http://{host}:{effective_port}/")
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(agent.run(), channels.start_all())
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            if health_server:
                health_server.close()
                await health_server.wait_closed()
            await agent.close_mcp()
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())


# ============================================================================
# Agent (interactive + single-message)
# ============================================================================

@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send"),
    session_id: str = typer.Option("cli:direct", "--session", "-s"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs"),
    no_deepthink: bool = typer.Option(False, "--no-deepthink", help="Disable DeepThink auto-routing"),
    dashboard_mode: bool = typer.Option(False, "--dashboard", "-d", help="Full-screen dashboard mode"),
):
    """Chat with K2DO agent."""
    from k2do.config.loader import load_config
    from k2do.bus.queue import MessageBus
    from k2do.agent.loop import AgentLoop
    from k2do.cli.dashboard import Dashboard, DashboardState
    from loguru import logger
    import time as _time

    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)

    if logs:
        logger.enable("k2do")
    else:
        logger.disable("k2do")

    # Dashboard state (shared between callbacks)
    dash_state = DashboardState()
    dash: Dashboard | None = None

    # DeepThink display for non-dashboard mode
    dt_display: DeepThinkDisplay | None = None

    def on_deepthink_progress(agent_name: str, status: str):
        nonlocal dt_display
        dash_state.update_agent(agent_name, status)
        if dash:
            dash.refresh()
        else:
            if dt_display is None:
                dt_display = DeepThinkDisplay()
                dt_display.start()
            dt_display.update(agent_name, status)

    def on_refine_round(round_num: int, role: str, status: str):
        dash_state.add_refine_round(round_num, role, status)
        if dash:
            dash.refresh()
        else:
            # Print inline for non-dashboard mode
            style = {"initial": "cyan", "critic": "yellow", "refiner": "green"}.get(role, "white")
            console.print(f"  [{style}]Round {round_num} ({role}): {status}[/{style}]")

    agent_loop = AgentLoop(
        bus=bus, provider=provider, workspace=config.workspace_path,
        model=config.agents.defaults.model,
        fallback_model=config.agents.defaults.fallback_model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        context_max_chars=config.agents.defaults.context_max_chars,
        simple_route_max_complexity=config.agents.defaults.simple_route_max_complexity,
        store_reasoning=config.agents.defaults.store_reasoning,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        deepthink_enabled=not no_deepthink and config.agents.deepthink.enabled,
        complexity_threshold=config.agents.deepthink.complexity_threshold,
        deepthink_agents=[
            {
                "name": a.name,
                "model": a.model,
                "temperature": a.temperature,
                "role_prompt": a.role_prompt,
            }
            for a in config.agents.deepthink.agents
        ] if config.agents.deepthink.agents else None,
        deepthink_max_agents=config.agents.deepthink.max_agents,
        deepthink_judge_model=config.agents.deepthink.judge_model or None,
        deepthink_thinker_timeout_s=config.agents.deepthink.thinker_timeout_s,
        deepthink_judge_timeout_s=config.agents.deepthink.judge_timeout_s,
        on_deepthink_progress=on_deepthink_progress,
        on_refine_round=on_refine_round,
    )

    def _thinking_ctx():
        if logs or dashboard_mode:
            from contextlib import nullcontext
            return nullcontext()
        return console.status(f"[dim]{__logo__} K2DO is thinking...[/dim]", spinner="dots")

    def _finish_overlays():
        nonlocal dt_display
        if dt_display:
            dt_display.stop()
            dt_display = None

    if message:
        async def run_once():
            start_t = _time.monotonic()
            with _thinking_ctx():
                response = await agent_loop.process_direct(message, session_id)
            _finish_overlays()
            duration = (_time.monotonic() - start_t) * 1000
            _print_agent_response(response, markdown, agent_loop._last_route, agent_loop._last_complexity)
            await agent_loop.close_mcp()
        asyncio.run(run_once())
    else:
        _init_prompt_session()

        if dashboard_mode:
            # Full-screen dashboard
            dash = Dashboard(dash_state)
            dash.start()
        else:
            console.print(Panel(
                f"[bold cyan]{__logo__} K2DO[/bold cyan] Interactive Mode\n"
                f"[dim]DeepThink: {'ON' if not no_deepthink else 'OFF'} | "
                f"[bold]/deepthink[/bold] multi-agent | "
                f"[bold]/refine[/bold] 3-round refinement | "
                f"[bold]exit[/bold] to quit[/dim]",
                border_style="cyan",
            ))

        def _exit_on_sigint(signum, frame):
            if dash:
                dash.stop()
            _restore_terminal()
            console.print("\nGoodbye!")
            os._exit(0)
        signal.signal(signal.SIGINT, _exit_on_sigint)

        async def run_interactive():
            nonlocal dt_display
            try:
                while True:
                    try:
                        _flush_pending_tty_input()

                        if dashboard_mode and dash:
                            # Temporarily stop dashboard for input
                            dash.stop()

                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()

                        if dashboard_mode and dash:
                            dash.start()

                        if not command:
                            continue
                        if _is_exit_command(command):
                            if dash:
                                dash.stop()
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        # Update dashboard state
                        route = "simple"
                        from k2do.agent.router import classify_query, compute_complexity
                        if command.startswith("/deepthink"):
                            route = "deepthink"
                        elif command.startswith("/refine"):
                            route = "refine"
                        else:
                            route = classify_query(command)
                        dash_state.set_mode(route, command[:60])
                        if dash:
                            dash.refresh()

                        start_t = _time.monotonic()
                        with _thinking_ctx():
                            response = await agent_loop.process_direct(user_input, session_id)
                        duration = (_time.monotonic() - start_t) * 1000
                        _finish_overlays()

                        # Update dashboard history
                        dash_state.finish_request(
                            command, response[:40] if response else "",
                            duration, agent_loop._last_route,
                        )
                        if dash:
                            dash.refresh()

                        if not dashboard_mode:
                            _print_agent_response(response, markdown, agent_loop._last_route, agent_loop._last_complexity)
                        else:
                            # In dashboard mode, show response briefly
                            if dash:
                                dash.stop()
                            _print_agent_response(response, markdown, agent_loop._last_route, agent_loop._last_complexity)
                            if dash:
                                dash.start()

                    except KeyboardInterrupt:
                        if dash:
                            dash.stop()
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        if dash:
                            dash.stop()
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                if dash:
                    dash.stop()
                await agent_loop.close_mcp()
        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================

channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")

@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from k2do.config.loader import load_config
    config = load_config()

    table = Table(title=f"{__logo__} Channel Status", box=box.ROUNDED, border_style="cyan")
    table.add_column("Channel", style="cyan bold")
    table.add_column("Enabled", style="green")
    table.add_column("Config", style="yellow")

    tg = config.channels.telegram
    tg_cfg = f"token: {tg.token[:10]}..." if tg.token else "[dim]not set[/dim]"
    table.add_row("Telegram", "ON" if tg.enabled else "[dim]OFF[/dim]", tg_cfg)

    console.print(table)


# ============================================================================
# Status
# ============================================================================

@app.command()
def status():
    """Show K2DO status."""
    from k2do.config.loader import load_config, get_config_path

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(Panel(
        f"[bold cyan]{__logo__} K2DO Status[/bold cyan]",
        border_style="cyan",
    ))

    console.print(f"Config: {config_path} {'[green]OK[/green]' if config_path.exists() else '[red]MISSING[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]OK[/green]' if workspace.exists() else '[red]MISSING[/red]'}")
    console.print(f"Model: [cyan]{config.agents.defaults.model}[/cyan]")
    console.print(f"Fallback: [cyan]{config.agents.defaults.fallback_model}[/cyan]")
    console.print(f"DeepThink: {'[green]ON[/green]' if config.agents.deepthink.enabled else '[dim]OFF[/dim]'}")
    console.print(f"Complexity threshold: {config.agents.deepthink.complexity_threshold:.0%}")

    console.print("\n[bold]Providers:[/bold]")
    from k2do.providers.registry import PROVIDERS
    for spec in PROVIDERS:
        p = getattr(config.providers, spec.name, None)
        if p is None:
            continue
        if spec.is_local:
            if p.api_base:
                console.print(f"  {spec.label}: [green]OK {p.api_base}[/green]")
            else:
                console.print(f"  {spec.label}: [dim]not set[/dim]")
        else:
            has_key = bool(p.api_key)
            console.print(f"  {spec.label}: {'[green]OK[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# Cron Commands
# ============================================================================

cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(all: bool = typer.Option(False, "--all", "-a")):
    """List scheduled jobs."""
    from k2do.config.loader import get_data_dir
    from k2do.cron.service import CronService

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    jobs = service.list_jobs(include_disabled=all)
    if not jobs:
        console.print("No scheduled jobs.")
        return
    table = Table(title="Scheduled Jobs", box=box.ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("Status")
    for job in jobs:
        if job.schedule.kind == "every":
            sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
        elif job.schedule.kind == "cron":
            sched = job.schedule.expr or ""
        else:
            sched = "one-time"
        st = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"
        table.add_row(job.id, job.name, sched, st)
    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n"),
    message: str = typer.Option(..., "--message", "-m"),
    every: int = typer.Option(None, "--every", "-e"),
    cron_expr: str = typer.Option(None, "--cron", "-c"),
    tz: str | None = typer.Option(None, "--tz"),
    at: str = typer.Option(None, "--at"),
    deliver: bool = typer.Option(False, "--deliver", "-d"),
    to: str = typer.Option(None, "--to"),
    channel: str = typer.Option(None, "--channel"),
):
    """Add a scheduled job."""
    from k2do.config.loader import get_data_dir
    from k2do.cron.service import CronService
    from k2do.cron.types import CronSchedule

    if every:
        if every <= 0:
            console.print("[red]--every must be > 0[/red]")
            raise typer.Exit(1)
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    job = service.add_job(name=name, schedule=schedule, message=message, deliver=deliver, to=to, channel=channel)
    console.print(f"[green]OK[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(job_id: str = typer.Argument(...)):
    """Remove a scheduled job."""
    from k2do.config.loader import get_data_dir
    from k2do.cron.service import CronService
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    if service.remove_job(job_id):
        console.print(f"[green]OK[/green] Removed {job_id}")
    else:
        console.print(f"[red]Not found: {job_id}[/red]")


if __name__ == "__main__":
    app()
