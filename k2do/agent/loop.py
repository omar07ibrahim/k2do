"""K2DO Agent Loop — core processing engine with DeepThink integration."""

import asyncio
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from k2do.bus.events import InboundMessage, OutboundMessage
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider
from k2do.agent.context import ContextBuilder
from k2do.agent.tools.registry import ToolRegistry
from k2do.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from k2do.agent.tools.shell import ExecTool
from k2do.agent.tools.web import WebSearchTool, WebFetchTool
from k2do.agent.tools.message import MessageTool
from k2do.agent.tools.spawn import SpawnTool
from k2do.agent.tools.cron import CronTool
from k2do.agent.tools.deepthink_tool import DeepThinkTool, RefineTool
from k2do.agent.memory import MemoryStore
from k2do.agent.subagent import SubagentManager
from k2do.agent.router import classify_query, compute_complexity
from k2do.agent.deepthink import DeepThinkEngine, DeepThinkResult
from k2do.agent.refine import RefinementEngine, RefinementResult
from k2do.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from k2do.config.schema import ExecToolConfig
    from k2do.cron.service import CronService


_INLINE_TOOL_CALL_RE = re.compile(
    r"<\s*tool_call\s*>(?P<body>.*?)</\s*tool_call\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ACTION_HINTS = (
    "create",
    "write",
    "run",
    "execute",
    "generate",
    "save",
    "make",
    "file",
    "script",
    "code",
    "build",
    "создай",
    "сделай",
    "сдела",
    "сгенерируй",
    "сгенерир",
    "запусти",
    "запуст",
    "выполни",
    "выполн",
    "файл",
    "скрипт",
    "код",
)
_PLAN_RESPONSE_HINTS = (
    "next steps",
    "immediate next action",
    "reflection",
    "plan",
    "краткий план",
    "следующие шаги",
    "следующий шаг",
    "план",
)


class AgentLoop:
    """
    K2DO Agent Loop with Smart Routing and DeepThink.

    Flow:
    1. Receive message
    2. Router classifies: simple or complex?
    3. Simple → standard agent loop (fast)
    4. Complex → DeepThink (parallel agents + judge)
    5. Send response
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        fallback_model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        context_max_chars: int = 24000,
        simple_route_max_complexity: float = 0.2,
        store_reasoning: bool = False,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        deepthink_enabled: bool = True,
        complexity_threshold: float = 0.6,
        deepthink_agents: list[dict[str, Any]] | None = None,
        deepthink_max_agents: int = 3,
        deepthink_judge_model: str | None = None,
        deepthink_thinker_timeout_s: float = 90.0,
        deepthink_judge_timeout_s: float = 60.0,
        on_deepthink_progress: Callable[[str, str], None] | None = None,
        on_refine_round: Callable[[int, str, str], None] | None = None,
    ):
        from k2do.config.schema import ExecToolConfig
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.think_model = model or provider.get_default_model()
        self.simple_model = fallback_model or self.think_model
        self.model = self.think_model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.context_max_chars = context_max_chars
        self.simple_route_max_complexity = simple_route_max_complexity
        self.store_reasoning = store_reasoning
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(
            workspace,
            store_reasoning=store_reasoning,
            max_context_chars=context_max_chars,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider, workspace=workspace, bus=bus,
            model=self.think_model, fallback_model=self.simple_model, temperature=self.temperature,
            max_tokens=self.max_tokens, brave_api_key=brave_api_key,
            exec_config=self.exec_config, restrict_to_workspace=restrict_to_workspace,
        )

        # DeepThink + Refinement
        self.deepthink_enabled = deepthink_enabled
        self.complexity_threshold = complexity_threshold
        self.on_deepthink_progress = on_deepthink_progress
        self.on_refine_round = on_refine_round
        self.deepthink = DeepThinkEngine(
            chat_fn=provider.chat,
            model=self.think_model,
            fallback_model=self.simple_model,
            thinkers=deepthink_agents,
            judge_model=deepthink_judge_model or self.think_model,
            max_agents=deepthink_max_agents,
            thinker_timeout_s=deepthink_thinker_timeout_s,
            judge_timeout_s=deepthink_judge_timeout_s,
        )
        self.refiner = RefinementEngine(
            chat_fn=provider.chat,
            model=self.think_model,
            critic_model=deepthink_judge_model or self.think_model,
            refiner_model=self.think_model,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._last_route: str = "simple"
        self._last_complexity: float = 0.0
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._consolidation_tasks: dict[str, asyncio.Task[None]] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            workspace_root=str(self.workspace),
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # DeepThink + Refine tools (agent can call these autonomously)
        self._deepthink_tool = DeepThinkTool(engine=self.deepthink)
        self._deepthink_tool.set_progress_callback(self.on_deepthink_progress)
        self.tools.register(self._deepthink_tool)

        self._refine_tool = RefineTool(engine=self.refiner)
        self._refine_tool.set_round_callback(self.on_refine_round)
        self.tools.register(self._refine_tool)

    async def _connect_mcp(self) -> None:
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from k2do.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)
        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)
        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        model: str | None = None,
        require_concrete_tools: bool = False,
        allow_reasoning_tools: bool = True,
        max_reasoning_tool_calls: int = 3,
        reflect_mode: bool = False,
    ) -> tuple[str | None, list[str]]:
        """Standard agent loop (for simple queries)."""
        # Pass system prompt to deepthink/refine tools so sub-agents have context
        if initial_messages and initial_messages[0].get("role") == "system":
            sys_prompt = initial_messages[0].get("content", "")
            self._deepthink_tool.set_system_prompt(sys_prompt)
            self._refine_tool.set_system_prompt(sys_prompt)

        def _filter_reasoning_tools(defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                d for d in defs
                if d.get("function", {}).get("name") not in {"deepthink", "refine"}
            ]

        base_tool_definitions = self.tools.get_definitions()
        tool_definitions = (
            base_tool_definitions
            if allow_reasoning_tools
            else _filter_reasoning_tools(base_tool_definitions)
        )
        reasoning_tool_calls = 0
        concrete_tools_since_reasoning = True

        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        concrete_tools_prompted = False

        while iteration < self.max_iterations:
            iteration += 1
            response = await self._chat_with_fallback(
                messages=messages,
                model=model or self.think_model,
                tools=tool_definitions,
            )
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                for tool_call in response.tool_calls:
                    is_reasoning_tool = tool_call.name in {"deepthink", "refine"}
                    if (
                        (not allow_reasoning_tools and is_reasoning_tool)
                        or (
                            is_reasoning_tool
                            and reasoning_tool_calls >= max(1, max_reasoning_tool_calls)
                        )
                        or (
                            is_reasoning_tool
                            and reasoning_tool_calls > 0
                            and not concrete_tools_since_reasoning
                        )
                    ):
                        if not allow_reasoning_tools:
                            err_text = (
                                f"Error: Tool '{tool_call.name}' is disabled for this phase. "
                                "Execute concrete tools (write_file/edit_file/exec/message) instead."
                            )
                        elif reasoning_tool_calls >= max(1, max_reasoning_tool_calls):
                            err_text = (
                                f"Error: Tool '{tool_call.name}' limit reached for this request. "
                                "Continue with concrete tools and finish the task."
                            )
                        else:
                            err_text = (
                                f"Error: Tool '{tool_call.name}' can be used again only after at least one "
                                "concrete tool action (write_file/edit_file/exec/message)."
                            )
                        messages = self.context.add_tool_result(
                            messages,
                            tool_call.id,
                            tool_call.name,
                            err_text,
                        )
                        continue
                    if is_reasoning_tool:
                        reasoning_tool_calls += 1
                        concrete_tools_since_reasoning = False
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    if not is_reasoning_tool:
                        concrete_tools_since_reasoning = True
                if reflect_mode:
                    messages.append({
                        "role": "user",
                        "content": "Reflect on the results and decide next steps.",
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Use the tool results to continue execution. "
                            "If the task is complete, return the final answer now. "
                            "Do not output planning/reflection sections."
                        ),
                    })
            else:
                inline_tool_call = self._extract_inline_tool_call(response.content)
                if inline_tool_call:
                    tool_name, tool_args = inline_tool_call
                    is_reasoning_tool = tool_name in {"deepthink", "refine"}
                    if (
                        (not allow_reasoning_tools and is_reasoning_tool)
                        or (
                            is_reasoning_tool
                            and reasoning_tool_calls >= max(1, max_reasoning_tool_calls)
                        )
                        or (
                            is_reasoning_tool
                            and reasoning_tool_calls > 0
                            and not concrete_tools_since_reasoning
                        )
                    ):
                        if not allow_reasoning_tools:
                            err_text = (
                                f"Error: Tool '{tool_name}' is disabled for this phase. "
                                "Execute concrete tools (write_file/edit_file/exec/message) instead."
                            )
                        elif reasoning_tool_calls >= max(1, max_reasoning_tool_calls):
                            err_text = (
                                f"Error: Tool '{tool_name}' limit reached for this request. "
                                "Continue with concrete tools and finish the task."
                            )
                        else:
                            err_text = (
                                f"Error: Tool '{tool_name}' can be used again only after at least one "
                                "concrete tool action (write_file/edit_file/exec/message)."
                            )
                        messages = self.context.add_assistant_message(
                            messages, response.content,
                            reasoning_content=response.reasoning_content,
                        )
                        messages = self.context.add_tool_result(
                            messages,
                            f"inline_{iteration}_{tool_name}",
                            tool_name,
                            err_text,
                        )
                        messages.append({
                            "role": "user",
                            "content": "Use concrete tools now and continue.",
                        })
                        continue
                    if is_reasoning_tool:
                        reasoning_tool_calls += 1
                        concrete_tools_since_reasoning = False
                    tools_used.append(tool_name)
                    args_str = json.dumps(tool_args, ensure_ascii=False)
                    logger.info(f"Inline tool call: {tool_name}({args_str[:200]})")
                    messages = self.context.add_assistant_message(
                        messages, response.content,
                        reasoning_content=response.reasoning_content,
                    )
                    result = await self.tools.execute(tool_name, tool_args)
                    messages = self.context.add_tool_result(
                        messages,
                        f"inline_{iteration}_{tool_name}",
                        tool_name,
                        result,
                    )
                    if not is_reasoning_tool:
                        concrete_tools_since_reasoning = True
                    if reflect_mode:
                        messages.append({
                            "role": "user",
                            "content": "Reflect on the tool result and continue until the task is complete.",
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": (
                                "Use the tool result and continue execution until complete. "
                                "When complete, return only the final answer without planning/reflection sections."
                            ),
                        })
                    continue

                if require_concrete_tools and not concrete_tools_prompted:
                    concrete_tools = [
                        t for t in tools_used
                        if t not in {"deepthink", "refine"}
                    ]
                    if not concrete_tools:
                        concrete_tools_prompted = True
                        messages = self.context.add_assistant_message(
                            messages,
                            response.content,
                            reasoning_content=response.reasoning_content,
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                "This is an action request. You must execute concrete tools "
                                "(filesystem/shell/message) and verify results before final answer. "
                                "Do not stop at planning/reflection."
                            ),
                        })
                        continue

                # If DeepThink was used but the model only produced a plan/
                # reflection (no concrete tools), force one execution pass.
                if (
                    not concrete_tools_prompted
                    and "deepthink" in tools_used
                    and self._looks_like_plan_response(response.content)
                    and not any(t not in {"deepthink", "refine"} for t in tools_used)
                ):
                    concrete_tools_prompted = True
                    messages = self.context.add_assistant_message(
                        messages,
                        response.content,
                        reasoning_content=response.reasoning_content,
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "DeepThink planning is complete. Now execute the plan with concrete tools "
                            "(write_file/edit_file/exec/message), verify outputs, then return final result."
                        ),
                    })
                    continue

                final_content = response.content
                break
        return final_content, tools_used

    @staticmethod
    def _extract_inline_tool_call(content: str | None) -> tuple[str, dict[str, Any]] | None:
        """Parse textual <tool_call>{...}</tool_call> fallback into a real tool call."""
        if not content:
            return None

        match = _INLINE_TOOL_CALL_RE.search(content)
        if match:
            body = match.group("body").strip()
        else:
            body = content.strip()

            # Tolerate malformed wrapper variants only when tool_call tags are
            # explicitly present.
            has_tool_tags = "<tool_call" in body.lower() or "</tool_call>" in body.lower()
            if has_tool_tags:
                body = re.sub(r"^\s*<\s*tool_call\s*>\s*", "", body, flags=re.IGNORECASE)
                body = re.sub(r"\s*</\s*tool_call\s*>\s*$", "", body, flags=re.IGNORECASE)
            else:
                # Strict fallback: only parse if the whole response is a JSON object,
                # not arbitrary text containing JSON snippets.
                if body.startswith("```"):
                    body = body.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                if not (body.startswith("{") and body.endswith("}")):
                    return None

        if not body:
            return None

        try:
            parsed = json_repair.loads(body)
        except Exception:
            return None

        if not isinstance(parsed, dict):
            return None

        tool_name = parsed.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return None

        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json_repair.loads(arguments)
            except Exception:
                arguments = {}

        if not isinstance(arguments, dict):
            arguments = {}

        return tool_name.strip(), arguments

    async def _chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Try the requested model first, then one fallback model if available."""
        model_order: list[str] = [model]
        if model != self.simple_model:
            model_order.append(self.simple_model)
        if model != self.think_model:
            model_order.append(self.think_model)

        # Deduplicate while keeping order.
        unique_models: list[str] = []
        for m in model_order:
            if m and m not in unique_models:
                unique_models.append(m)

        last_error: Exception | None = None
        retries_per_model = 2
        for idx, selected_model in enumerate(unique_models):
            for attempt in range(1, retries_per_model + 1):
                try:
                    return await self.provider.chat(
                        messages=messages,
                        tools=tools if tools is not None else self.tools.get_definitions(),
                        model=selected_model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                except Exception as e:
                    last_error = e
                    if attempt < retries_per_model:
                        await asyncio.sleep(0.35 * attempt)
                        continue

                    if idx < len(unique_models) - 1:
                        logger.warning(
                            f"Model call failed on {selected_model}, retrying with fallback: {e}"
                        )
                    else:
                        logger.error(f"Model call failed on {selected_model}: {e}")

        raise RuntimeError("Model backend unavailable after retry") from last_error

    async def _run_deepthink(self, query: str, system_prompt: str) -> DeepThinkResult:
        """Run DeepThink multi-agent reasoning."""
        return await self.deepthink.think(
            query=query,
            system_prompt=system_prompt,
            on_progress=self.on_deepthink_progress,
        )

    async def run(self) -> None:
        self._running = True
        await self._connect_mcp()
        logger.info("K2DO Agent loop started")
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Sorry, I hit an internal error while processing your request."
                    ))
            except asyncio.TimeoutError:
                continue

    async def close_mcp(self) -> None:
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass
            self._mcp_stack = None

    def _schedule_consolidation(self, session: Session, archive_all: bool = False) -> None:
        """Schedule background memory consolidation, deduplicated per session."""
        task_key = f"{session.key}::archive" if archive_all else session.key
        existing = self._consolidation_tasks.get(task_key)
        if existing and not existing.done():
            return

        async def _runner() -> None:
            try:
                await self._consolidate_memory(session, archive_all=archive_all)
            finally:
                self._consolidation_tasks.pop(task_key, None)

        self._consolidation_tasks[task_key] = asyncio.create_task(_runner())

    def stop(self) -> None:
        self._running = False
        for task in list(self._consolidation_tasks.values()):
            task.cancel()
        self._consolidation_tasks.clear()
        logger.info("K2DO Agent loop stopping")

    @staticmethod
    def _is_reflect_mode(session: Session) -> bool:
        return bool(session.metadata.get("reflect_mode", False))

    def _set_reflect_mode(self, session: Session, enabled: bool) -> None:
        from datetime import datetime

        session.metadata["reflect_mode"] = bool(enabled)
        session.updated_at = datetime.now()
        self.sessions.save(session)

    def _handle_reflect_command(
        self,
        raw_cmd: str,
        msg: InboundMessage,
        session: Session,
    ) -> OutboundMessage:
        arg = raw_cmd[len("/reflect"):].strip().lower()
        current = self._is_reflect_mode(session)

        if arg in {"", "toggle"}:
            new_state = not current
            self._set_reflect_mode(session, new_state)
            state = "ON" if new_state else "OFF"
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Reflect mode: {state}",
            )

        if arg in {"on", "1", "true", "enable", "enabled"}:
            self._set_reflect_mode(session, True)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Reflect mode: ON",
            )

        if arg in {"off", "0", "false", "disable", "disabled"}:
            self._set_reflect_mode(session, False)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Reflect mode: OFF",
            )

        if arg in {"status", "?", "state"}:
            state = "ON" if current else "OFF"
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Reflect mode: {state}",
            )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Usage: /reflect [on|off|toggle|status]",
        )

    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        if msg.channel == "system":
            return await self._process_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing: {msg.channel}:{msg.sender_id}: {preview}")

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        raw_cmd = msg.content.strip()
        cmd = raw_cmd.lower()
        if cmd == "/new":
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            temp = Session(key=session.key)
            temp.messages = messages_to_archive
            self._schedule_consolidation(temp, archive_all=True)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory saved.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="K2DO commands:\n/new — New conversation\n/reflect [on|off|toggle|status] — Toggle reflection-style responses\n/deepthink <query> — Force DeepThink mode\n/refine <query> — Multi-round refinement (3 rounds)\n/help — This message")
        if cmd == "/reflect" or cmd.startswith("/reflect "):
            return self._handle_reflect_command(raw_cmd, msg, session)
        if cmd.startswith("/refine "):
            return await self._process_refine_forced(msg, session)
        if cmd.startswith("/deepthink "):
            # Force DeepThink mode
            return await self._process_deepthink_forced(msg, session)

        if len(session.messages) > self.memory_window:
            self._schedule_consolidation(session)

        # Smart routing
        self._last_complexity = compute_complexity(msg.content)
        requested_route = classify_query(msg.content, self.complexity_threshold)

        self._set_tool_context(msg.channel, msg.chat_id)

        if self.deepthink_enabled and requested_route == "deepthink":
            self._last_route = "deepthink"
            return await self._process_deepthink(msg, session)

        # If DeepThink is disabled, force effective route to simple so UI/telemetry
        # reflect the real execution path.
        self._last_route = "simple"
        return await self._process_simple(msg, session)

    async def _process_simple(self, msg: InboundMessage, session: Session) -> OutboundMessage:
        """Standard single-agent processing."""
        is_action_request = self._looks_like_action_request(msg.content)
        # Keep "simple" path fast for truly trivial requests, but avoid routing
        # non-trivial/action tasks to the weak fast model.
        use_fast_model = (
            self._last_complexity <= self.simple_route_max_complexity
            and not is_action_request
        )
        target_model = self.simple_model if use_fast_model else self.think_model

        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )
        final_content, tools_used = await self._run_agent_loop(
            initial_messages,
            model=target_model,
            require_concrete_tools=is_action_request,
            reflect_mode=self._is_reflect_mode(session),
        )
        if final_content is None:
            final_content = "I couldn't generate a response. Please try again."

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content, tools_used=tools_used or None)
        self.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    async def _process_deepthink(self, msg: InboundMessage, session: Session) -> OutboundMessage:
        """Multi-agent DeepThink processing with execution handoff."""
        system_prompt = self.context.build_system_prompt()
        result = await self._run_deepthink(msg.content, system_prompt)

        # Handoff: use DeepThink verdict as guidance, then execute via normal tool loop.
        deepthink_verdict = self._format_deepthink_result(result)
        execution_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        execution_messages.append({
            "role": "assistant",
            "content": (
                "DeepThink guidance for this task:\n"
                f"{deepthink_verdict}\n\n"
                "Use it as planning input."
            ),
        })
        execution_messages.append({
            "role": "user",
            "content": (
                "Now complete the original request end-to-end. "
                "Use tools when needed. "
                "Do not claim any command/file action unless it was actually executed."
            ),
        })
        response, exec_tools = await self._run_agent_loop(
            execution_messages,
            model=self.think_model,
            require_concrete_tools=self._looks_like_action_request(msg.content),
            allow_reasoning_tools=True,
            max_reasoning_tool_calls=2,
            reflect_mode=self._is_reflect_mode(session),
        )
        non_reasoning_tools = [
            t for t in (exec_tools or [])
            if t not in {"deepthink", "refine"}
        ]
        if (
            not non_reasoning_tools
            and self._looks_like_action_request(msg.content)
        ):
            # Force one additional pass when the model discussed a plan but
            # did not execute any concrete tool actions.
            retry_messages = list(execution_messages)
            retry_messages.append({
                "role": "assistant",
                "content": response or "",
            })
            retry_messages.append({
                "role": "user",
                "content": (
                    "You have not executed any concrete tool action yet. "
                    "Now MUST execute the task using filesystem/shell/message tools, "
                    "verify results from tool outputs, then return the final answer."
                ),
            })
            retry_response, retry_tools = await self._run_agent_loop(
                retry_messages,
                model=self.think_model,
                allow_reasoning_tools=False,
                reflect_mode=self._is_reflect_mode(session),
            )
            if retry_response:
                response = retry_response
            exec_tools = (exec_tools or []) + (retry_tools or [])

        if response is None:
            response = deepthink_verdict
        tools_used = ["deepthink"] + [t for t in (exec_tools or []) if t != "deepthink"]

        session.add_message("user", msg.content)
        session.add_message("assistant", response, tools_used=tools_used)
        self.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=response,
            metadata=msg.metadata or {},
        )

    @staticmethod
    def _looks_like_action_request(text: str) -> bool:
        """Heuristic: task asks to produce/execute something concrete."""
        lowered = (text or "").lower()
        return any(hint in lowered for hint in _ACTION_HINTS)

    @staticmethod
    def _looks_like_plan_response(text: str | None) -> bool:
        """Detect reflective/planning text that should not be a final answer."""
        lowered = (text or "").lower()
        return any(hint in lowered for hint in _PLAN_RESPONSE_HINTS)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Normalize rich content payloads into plain text for logs/memory."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (int, float, bool)):
            return str(content)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    async def _process_deepthink_forced(self, msg: InboundMessage, session: Session) -> OutboundMessage:
        """Force DeepThink via /deepthink command."""
        query = msg.content.strip()[len("/deepthink "):].strip()
        self._last_route = "deepthink"
        self._last_complexity = 1.0
        system_prompt = self.context.build_system_prompt()
        result = await self._run_deepthink(query, system_prompt)
        response = self._format_deepthink_result(result)
        session.add_message("user", query)
        session.add_message("assistant", response, tools_used=["deepthink"])
        self.sessions.save(session)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=response, metadata=msg.metadata or {})

    async def _process_refine_forced(self, msg: InboundMessage, session: Session) -> OutboundMessage:
        """Force refinement via /refine command."""
        query = msg.content.strip()[len("/refine "):].strip()
        self._last_route = "refine"
        self._last_complexity = 1.0
        system_prompt = self.context.build_system_prompt()
        result = await self.refiner.refine(
            query=query,
            system_prompt=system_prompt,
            on_round=self.on_refine_round,
        )
        response = self._format_refine_result(result)
        session.add_message("user", query)
        session.add_message("assistant", response, tools_used=["refine"])
        self.sessions.save(session)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=response, metadata=msg.metadata or {})

    def _format_refine_result(self, result: RefinementResult) -> str:
        """Format refinement result — return the final refined answer."""
        if len(result.rounds) >= 3:
            return result.rounds[2].content
        elif result.rounds:
            return result.rounds[-1].content
        return "Refinement completed but no response was generated."

    def _format_deepthink_result(self, result: DeepThinkResult) -> str:
        """Format DeepThink result for output."""
        parts = []

        # Individual thinker summaries
        for r in result.thinker_results:
            status = f"{r.duration_ms}ms" if not r.error else f"ERROR: {r.error}"
            parts.append(f"**{r.name}** (temp={r.temperature}, {status})")

        # Judge verdict is the main response
        if result.judge_verdict:
            return result.judge_verdict

        # Fallback: best individual response
        best = max(result.thinker_results, key=lambda r: len(r.response)) if result.thinker_results else None
        return best.response if best else "DeepThink completed but no response was generated."

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel, origin_chat_id = parts[0], parts[1]
        else:
            origin_channel, origin_chat_id = "cli", msg.chat_id

        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content, channel=origin_channel, chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages, model=self.think_model)
        if final_content is None:
            final_content = "Background task completed, but no final text was generated."
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        return OutboundMessage(channel=origin_channel, chat_id=origin_chat_id, content=final_content)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        session_key = str(getattr(session, "key", "unknown"))
        lock = self._consolidation_locks.setdefault(session_key, asyncio.Lock())
        if lock.locked() and not archive_all:
            return

        async with lock:
            memory = MemoryStore(self.workspace)
            if archive_all:
                old_messages = session.messages
                keep_count = 0
            else:
                keep_count = self.memory_window // 2
                if len(session.messages) <= keep_count:
                    return
                messages_to_process = len(session.messages) - session.last_consolidated
                if messages_to_process <= 0:
                    return
                old_messages = session.messages[session.last_consolidated:-keep_count]
                if not old_messages:
                    return

            lines = []
            for m in old_messages:
                content_text = self._content_to_text(m.get("content")).strip()
                if not content_text:
                    continue
                tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
                ts = str(m.get("timestamp", "?"))[:16]
                role = str(m.get("role", "assistant")).upper()
                lines.append(f"[{ts}] {role}{tools}: {content_text}")
            conversation = "\n".join(lines)
            current_memory = memory.read_long_term()

            prompt = f"""You are a memory consolidation agent. Process this conversation and return JSON with:
1. "history_entry": 2-5 sentence summary with timestamp
2. "memory_update": Updated long-term memory with new facts

## Current Memory
{current_memory or "(empty)"}

## Conversation
{conversation}

Respond with ONLY valid JSON."""

            try:
                response = await self.provider.chat(
                    messages=[
                        {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                )
                raw_content = response.content
                text = self._content_to_text(raw_content).strip()
                if not text:
                    return
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                result = json_repair.loads(text)
                if not isinstance(result, dict):
                    return
                entry = self._content_to_text(result.get("history_entry")).strip()
                if entry:
                    memory.append_history(entry)

                update_value = result.get("memory_update")
                if isinstance(update_value, (dict, list)):
                    update = json.dumps(update_value, ensure_ascii=False, indent=2)
                else:
                    update = self._content_to_text(update_value).strip()
                if update and update != current_memory:
                    memory.write_long_term(update)
                if archive_all:
                    session.last_consolidated = 0
                else:
                    session.last_consolidated = len(session.messages) - keep_count
                    self.sessions.save(session)
            except Exception as e:
                logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(self, content, session_key="cli:direct", channel="cli", chat_id="direct"):
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        try:
            response = await self._process_message(msg, session_key=session_key)
            return response.content if response else ""
        except Exception as e:
            logger.error(f"Direct processing failed: {e}")
            return "Sorry, I hit an internal error while processing your request."
