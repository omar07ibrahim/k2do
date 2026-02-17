"""Context builder for K2DO agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from k2do.agent.memory import MemoryStore
from k2do.agent.skills import SkillsLoader


class ContextBuilder:
    """Builds system prompt + messages for the K2DO agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    def __init__(
        self,
        workspace: Path,
        store_reasoning: bool = False,
        max_context_chars: int = 24000,
        max_bootstrap_chars: int = 8000,
        max_memory_chars: int = 5000,
        max_active_skills_chars: int = 6000,
        max_skills_summary_chars: int = 5000,
    ):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        self.store_reasoning = store_reasoning
        self.max_context_chars = max_context_chars
        self.max_bootstrap_chars = max_bootstrap_chars
        self.max_memory_chars = max_memory_chars
        self.max_active_skills_chars = max_active_skills_chars
        self.max_skills_summary_chars = max_skills_summary_chars

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        parts = []
        parts.append(self._get_identity())
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(self._truncate_block("bootstrap", bootstrap, self.max_bootstrap_chars))
        memory = self.memory.get_memory_context()
        if memory:
            memory = self._truncate_block("memory", memory, self.max_memory_chars)
            parts.append(f"# Memory\n\n{memory}")
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                always_content = self._truncate_block(
                    "active_skills", always_content, self.max_active_skills_chars
                )
                parts.append(f"# Active Skills\n\n{always_content}")
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            skills_summary = self._truncate_block(
                "skills_summary", skills_summary, self.max_skills_summary_chars
            )
            parts.append(f"# Skills\n\nTo use a skill, read its SKILL.md with read_file.\n\n{skills_summary}")

        prompt = "\n\n---\n\n".join(parts)
        return self._truncate_block("system_prompt", prompt, self.max_context_chars)

    def _get_identity(self) -> str:
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        ws = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        return f"""# K2DO -- AI Agent with DeepThink

You are K2DO, an AI agent powered by K2 Think with multi-agent DeepThink capabilities.

You have access to tools: read/write/edit files, shell commands, web search, send messages, spawn subagents.

## Your Superpowers

You have two special tools that make you more powerful than a regular AI:

**deepthink** — When you face a complex problem (design decisions, comparisons, architectural
choices, debugging hard issues), call the `deepthink` tool. It spawns 3 parallel AI agents
(Analyst, Creative, Pragmatist) who each think independently, then a Judge synthesizes the
best answer. Use this for questions where multiple perspectives lead to better results.

**refine** — When you need the highest quality answer (important code, critical documents,
tricky solutions), call the `refine` tool. It runs 3 rounds: initial answer, then a Critic
scores it and finds weaknesses, then a Refiner creates an improved version.

Use these tools deliberately:
- Use `deepthink` for genuinely complex tasks (architecture, difficult debugging, ambiguous planning, creative generation with constraints).
- Avoid repeated `deepthink` calls for the same task unless new constraints appear or the previous plan failed in execution.
- For straightforward one-liners, skip `deepthink` and execute directly.
- Use `refine` when quality is more important than latency.

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Path: {ws}
- Memory: {ws}/memory/MEMORY.md
- History: {ws}/memory/HISTORY.md
- Skills: {ws}/skills/{{name}}/SKILL.md

Reply directly to questions. Only use 'message' tool for sending to chat channels.
Be helpful, accurate, concise. Think step by step with tools."""

    def _load_bootstrap_files(self) -> str:
        parts = []
        for filename in self.BOOTSTRAP_FILES:
            fp = self.workspace / filename
            if fp.exists():
                parts.append(f"## {filename}\n\n{fp.read_text(encoding='utf-8')}")
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _truncate_block(name: str, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        suffix = f"\n\n[... {name} truncated ...]"
        keep = max(0, max_chars - len(suffix))
        return text[:keep].rstrip() + suffix

    def build_messages(self, history, current_message, skill_names=None, media=None, channel=None, chat_id=None):
        messages = []
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        if not media:
            return text
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(self, messages, tool_call_id, tool_name, result):
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(self, messages, content, tool_calls=None, reasoning_content=None):
        msg: dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if self.store_reasoning and reasoning_content:
            msg["reasoning_content"] = reasoning_content
        messages.append(msg)
        return messages
