"""Tool registry for dynamic tool management."""

from typing import Any

from k2do.agent.tools.base import Tool

MCP_INVALID_NAME_MARKER = "mcp_[invalid]"
_MCP_TOOL_PREFIX = "mcp_"
_MCP_TOOL_DIGEST_LENGTH = 32
_LOWER_HEX = frozenset("0123456789abcdef")


def is_mcp_tool_attempt(name: object) -> bool:
    """Return whether an untrusted tool name enters the MCP trust boundary."""
    return type(name) is str and name.startswith(_MCP_TOOL_PREFIX)


def is_mcp_tool_name(name: object) -> bool:
    """Recognize only the non-disclosing public MCP identity grammar."""
    return (
        type(name) is str
        and len(name) == len(_MCP_TOOL_PREFIX) + _MCP_TOOL_DIGEST_LENGTH
        and name.startswith(_MCP_TOOL_PREFIX)
        and all(character in _LOWER_HEX for character in name[len(_MCP_TOOL_PREFIX) :])
    )


def safe_tool_name(name: str) -> str:
    """Replace malformed MCP-shaped names before logging or persistence."""
    if is_mcp_tool_attempt(name) and not is_mcp_tool_name(name):
        return MCP_INVALID_NAME_MARKER
    return name


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool without replacing an existing capability."""
        if is_mcp_tool_attempt(tool.name) and not is_mcp_tool_name(tool.name):
            raise ValueError("Malformed MCP tool identity")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> tuple[str, ...]:
        """Register one closed batch, or leave the registry unchanged."""
        names = tuple(tool.name for tool in tools)
        if any(is_mcp_tool_attempt(name) and not is_mcp_tool_name(name) for name in names):
            raise ValueError("Tool batch contains a malformed MCP identity")
        if len(names) != len(set(names)):
            raise ValueError("Tool batch contains duplicate names")
        collisions = tuple(name for name in names if name in self._tools)
        if collisions:
            raise ValueError("Tool batch collides with registered names")
        self._tools.update(zip(names, tools, strict=True))
        return names

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def unregister_many(
        self,
        names: tuple[str, ...],
        expected_tools: tuple[Tool, ...] | None = None,
    ) -> None:
        """Unregister a batch, optionally only when object identities still match."""
        if expected_tools is not None and len(names) != len(expected_tools):
            raise ValueError("Tool unregister batch length differs")
        for position, name in enumerate(names):
            if expected_tools is None or self._tools.get(name) is expected_tools[position]:
                self.unregister(name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a tool by name with given parameters.

        Args:
            name: Tool name.
            params: Tool parameters.

        Returns:
            Tool execution result as string.

        Raises:
            KeyError: If tool not found.
        """
        if is_mcp_tool_attempt(name) and not is_mcp_tool_name(name):
            return "Error: MCP tool not found"
        tool = self._tools.get(name)
        if not tool:
            if is_mcp_tool_attempt(name):
                return "Error: MCP tool not found"
            return f"Error: Tool '{name}' not found"

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            return await tool.execute(**params)
        except Exception as e:
            if is_mcp_tool_attempt(name):
                return "Error: MCP tool execution failed"
            return f"Error executing {name}: {str(e)}"

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
