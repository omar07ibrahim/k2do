"""
Smart Router — decides whether to use simple mode or DeepThink.

Analyzes query complexity using heuristics:
- Length of query
- Presence of reasoning keywords
- Multi-part questions
- Code/technical complexity markers

Returns: "simple" | "deepthink"
"""

from __future__ import annotations

import re

# Keywords that suggest complex reasoning is needed
COMPLEX_KEYWORDS = {
    # Reasoning
    "explain", "why", "how does", "compare", "analyze", "evaluate",
    "trade-off", "tradeoff", "pros and cons", "advantages",
    "design", "architect", "plan", "strategy",
    # Multi-step
    "step by step", "first then", "implement", "build", "generate", "compose",
    "refactor", "optimize", "debug", "fix the bug",
    # Code
    "algorithm", "data structure", "complexity", "music", "track", "song",
    "write a function", "write a class", "create a module",
    "full implementation", "complete solution",
    # Analysis
    "what would happen if", "predict", "estimate",
    "review", "audit", "assess",
    # Russian keywords for our hackathon demo
    "объясни", "почему", "сравни", "проанализируй",
    "как работает", "реализуй", "напиши", "создай", "сгенерируй", "музыку", "трек", "песню",
    "оптимизируй", "исправь", "спроектируй",
}

# Keywords that suggest simple/direct response
SIMPLE_KEYWORDS = {
    "hello", "hi", "hey", "thanks", "thank you",
    "what time", "what date", "weather",
    "привет", "спасибо", "который час", "погода",
}

# Explicit action/tool intents should stay in the regular agent loop so tools
# can actually execute (write_file, read_file, exec, etc.).
TOOL_INTENT_KEYWORDS = {
    "write_file", "read_file", "edit_file", "list_dir", "exec",
    "create file", "write file", "edit file", "read file",
    "создай файл", "запиши в файл", "измени файл", "прочитай файл",
    "выполни команду", "запусти команду",
}


def has_tool_intent(query: str) -> bool:
    """Return True when query is an explicit action likely requiring tools."""
    query_lower = query.lower()

    if any(kw in query_lower for kw in TOOL_INTENT_KEYWORDS):
        return True

    # Generic English patterns.
    if re.search(r"\b(create|write|edit|read)\s+.+\b(file|folder|directory)\b", query_lower):
        return True
    if re.search(r"\b(run|execute)\s+.+\b(command|script)\b", query_lower):
        return True

    return False


def classify_query(query: str, threshold: float = 0.45) -> str:
    """
    Classify a query as "simple" or "deepthink".

    Returns "deepthink" when complexity score exceeds threshold.
    """
    score = compute_complexity(query)
    if has_tool_intent(query):
        # Action/tool requests are often multi-step and benefit from DeepThink.
        score = min(1.0, score + 0.25)
    return "deepthink" if score >= threshold else "simple"


def compute_complexity(query: str) -> float:
    """
    Compute a complexity score from 0.0 to 1.0.

    Factors:
    - Query length
    - Complex keywords
    - Question marks count
    - Code block presence
    - Multi-sentence structure
    """
    query_lower = query.lower().strip()
    score = 0.0

    # Simple keyword override
    for kw in SIMPLE_KEYWORDS:
        if query_lower.startswith(kw) and len(query_lower) < 50:
            return 0.1

    # Length factor (longer = more complex)
    length = len(query)
    if length > 500:
        score += 0.3
    elif length > 200:
        score += 0.2
    elif length > 100:
        score += 0.1

    # Complex keywords — weighted more heavily
    keyword_hits = sum(1 for kw in COMPLEX_KEYWORDS if kw in query_lower)
    score += min(keyword_hits * 0.25, 0.6)

    # Multiple questions
    question_marks = query.count("?")
    if question_marks >= 2:
        score += 0.2
    elif question_marks == 1:
        score += 0.05

    # Code blocks present
    if "```" in query or "def " in query or "class " in query:
        score += 0.2

    # Multi-line / multi-sentence
    sentences = len(re.split(r'[.!?\n]', query))
    if sentences >= 5:
        score += 0.15
    elif sentences >= 3:
        score += 0.05

    # Contains numbered list or bullet points
    if re.search(r'^\s*[\d\-\*]\s', query, re.MULTILINE):
        score += 0.1

    return min(score, 1.0)


def get_route_label(route: str) -> str:
    """Human-readable label for the route."""
    return {
        "simple": "K2 Think (Single-Agent)",
        "deepthink": "DeepThink (Multi-Agent)",
        "mcp": "MCP (Isolated Retrieval)",
    }.get(route, route)
