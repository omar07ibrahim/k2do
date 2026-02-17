# K2DO -- AI Agent with Multi-Agent DeepThink

**K2DO** is an AI agent framework powered by **K2 Think** (LLM360) with a unique **DeepThink** multi-agent reasoning system.

## What Makes K2DO Special

### DeepThink: Multi-Agent Parallel Reasoning

When a complex question comes in, K2DO doesn't just ask one model — it spawns **multiple AI agents** that think in parallel from different perspectives, then a **Judge** synthesizes the best answer.

```
User Query --> Smart Router
                |
                +--> Simple? --> K2-Instruct (fast path)
                |
                +--> Complex? --> DeepThink Mode:
                      |
                      +--> Analyst (temp=0.3, rigorous logic)
                      +--> Creative (temp=0.9, innovative ideas)
                      +--> Pragmatist (temp=0.5, practical focus)
                      |
                      +--> Judge --> Synthesized Best Answer
```

### Smart Router

Automatically detects query complexity using NLP heuristics and routes to the right processing mode:
- **Simple queries** (greetings, quick facts) go straight to K2-Instruct for speed
- **Complex queries** (design, analysis, comparison) trigger DeepThink multi-agent mode

### Live DeepThink Visualization

Beautiful terminal UI shows all agents thinking in parallel in real-time:

```
+--------------------------------------------+
| DeepThink -- Multi-Agent Reasoning         |
+--------------------------------------------+
| Agent      | Status                         |
|------------|--------------------------------|
| Analyst    | Thinking...                    |
| Creative   | Done (1240ms)                  |
| Pragmatist | Thinking...                    |
| Judge      | Waiting for all agents...      |
+--------------------------------------------+
```

## Quick Start

```bash
# Install
pip install -e .

# Setup
k2do onboard

# Edit config with your K2 API key
vim ~/.k2do/config.json

# Chat (auto DeepThink routing)
k2do agent

# Single message
k2do agent -m "Compare Python vs Rust for backend development"

# Force DeepThink mode
k2do agent
> /deepthink Design a microservices architecture for a social media app

# Start gateway (Telegram)
k2do gateway
```

## Architecture

```
k2do/
  agent/
    loop.py          # Core agent loop with DeepThink integration
    router.py        # Smart query complexity router
    deepthink.py     # Multi-agent parallel reasoning engine
    context.py       # System prompt builder
    memory.py        # Two-layer memory (MEMORY.md + HISTORY.md)
    subagent.py      # Background task agents
    tools/           # File, shell, web, message, spawn, cron
  providers/
    registry.py      # K2 Think + K2 Instruct + other providers
    litellm_provider.py
  channels/          # Telegram
  config/            # Pydantic schema + JSON loader
  cli/               # Typer CLI with Rich UI
  bus/               # Async message queue
  session/           # JSONL session persistence
  cron/              # Scheduled tasks
  heartbeat/         # Periodic autonomous checks
```

## K2 Models

| Model | Purpose | Speed |
|-------|---------|-------|
| K2-Think-V2 | Deep reasoning, planning, analysis | Slower, thorough |
| K2-V2-Instruct | Fast responses and lightweight execution | Fast |

## Config Example

```json
{
  "providers": {
    "k2Think": {
      "apiKey": "YOUR_K2_API_KEY",
      "apiBase": "https://build-api.k2think.ai/v1"
    },
    "k2Instruct": {
      "apiKey": "YOUR_K2_API_KEY",
      "apiBase": "https://build-api.k2think.ai/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "k2-think-v2/LLM360/K2-Think-V2",
      "fallbackModel": "k2-v2-instruct/LLM360/K2-V2-Instruct"
    },
    "deepthink": {
      "enabled": true,
      "complexityThreshold": 0.6,
      "maxAgents": 3
    }
  }
}
```

## Commands

| Command | Description |
|---------|-------------|
| `k2do onboard` | Initialize config and workspace |
| `k2do agent` | Interactive chat with auto-routing |
| `k2do agent -m "msg"` | Single message mode |
| `k2do gateway` | Start multi-channel server |
| `k2do status` | Show configuration and provider status |
| `k2do channels status` | Show channel configuration |
| `k2do cron list` | List scheduled jobs |

## Built With

- **K2 Think / K2 Instruct** (LLM360) - Primary AI models
- **LiteLLM** - Universal LLM provider routing
- **Rich** - Beautiful terminal UI
- **Typer** - CLI framework
- **Pydantic** - Configuration management
