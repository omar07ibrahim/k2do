# K2DO — evidence-led DeepThink orchestration for K2

K2DO adapts the MIT-licensed
[HKUDS/nanobot](https://github.com/HKUDS/nanobot) runtime for K2 Think/Instruct
models and adds a K2-specific reasoning control plane: deterministic query
routing, concurrent role-configured thinker calls, model fallback, bounded
timeouts, judge synthesis, and cancellation cleanup.

> [!IMPORTANT]
> K2DO is a modified derivative, not a from-scratch agent framework. The
> generic message bus, tool shell, persistence, scheduling, provider
> abstractions, and Telegram transport descend from nanobot. The contribution
> boundary is documented in the
> [provenance and contribution map](docs/provenance.md).

This README starts with the credential-free paths that can be checked today.
The primary receipt now follows a routed request through the production
`MessageBus`, classifier, `AgentLoop._process_message` path, parallel
DeepThink/Judge phase, real workspace-restricted filesystem tools, disk-backed
session persistence, and the outbound queue. Live K2 setup comes later and is
deliberately not presented as benchmark evidence.

## Run the verified offline path

Both offline labs use strict scripted providers. They need no external
credentials and do not send model requests. The first verifies the routed tool
handoff end to end; the second isolates DeepThink fallback, timeout,
degradation, and cancellation behavior.

```mermaid
flowchart LR
    V["Create Python 3.11+ venv"] --> I["Install dev + pinned evidence renderer"]
    I --> H["Run routed handoff lab"]
    H --> HC["Check handoff bundle"]
    HC --> D["Run direct-engine lab"]
    D --> DC["Check direct-engine bundle"]
```

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip install -r requirements-evidence.txt

python -m k2do.labs.agent_handoff_trace
python -m k2do.labs.handoff_evidence_recorder check
python -m k2do.labs.deepthink_trace
python -m k2do.labs.trace_evidence_recorder check
python -m pytest -q
```

The setup is supported on CPython 3.11 or newer. It is not yet byte-for-byte
environment reproducible: application dependencies have lower bounds and the
repository has no lockfile. The raster evidence path pins Pillow `12.3.0`; its
manifest also records Python, OS, architecture, FreeType, zlib, and K2DO
versions.

![Raster rendering of the exact captured routed-handoff stdout](docs/agent-handoff-evidence/terminal.png)

*Receipt-derived raster terminal view. Its text is the exact captured command
and canonical stdout also stored byte-for-byte in
[`handoff-lab.txt`](docs/agent-handoff-evidence/handoff-lab.txt); it is not a
photograph of an OS terminal. The public payload contains labels, counts,
relative fixture names, and digests—not prompts, model responses, credentials,
endpoints, absolute paths, or elapsed timings. Receipt SHA-256:
`f219e1cf45671cf91c57c6bfdd9ef18b42f7d704c2b4c9b07eaefaff33f37486`.*

## Routed reasoning-to-action handoff

![Animated receipt-derived replay of the routed handoff](docs/agent-handoff-evidence/workflow-demo.gif)

*Nine-frame replay generated from the receipt's ordered workflow nodes. Frame
delays are illustrative and make no latency claim; this is not a live screen
recording.*

![Receipt-derived routed handoff architecture](docs/agent-handoff-evidence/architecture.svg)

The laboratory publishes a real inbound message, consumes it from the real
queue, and lets the production router select `deepthink` at the lab's fixed
`0.6` threshold. Three real `DeepThinkEngine` coroutines cross a strict
barrier, the Judge is forbidden to start until all three calls are terminal,
and the Judge verdict becomes guidance for the normal `AgentLoop` tool loop.
That loop then executes the registered `write_file` and `read_file`
implementations inside a private restricted workspace. A fresh
`SessionManager` reload proves the assistant record and
`deepthink → write_file → read_file` tool list reached disk before the outbound
message is consumed.

![Receipt-derived handoff and tool timeline](docs/agent-handoff-evidence/tool-timeline.svg)

The provider is fail-closed rather than permissive: it validates model,
temperature, token limit, full message shapes, normalized dynamic context,
Judge input, all eleven tool schemas in order, exact tool-result messages, and
the three-turn execution contract before returning the next scripted action.
The real artifact is constrained to the temporary workspace, read back through
the production tool, hashed, and removed with that workspace.

![Verified routed handoff contract matrix](docs/agent-handoff-evidence/contract-matrix.svg)

| Observed surface | Receipt result |
| --- | --- |
| Router and queues | 1 inbound + 1 outbound message; route `deepthink`; both queues drained |
| Parallel reasoning | 3 thinkers; peak 3 active provider calls; Judge gate after all thinkers terminal |
| Execution handoff | 3 sequential provider turns with the complete production tool catalog |
| Concrete tools | Real `write_file → read_file`; 32-byte bounded ASCII artifact; exact read-back |
| Persistence | Fresh disk reload yields roles `user, assistant` and tools `deepthink, write_file, read_file` |
| Cleanup | 0 active provider calls; temporary workspace removed |
| Communication observation | 0 calls observed in the manifest's listed Linux `strace` communication-syscall set |

The handoff manifest binds `pyproject.toml`, `requirements-evidence.txt`, and
every committed blob under `k2do/`—67 source files for the current capture—by
Git mode, blob ID, byte count, and SHA-256. `check` regenerates all eight
artifacts and, on this Linux host, repeats both the canonical capture and the
bounded `strace` observation.
This is routed control-flow and side-effect evidence. It is not a live-provider
quality benchmark, an MCP lifecycle test, a network sandbox, or a latency
measurement.

See the [evidence protocol and boundaries](docs/evidence.md) and the
[handoff manifest](docs/agent-handoff-evidence/manifest.json).

## Direct engine fault-path evidence

![Genuine terminal capture of the K2DO offline trace lab](docs/deepthink-trace-evidence/trace-lab.svg)

*Genuine captured stdout rendered from the canonical receipt. The payload
contains labels, counts, and digests—not prompts, model responses, credentials,
endpoints, absolute paths, or timing claims. Receipt SHA-256:
`f66e1db30dd79d77318a20ae86a0aa450a663e8f5cdff351557ca6fd1d0da80d`.*

![Receipt-derived K2DO synthesis call DAG](docs/deepthink-trace-evidence/call-dag.svg)

*Receipt-derived synthesis path through the production router and direct
`DeepThinkEngine`. The strict synthetic provider forces an Analyst primary
error and fallback, a Pragmatist timeout/cancellation path, and a Judge call
after every selected thinker is terminal. This is control-flow evidence, not an
answer-quality or latency claim.*

The canonical receipt covers these deterministic scenarios:

| Scenario | What was observed |
| --- | --- |
| Router contract | The complex architecture fixture routes to `deepthink`; the simple greeting fixture routes to `simple`. |
| Synthesis path | Three of four configured thinkers are selected; provider-call peak is 3; Analyst primary fails then fallback succeeds; Pragmatist reaches the categorical timeout path; Judge runs after all selected thinkers are terminal; cleanup ends with 0 active calls. |
| Judge degradation | Provider-call peak is 2; Judge primary and model fallback both error; the engine returns the longest successful synthetic thinker response; cleanup ends with 0 active calls. |
| Caller cancellation | Three blocked provider calls are cancelled; Judge call count is 0; the root task is cancelled; cleanup ends with 0 active calls and 0 incomplete barriers. |

The word “thinker” here means a concurrent, role-configured call through one
provider abstraction. The lab does not claim separate autonomous processes or
independent model backends.

## Measured properties

![Receipt-derived K2DO orchestration properties](docs/deepthink-trace-evidence/orchestration-properties.svg)

*All values are derived from the same receipt. Peaks `3 / 2 / 3` describe the
three scripted scenarios, not throughput. The Judge-degradation result is the
longest successful synthetic response, not a measured “best” answer.*

The evidence recorder makes the visuals reviewable rather than decorative:

- it materializes a private snapshot from committed Git blobs and runs that
  snapshot with `python -I -S -B`;
- it binds ten capture-relevant committed paths—the lab/engine modules,
  recorder, package initializers, and project metadata—by Git mode, blob ID,
  SHA-256, and byte count;
- it records zero observed communication syscalls in one Linux `strace` run
  over an explicit syscall set; this is an observation, not network isolation;
- it checks exact file sets, hashes, media types, JSON structure, explicit
  forbidden SVG patterns, and the SVG root contract;
- it deterministically re-renders the transcript and visuals during
  `trace_evidence_recorder check`.

See the [evidence protocol and boundaries](docs/evidence.md) and the
[machine-readable manifest](docs/deepthink-trace-evidence/manifest.json).

## Technical decisions

| Concern | Implementation | Evidence boundary |
| --- | --- | --- |
| Query routing | Deterministic heuristic score in `k2do/agent/router.py` | Two fixed route decisions are captured; general routing accuracy is not measured. |
| Parallel work | Selected role configurations run concurrently with `asyncio.gather` | Barrier order and peak active provider calls are captured. |
| Model fallback | A failed primary call retries once on the configured fallback model | Thinker fallback and double Judge failure are forced by the strict provider. |
| Time bounds | Thinker and Judge calls are wrapped with `asyncio.wait_for` | A categorical thinker timeout is captured; elapsed time is intentionally excluded. |
| Cancellation | Caller cancellation propagates through active thinker calls | Cancelled-call counts and zero-active cleanup are captured. |
| Judge degradation | If Judge execution fails, the longest raw thinker response is selected; a fixed message is used if that selected response is blank | The deterministic successful-response selection is captured; semantic quality is not assessed. |
| Reasoning-to-action handoff | The routed Judge verdict is appended as guidance before the normal `AgentLoop` tool loop | The offline handoff receipt verifies the exact message contract and three execution turns; it does not assess verdict quality. |
| Tool confinement | Filesystem tools resolve relative paths against a restricted temporary workspace | A real write/read-back is verified; shell, web, message, spawn, MCP, and arbitrary-path behavior are outside this receipt. |
| Session persistence | User and assistant records are written to JSONL with the assistant's tool list | A fresh manager reload validates two records and normalized timestamps; consolidation and long-term memory are outside this receipt. |
| Evidence integrity | Source-bound receipt, deterministic renderer, atomic no-replace publication | Adversarial tests cover dirty or changing sources, artifact tampering/symlinks/extras, hostile output leaves, publish races, output caps, timeout cleanup, and squash-safe verification. |

## What is and is not verified

| Surface | Current evidence |
| --- | --- |
| Default `classify_query` behavior for two fixed fixtures | Verified by the offline receipt |
| Direct `DeepThinkEngine` concurrency, fallback, timeout, Judge degradation, and cancellation | Verified by the offline receipt |
| `MessageBus → router → AgentLoop._process_message → DeepThink → tool loop → session → MessageBus` routed handoff | Verified by the handoff receipt |
| Workspace-restricted `write_file → read_file` and fresh session reload | Verified by the handoff receipt |
| Artifact provenance and renderer-runtime-bound deterministic rendering | Verified by both committed manifests and checkers |
| Background `AgentLoop.run()`, MCP lifecycle, config loader wiring, refinement, memory consolidation, gateway, and Telegram paths | Present or unit-tested in parts; outside both receipts |
| Live K2 provider behavior | Not captured |
| Answer quality, token cost, throughput, or latency | Not benchmarked |
| Locked dependency environment and CI | Not yet available |

## Optional live K2 setup

This credentialed configuration path can incur provider usage. It is not part
of the offline evidence above.

```bash
k2do onboard
# Add your key to ~/.k2do/config.json; never commit that file.
k2do status
k2do agent -m "Compare two deployment designs and state the trade-offs."
```

The generated config uses this shape:

```json
{
  "providers": {
    "k2Think": {
      "apiKey": "<K2_API_KEY>",
      "apiBase": "https://build-api.k2think.ai/v1"
    },
    "k2Instruct": {
      "apiKey": "<K2_API_KEY>",
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
      "complexityThreshold": 0.45,
      "maxAgents": 3
    }
  }
}
```

Interactive commands include `/deepthink <query>` and `/refine <query>`.
`k2do gateway` starts the configured channel gateway. No credentialed output,
live-provider result, Telegram exchange, or full-screen dashboard image is used
as portfolio evidence yet.

## Repository map

```text
k2do/
  agent/
    router.py                  # deterministic query classification
    deepthink.py               # thinker concurrency, fallback, Judge, cleanup
    loop.py                    # generic/adapted runtime integration
    refine.py                  # three-stage refinement path
  labs/
    agent_handoff_trace.py       # routed bus/tool/session receipt
    handoff_evidence_recorder.py # source-bound raster/SVG/GIF capture
    deepthink_trace.py           # direct-engine fault-path receipt
    trace_evidence_recorder.py   # direct-engine capture and renderer
  cli/                         # Typer/Rich live interface
  channels/                    # adapted channel integrations
docs/
  agent-handoff-evidence/      # receipt, real stdout, SVG/PNG/GIF, manifest
  deepthink-trace-evidence/    # direct-engine receipt and SVG evidence
  evidence.md                  # verification and maintenance protocols
  provenance.md                # upstream/contribution boundary
tests/                         # unit and adversarial evidence tests
```

The next evidence milestones are a locked application dependency set with CI,
config-loader/MCP lifecycle evidence, deterministic refinement and memory
traces, and an opt-in sanitized live-provider capture. They are tracked as
remaining work, not described as completed features.

## License and attribution

K2DO is distributed under the MIT License. The repository preserves the
nanobot contributors’ notice; see [LICENSE](LICENSE) and
[docs/provenance.md](docs/provenance.md) before evaluating or reusing the
project.
