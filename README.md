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

The credential-free verification path has three layers. Strict scripted
providers verify the bounded routed-processing handoff and isolate DeepThink
fallback, timeout, degradation, and cancellation behavior. A third laboratory
uses the production MCP client against real strict stdio subprocesses to force
discovery, protocol-error, repeated-cancellation, owner-scope, cleanup, and
same-loop restart paths. None of the three needs external credentials or sends
a model request.

```mermaid
flowchart LR
    V["Create CPython 3.12.13 venv"] --> I["Install 96 hash-locked requirements"]
    I --> M["Run production MCP fault lab"]
    M --> MC["Fresh-check MCP visual bundle"]
    MC --> H["Run routed handoff lab"]
    H --> HC["Check handoff bundle"]
    HC --> D["Run direct-engine lab"]
    D --> DC["Check direct-engine bundle"]
```

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-ci-py312.lock
python -m pip install --no-build-isolation --no-deps -e .

python -m k2do.labs.mcp_fault_lab
python -m k2do.labs.mcp_evidence_recorder check --fresh
python -m k2do.labs.agent_handoff_trace
python -m k2do.labs.handoff_evidence_recorder check
python -m k2do.labs.deepthink_trace
python -m k2do.labs.trace_evidence_recorder check
python -m pytest -q
```

The application supports CPython 3.11 or newer; the canonical hosted evidence
path fixes Ubuntu 24.04 and CPython 3.12.13. Its lock contains 96 exact
requirements guarded by 2,082 SHA-256 hashes and is installed with
`--require-hashes`. The adjacent provenance document binds the lock to pip
26.1.2, pip-tools 7.6.0, and its three compiler inputs. Other Python and OS
combinations remain supported application targets, not claims made by this CI
lock.

## Production MCP lifecycle under faults

![Receipt-derived replay of six real MCP fault scenarios](docs/mcp-fault-evidence/workflow-demo.gif)

*Six-frame replay generated from the ordered receipt. The underlying scenarios
use real subprocesses and the production lifecycle; frame delays are
illustrative and make no latency claim.*

![Source-bound MCP lifecycle architecture](docs/mcp-fault-evidence/architecture.svg)

This laboratory does not replace the MCP client with a mock. It starts strict,
independent stdio servers and connects them through
<code>AgentLoop.mcp_lifespan</code>, <code>connect_mcp_servers</code>,
<code>MCPToolWrapper.execute</code>, and the real tool registry. The pinned
<code>mcp==2.0.0</code> client negotiates protocol <code>2026-07-28</code>
through <code>server/discover</code> and NDJSON stdio.

![Rendered terminal view of byte-exact MCP fault-lab stdout](docs/mcp-fault-evidence/terminal.png)

*This is a raster rendering of the byte-exact captured CLI stdout, not a
hand-written mockup. The same bytes are committed as
[the terminal transcript](docs/mcp-fault-evidence/mcp-fault-lab.txt) and
[canonical receipt](docs/mcp-fault-evidence/receipt.json). Receipt SHA-256:
<code>e69543846b4563901637ec7d8a35035a5efad00eda0f6c3e7d36480fe152346c</code>.*

![Verified MCP fault and recovery matrix](docs/mcp-fault-evidence/fault-matrix.svg)

| Real scenario | Verified result |
| --- | --- |
| Discovery, call, close, restart | Stable catalog/result digests across two generations; first and repeated manual close both succeed |
| Broken catalog beside a healthy peer | Failed catalog stays unpublished while the healthy peer remains callable |
| Remote protocol error | Public result is categorical; raw fixture fault is absent; the same connection then succeeds |
| Two cancelled calls | Caller cancellation propagates, two courtesy cancellation notifications are observed, and the same connection recovers |
| Cancelled startup | The first process is reaped, no catalog survives, and the same loop starts a healthy replacement |
| Cancelled owner scope | The inherited borrower is cancelled and drained before the same loop restarts |

![Observed MCP cancellation and recovery timeline](docs/mcp-fault-evidence/cancellation-timeline.svg)

Across the matrix, all 10 fresh process generations report stdin EOF, are
reaped, and leave the tool registry at its baseline. Each scenario has a
15-second cancellation boundary, and the hosted evidence job repeats the
capture before publishing. The public projection is intentionally limited to
labels, counts, booleans, and SHA-256 digests: no token, PID, request ID, raw
argument, raw error, absolute path, or wall-clock measurement is emitted.

Run the same credential-free path:

    python -m k2do.labs.mcp_fault_lab
    python -m k2do.labs.mcp_evidence_recorder check --fresh

The [MCP evidence manifest](docs/mcp-fault-evidence/manifest.json) binds 12
capture-critical workflow, implementation, fixture, recorder, and test blobs
to source commit <code>c77bc34</code>. It records Python 3.12.13, MCP 2.0.0,
and Pillow 12.3.0, verifies every artifact hash/media type, deterministically
re-renders the PNG/SVG/GIF set, and compares a fresh receipt. The fixture is
designed around stdio plus a private authenticated AF_UNIX lifecycle oracle;
the recorder deliberately makes no independent syscall-level network-isolation
claim.

![Raster rendering of the routed-handoff reader banner and captured stdout](docs/agent-handoff-evidence/terminal.png)

*Receipt-derived raster terminal view. Its text is a reader-command banner plus
the byte-exact captured canonical stdout also stored in
[`handoff-lab.txt`](docs/agent-handoff-evidence/handoff-lab.txt); it is not a
photograph of an OS terminal. The public payload contains labels, counts,
relative fixture names, and digests—not prompts, model responses, credentials,
endpoints, absolute paths, or elapsed timings. Receipt SHA-256:
`12b6c89a40bf2ca4f775a9ffa0c68b0525cde3f853dd4cf089ebb5e333f91db8`.*

## Routed reasoning-to-action handoff

![Animated receipt-derived replay of the routed handoff](docs/agent-handoff-evidence/workflow-demo.gif)

*Nine-frame replay generated from the receipt's ordered workflow nodes. Frame
delays are illustrative and make no latency claim; this is not a live screen
recording.*

![Receipt-derived routed handoff architecture](docs/agent-handoff-evidence/architecture.svg)

The laboratory publishes a real inbound message, consumes it from the real
queue, and lets the production router select `deepthink` at the lab's fixed
`0.6` threshold. Three thinker coroutines scheduled by the production
`DeepThinkEngine` cross a strict barrier, the Judge is forbidden to start until
all three calls are terminal, and the Judge verdict becomes guidance for the
normal `AgentLoop` tool loop.
That loop then executes the registered `write_file` and `read_file`
implementations inside a private restricted workspace. The handler persists
before returning; after the outbound queue is consumed, a fresh
`SessionManager` reload verifies the assistant record and
`deepthink → write_file → read_file` tool list on disk.

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
| Execution handoff | 3 sequential provider turns; all 11 tools registered for the lab configuration supplied each turn |
| Concrete tools | Real `write_file → read_file`; 32-byte bounded ASCII artifact; exact read-back |
| Persistence | Fresh disk reload yields roles `user, assistant` and tools `deepthink, write_file, read_file` |
| Cleanup | 0 active provider calls; temporary workspace removed |
| Communication observation | 0 calls observed in the manifest's listed Linux `strace` communication-syscall set |

The handoff manifest binds every committed blob under `k2do/`, project and
renderer inputs, plus the lock and its provenance at source commit
`165b7c1`—72 source files in total—by Git mode, blob ID, byte count, and
SHA-256. `check` regenerates all eight
artifacts and, on this Linux host, repeats both the canonical capture and the
bounded `strace` observation.
This is routed control-flow and side-effect evidence. It is not a live-provider
quality benchmark, an MCP lifecycle test, a network sandbox, or a latency
measurement.

See the [evidence protocol and boundaries](docs/evidence.md) and the
[handoff manifest](docs/agent-handoff-evidence/manifest.json).

## Direct engine fault-path evidence

![Receipt-derived terminal rendering of the K2DO offline trace lab](docs/deepthink-trace-evidence/trace-lab.svg)

*Captured canonical stdout rendered as a terminal view. The payload
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
- it binds twelve capture-relevant committed paths—the lab/engine modules,
  recorder, package initializers, project metadata, lock, and lock
  provenance—by Git mode, blob ID, SHA-256, and byte count;
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
| Production MCP discovery, catalog isolation, call recovery, cancellation, cleanup, and same-loop restart | Verified by the MCP fault receipt across 10 real subprocess generations |
| Artifact provenance and renderer-runtime-bound deterministic rendering | Verified by all three committed manifests and checkers |
| Background `AgentLoop.run()`, config loader wiring, refinement, memory consolidation, gateway, and Telegram paths | Adversarially unit-tested in the full suite; outside the current receipts |
| Live K2 provider behavior | Not captured |
| Answer quality, token cost, throughput, or latency | Not benchmarked |
| Hosted clean-runner CI | 424-test offline suite, hardened-boundary Ruff checks, fresh MCP capture, and read-only verification of all three visual bundles |
| Locked application dependency environment | Ubuntu 24.04 + CPython 3.12.13; 96 exact requirements, 2,082 SHA-256 hashes, and source-bound lock provenance |

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
requirements-ci-py312.lock             # 96 exact CPython 3.12 CI requirements
requirements-ci-py312.provenance.json   # generator, inputs, runtime, and lock digest
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
    mcp_fault_lab.py             # real stdio subprocess lifecycle matrix
    strict_mcp_stdio_server.py   # independent strict protocol fixture
    mcp_evidence_recorder.py     # source-bound PNG/SVG/GIF capture
  cli/                         # Typer/Rich live interface
  channels/                    # adapted channel integrations
docs/
  agent-handoff-evidence/      # receipt, real stdout, SVG/PNG/GIF, manifest
  deepthink-trace-evidence/    # direct-engine receipt and SVG evidence
  mcp-fault-evidence/          # real MCP stdout, matrix, timeline, GIF, manifest
  evidence.md                  # verification and maintenance protocols
  provenance.md                # upstream/contribution boundary
tests/                         # unit and adversarial evidence tests
```

The next evidence milestones are config-loader evidence, deterministic
refinement and memory traces, an explicit upstream-diff ledger, and an opt-in
sanitized live-provider capture. They remain tracked work, not completed
claims.

## License and attribution

K2DO is distributed under the MIT License. The repository preserves the
nanobot contributors’ notice; see [LICENSE](LICENSE) and
[docs/provenance.md](docs/provenance.md) before evaluating or reusing the
project.
