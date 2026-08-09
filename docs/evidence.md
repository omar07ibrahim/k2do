# K2DO evidence protocol

This document defines what the committed offline evidence proves, how a reader
can verify it, and how a maintainer can replace it without hand-editing
generated output.

## Published bundles

### Routed handoff

`docs/agent-handoff-evidence/` contains:

| File | Purpose |
| --- | --- |
| `receipt.json` | Canonical routed-workflow receipt |
| `handoff-lab.txt` | Reader-command banner followed by byte-exact captured stdout |
| `terminal.svg` | Receipt-derived vector terminal rendering |
| `terminal.png` | Receipt-derived raster terminal rendering |
| `architecture.svg` | Ordered routed-processing architecture from receipt nodes |
| `tool-timeline.svg` | Reasoning, tool, and persistence sequence from the receipt |
| `contract-matrix.svg` | Verified contract fields and bounded communication observation |
| `workflow-demo.gif` | Nine-frame illustrative replay of receipt workflow nodes |
| `manifest.json` | Source bindings, renderer runtime, artifact hashes, sizes, and media types |

The routed receipt SHA-256 is
`12b6c89a40bf2ca4f775a9ffa0c68b0525cde3f853dd4cf089ebb5e333f91db8`.
The manifest binds all committed blobs under `k2do/`, project and renderer
inputs, plus `requirements-ci-py312.lock` and its provenance: 72 paths for
the current capture.

### Direct DeepThink fault paths

`docs/deepthink-trace-evidence/` contains:

| File | Purpose |
| --- | --- |
| `receipt.json` | Canonical direct-engine scenario receipt |
| `trace-lab.txt` | Reader-command banner followed by byte-exact captured stdout |
| `trace-lab.svg` | Receipt-derived vector terminal rendering |
| `call-dag.svg` | Direct synthesis call graph from receipt nodes and edges |
| `orchestration-properties.svg` | Fallback, timeout, degradation, and cleanup cards |
| `manifest.json` | Selected source bindings, artifact hashes, sizes, and runtime |

The direct-engine receipt SHA-256 is
`f66e1db30dd79d77318a20ae86a0aa450a663e8f5cdff351557ca6fd1d0da80d`.
Its manifest binds twelve explicitly selected source blobs, including the
CPython 3.12 lock and its provenance. It does not claim to inventory every
tracked repository file.

### Production MCP fault lifecycle

`docs/mcp-fault-evidence/` contains a canonical receipt and byte-exact CLI
transcript, a real 1600×3918 terminal PNG, four source-derived SVG views, a
six-frame GIF replay, and a manifest. The receipt SHA-256 is
`e69543846b4563901637ec7d8a35035a5efad00eda0f6c3e7d36480fe152346c`.
Its 12 source bindings cover the read-only workflows, production MCP client,
strict stdio fixtures, recorder, and focused tests. The lab starts ten real
subprocess generations and verifies discovery, protocol failure, repeated
cancellation, cleanup, and same-loop recovery without credentials.

The source commit in each manifest produced its bundle. Each bundle is
published in a later commit so its capture starts from a clean source commit.

## Reader verification

Run these commands from a clone of this repository:

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

`requirements-ci-py312.lock` is the reviewed Ubuntu 24.04 / CPython 3.12
evidence environment; its provenance records the exact generator, inputs, and
digest. `requirements-evidence.txt` remains one compiler input and is not
promised as part of a built source distribution, so verification intentionally
starts from a clone.

Neither `check` command alters tracked files or a published bundle. During
verification it may create and remove transient private snapshots and fixture
files. Depending on the bundle, the checker rejects:

- missing, extra, non-regular, executable, or symlinked bundle entries;
- duplicate JSON keys, unsupported schemas, wrong media types, sizes, or
  SHA-256 hashes;
- source modes, Git blob IDs, content hashes, or byte counts that differ from
  the manifest;
- transcript bytes that differ from the reader banner plus canonical receipt;
- SVG, PNG, or GIF bytes that differ from a fresh deterministic render;
- unsafe SVG constructs, malformed XML, invalid raster formats or dimensions,
  and unexpected GIF frame contracts;
- dirty or changing capture sources and replaced capture objects.

On supported Linux hosts, each checker also recaptures its scenario from a
private committed-`HEAD` snapshot and repeats the bounded `strace`
observation. When the capture commit still exists, its object and tree are
verified as additional provenance. The squash-safe authority is the complete
recorded source inventory; ancestry alone is never accepted as proof.

## Evidence boundaries

### What the routed receipt exercises

The handoff lab:

- publishes and consumes one real `MessageBus` inbound message;
- invokes the production `AgentLoop._process_message` routed-processing path;
- uses the real context builder and production classifier with the lab's fixed
  `0.6` threshold;
- runs three thinker coroutines scheduled by the production `DeepThinkEngine`
  against a strict scripted provider and gates the Judge until every thinker
  is terminal;
- hands the Judge result to the normal tool loop with all eleven tool schemas
  registered for the lab configuration;
- executes the real workspace-restricted `write_file` and `read_file`
  implementations, verifies exact read-back, and removes the private fixture
  workspace;
- persists user and assistant session records, publishes and consumes the
  outbound queue entry, then verifies the records with a fresh
  `SessionManager`.

The lab deliberately calls `_process_message` and publishes the returned
outbound message itself. It does not execute the background
`AgentLoop.run()` loop or a channel's outbound dispatcher. It also does not
exercise MCP lifecycle, config-file loading, shell/web/message/spawn tools,
refinement, memory consolidation, gateway, Telegram, or live providers.

### What the direct-engine receipt exercises

The direct lab covers the default `classify_query` function and an isolated
production `DeepThinkEngine`: bounded parallel thinker selection, primary
model failure followed by fallback, a categorical thinker timeout, Judge
gating, Judge-failure degradation, caller cancellation, and zero-active-call
cleanup.

### Shared limits

Both providers are strict scripted fakes. They validate request contracts and
return synthetic fixtures. The receipts exclude raw prompts, model responses,
credentials, provider endpoints, absolute paths, and elapsed timings. Neither
bundle supports claims about answer quality, live-provider behavior, token
cost, throughput, or latency.

Each manifest reports zero calls observed in one Linux `strace` run for its
explicit communication-syscall set. That is a bounded observation, not a
sandbox, firewall, or proof about arbitrary future code. Passive event-loop
bookkeeping is outside the listed set.

The PNG and GIF are generated from the canonical receipt under the recorded
Pillow, FreeType, and zlib runtime. They are reproducible evidence renderings,
not photographs of a terminal or live screen recordings. GIF frame delays are
illustrative and make no timing claim.

The hosted evidence path fixes CPython 3.12.13 and installs 96 exact
requirements under 2,082 SHA-256 hashes; Pillow `12.3.0` is part of that
lock. This is a hash-verified dependency contract for the named runner and
runtime, not a claim that operating-system images or arbitrary Python targets
are byte-for-byte identical.

## Maintainer replacement protocol

Generated bundle files must never be hand-edited.

1. Change production code, lab, recorder, tests, or capture dependencies.
2. Identify every bundle whose source binding or renderer/runtime contract is
   affected.
3. Remove each affected generated bundle and commit the source change and
   bundle removal.
4. Run focused tests and the full suite.
5. Confirm that `HEAD` is committed, the worktree is clean, and the affected
   output directory is absent.
6. Run exactly one appropriate production capture:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 \
     python -m k2do.labs.handoff_evidence_recorder record

   PYTHONDONTWRITEBYTECODE=1 \
     python -m k2do.labs.trace_evidence_recorder record

   PYTHONDONTWRITEBYTECODE=1 \
     python -m k2do.labs.mcp_evidence_recorder record --replace
   ```

   Run only the command for each bundle being replaced. The handoff and direct
   recorders require an absent destination; the MCP recorder performs an
   atomic replacement in its clean checkout.

7. Do not run `record` twice for the same source commit. Run the verification
   checker, inspect every visual at rendered size, and scan all public bytes
   for secrets or personal data.
8. If a visual is defective, discard the unpublished bundle, fix the renderer,
   commit a new clean source state, and capture from that commit.
9. Commit only the reviewed generated bundle as a separate evidence commit.

The handoff and direct recorders use private committed-`HEAD` snapshots,
bounded process output, process-group cleanup, descriptor-relative no-follow
reads, exclusive writes, `fsync`, and atomic no-replace publication. The MCP
recorder uses an atomic backup-and-replace boundary in its read-only workflow. Adversarial tests cover source
mutation, dirty trees, divergent histories with identical blobs, hostile
output leaves, symlinks and FIFOs, publish races, output overflow, timeouts,
duplicate keys, artifact tampering, extra files, and renderer determinism.
