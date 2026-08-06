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
The manifest binds `pyproject.toml`, `requirements-evidence.txt`, and all
committed blobs under `k2do/`: 67 paths for the current capture.

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
Its manifest binds the ten explicitly selected source blobs that define that
capture. It does not claim to inventory every tracked repository file.

The source commit in each manifest produced its bundle. Each bundle is
published in a later commit so its capture starts from a clean source commit.

## Reader verification

Run these commands from a clone of this repository:

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

`requirements-evidence.txt` is a repository capture dependency, not a
promise that it is included in a built source distribution. The commands above
therefore intentionally start from a clone.

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

Pillow `12.3.0` is exactly pinned for the raster renderer. Application
dependencies still use lower bounds and the repository has no lockfile, so the
project does not yet claim a byte-for-byte reproducible package environment.

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
   ```

   Run only the command for each bundle being replaced.

7. Do not run `record` twice for the same source commit. Run the verification
   checker, inspect every visual at rendered size, and scan all public bytes
   for secrets or personal data.
8. If a visual is defective, discard the unpublished bundle, fix the renderer,
   commit a new clean source state, and capture from that commit.
9. Commit only the reviewed generated bundle as a separate evidence commit.

The recorders use private committed-`HEAD` snapshots, bounded process output,
process-group cleanup, descriptor-relative no-follow reads, exclusive writes,
`fsync`, and atomic no-replace publication. Adversarial tests cover source
mutation, dirty trees, divergent histories with identical blobs, hostile
output leaves, symlinks and FIFOs, publish races, output overflow, timeouts,
duplicate keys, artifact tampering, extra files, and renderer determinism.
