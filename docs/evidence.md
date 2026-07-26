# DeepThink evidence protocol

This document explains what the committed evidence bundle proves, how a reader
can verify it, and how a maintainer can replace it without hand-editing
generated output.

## Published bundle

`docs/deepthink-trace-evidence/` contains:

| File | Purpose |
| --- | --- |
| `receipt.json` | Canonical output of the deterministic offline trace lab |
| `trace-lab.txt` | Exact command banner followed by byte-exact receipt output |
| `trace-lab.svg` | Terminal rendering derived from the receipt |
| `call-dag.svg` | Synthesis call graph derived from receipt nodes and edges |
| `orchestration-properties.svg` | Scenario cards derived from receipt fields |
| `manifest.json` | Source bindings, runtime observation, artifact hashes, sizes, and media types |

The receipt SHA-256 for the current scenario contract is
`f66e1db30dd79d77318a20ae86a0aa450a663e8f5cdff351557ca6fd1d0da80d`.
The manifest records the source commit that produced the bundle; the evidence
commit itself is intentionally a later commit.

## Reader verification

From an installed development environment:

```bash
python -m k2do.labs.deepthink_trace
python -m pytest -q tests/test_deepthink_trace_lab.py tests/test_trace_evidence_recorder.py
python -m k2do.labs.trace_evidence_recorder check
```

`check` is read-only. It rejects:

- missing, extra, non-regular, or symlinked bundle entries;
- duplicate JSON keys, unsupported schemas, wrong sizes, or wrong hashes;
- source modes, Git blobs, content hashes, or byte counts that differ from the
  recorded source inventory;
- transcript bytes that differ from the command plus canonical receipt;
- SVG bytes that differ from a fresh deterministic render;
- the recorder's explicit forbidden SVG patterns: doctypes/entities, scripts,
  `foreignObject`, metadata, `href` references, JavaScript/data references, and
  external CSS URLs;
- invalid XML or a root element that is not an SVG with `role="img"`.

When the capture commit still exists, the checker verifies that object and its
tree as additional provenance. The squash-safe authority is the recorder's
complete selected capture-source inventory: path, mode, Git blob ID, SHA-256,
and byte count. It is not an inventory of every tracked repository file. An
ancestry relationship alone is never accepted as proof.

## Evidence boundary

The lab exercises:

- the default production `classify_query` function;
- a direct production `DeepThinkEngine`;
- parallel role-configured provider calls;
- primary/fallback model paths;
- thinker timeout and cancellation propagation;
- Judge synthesis and Judge-failure degradation;
- cleanup counters after completion or caller cancellation.

The provider is a strict scripted fake. It validates request contracts and
returns synthetic labels rather than model content. Prompts, response text,
credentials, provider endpoints, absolute paths, and elapsed timings are
excluded from the receipt.

The capture records zero communication syscalls observed in one Linux
`strace` run of the lab and its child threads for the explicit set listed in
the manifest. Local event-loop socketpair creation is outside that set. This is
not a sandbox, firewall, or proof that arbitrary future code cannot communicate.

The runtime version is recorded, but dependency versions are not lock-pinned.
The bundle therefore supports source and output verification, not a claim of a
byte-for-byte reproducible package environment.

The bundle does not exercise live model providers, AgentLoop configuration and
automatic handoff, tools, refinement, memory, gateway, Telegram, or dashboard
rendering. It makes no answer-quality, token-cost, throughput, or latency claim.

## Maintainer replacement protocol

Generated artifacts must never be hand-edited.

1. Change the lab, recorder, production surface, or tests as needed.
2. Remove the previous generated bundle.
3. Run the focused and full test suites.
4. Commit the source changes and bundle removal.
5. Confirm that `HEAD` is committed, the worktree is clean, and
   `docs/deepthink-trace-evidence/` is absent.
6. Run exactly one production capture:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 \
     python -m k2do.labs.trace_evidence_recorder record
   ```

7. Do not re-run `record` for that source commit. Run the read-only checker,
   inspect every visual at its rendered size, and scan for secrets or personal
   data.
8. If any generated visual is defective, discard the unpublished bundle,
   correct the renderer, commit a new clean source state, and capture from that
   new commit.
9. Commit only the reviewed generated bundle as a separate evidence commit.

The recorder uses a private committed-HEAD snapshot, bounded stdout/stderr
drains, process-group termination, descriptor-relative no-follow file access,
exclusive creation, directory/file `fsync`, and atomic no-replace publication.
Its current adversarial tests cover dirty trees and mid-capture source
mutation; squash/divergent histories with identical selected blobs; hostile
file, directory, symlink, and FIFO output leaves; no-replace publish races;
extra staging entries; stdout overflow and timeout cleanup of TERM-ignoring
descendants; artifact tampering and symlinks; duplicate manifest keys; extra
published files; and verification from a restricted-umask clone. A separate
Linux test runs the real capture and asserts that its explicit communication
syscall trace is empty.
