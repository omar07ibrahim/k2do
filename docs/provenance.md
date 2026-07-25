# K2DO provenance and contribution map

K2DO is a modified derivative of
[HKUDS/nanobot](https://github.com/HKUDS/nanobot), distributed under the MIT
License. The existing `LICENSE` preserves the nanobot contributors' copyright
notice and permission text.

This boundary matters for portfolio review: the underlying generic agent
framework must not be presented as original K2DO work.

## Reconstructed baseline

The initial K2DO commit (`4c60c637804c4659e633c23d0d96a2a5ae150624`) was
published as one squashed hackathon snapshot, so it does not retain the
upstream commit graph. A generated audit file in that snapshot explicitly
recorded that the working repository had been cloned from
`https://github.com/HKUDS/nanobot` and that its source notes were generated
against nanobot commit
[`4617043d2c91452a456ce5b7283789dacf42fc3c`](https://github.com/HKUDS/nanobot/commit/4617043d2c91452a456ce5b7283789dacf42fc3c).
That commit is a provenance anchor for the old notes, not a claim that every
K2DO file was copied from exactly that tree.

An additional comparison against nanobot commit
[`16127d49f98cccb47fdf99306f41c38934821bc9`](https://github.com/HKUDS/nanobot/commit/16127d49f98cccb47fdf99306f41c38934821bc9),
the last upstream commit before the K2DO publication time, confirms substantial
shared implementation in the generic runtime. The precise per-file source
snapshot still needs a history reconstruction; this repository intentionally
does not invent one.

## Upstream-derived framework boundary

The following areas retain substantial nanobot structure or implementation and
must be reviewed as adapted upstream code:

- asynchronous inbound/outbound message bus;
- agent context, memory, tool registry, filesystem/shell/web tools, and
  subagent shell;
- channel base classes and Telegram transport;
- session persistence, cron scheduling, and heartbeat service;
- generic provider interfaces, LiteLLM adapter, transcription adapter, and
  CLI/onboarding shell;
- bundled nanobot-style skills and their helper scripts.

K2DO keeps the upstream MIT notice for these portions.

## K2DO-specific slice in the hackathon baseline

The initial snapshot adds or materially changes these areas:

- dual K2 Think/K2 Instruct configuration and model fallback selection;
- query complexity routing and explicit simple/deepthink/refine modes;
- parallel thinker roles followed by a judge synthesis step;
- iterative critic/refiner orchestration;
- timeouts and handoff guards around DeepThink execution;
- K2 response sanitization and inline tool-call recovery;
- tests for routing, handoff, fallback, memory consolidation, and reasoning
  control flow.

These are code-presence statements, not quality or originality claims. Each
area still requires isolated tests, reproducible traces, and a reviewed diff
against the reconstructed upstream baseline before it is featured as portfolio
evidence.

## Removed generated audit dump

The original root-level `dock.txt` was an unreviewed, automatically generated
nanobot wiki dump. It included host-specific paths, stale upstream descriptions,
and copied documentation that did not describe K2DO's verified behavior. It was
removed in favor of this concise, source-linked provenance record.

## Next evidence milestones

1. Pin a reproducible Python toolchain and add CI.
2. Add a deterministic fake-provider harness that exercises router,
   thinker/judge, fallback, cancellation, and tool boundaries without API keys.
3. Measure only properties the harness can prove: route decisions, call DAG,
   cancellation behavior, bounded concurrency, and deterministic receipts.
4. Generate real CLI transcripts and diagrams from that harness.
5. Maintain an explicit upstream-diff ledger for every retained generic module.
