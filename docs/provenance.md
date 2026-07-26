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
area still requires a reviewed diff against the reconstructed upstream
baseline before it is treated as an originality claim.

The deterministic offline lab now supplies narrower behavioral evidence for the
default query router and a direct `DeepThinkEngine`: selected thinker
concurrency, thinker model fallback, categorical timeout, Judge gating and
degradation, caller cancellation, and zero-active cleanup. Its strict scripted
provider, canonical receipt, source-bound manifest, transcript, and generated
visuals are documented in [the evidence protocol](evidence.md).

That receipt does not exercise the generic AgentLoop, configuration wiring,
automatic execution handoff, tools, refinement, memory, live providers,
gateway, Telegram, or dashboard. Unit tests for some of those paths do not
expand the receipt's evidence boundary.

## Removed generated audit dump

The original root-level `dock.txt` was an unreviewed, automatically generated
nanobot wiki dump. It included host-specific paths, stale upstream descriptions,
and copied documentation that did not describe K2DO's verified behavior. It was
removed in favor of this concise, source-linked provenance record.

## Evidence milestones

Completed:

1. A deterministic credential-free provider harness exercises the router and
   direct thinker/Judge engine.
2. The canonical receipt measures route decisions, call DAG, model fallback,
   categorical timeout, cancellation, bounded concurrency, and cleanup.
3. A real captured transcript and receipt-derived SVGs are bound to committed
   source blobs and checked by a deterministic renderer.

Remaining:

1. Pin a reproducible Python dependency set and publish CI.
2. Add source-bound receipts for AgentLoop/config handoff, tool execution, and
   refinement.
3. Add an opt-in sanitized live-provider capture without publishing prompts,
   responses, credentials, endpoints, or personal data.
4. Maintain an explicit upstream-diff ledger for every retained generic module.
