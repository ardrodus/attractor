# Attractor [ARCHIVED]

> **This repository has been archived.** The Attractor pipeline engine has been fully ported to [Dark Factory](https://github.com/ardrodus/dark-factory) and now lives at `factory/engine/`. All future development happens there.
>
> **New home:** [ardrodus/dark-factory](https://github.com/ardrodus/dark-factory) -- `factory/engine/` directory

---

## What was this?

An implementation of [StrongDM's Attractor](https://github.com/strongdm/attractor) -- a DOT-based pipeline runner for orchestrating multi-stage AI workflows. The three-layer architecture (Unified LLM Client, Coding Agent Loop, Pipeline Engine) achieved 100% spec coverage.

## Migration

All engine code has been ported to Dark Factory with local type definitions (no external dependency on this repo):

| Original (this repo) | New home (Dark Factory) |
|---|---|
| `src/attractor_pipeline/` | `factory/engine/` |
| `src/attractor_agent/` | `factory/engine/agent/` |
| `src/attractor_llm/types.py` | `factory/engine/types.py` |
| `tests/` | `factory/tests/test_engine/` |

All 381 engine unit tests and 24 end-to-end tests pass in the new location.

## Original README

The original README content is preserved in git history. See the commit prior to archival for the full documentation.

## Credits

This is an implementation of the [Attractor nlspec](https://github.com/strongdm/attractor) published by [StrongDM](https://www.strongdm.com/). Built using [Amplifier](https://github.com/microsoft/amplifier) with multi-model peer review across Claude, GPT, and Gemini.

## License

This implementation is provided as-is. The original Attractor specifications are licensed under [Apache License 2.0](https://github.com/strongdm/attractor/blob/main/LICENSE) by StrongDM.
