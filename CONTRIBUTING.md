# Contributing

Contributions are welcome, but please **open an issue first** — do not open pull requests directly.

## Issues, not PRs

1. **Open an issue** describing the bug, feature, or improvement you have in mind.
2. Wait for discussion and alignment on the approach.
3. Only submit a PR once the issue has been acknowledged and the direction agreed upon.

PRs that are not linked to a prior discussion in an issue will be closed.

## Why?

This keeps the project cohesive. Many changes touch the agent loop, tool registry, or memory subsystem where small surface-level edits can have subtle effects on LLM behaviour. Discussing first saves everyone time.

## Bug reports

Include:
- What you did (exact command or tool call)
- What you expected
- What happened (full stderr output — telemetry goes there)
- Your `config.json` (redact API keys)
- The model and endpoint you're using
