# Twitter Agent Kit (ElizaOS + Research Backend + Twitter)

This directory contains the **agent and scripts** that connect **ElizaOS/ElizaOS**, a **research backend API**, and **Twitter**. It is a neutral reference implementation for running a neuroscience-oriented social or research agent.

## What's included

| Path | Description |
|------|-------------|
| **agent/** | ElizaOS agent project: character, research-backend plugin, and plugin registration. Uses `@elizaos/plugin-twitter` and `@elizaos/core` from a vendored ElizaOS monorepo (see below). |
| **scripts/** | Setup and verification: backend API (`setup-research-api.sh`, `verify-research-api.sh`, `check-research-agent-status.ts`) and Twitter credentials (`update-twitter-credentials.sh`, `.ps1`). |
| **config/** | Example config files (retention, typing fallback). |
| **INTEGRATION_MAP.md** | Map of where each integration (ElizaOS, research backend, Twitter) is wired. |

## Integrations

1. **ElizaOS / ElizaOS** — Agent runtime. The `agent/` project depends on packages from a **vendored** ElizaOS monorepo (e.g. `@elizaos/core`, `@elizaos/plugin-twitter`, `@elizaos/plugin-bootstrap`, `@elizaos/plugin-sql`). In the original repo these are `file:../vendor/eliza-vendor/packages/...`. To run this kit you must either clone the full integration repo (which has `vendor/eliza-vendor`) or install/link those packages yourself.
2. **Research backend** — Custom plugin in `agent/src/plugins/research-api/`. Calls `RESEARCH_API_URL` (e.g. `http://localhost:3000`) for research summaries and injects them into context (`RESEARCH_CONTEXT`). Optional: agent works with a fallback if the API is unavailable.
3. **Twitter** — Posting and replies via `@elizaos/plugin-twitter` from the vendored monorepo. Configure `TWITTER_*` credentials in `agent/.env`.

## Quick setup (when vendor is available)

1. Clone the full integration repo (or ensure `vendor/eliza-vendor` is present and `agent/package.json` points to it).
2. In `agent/`, copy `.env.example` to `.env` and set:
   - Model provider (e.g. `OPENROUTER_API_KEY`, `OPENAI_API_KEY`)
   - Twitter OAuth 1.0a: `TWITTER_API_KEY`, `TWITTER_API_SECRET_KEY`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET`
   - Optional: `RESEARCH_API_URL=http://localhost:3000` (and run `./scripts/setup-research-api.sh` from repo root).
3. Install and run:
   ```bash
   cd agent
   bun install
   bun run dev
   ```

## Docs in this repo

- **Integration map:** `twitter-agent-kit/INTEGRATION_MAP.md`
- **Backend / architecture references:** `docs/external-specs/` (e.g. `AGENTKIT_EXPLAINED.md`, `architecture.md`, `DEEP_ANALYSIS.md`)

## License and origin

This kit is derived from earlier integrations that connected ElizaOS, a research backend, and Twitter. Vendor dependencies (ElizaOS monorepo, optional backend services) have their own licenses and repos.
