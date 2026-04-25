# Integration Map: ElizaOS + Research Backend + Twitter

This document maps where the connection code lives for the three integrations: **ElizaOS/ElizaOS** (agent runtime), a **research backend API**, and **Twitter** (posting and engagement).

---

## 1. High-level layout

| Integration | Role | Where it lives (in this kit) |
|-------------|------|------------------------------|
| **ElizaOS / ElizaOS** | Agent runtime (core, CLI, plugins) | Vendored externally: `vendor/eliza-vendor/`; agent code in `agent/` |
| **Research backend** | Research summaries and Q&A API | `agent/src/plugins/research-api/` + env `RESEARCH_API_URL` |
| **Twitter** | Posting and replies (OAuth 1.0a) | Vendored: `vendor/eliza-vendor/packages/plugin-twitter/` + `agent/.env` credentials |

---

## 2. ElizaOS / ElizaOS

| What | Location | Notes |
|------|----------|--------|
| **Core runtime** | (vendor) `vendor/eliza-vendor/packages/core` | `agent/package.json` → `file:../vendor/eliza-vendor/packages/core` |
| **CLI** | (vendor) `vendor/eliza-vendor/packages/cli` | `elizaos start` / `elizaos dev` |
| **Agent entry** | `agent/src/index.ts` | Imports character, registers plugins (research-api, Twitter) |
| **Character** | `agent/characters/neuroscience_agent.json` | Name, system prompt, postExamples, topics |
| **Character loader** | `agent/src/character.ts` | Loads JSON from `AGENT_CONFIG_PATH` or `characters/neuroscience_agent.json` |
| **Other Eliza plugins** | (vendor) plugin-bootstrap, plugin-sql | In `agent/package.json` as `file:../vendor/...` |

The agent **project** is `agent/`; the **runtime and Twitter plugin** come from the vendored eliza-vendor monorepo (not included in this kit; clone the full integration repo or link packages).

---

## 3. Research backend

| What | Location | Notes |
|------|----------|--------|
| **Plugin (custom)** | `agent/src/plugins/research-api/` | Only custom backend integration code in the agent |
| **Plugin entry** | `agent/src/plugins/research-api/index.ts` | Registers service, `RESEARCH_CONTEXT` provider, `ASK_RESEARCH` action |
| **API client** | `agent/src/plugins/research-api/client/request.ts` | Calls `POST {baseUrl}/api/chat` |
| **Config / env** | `agent/src/plugins/research-api/client/validator.ts` | Reads `RESEARCH_API_URL`, `RESEARCH_BEARER_TOKEN`, etc. |
| **Registration** | `agent/src/index.ts` | `import research-agentApiPlugin from './plugins/research-api'` and adds to `plugins: [research-agentApiPlugin, TwitterPlugin]` |

Env vars (in `agent/.env` or `.env.example`): `RESEARCH_API_URL`, `RESEARCH_BEARER_TOKEN`, `RESEARCH_SUMMARY_PROMPT`, `RESEARCH_REQUEST_TIMEOUT_MS`, `RESEARCH_SUMMARY_TTL_MINUTES`, `RESEARCH_CONVERSATION_ID`.

---

## 4. Twitter

| What | Location | Notes |
|------|----------|--------|
| **Plugin (vendored)** | (vendor) `vendor/eliza-vendor/packages/plugin-twitter/` | Patched in the full integration repo |
| **Registration** | `agent/src/index.ts` | `import TwitterPlugin from '@elizaos/plugin-twitter'` (package.json `file:../vendor/...`) |
| **Credentials** | `agent/.env` | `TWITTER_API_KEY`, `TWITTER_API_SECRET_KEY`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET` |
| **Behavior** | `agent/.env` | `TWITTER_DRY_RUN`, `TWITTER_ENABLE_POST`, `TWITTER_POST_INTERVAL_MIN/MAX`, `TWITTER_ENABLE_REPLIES`, etc. |

---

## 5. Scripts (in this kit)

| Script | Purpose |
|--------|---------|
| `scripts/setup-research-api.sh` | Interactive setup for `RESEARCH_API_URL` and related env |
| `scripts/verify-research-api.sh` | Check backend config and connectivity |
| `scripts/check-research-agent-status.ts` | Programmatic backend status check |
| `scripts/update-twitter-credentials.sh` (and `.ps1`) | Refresh Twitter OAuth tokens |

---

## 6. Summary

- **Custom code in this kit:** `agent/src/plugins/research-api/` (backend API client, context provider, action).
- **Wiring:** `agent/src/index.ts`, `agent/src/character.ts`, `agent/characters/`, `agent/package.json`, `agent/.env.example`.
- **Vendor required to run:** ElizaOS monorepo (core, CLI, plugin-twitter, plugin-bootstrap, plugin-sql). In the full integration repo that is `vendor/eliza-vendor`.
