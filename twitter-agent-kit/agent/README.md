# Twitter agent (ElizaOS)

This project runs the community research agent on ElizaOS with Twitter integration and
ResearchAgent research context injection.

## What It Includes

- `characters/neuroscience_agent.json`: the agent personality and posting style.
- `src/plugins/research-api`: pulls summaries from ResearchAgent AgentKit.
- `@elizaos/plugin-twitter`: posting, replies, and engagement automation.
- `@elizaos/plugin-openai`: text generation provider (OpenAI API).

## Quickstart

1. Copy `.env.example` to `.env` and fill in:
   - Twitter OAuth 1.0a credentials.
   - `OPENAI_API_KEY` for text generation.
   - `RESEARCH_API_URL` for the ResearchAgent API server.
2. Install dependencies and start the agent:
   - `bun install`
   - `bun run dev`

## Customization

- Update `characters/neuroscience_agent.json` to refine voice, topics, and post examples.
- Adjust the ResearchAgent summary prompt via `RESEARCH_SUMMARY_PROMPT`.
- Toggle Twitter automation via `TWITTER_ENABLE_POST`, `TWITTER_ENABLE_REPLIES`, and `TWITTER_DRY_RUN`.

## Key Files

- `src/index.ts`: project entry point and plugin wiring.
- `src/character.ts`: loads the local character config.
- `src/plugins/research-api`: API client + provider + action.
