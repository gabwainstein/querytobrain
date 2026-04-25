import { type IAgentRuntime } from '@elizaos/core';

export function getResearchBaseUrl(runtime: IAgentRuntime): string | null {
  const baseUrl =
    (runtime.getSetting?.('RESEARCH_API_URL') as string | null) || process.env.RESEARCH_API_URL;
  if (!baseUrl) {
    return null;
  }
  return baseUrl.replace(/\/+$/, '');
}

export function getResearchBearerToken(runtime: IAgentRuntime): string | null {
  const token =
    (runtime.getSetting?.('RESEARCH_BEARER_TOKEN') as string | null) ||
    process.env.RESEARCH_BEARER_TOKEN;
  return token || null;
}

export function getResearchConversationId(runtime: IAgentRuntime): string | null {
  const conversationId =
    (runtime.getSetting?.('RESEARCH_CONVERSATION_ID') as string | null) ||
    process.env.RESEARCH_CONVERSATION_ID;
  return conversationId || null;
}

export function getResearchTimeoutMs(runtime: IAgentRuntime, fallback: number): number {
  const raw =
    (runtime.getSetting?.('RESEARCH_REQUEST_TIMEOUT_MS') as string | null) ||
    process.env.RESEARCH_REQUEST_TIMEOUT_MS;
  if (!raw) {
    return fallback;
  }
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function getSummaryTtlMinutes(runtime: IAgentRuntime, fallback: number): number {
  const raw =
    (runtime.getSetting?.('RESEARCH_SUMMARY_TTL_MINUTES') as string | null) ||
    process.env.RESEARCH_SUMMARY_TTL_MINUTES;
  if (!raw) {
    return fallback;
  }
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function getSummaryPrompt(runtime: IAgentRuntime, fallback: string): string {
  const prompt =
    (runtime.getSetting?.('RESEARCH_SUMMARY_PROMPT') as string | null) ||
    process.env.RESEARCH_SUMMARY_PROMPT;
  return prompt?.trim() ? prompt : fallback;
}

export function validatePrompt(prompt: string): void {
  if (!prompt || !prompt.trim()) {
    throw new Error('ResearchAgent prompt cannot be empty.');
  }
}



