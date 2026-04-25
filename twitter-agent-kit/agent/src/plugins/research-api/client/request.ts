import { type ResearchChatRequest, type ResearchChatResponse } from './types';

type ChatRequestOptions = {
  baseUrl: string;
  bearerToken?: string | null;
  timeoutMs: number;
  payload: ResearchChatRequest;
};

export async function callResearchChat({
  baseUrl,
  bearerToken,
  timeoutMs,
  payload,
}: ChatRequestOptions): Promise<ResearchChatResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(bearerToken ? { Authorization: `Bearer ${bearerToken}` } : {}),
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ResearchAgent API error (${response.status}): ${errorText}`);
    }

    const data = (await response.json()) as ResearchChatResponse;
    return data;
  } finally {
    clearTimeout(timeout);
  }
}



