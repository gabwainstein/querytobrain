import {
  type Action,
  type Content,
  type HandlerCallback,
  type IAgentRuntime,
  type Memory,
  type Plugin,
  type Provider,
  type ProviderResult,
  Service,
  type State,
  logger,
} from '@elizaos/core';
import { callResearchChat } from './client/request';
import {
  getResearchBaseUrl,
  getResearchBearerToken,
  getResearchConversationId,
  getResearchTimeoutMs,
  getSummaryPrompt,
  getSummaryTtlMinutes,
  validatePrompt,
} from './client/validator';

const DEFAULT_TIMEOUT_MS = 20000;
const DEFAULT_SUMMARY_TTL_MINUTES = 60;
const DEFAULT_SUMMARY_PROMPT =
  'Summarize the most important evidence-based insights on nootropics, cognitive health, ' +
  'and longevity. Use short bullet points and avoid medical claims.';

type SummaryCache = {
  text: string;
  createdAt: number;
  prompt: string;
};

type ResearchConfig = {
  baseUrl: string | null;
  bearerToken: string | null;
  conversationId: string | null;
  timeoutMs: number;
  summaryPrompt: string;
  summaryTtlMs: number;
};

export class ResearchApiService extends Service {
  static serviceType = 'research-agent_api';
  capabilityDescription = 'Fetches research answers and summaries from ResearchAgent AgentKit.';

  private apiConfig!: ResearchConfig;
  private summaryCache: SummaryCache | null = null;

  constructor(runtime?: IAgentRuntime) {
    super(runtime);
  }

  static async start(runtime: IAgentRuntime) {
    const config: ResearchConfig = {
      baseUrl: getResearchBaseUrl(runtime),
      bearerToken: getResearchBearerToken(runtime),
      conversationId: getResearchConversationId(runtime),
      timeoutMs: getResearchTimeoutMs(runtime, DEFAULT_TIMEOUT_MS),
      summaryPrompt: getSummaryPrompt(runtime, DEFAULT_SUMMARY_PROMPT),
      summaryTtlMs: getSummaryTtlMinutes(runtime, DEFAULT_SUMMARY_TTL_MINUTES) * 60 * 1000,
    };

    const service = new ResearchApiService(runtime);
    service.apiConfig = config;

    if (!config.baseUrl) {
      logger.warn(
        'RESEARCH_API_URL is not configured; ResearchAgent API integration will be disabled.'
      );
      logger.info(
        'To enable: Set RESEARCH_API_URL in .env (e.g., http://localhost:3000) or run ./scripts/setup-research-api.sh'
      );
    } else {
      logger.info(`ResearchAgent API configured: ${config.baseUrl}`);
      logger.info('ResearchAgent integration is enabled. Research context will be injected when API is accessible.');
    }

    return service;
  }

  async stop(): Promise<void> {
    // No long-lived connections right now; clear caches for cleanliness.
    this.summaryCache = null;
  }

  async ask(prompt: string, conversationId?: string): Promise<string> {
    validatePrompt(prompt);
    if (!this.apiConfig.baseUrl) {
      throw new Error('RESEARCH_API_URL is not configured.');
    }

    const response = await callResearchChat({
      baseUrl: this.apiConfig.baseUrl,
      bearerToken: this.apiConfig.bearerToken,
      timeoutMs: this.apiConfig.timeoutMs,
      payload: {
        message: prompt,
        conversationId: conversationId || this.apiConfig.conversationId || undefined,
      },
    });

    return response.text?.trim() || '';
  }

  async getSummary(forceRefresh = false): Promise<string | null> {
    if (!this.apiConfig.baseUrl) {
      return null;
    }

    const now = Date.now();
    if (!forceRefresh && this.summaryCache) {
      const isValid =
        this.summaryCache.prompt === this.apiConfig.summaryPrompt &&
        now - this.summaryCache.createdAt < this.apiConfig.summaryTtlMs;
      if (isValid) {
        return this.summaryCache.text;
      }
    }

    const summary = await this.ask(this.apiConfig.summaryPrompt);
    if (!summary) {
      return null;
    }

    this.summaryCache = {
      text: summary,
      createdAt: now,
      prompt: this.apiConfig.summaryPrompt,
    };

    return summary;
  }
}

const researchAgentContextProvider: Provider = {
  name: 'RESEARCH_CONTEXT',
  description: 'Injects a cached ResearchAgent research summary into agent context.',
  get: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State
  ): Promise<ProviderResult> => {
    try {
      const service = runtime.getService(
        ResearchApiService.serviceType
      ) as ResearchApiService | null;
      if (!service) {
        return { text: '' };
      }

      const summary = await service.getSummary(false);
      if (!summary) {
        return { text: '' };
      }

      logger.debug('RESEARCH_CONTEXT: Research summary injected into agent context');
      return {
        text: `[RESEARCH_CONTEXT]\n${summary}\n[/RESEARCH_CONTEXT]`,
      };
    } catch (error) {
      logger.warn('RESEARCH_CONTEXT provider failed:', error);
      logger.debug('Agent will continue without ResearchAgent research context (using fallback)');
      return { text: '' };
    }
  },
};

const askResearchAction: Action = {
  name: 'ASK_RESEARCH',
  description: 'Query the ResearchAgent AgentKit API for a research answer.',
  examples: [
    [
      {
        name: '{{name1}}',
        content: { text: 'Summarize the evidence for creatine and cognition.' },
      },
      {
        name: '{{name2}}',
        content: {
          text: 'I will fetch a concise evidence summary from ResearchAgent.',
          actions: ['ASK_RESEARCH'],
        },
      },
    ],
  ],
  validate: async (runtime: IAgentRuntime) => {
    const service = runtime.getService(ResearchApiService.serviceType);
    return Boolean(service);
  },
  handler: async (
    runtime: IAgentRuntime,
    message: Memory,
    _state: State,
    _options: unknown,
    callback: HandlerCallback
  ) => {
    const service = runtime.getService(
      ResearchApiService.serviceType
    ) as ResearchApiService | null;
    if (!service) {
      throw new Error('ResearchApiService not available.');
    }

    const prompt = message.content.text?.trim() || '';
    validatePrompt(prompt);

    const answer = await service.ask(prompt);
    const responseContent: Content = {
      text: answer || 'ResearchAgent returned an empty response.',
      actions: ['ASK_RESEARCH'],
      source: message.content.source,
    };

    await callback(responseContent);
    return responseContent;
  },
};

const researchAgentApiPlugin: Plugin = {
  name: 'querytobrain-research-agent',
  description: 'Integrates ResearchAgent AgentKit for research summaries and Q&A.',
  services: [ResearchApiService],
  providers: [researchAgentContextProvider],
  actions: [askResearchAction],
};

export default researchAgentApiPlugin;

