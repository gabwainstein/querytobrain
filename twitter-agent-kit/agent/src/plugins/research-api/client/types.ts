export type ResearchChatRequest = {
  message: string;
  conversationId?: string;
};

export type ResearchChatResponse = {
  text?: string;
  userId?: string;
  [key: string]: unknown;
};



