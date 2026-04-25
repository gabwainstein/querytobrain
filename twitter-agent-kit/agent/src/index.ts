import dotenv from 'dotenv';
import { logger, type IAgentRuntime, type Project, type ProjectAgent } from '@elizaos/core';
import { character } from './character';
import researchAgentApiPlugin from './plugins/research-api';
// @ts-ignore - Workspace dependency
import TwitterPlugin from '@elizaos/plugin-twitter';

dotenv.config();

const initCharacter = async ({ runtime }: { runtime: IAgentRuntime }) => {
  logger.info('Initializing Agent character');
  logger.info(`Name: ${character.name}`);
};

export const scientificAgent: ProjectAgent = {
  character,
  plugins: [researchAgentApiPlugin, TwitterPlugin as any],
  init: async (runtime: IAgentRuntime) => await initCharacter({ runtime }),
};

const project: Project = {
  agents: [scientificAgent],
};

// Export character for use in other files
export { character } from './character';

export default project;
