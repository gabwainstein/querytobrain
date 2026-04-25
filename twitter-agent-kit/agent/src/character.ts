import fs from 'node:fs';
import path from 'node:path';
import { type Character } from '@elizaos/core';

const defaultConfigPath = 'characters/neuroscience_agent.json';
const configPath = process.env.AGENT_CONFIG_PATH || defaultConfigPath;
const resolvedConfigPath = path.resolve(configPath);

if (!fs.existsSync(resolvedConfigPath)) {
  throw new Error(
    `Character config not found at ${resolvedConfigPath}. ` +
      `Set AGENT_CONFIG_PATH or create ${defaultConfigPath}.`
  );
}

const characterJson = fs.readFileSync(resolvedConfigPath, 'utf8');
export const character = JSON.parse(characterJson) as Character;
