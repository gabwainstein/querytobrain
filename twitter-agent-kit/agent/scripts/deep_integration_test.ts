import { ResearchApiService } from '../src/plugins/research-api/index.ts';
import { TwitterService } from '../../vendor/eliza-vendor/packages/plugin-twitter/src/services/twitter.service.ts';

function createRuntime() {
  return {
    getSetting: (key: string) => process.env[key] ?? null,
  } as any;
}

async function run() {
  const runtime = createRuntime();

  const researchAgentService = await ResearchApiService.start(runtime);
  const summary = await researchAgentService.getSummary(true);
  if (!summary || !summary.includes('mock response')) {
    throw new Error(`Unexpected summary: ${summary}`);
  }

  const answer = await researchAgentService.ask('Deep test question');
  if (!answer || !answer.includes('mock response')) {
    throw new Error(`Unexpected answer: ${answer}`);
  }

  await TwitterService.start(runtime);

  console.log('✅ Deep integration test passed.');
}

run().catch((error) => {
  console.error('❌ Deep integration test failed:', error);
  process.exit(1);
});

