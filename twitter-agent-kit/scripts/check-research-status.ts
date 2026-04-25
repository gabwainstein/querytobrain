#!/usr/bin/env bun
/**
 * ResearchAgent API Status Checker
 * 
 * Checks if ResearchAgent AgentKit API is configured and accessible.
 * Can be run independently or as part of agent startup.
 */

import { readFileSync } from 'fs';
import { join } from 'path';

interface ResearchConfig {
  apiUrl: string | null;
  bearerToken: string | null;
  configured: boolean;
  accessible: boolean;
  error?: string;
}

async function checkResearchStatus(): Promise<ResearchConfig> {
  const agentDir = join(import.meta.dir, '..', 'agent');
  const envPath = join(agentDir, '.env');
  
  let apiUrl: string | null = null;
  let bearerToken: string | null = null;

  // Read .env file
  try {
    const envContent = readFileSync(envPath, 'utf-8');
    const lines = envContent.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('RESEARCH_API_URL=')) {
        apiUrl = line.split('=')[1]?.trim().replace(/^["']|["']$/g, '') || null;
      }
      if (line.startsWith('RESEARCH_BEARER_TOKEN=')) {
        bearerToken = line.split('=')[1]?.trim().replace(/^["']|["']$/g, '') || null;
      }
    }
  } catch (error) {
    return {
      apiUrl: null,
      bearerToken: null,
      configured: false,
      accessible: false,
      error: `.env file not found or unreadable: ${error}`,
    };
  }

  const configured = apiUrl !== null && apiUrl !== '';

  if (!configured) {
    return {
      apiUrl: null,
      bearerToken: null,
      configured: false,
      accessible: false,
      error: 'RESEARCH_API_URL not configured',
    };
  }

  // Test API accessibility
  let accessible = false;
  let error: string | undefined;

  try {
    const healthUrl = `${apiUrl}/api/health`;
    const response = await fetch(healthUrl, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      accessible = true;
    } else {
      error = `Health check failed: HTTP ${response.status}`;
    }
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  }

  return {
    apiUrl,
    bearerToken: bearerToken || null,
    configured: true,
    accessible,
    error,
  };
}

// Main execution
async function main() {
  console.log('🔍 Checking ResearchAgent API status...\n');

  const status = await checkResearchStatus();

  console.log('Configuration:');
  console.log(`  API URL: ${status.apiUrl || '❌ NOT SET'}`);
  console.log(`  Bearer Token: ${status.bearerToken ? '✅ SET' : '⚠️  NOT SET (optional)'}`);
  console.log(`  Configured: ${status.configured ? '✅ YES' : '❌ NO'}`);
  console.log(`  Accessible: ${status.accessible ? '✅ YES' : '❌ NO'}`);

  if (status.error) {
    console.log(`\n⚠️  Error: ${status.error}`);
  }

  console.log('\n' + '='.repeat(50));

  if (status.configured && status.accessible) {
    console.log('✅ ResearchAgent API is ready!');
    console.log('   The agent will use ResearchAgent for research context.');
    process.exit(0);
  } else if (status.configured && !status.accessible) {
    console.log('⚠️  ResearchAgent API is configured but not accessible.');
    console.log('   Start AgentKit: cd vendor/ResearchAgent && bun run dev');
    console.log('   The agent will work without ResearchAgent (using fallback).');
    process.exit(1);
  } else {
    console.log('❌ ResearchAgent API is not configured.');
    console.log('   Run: ./scripts/setup-research-api.sh');
    console.log('   The agent will work without ResearchAgent (using fallback).');
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('❌ Error checking ResearchAgent status:', error);
  process.exit(1);
});

