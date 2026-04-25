#!/bin/bash
# Verification script for ResearchAgent AgentKit API integration
# Checks if the API is configured and accessible

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR/../agent"

echo "=========================================="
echo "ResearchAgent API Verification"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f "$AGENT_DIR/.env" ]; then
    echo "❌ .env file not found in $AGENT_DIR"
    exit 1
fi

# Read API URL
API_URL=$(grep "^RESEARCH_API_URL=" "$AGENT_DIR/.env" 2>/dev/null | cut -d '=' -f2- | tr -d '"' | tr -d "'" || echo "")

if [ -z "$API_URL" ]; then
    echo "❌ RESEARCH_API_URL is not configured"
    echo ""
    echo "To configure:"
    echo "  ./scripts/setup-research-api.sh"
    exit 1
fi

echo "✅ Configuration found:"
echo "   RESEARCH_API_URL: $API_URL"

# Check optional configs
BEARER_TOKEN=$(grep "^RESEARCH_BEARER_TOKEN=" "$AGENT_DIR/.env" 2>/dev/null | cut -d '=' -f2- | tr -d '"' | tr -d "'" || echo "")
if [ -n "$BEARER_TOKEN" ]; then
    echo "   RESEARCH_BEARER_TOKEN: [SET]"
else
    echo "   RESEARCH_BEARER_TOKEN: [NOT SET] (optional)"
fi

echo ""
echo "Testing API connection..."
echo "-----------------------------------"

# Test health endpoint
HEALTH_URL="${API_URL}/api/health"
echo "1. Health check: $HEALTH_URL"

if command -v curl &> /dev/null; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "$HEALTH_URL" 2>/dev/null || echo "000")
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "   ✅ Health check passed (HTTP $HTTP_CODE)"
    else
        echo "   ❌ Health check failed (HTTP $HTTP_CODE)"
        echo "   AgentKit may not be running"
    fi
else
    echo "   ⚠️  curl not found, skipping health check"
fi

# Test chat endpoint
CHAT_URL="${API_URL}/api/chat"
echo ""
echo "2. Chat endpoint: $CHAT_URL"

if command -v curl &> /dev/null; then
    # Try a simple request
    RESPONSE=$(curl -s -w "\n%{http_code}" -m 5 \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"message":"test"}' \
        "$CHAT_URL" 2>/dev/null || echo -e "\n000")
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n-1)
    
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
        echo "   ✅ Chat endpoint accessible (HTTP $HTTP_CODE)"
    elif [ "$HTTP_CODE" = "404" ]; then
        echo "   ❌ Chat endpoint not found (HTTP 404)"
        echo "   AgentKit may not be running or endpoint changed"
    elif [ "$HTTP_CODE" = "000" ]; then
        echo "   ❌ Connection failed (timeout or unreachable)"
        echo "   Check if AgentKit is running on $API_URL"
    else
        echo "   ⚠️  Unexpected response (HTTP $HTTP_CODE)"
        echo "   Response: ${BODY:0:100}"
    fi
else
    echo "   ⚠️  curl not found, skipping chat endpoint test"
fi

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    echo "✅ ResearchAgent API is configured and accessible!"
    echo ""
    echo "The agent will use ResearchAgent for research context."
    echo "Restart the agent to enable the integration."
else
    echo "⚠️  ResearchAgent API is configured but not accessible"
    echo ""
    echo "To start AgentKit:"
    echo "  cd vendor/ResearchAgent"
    echo "  bun run dev"
    echo ""
    echo "The agent will work without ResearchAgent (using fallback),"
    echo "but won't have research context until AgentKit is running."
fi
echo ""

