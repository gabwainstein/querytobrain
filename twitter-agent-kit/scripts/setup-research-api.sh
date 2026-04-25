#!/bin/bash
# Setup script for ResearchAgent AgentKit API integration
# This script helps configure and verify the ResearchAgent API connection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR/../agent"
VENDOR_DIR="$SCRIPT_DIR/../vendor/ResearchAgent"

echo "=========================================="
echo "ResearchAgent AgentKit API Setup"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f "$AGENT_DIR/.env" ]; then
    echo "❌ .env file not found in $AGENT_DIR"
    echo "   Please create it from .env.example first"
    exit 1
fi

# Function to check if variable is set
check_env_var() {
    local var_name=$1
    local var_value=$(grep "^${var_name}=" "$AGENT_DIR/.env" 2>/dev/null | cut -d '=' -f2- | tr -d '"' | tr -d "'" || echo "")
    
    if [ -z "$var_value" ] || [ "$var_value" = "" ]; then
        return 1
    fi
    return 0
}

# Function to set env var
set_env_var() {
    local var_name=$1
    local default_value=$2
    local description=$3
    
    if check_env_var "$var_name"; then
        local current=$(grep "^${var_name}=" "$AGENT_DIR/.env" | cut -d '=' -f2- | tr -d '"' | tr -d "'")
        echo "✅ $var_name is set: $current"
        return 0
    fi
    
    echo ""
    echo "📝 $description"
    if [ -n "$default_value" ]; then
        read -p "   Enter value (default: $default_value): " value
        value=${value:-$default_value}
    else
        read -p "   Enter value: " value
    fi
    
    if [ -z "$value" ]; then
        echo "   ⚠️  Skipping $var_name (optional)"
        return 1
    fi
    
    # Remove existing line if present
    sed -i "/^${var_name}=/d" "$AGENT_DIR/.env" 2>/dev/null || true
    
    # Add new line
    echo "${var_name}=${value}" >> "$AGENT_DIR/.env"
    echo "   ✅ Set $var_name=$value"
    return 0
}

echo "Step 1: Configure ResearchAgent API URL"
echo "-----------------------------------"
set_env_var "RESEARCH_API_URL" "http://localhost:3000" "ResearchAgent AgentKit API base URL"

echo ""
echo "Step 2: Optional Configuration"
echo "-----------------------------------"
set_env_var "RESEARCH_BEARER_TOKEN" "" "Bearer token for API authentication (optional)"
set_env_var "RESEARCH_SUMMARY_PROMPT" "" "Custom prompt for research summaries (optional)"
set_env_var "RESEARCH_REQUEST_TIMEOUT_MS" "20000" "Request timeout in milliseconds (optional)"
set_env_var "RESEARCH_SUMMARY_TTL_MINUTES" "60" "Summary cache TTL in minutes (optional)"

echo ""
echo "Step 3: Verify Configuration"
echo "-----------------------------------"

if check_env_var "RESEARCH_API_URL"; then
    API_URL=$(grep "^RESEARCH_API_URL=" "$AGENT_DIR/.env" | cut -d '=' -f2- | tr -d '"' | tr -d "'")
    echo "✅ RESEARCH_API_URL: $API_URL"
    
    # Test connection
    echo ""
    echo "Testing connection to AgentKit..."
    if command -v curl &> /dev/null; then
        HEALTH_URL="${API_URL}/api/health"
        echo "   Checking: $HEALTH_URL"
        
        if curl -s -f -m 5 "$HEALTH_URL" > /dev/null 2>&1; then
            echo "   ✅ AgentKit is running and accessible!"
        else
            echo "   ⚠️  AgentKit is not responding"
            echo "   This is OK if you haven't started it yet"
            echo ""
            echo "   To start AgentKit:"
            echo "   cd $VENDOR_DIR"
            echo "   bun install"
            echo "   bun run migrate"
            echo "   bun run dev"
        fi
    else
        echo "   ⚠️  curl not found, skipping connection test"
    fi
else
    echo "❌ RESEARCH_API_URL is not set"
    echo "   ResearchAgent integration will be disabled"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start ResearchAgent AgentKit (if not running):"
echo "   cd $VENDOR_DIR"
echo "   bun run dev"
echo ""
echo "2. Restart the agent:"
echo "   cd $AGENT_DIR"
echo "   bun run dev"
echo ""
echo "3. Verify integration:"
echo "   ./scripts/verify-research-api.sh"
echo ""

