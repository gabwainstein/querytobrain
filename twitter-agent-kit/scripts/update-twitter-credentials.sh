#!/bin/bash
# Bash script to update Twitter credentials in .env file
# Run this from WSL: bash scripts/update-twitter-credentials.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$KIT_ROOT/agent/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: .env file not found at $ENV_FILE"
    exit 1
fi

echo "🔑 Twitter Credentials Updater"
echo "================================"
echo ""
echo "Enter your Twitter credentials from the Developer Portal:"
echo ""

# Backup original file
BACKUP_FILE="${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$ENV_FILE" "$BACKUP_FILE"
echo "✅ Created backup: $BACKUP_FILE"
echo ""

# Get credentials from user
read -p "1. TWITTER_API_KEY (Consumer Key): " API_KEY
read -p "2. TWITTER_API_SECRET_KEY (Consumer Secret): " API_SECRET_KEY
read -p "3. TWITTER_ACCESS_TOKEN (OAuth 1.0a Access Token): " ACCESS_TOKEN
read -p "4. TWITTER_ACCESS_TOKEN_SECRET (OAuth 1.0a Access Token Secret): " ACCESS_TOKEN_SECRET

if [ -z "$API_KEY" ] || [ -z "$API_SECRET_KEY" ] || [ -z "$ACCESS_TOKEN" ] || [ -z "$ACCESS_TOKEN_SECRET" ]; then
    echo "❌ Error: All credentials are required"
    exit 1
fi

# Update or add each credential
update_env_var() {
    local key=$1
    local value=$2
    
    if grep -q "^${key}=" "$ENV_FILE"; then
        # Replace existing line (works on both Linux and WSL)
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
            # Windows/Git Bash
            sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
        else
            # Linux/WSL
            sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
        fi
        echo "✅ Updated $key"
    else
        # Add new line
        echo "${key}=${value}" >> "$ENV_FILE"
        echo "✅ Added $key"
    fi
}

update_env_var "TWITTER_API_KEY" "$API_KEY"
update_env_var "TWITTER_API_SECRET_KEY" "$API_SECRET_KEY"
update_env_var "TWITTER_ACCESS_TOKEN" "$ACCESS_TOKEN"
update_env_var "TWITTER_ACCESS_TOKEN_SECRET" "$ACCESS_TOKEN_SECRET"

echo ""
echo "✅ Successfully updated Twitter credentials in .env file!"
echo ""
echo "⚠️  Remember to restart the agent for changes to take effect"
echo ""


