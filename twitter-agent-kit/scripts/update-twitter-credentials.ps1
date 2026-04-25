# PowerShell script to update Twitter credentials in .env file
# Run this from PowerShell: .\scripts\update-twitter-credentials.ps1

$kitRoot = Split-Path $PSScriptRoot -Parent
$envFile = Join-Path $kitRoot "agent\.env"

if (-not (Test-Path $envFile)) {
    Write-Host "Error: .env file not found at $envFile" -ForegroundColor Red
    exit 1
}

Write-Host "Twitter Credentials Updater" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Enter your Twitter credentials from the Developer Portal:" -ForegroundColor Yellow
Write-Host ""

# Backup original file
$dateFormat = "yyyyMMdd_HHmmss"
$timestamp = Get-Date -Format $dateFormat
$backupFile = "$envFile.backup.$timestamp"
Copy-Item $envFile $backupFile
Write-Host "Created backup: $backupFile" -ForegroundColor Green
Write-Host ""

# Get credentials from user
$apiKey = Read-Host "1. TWITTER_API_KEY (Consumer Key)"
$apiSecretKey = Read-Host "2. TWITTER_API_SECRET_KEY (Consumer Secret)"
$accessToken = Read-Host "3. TWITTER_ACCESS_TOKEN (OAuth 1.0a Access Token)"
$accessTokenSecret = Read-Host "4. TWITTER_ACCESS_TOKEN_SECRET (OAuth 1.0a Access Token Secret)"

if ([string]::IsNullOrWhiteSpace($apiKey) -or 
    [string]::IsNullOrWhiteSpace($apiSecretKey) -or 
    [string]::IsNullOrWhiteSpace($accessToken) -or 
    [string]::IsNullOrWhiteSpace($accessTokenSecret)) {
    Write-Host "Error: All credentials are required" -ForegroundColor Red
    exit 1
}

# Read current .env file
$content = Get-Content $envFile -Raw

# Update or add each credential
$updates = @{
    "TWITTER_API_KEY" = $apiKey
    "TWITTER_API_SECRET_KEY" = $apiSecretKey
    "TWITTER_ACCESS_TOKEN" = $accessToken
    "TWITTER_ACCESS_TOKEN_SECRET" = $accessTokenSecret
}

foreach ($key in $updates.Keys) {
    $value = $updates[$key]
    $pattern = "^$key=.*"
    
    if ($content -match $pattern) {
        # Replace existing line
        $content = $content -replace $pattern, "$key=$value"
        Write-Host "Updated $key" -ForegroundColor Green
    } else {
        # Add new line
        $content += "`n$key=$value"
        Write-Host "Added $key" -ForegroundColor Green
    }
}

# Write updated content
Set-Content -Path $envFile -Value $content -NoNewline

Write-Host ""
Write-Host "Successfully updated Twitter credentials in .env file!" -ForegroundColor Green
Write-Host ""
Write-Host "Remember to restart the agent for changes to take effect" -ForegroundColor Yellow
Write-Host ""
