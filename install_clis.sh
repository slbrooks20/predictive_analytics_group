#!/usr/bin/env bash
set -euo pipefail

echo "Installing benchmark CLIs for macOS / Linux / WSL..."

if ! command -v node >/dev/null 2>&1; then
  echo "Error: Node.js is required for Gemini CLI and Codex CLI."
  echo "Install Node.js first, then rerun this script."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is required but was not found."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required but was not found."
  exit 1
fi

echo "Installing Gemini CLI..."
npm install -g @google/gemini-cli

echo "Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash

echo "Installing Codex CLI..."
npm i -g @openai/codex

echo "Installing Mistral Vibe..."
curl -LsSf https://mistral.ai/vibe/install.sh | bash

echo
echo "CLI installation complete."
echo "Next, authenticate the CLIs you plan to use."
echo "Windows users should use the platform-specific commands in RERUN_SETUP.md."
