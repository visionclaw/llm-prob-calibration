#!/bin/bash
# Setup script for llm-prob-calibration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up llm-prob-calibration ==="

# Create venv with uv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
