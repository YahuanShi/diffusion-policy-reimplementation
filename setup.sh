#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="DP"

# Check uv is installed
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating environment '$ENV_NAME' ..."
uv venv "$ENV_NAME" --python 3.10

echo "Installing dependencies ..."
UV_PROJECT_ENVIRONMENT="$ENV_NAME" uv sync --extra dev

echo ""
echo "Done. Activate with:"
echo "  source $ENV_NAME/bin/activate"
