#!/bin/bash
set -e

echo "=== VeriDoc AI v2.0 Startup ==="
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p uploads caches logs config utils/__pycache__
chmod -R 755 uploads caches logs config utils

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Verify critical imports
python -c "import fastapi, uvicorn, tiktoken, openai; print('✅ Dependencies OK')"

# Set port
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

# Start application
exec python -m uvicorn simple_web_interface_v2:app --host $HOST --port $PORT --workers 1
