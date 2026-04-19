#!/usr/bin/env bash
set -e

echo "Starting internal update API on port 8001..."
python -m uvicorn src.internal_update_api:app --host 0.0.0.0 --port 8001 &

echo "Starting Streamlit on port ${PORT:-8080}..."
python -m streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8080}"