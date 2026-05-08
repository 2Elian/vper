#!/bin/bash
set -e

echo "=== DABench Submission Started ==="
echo "MODEL_API_URL: ${MODEL_API_URL}"
echo "MODEL_NAME: ${MODEL_NAME:-qwen3.5-35b-a3b}"

mkdir -p /output /logs

uv run dabench submit --config configs/react_baseline.example.yaml 2>&1 | tee /logs/runtime.log

echo "=== DABench Submission Finished ==="
