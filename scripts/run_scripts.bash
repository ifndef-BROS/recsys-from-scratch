#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting pipeline..."

for script in "$SCRIPT_DIR"/[0-9][0-9]_*.py; do
    echo "========================================"
    echo "Running: $(basename "$script")"
    start=$(date +%s)

    uv run "$script"

    end=$(date +%s)
    echo "Completed: $(basename "$script") in $((end - start))s"
done

echo "========================================"
echo "Pipeline completed successfully!"