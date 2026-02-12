#!/usr/bin/env bash
# VP Capital Cycle Analysis Pipeline Runner
# Usage:
#   ./run_pipeline.sh               Run pipeline using cached data
#   ./run_pipeline.sh --clear-cache Clear all cached data before running

set -e  # Exit on any error

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  VP CAPITAL CYCLE ANALYSIS - PIPELINE RUNNER                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Get script directory (works even if called from elsewhere)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "► Working directory: $SCRIPT_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Clear cache (only if --clear-cache flag is passed)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$1" = "--clear-cache" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STEP 1: CLEARING LOCAL CACHE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cache_dirs=(
        "data/raw/macro"
        "data/processed"
        "outputs/data"
        "outputs/tables"
    )

    for dir in "${cache_dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo "  ✓ Clearing: $dir"
            rm -rf "$dir"/*
        else
            echo "  ⊘ Not found (skipping): $dir"
        fi
    done

    echo ""
    echo "✓ Cache cleared"
    echo ""
else
    echo "► Using cached data (pass --clear-cache to force a fresh fetch)"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Verify virtual environment
# ─────────────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: VERIFYING VIRTUAL ENVIRONMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d ".venv" ]; then
    echo "  ✗ Virtual environment not found!"
    echo "  → Creating virtual environment..."
    python3 -m venv .venv || uv venv
fi

# Determine Python executable
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    echo "  ✓ Using virtual environment: $PYTHON"
elif [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
    echo "  ✓ Using virtual environment: $PYTHON"
else
    echo "  ✗ Virtual environment Python not found!"
    exit 1
fi

# Show Python version
PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "  → Python version: $PYTHON_VERSION"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: RUNNING ANALYSIS PIPELINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON src/main.py

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PIPELINE COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Outputs generated:"
echo "  • Chart data:     outputs/data/ ($(ls -1 outputs/data/*.json 2>/dev/null | wc -l | xargs) JSON files)"
echo "  • Summary table:  outputs/tables/summary_metrics.csv"
echo "  • Master dataset: data/processed/master_dataset.parquet"
echo ""
echo "To view outputs:"
echo "  python -m http.server 8000"
echo "  → Then open: http://localhost:8000/dashboard.html"
echo ""
echo "✓ Done!"
