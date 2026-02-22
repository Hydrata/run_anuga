#!/usr/bin/env bash
# Phase 1: Core Install + CLI (no geo deps)
# Tests: pip install, CLI entry point, validate, info

echo "=== Phase 1: Core Install + CLI ==="
echo "README says: pip install run_anuga"
echo "Expected: core-only install, CLI works, validate/info work without geo deps"
echo ""

# Copy source to writable location (bind mount is :ro)
cp -r /app /tmp/run_anuga_src

# Step 1: pip install run_anuga (core only, from local source)
test_step 1 "pip install run_anuga (core only)" \
    "pip install /tmp/run_anuga_src 2>&1 | tail -10"

# Step 2: run-anuga --help
test_step 2 "run-anuga --help" \
    "run-anuga --help"

# Step 3: run-anuga validate
test_step 3 "run-anuga validate examples/australian_floodplain/" \
    "run-anuga validate /app/examples/australian_floodplain/"

# Step 4: run-anuga info
test_step 4 "run-anuga info examples/australian_floodplain/" \
    "run-anuga info /app/examples/australian_floodplain/"

print_summary
