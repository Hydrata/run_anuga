#!/usr/bin/env bash
#
# test_readme.sh — Test run_anuga README instructions as a new user
#
# Runs 3 phases in separate Docker containers (python:3.12-bookworm).
# The repo is bind-mounted at /app (simulating git clone).
# Each phase tests only commands from the README + documents gaps.
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="python:3.12-bookworm"
LOG_DIR="$REPO_DIR/test-docker/logs"

mkdir -p "$LOG_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

TOTAL_PASS=0
TOTAL_FAIL=0

# ─── Helpers ──────────────────────────────────────────────────────────

log_phase() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Phase $1: $2${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}\n"
}

run_in_container() {
    local phase_name="$1"
    local script_file="$2"
    local log_file="$LOG_DIR/${phase_name}.log"

    echo "--- Container: $phase_name ---" > "$log_file"
    echo "--- Started: $(date -Iseconds) ---" >> "$log_file"
    echo "" >> "$log_file"

    docker run --rm \
        -v "$REPO_DIR:/app:ro" \
        -w /tmp/workdir \
        -e DEBIAN_FRONTEND=noninteractive \
        "$IMAGE" \
        bash -c "source /app/test-docker/helpers.sh && source /app/test-docker/$script_file" 2>&1 | tee -a "$log_file"

    local exit_code=${PIPESTATUS[0]}
    echo "" >> "$log_file"
    echo "--- Finished: $(date -Iseconds) --- exit=$exit_code ---" >> "$log_file"
    return $exit_code
}

# Parse RESULTS line from container output
parse_results() {
    local output="$1"
    local pass fail
    pass=$(echo "$output" | grep -oP 'pass=\K[0-9]+' | tail -1 || echo 0)
    fail=$(echo "$output" | grep -oP 'fail=\K[0-9]+' | tail -1 || echo 0)
    TOTAL_PASS=$((TOTAL_PASS + pass))
    TOTAL_FAIL=$((TOTAL_FAIL + fail))
}


# ─── Phase 1: Core Install + CLI ─────────────────────────────────────

test_phase_1() {
    log_phase 1 "Core Install + CLI (no geo deps)"

    local output
    output=$(run_in_container "phase1" "phase1.sh") || true
    echo "$output"
    parse_results "$output"
}


# ─── Phase 2: Sim Install + Run ──────────────────────────────────────

test_phase_2() {
    log_phase 2 "Sim Install + Run (README commands only, then with sys deps)"

    local output
    output=$(run_in_container "phase2" "phase2.sh") || true
    echo "$output"
    parse_results "$output"
}


# ─── Phase 3: Post-process ───────────────────────────────────────────

test_phase_3() {
    log_phase 3 "Post-process (SWW to GeoTIFF)"

    local output
    output=$(run_in_container "phase3" "phase3.sh") || true
    echo "$output"
    parse_results "$output"
}


# ─── Phase 4: MPI Parallel Run ──────────────────────────────────────

test_phase_4() {
    log_phase 4 "MPI Parallel Run (2 processes)"

    local output
    output=$(run_in_container "phase4" "phase4.sh") || true
    echo "$output"
    parse_results "$output"
}


# ─── Phase 5: Visualization ─────────────────────────────────────────

test_phase_5() {
    log_phase 5 "Visualization (video from TIFFs)"

    local output
    output=$(run_in_container "phase5" "phase5.sh") || true
    echo "$output"
    parse_results "$output"
}


# ─── Main ─────────────────────────────────────────────────────────────

main() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  run_anuga README Test — New User Experience                ║${NC}"
    echo -e "${CYAN}║  Image: $IMAGE                                ║${NC}"
    echo -e "${CYAN}║  Repo:  $REPO_DIR${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

    test_phase_1
    test_phase_2
    test_phase_3
    test_phase_4
    test_phase_5

    # ─── Final Report ─────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  FINAL RESULTS                                             ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${GREEN}Passed: $TOTAL_PASS${NC}"
    echo -e "  ${RED}Failed: $TOTAL_FAIL${NC}"
    echo ""

    if [ $TOTAL_FAIL -gt 0 ]; then
        echo -e "${YELLOW}Some tests failed. Check logs in: $LOG_DIR/${NC}"
    else
        echo -e "${GREEN}All tests passed!${NC}"
    fi

    echo ""
    echo "Logs saved to: $LOG_DIR/"
    echo "  phase1.log  — Core install + CLI"
    echo "  phase2.log  — Sim install + run"
    echo "  phase3.log  — Post-process"
    echo "  phase4.log  — MPI parallel run"
    echo "  phase5.log  — Visualization"
}

main "$@"
