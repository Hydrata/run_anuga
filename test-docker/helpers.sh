#!/usr/bin/env bash
# Container-side test helpers (sourced inside each Docker container)

count_pass=0
count_fail=0
last_rc=0

test_step() {
    local num="$1"
    local desc="$2"
    shift 2
    local cmd="$*"

    echo ""
    echo "--- Step $num: $desc"
    echo "    Command: $cmd"
    echo "    $(date -Iseconds)"

    # Run command in subshell with pipefail.
    # Write exit code to temp file since pipe to sed masks it.
    local rc_file
    rc_file=$(mktemp)
    ( set -o pipefail; eval "$cmd" 2>&1; echo $? > "$rc_file" ) | sed 's/^/    /'
    last_rc=$(cat "$rc_file")
    rm -f "$rc_file"

    if [ "$last_rc" -eq 0 ]; then
        echo "    >> PASS (exit $last_rc)"
        count_pass=$((count_pass + 1))
    else
        echo "    >> FAIL (exit $last_rc)"
        count_fail=$((count_fail + 1))
    fi
}

print_summary() {
    echo ""
    echo "--- Phase Summary: $count_pass passed, $count_fail failed ---"
    echo "RESULTS:pass=$count_pass,fail=$count_fail"
}
