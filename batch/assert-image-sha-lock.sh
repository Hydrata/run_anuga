#!/usr/bin/env bash
# assert-image-sha-lock.sh — prove two built anuga images bake IDENTICAL
# {run_anuga, anuga_core, hydrata-leaves} SHAs (TASK-2671, epic 2662 D8).
#
# Usage: ./assert-image-sha-lock.sh <cpu-image-ref> <gpu-image-ref>
#
# Compares the three org.hydrata.*_sha LABELs (declared per-target in
# batch/Dockerfile from the SAME global ARGs) across both images. Any
# missing/empty/unequal label is a HARD FAIL — an image pair that cannot
# prove same-code must never be pushed/registered as a family.
#
# Called by deploy/scripts/rebuild-batch-image.sh after a build when the
# sibling target exists locally, by the run_anuga CI sha-lock job, and by
# operators ad hoc.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <cpu-image-ref> <gpu-image-ref>" >&2
  exit 2
fi
CPU_IMAGE="$1"
GPU_IMAGE="$2"

LABELS=(org.hydrata.run_anuga_sha org.hydrata.anuga_core_sha org.hydrata.gn_anuga_sha)

fail=0
for image in "$CPU_IMAGE" "$GPU_IMAGE"; do
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "[sha-lock] FAIL: image not found locally: $image" >&2
    exit 1
  fi
done

get_label() {  # $1=image $2=label
  docker image inspect "$1" --format "{{ index .Config.Labels \"$2\" }}"
}

echo "[sha-lock] comparing ${LABELS[*]}"
echo "[sha-lock]   cpu: $CPU_IMAGE"
echo "[sha-lock]   gpu: $GPU_IMAGE"
for label in "${LABELS[@]}"; do
  cpu_val=$(get_label "$CPU_IMAGE" "$label")
  gpu_val=$(get_label "$GPU_IMAGE" "$label")
  for v in "$cpu_val" "$gpu_val"; do
    if [ -z "$v" ] || [ "$v" = "<no value>" ] || [ "$v" = "unknown" ]; then
      echo "[sha-lock] FAIL: $label is missing/empty/unknown (cpu=${cpu_val:-∅} gpu=${gpu_val:-∅})" >&2
      fail=1
    fi
  done
  if [ "$cpu_val" != "$gpu_val" ]; then
    echo "[sha-lock] FAIL: $label differs — cpu=$cpu_val gpu=$gpu_val" >&2
    fail=1
  else
    echo "[sha-lock]   $label: $cpu_val == $gpu_val"
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[sha-lock] FAILED — the image pair does not bake identical SHAs; refusing (D8 same-code guarantee)." >&2
  exit 1
fi
echo "[sha-lock] PASS — both targets bake identical {run_anuga, anuga_core, hydrata} SHAs."
