#!/usr/bin/env bash

# Watch the systemd evaluation service until all three validated result sets
# exist. If the main service ever becomes inactive before that, start it again.
set -Eeuo pipefail

ROOT_DIR=${ROOT_DIR:-/root/opd_without_regret_wrx}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/opd_outputs}
EVAL_PROTOCOL_TAG=${EVAL_PROTOCOL_TAG:-valfix31744}
EVAL_OUT_ROOT=${EVAL_OUT_ROOT:-$OUTPUT_ROOT/justrl_eval_outputs_$EVAL_PROTOCOL_TAG}
MAIN_UNIT=${MAIN_UNIT:-opd-three-variant-step279-eval-main.service}
POLL_SECONDS=${POLL_SECONDS:-30}
EXPECTED_DETAILED_ROWS=2288
PYTHON=${PYTHON:-/usr/envs/wrx_env/bin/python3}
LOG_DIR=${LOG_DIR:-$ROOT_DIR/outputs/logs/eval}
LOG_FILE=${LOG_FILE:-$LOG_DIR/three_variants_step279_watchdog.log}
LOCK_FILE=${LOCK_FILE:-$ROOT_DIR/outputs/three_variants_step279_watchdog.lock}

mkdir -p "$LOG_DIR" "$(dirname "$LOCK_FILE")"
exec 8>"$LOCK_FILE"
if ! flock -n 8; then
    echo "Another evaluation watchdog is already running: $LOCK_FILE" >&2
    exit 1
fi
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

on_error() {
    local status=$?
    log "ERROR: watchdog failed with status $status (line ${BASH_LINENO[0]})."
    exit "$status"
}
trap on_error ERR

result_files_are_complete() {
    local method model_dir grading_file detailed_file detailed_rows
    for method in pruneopd exopd eopd; do
        model_dir=$EVAL_OUT_ROOT/${method}_trainseed0_root0_delta279_hf
        grading_file=$model_dir/grading_results.json
        detailed_file=$model_dir/detailed_results.jsonl
        if [ ! -s "$grading_file" ] || [ ! -s "$detailed_file" ]; then
            return 1
        fi
        detailed_rows=$(wc -l < "$detailed_file")
        if [ "$detailed_rows" -ne "$EXPECTED_DETAILED_ROWS" ]; then
            return 1
        fi
        if ! "$PYTHON" -c \
            'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); assert isinstance(data, list) and len(data) == 3' \
            "$grading_file"; then
            return 1
        fi
    done
}

if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    log "POLL_SECONDS must be a positive integer: $POLL_SECONDS"
    exit 1
fi
if [ ! -x "$PYTHON" ]; then
    log "Python executable is missing: $PYTHON"
    exit 1
fi

log "Evaluation watchdog started for $MAIN_UNIT."
local_status_countdown=0
while true; do
    if result_files_are_complete; then
        log "SUCCESS: all three grading files and all 3 x $EXPECTED_DETAILED_ROWS detailed rows are complete."
        exit 0
    fi

    unit_state=$(systemctl is-active "$MAIN_UNIT" 2>/dev/null || true)
    case "$unit_state" in
        active|activating|reloading)
            if (( local_status_countdown == 0 )); then
                completed_results=$(find "$EVAL_OUT_ROOT" -maxdepth 2 \
                    -path '*_trainseed0_root0_delta279_hf/grading_results.json' -type f -size +0c \
                    2>/dev/null | wc -l)
                log "Main service state=$unit_state; completed grading files=$completed_results/3."
                local_status_countdown=20
            fi
            ;;
        *)
            log "Main service state=${unit_state:-unknown} before results completed; restarting $MAIN_UNIT."
            systemctl reset-failed "$MAIN_UNIT" 2>/dev/null || true
            systemctl start "$MAIN_UNIT"
            local_status_countdown=0
            ;;
    esac

    sleep "$POLL_SECONDS"
    if (( local_status_countdown > 0 )); then
        local_status_countdown=$((local_status_countdown - 1))
    fi
done
