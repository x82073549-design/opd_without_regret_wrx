#!/usr/bin/env bash
# Source this file from the OPD repository root after activating the conda env:
#   conda activate verl-opd
#   source env.sh

export OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${OPD_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

hash -r 2>/dev/null || true
