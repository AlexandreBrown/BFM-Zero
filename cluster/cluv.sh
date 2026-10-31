#!/usr/bin/env bash
# Use the current checkout of cluv with the already installed CLI's interpreter.
# This avoids installing cluv (Python >=3.11) into BFM-Zero's Python 3.10 venv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CLUV_SOURCE=${BFM_CLUV_SOURCE:-$HOME/Documents/cluv}
[[ -f $CLUV_SOURCE/cluv/__main__.py ]] || { echo "Set BFM_CLUV_SOURCE to your current cluv checkout." >&2; exit 1; }
export PYTHONPATH="$CLUV_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
exec cluv "$@"
