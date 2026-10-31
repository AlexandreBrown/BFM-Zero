#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p .apptainer
export APPTAINER_CACHEDIR="$PWD/.apptainer/cache"
export APPTAINER_TMPDIR="$PWD/.apptainer/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
if [[ -e bfm-zero.sif ]]; then
    echo "bfm-zero.sif already exists. Move it aside before building a new image." >&2
    exit 1
fi
apptainer build --fakeroot bfm-zero.sif apptainer/bfm-zero.def
sha256sum bfm-zero.sif > bfm-zero.sif.sha256
echo "Built bfm-zero.sif. Run the GPU smoke job before longer experiments."
