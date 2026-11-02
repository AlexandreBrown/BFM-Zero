#!/usr/bin/env bash
# Explicit clusters only; transfers the image and two public motion files.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
(($#)) || { echo "Usage: bash apptainer/push.sh <ssh-cluster> [<ssh-cluster> ...]" >&2; exit 2; }
[[ -f bfm-zero.sif ]] || { echo "Build bfm-zero.sif first." >&2; exit 1; }
IMAGE_FILES=(bfm-zero.sif bfm-zero.sif.sha256)
for candidate in bfm-zero-wandb.sif bfm-zero-wandb-dev.sif; do
    if [[ -f $candidate ]]; then
        IMAGE_FILES+=("$candidate" "$candidate.sha256")
    fi
done
for image_file in "${IMAGE_FILES[@]}"; do
    [[ $image_file != *.sha256 ]] || sha256sum --check "$image_file"
done
for data in humanoidverse/data/lafan_29dof{,_10s-clipped}.pkl; do
    [[ -f $data && $(wc -c < "$data") -gt 1024 ]] || { echo "Missing data or LFS pointer: $data" >&2; exit 1; }
done
for cluster in "$@"; do
    [[ $cluster =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo "Use an SSH host alias." >&2; exit 2; }
    # A login shell is needed for cluster-provided SCRATCH.
    remote_root=$(ssh "$cluster" 'bash -lc '\''test -n "$SCRATCH" && printf "%s" "$SCRATCH/bfm-zero"'\')
    [[ $remote_root =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo "Could not resolve a simple absolute scratch path on $cluster." >&2; exit 1; }
    ssh "$cluster" "mkdir -p '$remote_root/containers' '$remote_root/data' '$remote_root/output'"
    rsync -avP "${IMAGE_FILES[@]}" "$cluster:$remote_root/containers/"
    for image_file in "${IMAGE_FILES[@]}"; do
        [[ $image_file != *.sha256 ]] || ssh "$cluster" "cd '$remote_root/containers' && sha256sum --check '$image_file'"
    done
    rsync -avP humanoidverse/data/lafan_29dof.pkl humanoidverse/data/lafan_29dof_10s-clipped.pkl "$cluster:$remote_root/data/"
done
