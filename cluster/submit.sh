#!/usr/bin/env bash
# Submit held, create cluv's Slurm log directory, then release the allocation.
# Slurm must be able to open stdout before it can execute cluster/job.sh.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLUSTER=${1:?Usage: bash cluster/submit.sh <cluster> [cluv submit arguments]}
shift
[[ $CLUSTER =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ && $CLUSTER != first ]] || {
    echo "Use one configured SSH cluster alias." >&2; exit 2;
}
JOB_SCRIPT=()
# cluv expects the optional script before Slurm flags; otherwise its parser can
# mistake the script path for a value belonging to --hold.
if (($#)) && [[ $1 != -* && $1 == *.sh ]]; then
    JOB_SCRIPT=("$1")
    shift
fi
KEEP_HELD=0
for arg in "$@"; do [[ $arg != --hold ]] || KEEP_HELD=1; done
JOB_ID=$(bash "$HERE/cluv.sh" submit "$CLUSTER" "${JOB_SCRIPT[@]}" --parsable --hold "$@")
[[ $JOB_ID =~ ^[0-9]+$ ]] || {
    echo "cluv did not return a job ID; no job was released." >&2; exit 1;
}
echo "Submitted held job $JOB_ID on $CLUSTER"
if ! ssh "$CLUSTER" "bash -lc 'mkdir -p \"\$SCRATCH/bfm-zero/output/${CLUSTER}_${JOB_ID}\"'"; then
    echo "Could not create the log directory. Job $JOB_ID remains held." >&2
    exit 1
fi
if ((KEEP_HELD)); then
    echo "Job $JOB_ID remains held as requested."
else
    ssh "$CLUSTER" "scontrol release $JOB_ID"
    echo "Released job $JOB_ID"
fi
