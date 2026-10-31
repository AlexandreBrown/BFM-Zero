#!/usr/bin/env bash
# cluv submit <cluster> -- train --preset smoke
# cluv submit <cluster> -- eval --model-folder /scratch/.../run
# Arguments after the mode are passed unchanged to the selected Python module.
set -euo pipefail
umask 077

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z ${SLURM_JOB_ID:-} && ${BFM_ALLOW_LOCAL:-0} != 1 ]]; then
    echo "Run inside a Slurm allocation, or set BFM_ALLOW_LOCAL=1 for a workstation." >&2
    exit 1
fi
: "${SCRATCH:?SCRATCH must point to persistent storage}"
MODE=${1:-train}
if (($#)); then shift; fi
case "$MODE" in
    train) MODULE=experiments.train ;;
    eval) MODULE=experiments.evaluate ;;
    tracking) MODULE=humanoidverse.tracking_inference ;;
    goal) MODULE=humanoidverse.goal_inference ;;
    reward) MODULE=humanoidverse.reward_inference ;;
    check) MODULE=experiments.preflight ;;
    log-videos) MODULE=experiments.log_videos ;;
    *) echo "Mode must be train, eval, tracking, goal, reward, check, or log-videos." >&2; exit 2 ;;
esac

BFM_SIF=${BFM_SIF:-$SCRATCH/bfm-zero/containers/bfm-zero.sif}
BFM_DATA_DIR=${BFM_DATA_DIR:-$SCRATCH/bfm-zero/data}
BFM_OUTPUT_DIR=${BFM_OUTPUT_DIR:-$SCRATCH/bfm-zero/output}
[[ -f $BFM_SIF ]] || { echo "Missing image: $BFM_SIF (see apptainer/README.md)" >&2; exit 1; }
[[ -d $BFM_DATA_DIR ]] || { echo "Missing data: $BFM_DATA_DIR" >&2; exit 1; }
if [[ -f $BFM_SIF.sha256 ]]; then
    read -r image_sha image_name < "$BFM_SIF.sha256"
    [[ $image_sha =~ ^[a-f0-9]{64}$ ]] || { echo "Invalid image checksum sidecar." >&2; exit 1; }
    export APPTAINERENV_BFM_IMAGE_SHA256="$image_sha" SINGULARITYENV_BFM_IMAGE_SHA256="$image_sha"
fi

if ! command -v apptainer >/dev/null && ! command -v singularity >/dev/null; then
    if type module >/dev/null 2>&1; then module load apptainer || module load singularity; fi
fi
if command -v apptainer >/dev/null; then ENGINE=apptainer; else ENGINE=singularity; fi

JOB_KEY=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}
JOB_BASE=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/bfm-${JOB_KEY}.XXXXXX")
mkdir -p "$JOB_BASE/home" "$JOB_BASE/cache" "$JOB_BASE/mpl" "$BFM_OUTPUT_DIR"
# Jobs use a snapshot at submission's GIT_COMMIT; later cluv syncs cannot alter it.
# Skip LFS here: large motion files are separately staged, never downloaded by jobs.
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --no-hardlinks --no-checkout "$REPO_ROOT" "$JOB_BASE/source"
GIT_LFS_SKIP_SMUDGE=1 git -C "$JOB_BASE/source" checkout --quiet --detach "${GIT_COMMIT:-HEAD}"

BINDS=(--bind "$JOB_BASE/source:/workspace/bfm-zero"
       --bind "$SCRATCH:$SCRATCH"
       --bind "$BFM_DATA_DIR:$BFM_DATA_DIR:ro"
       --bind "$BFM_OUTPUT_DIR:$BFM_OUTPUT_DIR"
       --bind "$JOB_BASE/cache:/home/bfm/.cache"
       --bind "$JOB_BASE/mpl:/tmp/matplotlib")
# Read-only credential mount, outside the checkout, image and synced results.
if [[ -f $HOME/.netrc ]]; then
    touch "$JOB_BASE/home/.netrc"
    BINDS+=(--bind "$HOME/.netrc:/home/bfm/.netrc:ro")
fi
# Pass values through environment, including an optional scheduler-injected key.
# No credential value is placed in argv, config, launch metadata or output.
for key in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_NAME WANDB_BASE_URL; do
    if [[ -v $key ]]; then
        export "APPTAINERENV_${key}=${!key}" "SINGULARITYENV_${key}=${!key}"
    fi
done
export APPTAINERENV_MUJOCO_GL=egl SINGULARITYENV_MUJOCO_GL=egl
export APPTAINERENV_OMP_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=1

ARGS=("$@")
if [[ $MODE == train ]]; then
    # Explicit flags override these defaults; arrays default to independent seeds.
    ARGS=(--preset smoke --seed "${SLURM_ARRAY_TASK_ID:-4728}"
          --work-dir "$BFM_OUTPUT_DIR/runs/$JOB_KEY"
          --data-path "$BFM_DATA_DIR/lafan_29dof_10s-clipped.pkl" "${ARGS[@]}")
elif [[ $MODE == eval || $MODE == tracking || $MODE == goal || $MODE == reward ]]; then
    ARGS=(--data-path "$BFM_DATA_DIR/lafan_29dof.pkl" "${ARGS[@]}")
elif [[ $MODE == check ]]; then
    ARGS=(--data-dir "$BFM_DATA_DIR" "${ARGS[@]}")
fi
echo "Mode=$MODE commit=$(git -C "$JOB_BASE/source" rev-parse HEAD) job=$JOB_KEY"
echo "Temporary workspace=$JOB_BASE"
# Don't exec: after normal exit print the scratch path. Slurm removes SLURM_TMPDIR.
set +e
"$ENGINE" exec --nv --cleanenv --contain --home "$JOB_BASE/home:/home/bfm" \
    --pwd /workspace/bfm-zero "${BINDS[@]}" "$BFM_SIF" python -m "$MODULE" "${ARGS[@]}"
STATUS=$?
set -e
echo "Exit=$STATUS persistent results=$BFM_OUTPUT_DIR"
exit "$STATUS"
