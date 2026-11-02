# BFM-Zero container

Run `bash apptainer/build.sh` from the workstation to build `bfm-zero.sif`, then
`bash apptainer/push.sh mila` to transfer the image and motion data. Nothing runs
on a cluster until you submit a job. Build needs internet, fakeroot support and
substantial free disk space (allow 60+ GB for downloads, build and cache).

The image uses Ubuntu 22.04, CUDA 12.4.1, Python 3.10, Isaac Sim 4.5.0 and
Isaac Lab 2.0.2, consuming the existing root `uv.lock` with `--frozen`. The source
is mounted at `/workspace/bfm-zero`; ordinary code changes need no image rebuild.
Rebuild when dependencies or the lock change. The locked W&B SDK is old
(`0.12.21`); first validate offline logging, then a tiny online run before a long
training job. We have not silently upgraded it or regenerated your modified lock.

For new-format API keys, use the separately built [modern W&B derivative](wandb-modern.md).
The cluster configuration selects `bfm-zero-wandb-dev.sif`, W&B 0.27.1 and the Forge
API endpoint. `push.sh` transfers this image and its checksum when present. The
base image and repository lock retain their original dependency versions.

Submit with `bash cluster/submit.sh mila ... -- train ...`: it holds the job,
creates Slurm's log directory, and releases it. This prevents startup failures
before the job script runs. `BFM_CODE_DIR` selects the normal home checkout,
bound directly at `/workspace/bfm-zero` inside the image. It is configured in
`cluster/pyproject.toml` alongside the image, data and results paths.

Only `pyproject.toml` and `uv.lock` enter the image. No source tree, `.netrc`, SSH
key, `.env`, user home, dataset or model is baked in. The runtime job reads an
external `.netrc` through a read-only mount, or accepts a scheduler-injected
`WANDB_API_KEY` through environment variables. Neither value enters arguments or
experiment config. Never enable shell tracing around credentials.

The build-time `%test` checks dependency imports only. It cannot validate a GPU
driver, PhysX stepping, rendering, throughput, network assets or convergence. On
an allocated L40S, run `cluster/job.sh check --require-gpu --boot-isaac`, followed
by the smoke preset. Apptainer's `--nv` uses the host driver: a container cannot
repair a missing or incompatible driver. NVIDIA lists non-RT GPUs such as A100
and H100 as unsupported for Isaac Sim; use L40S for the first supported path.

First Isaac startup can need extension/asset downloads. Before using compute
nodes without internet, validate the **actual smoke job** on that cluster's
network-accessible GPU allocation, and arrange local asset/extension caches if
needed. The included extension-cache packages reduce downloads; they do not
prove a fully air-gapped launch. If startup reports an unwritable path, bind that
specific Kit cache path to job-local storage. Do not upgrade Isaac to fix a mount
problem; that changes the simulator version under the experiment.

The job uses a temporary isolated home/cache, the directly mounted checkout,
motion data and persistent scratch outputs. Keep that checkout at the version
used for the experiment while it runs. It does not checkpoint on SIGTERM;
interruption recovers only the latest **completed** scheduled checkpoint. Replay
can be tens of GB, and checkpoint publication temporarily keeps both old and new
versions. Budget disk accordingly. The script leaves job-local cache cleanup to
Slurm, so workstation runs with `BFM_ALLOW_LOCAL=1` leave their printed temporary
directory until you remove it.

See [START_HERE.md](../docs/START_HERE.md) for commands, W&B and the experiment ladder.
