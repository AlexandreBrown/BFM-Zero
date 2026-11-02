# New-format W&B API keys

The upstream lock includes `wandb==0.12.21`, whose login rejects keys longer
than 40 characters. New W&B keys require a newer SDK in the training process,
not only a newer login command. Never truncate a key or store one in this repo.

Build the logging derivative after building the base image:

```bash
apptainer build --fakeroot bfm-zero-wandb-dev.sif apptainer/wandb-modern.def
sha256sum bfm-zero-wandb-dev.sif > bfm-zero-wandb-dev.sif.sha256
```

This updates W&B (0.27.1) and protobuf (4.25.8), and installs `python3.10-dev`
for PyTorch's runtime C++ compilation. The build test actually compiles and runs
a small kernel; dependency imports alone would not catch missing `Python.h`.
The base image, repository lock, simulator, PyTorch, and learning implementation
are unchanged. All 46 CPU regression tests passed in this derivative. GPU
compilation and actual training updates must also be checked after deployment.

On Mila, authenticate interactively outside the checkout:

```bash
cd "$HOME"
UV_CACHE_DIR="$SCRATCH/bfm-zero/uv-cache" \
  uvx --from 'wandb==0.27.1' wandb login --relogin \
  --host https://api.forge.coreweave.com
```

Paste the key only at the prompt. Credentials live in the external `~/.netrc`.
The existing job script mounts that file read-only, outside the checkout.
For live logging, select `--wandb-mode online`. The image and public endpoint are
configured under `[tool.cluv.env]` in `cluster/pyproject.toml`. The current cluv
forces Slurm's `--export=ALL`, so use its environment configuration rather than
trying to pass individual variables with a CLI `--export` override.

The image checksum sidecar is carried into run provenance by `cluster/job.sh`.
The API URL is different from the browser project URL. The W&B CLI settings
file is not relied upon inside the isolated compute-job home.
