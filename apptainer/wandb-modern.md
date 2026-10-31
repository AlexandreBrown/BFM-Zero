# New-format W&B API keys

The upstream lock includes `wandb==0.12.21`, whose login rejects keys longer
than 40 characters. New W&B keys require a newer SDK in the training process,
not only a newer login command. Never truncate a key or store one in this repo.

Build the logging derivative after building the base image:

```bash
apptainer build --fakeroot bfm-zero-wandb.sif apptainer/wandb-modern.def
sha256sum bfm-zero-wandb.sif > bfm-zero-wandb.sif.sha256
```

This changes only W&B (0.27.1) and protobuf (4.25.8) in the image. The base
image, repository lock, simulator, PyTorch, training source, and algorithm are
unchanged. All 44 CPU regression tests passed in this derivative. GPU startup
must also be checked after deploying it.

On Mila, authenticate interactively outside the checkout:

```bash
cd "$HOME"
UV_CACHE_DIR="$SCRATCH/bfm-zero/uv-cache" \
  uvx --from 'wandb==0.27.1' wandb login --relogin \
  --host https://api.forge.coreweave.com
```

Paste the key only at the prompt. Credentials live in the external `~/.netrc`.
The existing job script mounts that file read-only, outside the source snapshot.
For live logging, select `--wandb-mode online`. The image and public endpoint are
configured under `[tool.cluv.env]` in `cluster/pyproject.toml`. The current cluv
forces Slurm's `--export=ALL`, so use its environment configuration rather than
trying to pass individual variables with a CLI `--export` override.

The image checksum sidecar is carried into run provenance by `cluster/job.sh`.
The API URL is different from the browser project URL. The W&B CLI settings
file is not relied upon inside the isolated compute-job home.
