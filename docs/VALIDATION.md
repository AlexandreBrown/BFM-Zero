# Validation

## Cluster checks on 2026-11-02

The compiler-header fix is published in `55be14a`. The new
`bfm-zero-wandb-dev.sif` includes Python development headers and W&B 0.27.1;
job 11498045 passed real CPU and CUDA compiled forward/backward checks.

Reference job 11498106 then completed its initial evaluation but failed before
learning: CPU expert replay observations were passed directly into the CUDA
backward encoder when constructing tracking prompts. Slurm correctly reported
exit code 1 with the new failure reporting. The earlier run 11462675 had failed
on missing headers while incorrectly appearing successful. Neither run trained
a policy or produced a training checkpoint.

The pending patch transfers expert observations to the model device in
`_sample_tracking_z`. Losses and prompt mathematics are unchanged. A focused
compiled optimizer test also exposed the contained `/tmp` filling up, so
`cluster/job.sh` now directs temporary files and Inductor/Triton caches to the
mounted job-local disk, outside the repository and home quota.

Verification of these pending changes:

- Local suite: 49 tests run, 47 passed and 2 CUDA-only tests skipped.
- Mila L40S job **11498366**: all 3 device regression tests passed, Slurm exit
  code 0, elapsed 2 minutes 30 seconds. This used the published source with the
  candidate prompt helper loaded from a diagnostic file on scratch; it did not
  modify the remote checkout. The learning test used small networks and synthetic
  CPU replay, ran two compiled native FB/critic/discriminator/auxiliary/actor
  updates, checked finite metrics and verified actor weights changed. The prompt
  test used a real CUDA encoder and normalizer and checked CPU replay stayed on
  CPU. Compiler files used job-local disk.
- Shell and guide JavaScript syntax, source-browser regeneration and
  `git diff --check` passed.

These tests establish device and compiler readiness. A full simulation training
run, full-size update VRAM fit, throughput, convergence and policy videos remain
to be verified after the pending patch is reviewed, committed and pushed by the
user. The next launch keeps the 192M-step target and 2-day walltime, with more
frequent metric logging to confirm the first real learning updates promptly.

## Initial local checks on 2026-10-30

The local setup is implemented and left uncommitted. No cluster job was submitted,
no Git push was made, and no new policy or policy video was produced.

Verified:

- Built `bfm-zero.sif` successfully with `bash apptainer/build.sh` (about 7.9 GB).
  The image contains the frozen root lock's Python dependencies, not a checkout
  of the experiment source. Build-time imports passed for torch, numpy, W&B,
  pydantic, MuJoCo, and package presence for Isaac Sim / Isaac Lab.
- All **44 CPU regression tests passed** in the existing local venv and again
  inside the built image with an isolated home and mounted source.
- A synthetic CPU update using the smoke agent's complete update path returned
  **41 finite metrics**. This checks network shapes and update execution, not
  physical simulation or learning progress.
- Real W&B **offline** initialization, logging, finalization and event-file
  creation passed with the existing pinned SDK, using a temporary directory.
  The test logged only `healthcheck/value`, and did not upload a cloud run.
- All three presets validate without building a simulator. Reference preserves
  the native agent/environment configuration. Overrides, credentials-in-config
  rejection, resume compatibility and standalone tracking report serialization
  are covered by the added tests.
- Native Hydra physical configuration composes with the configured hard-waist G1.
- The current `~/Documents/cluv` parser accepts the Mila L40S configuration and
  its command-line help is reached through `cluster/cluv.sh`. No SSH submission
  was needed for that check.
- Shell syntax, Python compilation and guide JavaScript syntax passed. Browser
  QA exercised the training stepper, FB loss tabs, source-line navigation and
  update-ratio calculator. The guide renders locally without external scripts.

Not verified in this environment:

- NVIDIA driver/CUDA runtime access: `nvidia-smi` cannot communicate with the
  driver and `torch.cuda.is_available()` is false.
- Isaac startup, physics stepping, L40S VRAM fit, training throughput,
  convergence, checkpoint behavior under a real Slurm preemption, rendering,
  ONNX execution or policy quality.
- W&B online authentication/API compatibility with the provided Forge project.
- Live SSH access, scheduler resource names/availability, account eligibility,
  cluster filesystem mounts or offline-node asset availability.

The next real check is an allocated L40S startup job, followed by the smoke
training preset. See [START_HERE.md](START_HERE.md) for the exact commands.

Built image SHA-256:

```text
920bcb81774d43e19ef31e0b61a2ced6b7afb4f893cecf1e6e70fcdbf94c08e5
```

The checksum is also in ignored `bfm-zero.sif.sha256`. Rebuilding changes this
record; the source, system packages and base image are not a bit-reproducible build
specification even though Python dependencies use the supplied frozen lock.

## Expanded teaching notes

The guide now includes 18 teaching accordions, 58 explained code blocks, and a
four-environment prompt-scheduling example. JavaScript syntax and all new source
references passed validation. All 16 combinations of the example’s eight time
checkpoints and two sampler settings match the actual Python
`maybe_update_rollout_context()` function with deterministic mock samples.

The earlier browser QA above applies to the original guide. For this expansion,
the browser tool rejected access to the user’s `file://` page. The new notes were
validated from the local source and example logic; their live browser appearance
and controls were not rechecked through that tool. No training code changed in
this guide revision.
