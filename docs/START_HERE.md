# Start with one complete learning loop

Open [the interactive code guide](index.html), then follow the numbered ladder
below. The guide includes a local source browser, the tensor/data flow, each loss,
the W&B metric names, and places where the code and paper differ.

This layer calls the native `default_train_config()` and `Workspace`. The only
change under `humanoidverse/` is W&B run naming and moving its files under the run
directory. The networks, optimizers, rewards, environment and learning updates are
unchanged. The assistant does not make commits; you review and commit the infrastructure changes.

Current cluster defaults use the modern W&B image and online logging; see
[modern-key authentication](../apptainer/wandb-modern.md). The numbered ladder
below still recommends a smoke check before scientific comparisons.

Your project: [glen-berseth/bfm-zero](https://forge.coreweave.com/wandb/glen-berseth/bfm-zero).
The team/project are configured in `cluster/pyproject.toml`. Credentials are not.
The configured API endpoint is `https://api.forge.coreweave.com`, distinct from
the project web page. The modern container passed an online write check to this
project using the external, read-only `.netrc` mount. Use
[the personal experiment workspace](https://forge.coreweave.com/wandb/glen-berseth/bfm-zero?nw=nwuseralexandrebrown)
to view runs. The base image SDK rejects new-format API keys; use the modern
container described above for both training and login.

## The files you will actually touch

| File | Purpose |
|---|---|
| `experiments/train.py` | Validated CLI, overrides, explicit resume, run provenance |
| `experiments/presets/` | Smoke, pilot, native reference recipe |
| `experiments/ablations/` | Small JSON patches; each changes one stated factor |
| `cluster/pyproject.toml` | Cluster resources, paths, W&B team/project |
| `cluster/submit.sh` | Submit held, prepare Slurm logs, then release |
| `cluster/cluv.sh` | Runs the current `~/Documents/cluv` source checkout |
| `cluster/job.sh` | One GPU job, direct code bind, container mounts |
| `apptainer/` | Frozen dependency image and transfer helpers |
| `experiments/evaluate.py` | Numerical native tracking evaluation to JSON |
| `experiments/log_videos.py` | Optional video logging to a separate W&B run |

cluv requires Python >=3.11; BFM-Zero requires Python 3.10. They deliberately use
separate environments. The small `cluster/` project prevents cluv from installing
the entire Isaac stack on every login node. Its `project_dir` ends in `/cluster`;
the current cluv implementation correctly clones the whole parent repository.
Do not move that directory independently of the repository.

## Numbered experiment ladder

1. **Inspect the configuration before using a GPU.** From the BFM-Zero root:

   ```bash
   .venv/bin/python -m experiments.preflight
   .venv/bin/python -m experiments.train --preset smoke \
     --work-dir results/smoke-local --print-config
   MPLCONFIGDIR=/tmp/bfm-mpl .venv/bin/python -m unittest discover -s tests -v
   ```

   The motion files in this checkout are real LFS payloads. There is no pretrained
   checkpoint here; the `model/` folder contains example media. The current
   session reports no working NVIDIA driver. Config inspection and CPU regression
   tests work here; Isaac training must run in a working GPU environment.

2. **Build the environment once and inspect the cluster configuration.**

   ```bash
   test -f bfm-zero.sif || bash apptainer/build.sh
   test -f bfm-zero-wandb.sif || apptainer build --fakeroot bfm-zero-wandb.sif apptainer/wandb-modern.def
   sha256sum bfm-zero-wandb.sif > bfm-zero-wandb.sif.sha256
   bash cluster/cluv.sh submit --help
   ```

   Review `cluster/pyproject.toml`. Default: Mila, one L40S, 128 GB host RAM,
   8 CPUs, 2 hours, online W&B. Long runs override walltime explicitly. Account names for Alliance are copied from
   SceneGMT and need to match your current allocations. Change them in this one
   file or override resource flags on the CLI. A local 7.9 GB SIF was built and
   passed all 44 CPU tests; see [validation](VALIDATION.md). The modern derivative
   also passed those tests, online logging, and a 20-step CUDA physics diagnostic
   on Mila L40S; full training startup is being validated separately. Reserve substantial disk space for rebuilding and see
   [container notes](../apptainer/README.md).
   `bash cluster/cluv.sh ...` loads `~/Documents/cluv`, not an older installed
   package. Set `BFM_CLUV_SOURCE` if you move that checkout.

3. **Review and commit the files yourself, then stage the image and data.**

   cluv operates on Git commits, can push the branch, and does not transmit
   uncommitted additions. Your existing `uv.lock` and video modifications were
   preserved. Decide which belong in your commit; inspect `git diff` and staged
   changes. Never use cluv's `--autocommit` for this workflow.

   After your review and commit:

   ```bash
   bash cluster/cluv.sh login mila
   bash apptainer/push.sh mila
   ```

   These commands use your existing SSH aliases and external authentication.
   The transfer sends only the image and the two public motion files. Code goes
   through cluv/Git. No API key belongs in TOML, JSON, a command argument or the
   image. For online W&B, run `wandb login` interactively on the target cluster
   with the image's SDK; it stores credentials in your external `~/.netrc`.
   Offline compute jobs need no W&B credential.

4. **Prove simulator startup on an L40S, then run a tiny real training job.**

   ```bash
   bash cluster/submit.sh mila --time=00:30:00 -- check --require-gpu --boot-isaac
   bash cluster/submit.sh mila --time=01:00:00 -- train --preset smoke
   ```

   Startup and training are separate checks. First startup can compile shaders
   or request assets; restricted compute networking can still require cache
   preparation. Smoke uses 32 environments, batch 64, small networks, CPU replay,
   4096 transitions, no compilation, and no expensive tracking evaluation. It is
   deliberately a functional test. Pass when optimizer metrics are finite, a final
   checkpoint exists at step 4096, and W&B offline files exist. Do not judge policy
   quality at this stage. New jobs default to unique scratch run directories.

5. **Validate W&B and resume before spending hours.**

   With internet on the target node and `wandb login` completed externally:

   ```bash
   bash cluster/submit.sh mila --time=01:00:00 -- train --preset smoke \
     --wandb-mode online --group plumbing
   ```

   Confirm metrics reach `glen-berseth/bfm-zero`. The base lock has W&B 0.12.21;
   the selected derivative uses 0.27.1 and passed an online write check. If authentication
   fails, check the organization's API endpoint and SDK support before a long run.
   Do not change simulator/learning dependencies to fix a logging problem.

   To test resume, replace the path below with the smoke run's **outer directory**:

   ```bash
   bash cluster/submit.sh mila --time=01:00:00 -- train --resume \
     --work-dir /scratch/your-user/bfm-zero/output/runs/JOB_0 \
     --set num_env_steps=8192 --group plumbing
   ```

   Resume loads the saved recipe, optimizer and replay; the preset is ignored.
   Do not change seed, architecture, environment or learning parameters. The
   native resume validator rejects those changes. Continue the same path with a
   larger budget. Simulator state, RNG state and rollout context are not saved,
   so resume is statistically useful but not a bit-exact continuation. A
   persistent W&B ID is kept for online continuation; offline segments are
   separate files and do not support seamless offline resume.

6. **Run a full-network pilot on L40S and measure the cost of learning.**

   ```bash
   bash cluster/submit.sh mila --time=04:00:00 -- train --preset pilot \
     --seed 4728 --group pilot --wandb-mode offline
   ```

   Pilot retains the reference networks, 1024 environments, batch 1024 and
   16 updates per vector step, but uses 512k replay transitions on CPU, a 10.24M
   transition budget and 2.048M evaluation/checkpoint cadence. It is a learning
   pilot, not the full reproduction. Record host RAM, VRAM (`nvidia-smi` inside the
   allocation), FPS after compilation, checkpoint size and evaluation duration.
   The network configuration is large; neither memory fit nor throughput has
   been measured on L40S yet. If it OOMs, first keep replay on CPU and compile off
   (`--set agent.compile=false`). Reducing batch or network size changes the recipe;
   label those runs explicitly and preserve your first successful pilot config.

   Plot `eval/humanoidverse_tracking_eval/emd`,
   `eval/humanoidverse_tracking_eval/joint_angle_l2_rad`,
   `train/fb_offdiag`, `train/fb_diag`, `train/orth_loss`,
   `train/disc_expert_loss`, `train/disc_train_loss`, `train/critic_loss`,
   `train/aux_critic_loss`, `train/Q_fb`, `train/Q_discriminator`,
   `train/Q_aux`, `train/mean_aux_reward`, and `train/FPS`.
   Use W&B's Step axis: steps count **all environments**, not vector iterations.
   Native evaluation starts at step zero. Losses need not decrease monotonically,
   and total `fb_loss` can be negative. Favor better held-protocol evaluation and
   video quality over prettier optimization curves.

7. **Freeze an early checkpoint and evaluate it with a fixed protocol.**

   Training overwrites `checkpoint/`; copy the entire run or at least
   `config.json`, `config.yaml` and `checkpoint/` into a new scratch directory
   **while the training job is stopped**, before resuming. Reward inference needs
   replay too. Do not copy during checkpoint publication. Name snapshots by
   environment step (e.g. `step-10240000`). Then:

   ```bash
   bash cluster/submit.sh mila --time=02:00:00 -- eval \
     --model-folder /scratch/your-user/bfm-zero/snapshots/step-10240000 \
     --num-envs 32 --seed 0 --episodes-per-motion 1
   ```

   Output: timestamped JSON under `evaluations/`, per-motion metrics and
   aggregate metrics. This explicitly uses the full `lafan_29dof.pkl` and native
   one-frame lookahead, matching this fork's native evaluator. Full LAFAN motions
   share the training distribution; this is
   a consistent evaluation protocol, not a held-out-data claim. Evaluate the same
   snapshot again with seeds 1 and 2, then with `--disable-dr`; keep noise fixed
   unless testing it separately. Mean-plus-std over motions is not seed uncertainty.

8. **Make three kinds of prompt visible.**

   ```bash
   bash cluster/submit.sh mila --time=02:00:00 -- tracking \
     --model-folder /scratch/your-user/bfm-zero/snapshots/step-10240000 \
     --simulator mujoco --motion-list 0 1 25 --save-mp4
   bash cluster/submit.sh mila --time=02:00:00 -- goal \
     --model-folder /scratch/your-user/bfm-zero/snapshots/step-10240000 \
     --simulator mujoco --save-mp4
   bash cluster/submit.sh mila --time=04:00:00 -- reward \
     --model-folder /scratch/your-user/bfm-zero/snapshots/step-10240000 \
     --simulator mujoco --num-samples 150000 --n-inferences 1 --save-mp4
   ```

   These are the native scripts. Tracking exports a one-frame-prompt video,
   matching the native numeric evaluator; the model's generic method uses an
   8-frame window. Goal inference needs
   the goal JSON included under the robot assets. Reward inference relabels saved
   replay, calls `reward_wr_inference` and saves prompts/videos; it does **not**
   print a benchmark return table. Do not treat the environment's auxiliary reward
   or a plausible video as a downstream task score. Tracking JSON is the first
   quantitative benchmark here; measuring reward returns is a later extension.
   The renderers use MuJoCo; run `MUJOCO_GL=egl`, which `job.sh` already sets.
   Exported ONNX and videos are evidence of export/rendering only when those jobs
   actually succeed. No new trained videos have been produced in this setup.

   Optional upload, after reviewing the generated videos:

   ```bash
   bash cluster/submit.sh mila --time=01:00:00 -- log-videos \
     --model-folder /scratch/your-user/bfm-zero/snapshots/step-10240000 \
     --wandb-mode online
   ```

9. **Continue to a meaningful training horizon without changing several things.**

   Once the pilot works, start a **new** reference run with full CPU replay on L40S:

   ```bash
   bash cluster/submit.sh mila --time=24:00:00 -- train --preset reference \
     --config experiments/ablations/l40s_memory.json \
     --config experiments/ablations/paper_budget.json \
     --seed 4728 --group reference-l40s
   ```

   This uses 5.12M replay transitions, 192M total transitions, original networks
   and learning parameters, with CPU placement and smaller inference chunks.
   `reference` alone uses the repository's 384M budget and GPU replay. The paper's
   Appendix B reports ~192M and a steeper motion-priority exponent; there are also
   style-reward presentation inconsistencies. The guide identifies these rather
   than claiming exact paper reproduction. Compare this fork's fixed recipe first.

   At 50M and 100M, evaluate snapshots with the **same** motions, seeds, noise,
   DR and lookahead as step 7. Upstream README suggests EMD below 0.75 after
   50–100M on its training eval; that is a diagnostic hint, not a convergence
   guarantee or directly comparable to a different evaluation dataset/protocol.
   Continue while evaluations and diverse prompt videos improve. Budget from
   measured FPS: remaining hours = remaining transitions / FPS / 3600, plus
   evaluation and checkpoint overhead. Choose job walltime from that measurement.

10. **Measure variability before making algorithmic conclusions.** Run the same
    successful recipe with seeds 4728, 4729, 4730 in separate directories, ideally
    a throttled array:

    ```bash
    bash cluster/submit.sh mila --array=4728-4730%1 --time=24:00:00 -- train \
      --preset reference --config experiments/ablations/l40s_memory.json \
      --config experiments/ablations/paper_budget.json --group reference-seeds
    ```

    Array indices become seeds automatically. Each task has its own run path.
    Compare at equal transition budgets and also plot wall-clock cost. Keep the
    full config, commit, image checksum and motion-data hash with results.

11. **Ablate one mechanism at a time, with matched seeds/budgets.** Use a shorter
    fixed 50M–100M budget for screening; rerun promising observations longer.

    ```bash
    bash cluster/submit.sh mila --time=24:00:00 -- train --preset reference \
      --config experiments/ablations/l40s_memory.json \
      --config experiments/ablations/no_aux_actor.json \
      --set num_env_steps=51200000 --seed 4728 --group ablation-no-aux
    ```

    Compare against a matched baseline with the same 51.2M budget. `no_aux_actor`
    sets only the actor's auxiliary coefficient to zero; the critic still trains.
    Look for tracking tradeoffs, contacts, slipping and action jitter.
    `no_style_actor` similarly removes the actor's style term, while retaining the
    discriminator, expert latent mixture and expert-prompt rollouts. It is **not**
    pure FB. `no_dr` disables configured dynamics randomization but retains sensor
    noise; evaluate both nominal and randomized dynamics. `z64` changes the latent
    dimension and requires a new run. Delay history ablations: removing history
    requires changing multiple input filters and normalizer keys consistently.

12. **Explore inference before inventing new training objectives.** For a fixed
    frozen policy and replay, compare reward inference sample counts (10k, 50k,
    150k, 400k) and repeat each inference several times. Save which inference
    operator was used (`reward_wr_inference` is reward-softmax weighted), the
    sampling seed and selected prompt. Inspect failure modes per task, then add a
    numerical reward-return evaluator before selecting a method by reward score.
    Try interpolation between goal prompts, holding the same initial state;
    normalized interpolation changes prompt direction, not a calibrated blend of
    task achievement. Few-shot CEM adaptation, held-out motion splits, larger
    latent spaces and additional data are good follow-ups after this baseline.
    The released scripts here do not implement the paper's full CEM/DIAL workflow.

## Storage, logging and retrieval

A run contains `config.json` (training config), `config.yaml` (environment config),
`launch.json`, timestamped `launches/*.json`, `wandb-run-id.txt`, native CSV logs,
`_wandb/`, and `checkpoint/`. Provenance records commit/dirty status, software/GPU,
data and lock hashes, and resolved config. It never dumps environment variables.
The normal home checkout is bound directly from `BFM_CODE_DIR`; keep it at the
experiment's version while training runs. The image hash is written
next to the SIF by `build.sh`; archive it with your result notes.

W&B logs metrics/config; it does not automatically back up checkpoints or replay.
`checkpoint_buffer=true` is intentionally kept: reward inference and faithful
optimizer/replay continuation need that data. The full replay is roughly 44 GiB
of float payload with the current observation/history layout, before overhead.
Checkpoint publication needs space for two copies; host RAM needs headroom for
expert data, simulator state and serialization. L40S's 48 GB VRAM makes CPU replay
the sensible first test, not proof that the remaining model/activations fit.

For offline runs, sync from a network-accessible login node using the same SDK:

```bash
apptainer exec -B "$SCRATCH:$SCRATCH" \
  "$SCRATCH/bfm-zero/containers/bfm-zero.sif" wandb sync \
  "$SCRATCH/bfm-zero/output/runs/JOB_0/_wandb/wandb/"offline-run-*
```

Sync each offline segment intentionally. Avoid creating competing online writers
for the same run ID. Native online resume may already have logged steps newer than
the checkpoint; W&B can discard replayed older steps. That does not rewind the
server log. Use a new logging run if a clearly separated restart is preferable.

`cluv sync mila` can fetch all results, including very large replay checkpoints.
With the current cluv, use `--no-fetch-results` on submission when appropriate.
For selective retrieval, use rsync of the one run, excluding `checkpoint/buffers/`
if you only need metrics/videos/weights. Do not exclude replay from your only
backup if you intend reward inference or training resume later.

To switch GPU for a one-off **diagnostic** job, pass `--gres=gpu:<type>:1` before
`--`. To switch cluster, pass another configured alias and stage its image/data.
GPU names, partitions, driver support, allocations and networking remain cluster
specific. Native `Workspace` does not implement distributed training: requesting
multiple GPUs does not make this recipe distributed.

## Sources and scope

- [BFM-Zero paper, v2](https://arxiv.org/html/2511.04131v2), sections 2.1–2.2 and Appendix B.
- [Upstream repository](https://github.com/LeCAR-Lab/BFM-Zero); the local fork is authoritative for executable behavior.
- Current local cluv implementation: `~/Documents/cluv/cluv/config.py`,
  `cli/sync.py`, `cli/submit.py` and `docs/guides/submit/`.
- [Isaac Lab 2.0.2 installation](https://isaac-sim.github.io/IsaacLab/v2.0.2/source/setup/installation/pip_installation.html).
- [NVIDIA GPU requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
  and [L40S specifications](https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html).

Regenerate the source browser after code changes with
`python3 docs/build_source.py`. Open `docs/index.html` directly, or serve the
repository with `python3 -m http.server 8000` and visit `/docs/`.
