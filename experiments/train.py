"""A small, validated CLI around the repository's actual training recipe.

Run from the repository root: python -m experiments.train --help
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SECRET_FIELD = re.compile(r"(?:api[_-]?key|password|passwd|secret|credential|access[_-]?token|auth[_-]?token|private[_-]?key)", re.I)


def reject_secrets(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if SECRET_FIELD.search(key):
                raise ValueError("Credentials belong in the process environment or an external credential store, never experiment config.")
            reject_secrets(child)
    elif isinstance(value, list):
        for child in value:
            reject_secrets(child)


def merge(base, patch):
    """Merge dictionaries; replace lists and scalars. Schema validation follows."""
    result = copy.deepcopy(base)
    for key, value in patch.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def set_field(config, expression):
    key, sep, raw = expression.partition("=")
    if not sep:
        raise ValueError("--set requires dotted.path=JSON_VALUE (strings may be unquoted).")
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            raise ValueError(f"Unknown configuration path: {key}")
        target = target[part]
    if not isinstance(target, dict) or parts[-1] not in target:
        raise ValueError(f"Unknown configuration path: {key}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    reject_secrets({key: value})
    target[parts[-1]] = value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["smoke", "pilot", "reference"], default="smoke")
    p.add_argument("--config", type=Path, action="append", default=[], help="JSON override file; repeatable, applied in order")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    p.add_argument("--hydra", action="append", default=[], help="Replace/append an environment Hydra override by key")
    p.add_argument("--seed", type=int, default=None, help="Default 4728; resume preserves the saved seed")
    p.add_argument("--work-dir", type=Path, required=True, help="Unique run directory; reuse only with --resume")
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "offline"))
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or "glen-berseth")
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "bfm-zero"))
    p.add_argument("--group", default=None)
    p.add_argument("--resume", action="store_true", help="Load saved config and continue the same checkpoint")
    p.add_argument("--print-config", action="store_true", help="Validate and print config without creating a simulator or run")
    return p


def resolve_config(args):
    # Importing TrainConfig does not boot Isaac; only .build() does.
    from humanoidverse.train import TrainConfig, default_train_config

    work_dir = args.work_dir.expanduser().resolve()
    if args.resume:
        config = json.loads((work_dir / "config.json").read_text())
    else:
        config = default_train_config().model_dump(mode="json")
        config = merge(config, json.loads((ROOT / "experiments" / "presets" / f"{args.preset}.json").read_text()))
    for path in args.config:
        patch = json.loads(path.read_text())
        if not isinstance(patch, dict):
            raise ValueError("Override config must be a JSON object.")
        config = merge(config, patch)
    if args.data_path:
        config["env"]["lafan_tail_path"] = str(args.data_path.expanduser().resolve())
    for expression in args.overrides:
        set_field(config, expression)
    if args.seed is not None:
        config["seed"] = args.seed
    # CLI operational flags are authoritative, including during resume.
    config.update(work_dir=str(work_dir), use_wandb=args.wandb_mode != "disabled",
                  wandb_ename=args.entity, wandb_pname=args.project,
                  wandb_gname=args.group or (config["wandb_gname"] if args.resume else args.preset))
    hydra = config["env"]["hydra_overrides"]
    for expression in args.hydra:
        if "=" not in expression:
            raise ValueError("--hydra requires key=value")
        key = expression.split("=", 1)[0]
        reject_secrets({key: None})
        hydra[:] = [old for old in hydra if old.split("=", 1)[0] != key]
        hydra.append(expression)
    reject_secrets(config)
    cfg = TrainConfig.model_validate(config)
    validate_schedule(cfg)
    return cfg


def validate_schedule(cfg):
    n = cfg.online_parallel_envs
    if n <= 0 or cfg.num_agent_updates <= 0:
        raise ValueError("Environment and update counts must be positive.")
    for key in ("num_env_steps", "update_agent_every", "log_every_updates", "checkpoint_every_steps", "eval_every_steps", "buffer_size"):
        if getattr(cfg, key) <= 0 or getattr(cfg, key) % n:
            raise ValueError(f"{key} must be a positive multiple of online_parallel_envs={n}.")
    if cfg.num_seed_steps < 0 or cfg.num_seed_steps % n or cfg.num_seed_steps >= cfg.num_env_steps:
        raise ValueError("num_seed_steps must be a nonnegative environment-count multiple below num_env_steps.")
    if cfg.agent.train.batch_size < 2 or cfg.agent.train.batch_size % cfg.agent.model.seq_length:
        raise ValueError("batch_size must be >=2 and divisible by model.seq_length.")
    if cfg.buffer_size // n < cfg.agent.model.seq_length + 1:
        raise ValueError("Replay must contain more time steps per environment than seq_length.")
    if not 0 <= cfg.agent.train.train_goal_ratio + cfg.agent.train.expert_asm_ratio <= 1:
        raise ValueError("Goal and expert latent mixture probabilities must sum to <=1.")
    if not cfg.use_trajectory_buffer or not cfg.checkpoint_buffer:
        raise ValueError("This launcher keeps trajectory replay checkpointing for resume and reward inference.")


def check_data(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Motion data missing: {path}. Run git lfs pull or pass --data-path.")
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Motion data is an LFS pointer: {path}. Run git lfs pull.")


def git_output(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args, cfg):
    import torch
    from humanoidverse.train import checkpoint_path, validate_resume_config

    check_data(cfg.env.lafan_tail_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Isaac training requires a working NVIDIA GPU/driver. Use --print-config for CPU-only validation.")
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # A file lock also prevents two Slurm jobs sharing a resume directory.
    with (work / ".launch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cp = checkpoint_path(work)
        if args.resume:
            if not (cp / "buffers" / "train").is_dir():
                raise ValueError("Resume requires the saved replay buffer and checkpoint in this work directory.")
            validate_resume_config(work, cfg)
        elif (work / "config.json").exists() or (work / "launch.json").exists() or cp.exists():
            raise ValueError("Run directory already contains a run. Choose a new directory or use --resume.")

        os.environ["WANDB_MODE"] = args.wandb_mode
        os.environ["WANDB_DISABLE_CODE"] = "true"
        os.environ["WANDB_DISABLE_GIT"] = "true"
        run_id = work / "wandb-run-id.txt"
        if not run_id.exists():
            import uuid
            run_id.write_text(uuid.uuid4().hex[:8] + "\n")
        os.environ["WANDB_RUN_ID"] = run_id.read_text().strip()
        if args.wandb_mode == "online":
            os.environ["WANDB_RESUME"] = "allow"
        else:
            # Offline segments are separate files. W&B does not support offline resume.
            os.environ.pop("WANDB_RESUME", None)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        manifest = {
            "created_utc": stamp, "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "config": cfg.model_dump(mode="json"), "torch": torch.__version__,
            "python": sys.version.split()[0], "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "wandb_mode": args.wandb_mode,
            "lock_sha256": file_hash(ROOT / "uv.lock"),
            "image_sha256": os.environ.get("BFM_IMAGE_SHA256"),
            "data_sha256": file_hash(cfg.env.lafan_tail_path),
        }
        (work / "launches").mkdir(exist_ok=True)
        serialized = json.dumps(manifest, indent=2)
        (work / "launches" / f"{stamp}.json").write_text(serialized)
        if not args.resume:
            (work / "launch.json").write_text(serialized)
        print(f"Run: {work}\nW&B mode: {args.wandb_mode}", flush=True)
        try:
            cfg.build().train()
        finally:
            if cfg.use_wandb:
                import wandb
                wandb.finish()


def main(argv=None):
    args = parser().parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run from the BFM-Zero repository root (python -m experiments.train ...).")
    cfg = resolve_config(args)
    if args.print_config:
        print(cfg.model_dump_json(indent=2))
    else:
        run(args, cfg)


if __name__ == "__main__":
    main()
