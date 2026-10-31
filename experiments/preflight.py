"""Inspect runtime readiness without reading or printing credentials."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--boot-isaac", action="store_true", help="Boot a headless SimulationApp; requires NVIDIA GPU")
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args(argv)
    report = {"python": platform.python_version(), "packages": {}, "commands": {}}
    for package in ("torch", "isaacsim", "isaaclab", "wandb", "mujoco", "pydantic", "humenv"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    for command in ("uv", "cluv", "apptainer", "nvidia-smi", "ffmpeg", "git-lfs"):
        report["commands"][command] = bool(shutil.which(command))
    root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or root / "humanoidverse" / "data"
    report["data"] = {}
    for name in ("lafan_29dof.pkl", "lafan_29dof_10s-clipped.pkl"):
        path = data_dir / name
        pointer = False
        if path.is_file():
            with path.open("rb") as stream:
                pointer = stream.read(128).startswith(b"version https://git-lfs.github.com/spec/")
        report["data"][name] = {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0, "lfs_pointer": pointer}
    gpu = False
    if importlib.util.find_spec("torch"):
        import torch
        gpu = torch.cuda.is_available()
        report["cuda_available"] = gpu
        report["cuda_build"] = torch.version.cuda
        if gpu:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    print(json.dumps(report, indent=2), flush=True)
    if (args.require_gpu or args.boot_isaac) and not gpu:
        raise SystemExit("GPU unavailable. Configuration inspection and CPU tests can still run.")
    if args.boot_isaac:
        from humanoidverse.agents.envs.humanoidverse_isaac import instantiate_isaac_sim
        instantiate_isaac_sim(num_envs=1, headless=True)
        print("Headless Isaac startup passed. Next run the actual smoke preset.", flush=True)


if __name__ == "__main__":
    main()
