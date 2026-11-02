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
import sysconfig


def check_compile_toolchain():
    """Exercise the C++ and, when available, CUDA forward/backward compilers."""
    header = Path(sysconfig.get_path("include")) / "Python.h"
    if not header.is_file():
        raise RuntimeError(f"Missing {header}. Rebuild the container with python3.10-dev.")
    if not (shutil.which("c++") or shutil.which("g++")):
        raise RuntimeError("Missing C++ compiler. Rebuild the container with build-essential.")
    import torch

    compiled = torch.compile(lambda x: x.sin() + x * 2, fullgraph=True)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        x = torch.linspace(-1, 1, 16, device=device, requires_grad=True)
        y = compiled(x)
        torch.testing.assert_close(y, x.sin() + x * 2)
        (grad,) = torch.autograd.grad(y.sum(), x)
        torch.testing.assert_close(grad, x.cos() + 2)
        print(f"torch.compile forward/backward passed: {device}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--boot-isaac", action="store_true", help="Boot a headless SimulationApp; requires NVIDIA GPU")
    p.add_argument("--check-compile", action="store_true", help="Compile and run small CPU/CUDA forward/backward kernels")
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args(argv)
    report = {"python": platform.python_version(), "packages": {}, "commands": {}}
    report["python_header"] = str(Path(sysconfig.get_path("include")) / "Python.h")
    report["python_header_exists"] = Path(report["python_header"]).is_file()
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
    if args.check_compile:
        check_compile_toolchain()
    if args.boot_isaac:
        from humanoidverse.agents.envs.humanoidverse_isaac import instantiate_isaac_sim
        instantiate_isaac_sim(num_envs=1, headless=True)
        print("Headless Isaac startup passed. Next run the actual smoke preset.", flush=True)


if __name__ == "__main__":
    main()
