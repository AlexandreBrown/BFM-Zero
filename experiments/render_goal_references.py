"""Render the exact static reference poses used by a recorded goal rollout."""

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from PIL import Image

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.rollout / "rollout.json").read_text())
    config = json.loads((Path(protocol["model_folder"]) / "config.json").read_text())["env"]
    config.update(device="cpu", lafan_tail_path=protocol["data_path"],
                  disable_domain_randomization=True, disable_obs_noise=True)
    config["hydra_overrides"] += ["simulator=mujoco", "env.config.headless=True"]
    wrapped, _ = HumanoidVerseIsaacConfig(**config).build(1)
    env = wrapped._env
    assert np.isclose(env.dt, protocol["control_dt"])
    renderer = IsaacRendererWithMuJoco(render_size=protocol["render_size"])
    selected = {item["motion_name"]: item for item in json.loads(Path(protocol["goal_json"]).read_text())}
    args.output.mkdir(parents=True, exist_ok=True)
    goals = []
    last_motion = None
    for idx, name in enumerate(protocol["goal_names"]):
        motion, frame = name.rsplit("_", 1)
        frame = int(frame)
        if motion != last_motion:
            env.set_is_evaluating(selected[motion]["motion_id"])
            # The loaded slot is 0; JSON IDs index the global motion library.
            actual = str(env._motion_lib.curr_motion_keys[0])
            if actual != motion:
                raise ValueError(f"JSON motion {motion} points at loaded motion {actual}")
            last_motion = motion
        # Match get_backward_observation's time grid exactly (not source-file fps).
        times = torch.arange(int(np.ceil(float(env._motion_lib._motion_lengths[0]) / env.dt))) * env.dt
        state = env._motion_lib.get_motion_state(0, times[frame:frame + 1])
        xyz = state["root_pos"][0].numpy()
        xyzw = state["root_rot"][0].numpy()
        joints = state["dof_pos"][0].numpy()
        qpos = np.concatenate([xyz, xyzw[[3, 0, 1, 2]], joints])
        image = renderer.from_qpos([qpos])[0]
        file_name = f"goal-{idx:03d}.png"
        Image.fromarray(image).save(args.output / file_name)
        goals.append({"name": name, "image": file_name, "motion": motion,
                      "frame": frame, "time_seconds": float(times[frame]),
                      "qpos": qpos.tolist()})
    (args.output / "manifest.json").write_text(json.dumps({"goals": goals,
        "data_path": protocol["data_path"], "control_dt": env.dt,
        "note": "Exact kinematic target pose; no physics rollout. Velocity targets are zero."}, indent=2) + "\n")
    print(f"Rendered {len(goals)} reference poses to {args.output}")


if __name__ == "__main__":
    main()
