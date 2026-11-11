"""Run the unmodified tracking script; record states and vary only the B averaging window.

Rendering and ONNX export are deferred to avoid repeating them during simulation.
The native reset, observation, action and one-frame indexing paths are retained.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import humanoidverse.tracking_inference as native
import humanoidverse.agents.envs.humanoidverse_isaac as env_module
from humanoidverse.agents.utils import set_seed_everywhere


def average_features(features, horizon):
    if horizon == 1:
        return features
    # Read only from the original features, including shorter windows at the end.
    return torch.stack([features[i:min(i + horizon, len(features))].mean(0)
                        for i in range(len(features))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12])
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--short", action="store_true", help="Use the previous eight 8-second segments")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    load = native.load_model_from_checkpoint_dir
    # Load checkpoint copies on CPU to avoid transient duplicate GPU allocations.
    model = load(args.model_folder / "checkpoint", device="cpu").to("cuda")
    raw_backward = model.backward_map
    native.load_model_from_checkpoint_dir = lambda *a, **kw: model
    native.export_meta_policy_as_onnx = lambda *a, **kw: None
    # Native main writes its normal z artifacts to model_folder/tracking_inference.
    # The recorder's source trajectories and manifests go to a separate result directory.
    build = native.HumanoidVerseIsaacConfig.build
    active = {}
    native_get_observation = native.get_backward_observation
    starts = [2093, 405, 600, 152, 5459, 1050, 2488, 8944]
    motion_ids = [0, 1, 2, 3, 5, 7, 13, 21] if args.short else [25]

    def get_observation(*a, **kw):
        obs, details = native_get_observation(*a, **kw)
        if args.short:
            idx = active.get("motion_idx", 0)
            start = starts[idx]
            active["motion_idx"] = idx + 1
            # Keep future frames beyond the displayed clip for a full h=12 window.
            stop = start + 400 + max(args.horizons)
            obs = native.tree_map(lambda x: x[start:stop], obs)
            details = {k: v[start:stop] for k, v in details.items()}
        return obs, details

    native.get_backward_observation = get_observation

    def capture_build(config, *a, **kw):
        wrapped, info = build(config, *a, **kw)
        active["wrapped"] = wrapped
        return wrapped, info

    native.HumanoidVerseIsaacConfig.build = capture_build

    class StateRecorder:
        def __init__(self, **kwargs):
            self.states = []
            active["recorder"] = self
            active.setdefault("recorders", []).append(self)

        def from_qpos(self, qpos):
            self.reference = np.asarray(qpos).copy()
            return [np.zeros((1, 1, 3), dtype=np.uint8)] * len(qpos)

        def render(self, env, env_idx):
            root = env.simulator.robot_root_states[:, [0, 1, 2, 6, 3, 4, 5]].detach().cpu().numpy()
            joints = env.simulator.dof_pos.detach().cpu().numpy()
            self.states.append(np.concatenate([root, joints], axis=1)[0].copy())
            return [np.zeros((1, 1, 3), dtype=np.uint8)]

    native.IsaacRendererWithMuJoco = StateRecorder
    native.media.write_video = lambda *a, **kw: None
    data_path = Path(native.HUMANOIDVERSE_DIR) / "data/lafan_29dof.pkl"
    config = json.loads((args.model_folder / "config.json").read_text())
    source = Path(native.__file__)
    for horizon in args.horizons:
        if horizon < 1:
            raise ValueError("Horizons must be positive")
        print(f"BEGIN h={horizon}", flush=True)
        set_seed_everywhere(args.seed)
        model.backward_map = lambda obs, h=horizon: average_features(raw_backward(obs), h)[:400] if args.short else average_features(raw_backward(obs), h)
        # Construct the same native environment from the same seed for every window.
        env_module._humanoidverse_env_singleton = None
        active.clear()
        with torch.inference_mode():
            native.main(args.model_folder, data_path=data_path, simulator="mujoco",
                        save_mp4=True, motion_list=motion_ids)
        recorder = active["recorder"]
        if args.short:
            states = np.concatenate([np.stack(r.states)[1:] for r in active["recorders"]])
            reference = np.concatenate([r.reference[1:] for r in active["recorders"]])
        else:
            states = np.stack(recorder.states)
            reference = recorder.reference
        if states.shape != reference.shape:
            raise ValueError(f"Reference/policy lengths differ: {states.shape}/{reference.shape}")
        np.save(args.output / f"qpos-h{horizon}.npy", states)
        if (args.output / "reference.npy").exists():
            assert np.array_equal(np.load(args.output / "reference.npy"), reference)
        else:
            np.save(args.output / "reference.npy", reference)
        env = active["wrapped"]._env
        status = json.loads((args.model_folder / "checkpoint/train_status.json").read_text())
        metadata = {"horizon": horizon, "seed": args.seed, "motion_id": motion_ids,
                    "motion_name": "multi-motion" if args.short else str(env._motion_lib.curr_motion_keys[0]),
                    "frames": len(states), "control_dt": float(env.dt),
                    "training_steps": status["time"], "model_folder": str(args.model_folder.resolve()),
                    "native_script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "env_config": config["env"], "disable_dr": False, "disable_obs_noise": False,
                    "initialization": "Original tracking_inference.py reset and zero-action step"}
        if args.short:
            import joblib
            names = list(joblib.load(data_path))
            metadata["segments"] = [{"motion_id": m, "motion_name": names[m],
                "start_frame": s, "frame_start": i*400, "frames": 400}
                for i,(m,s) in enumerate(zip(motion_ids,starts))]
            metadata["initial_states"] = [r.states[0].tolist() for r in active["recorders"]]
        (args.output / f"h{horizon}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"COMPLETE h={horizon}: {len(states)} frames", flush=True)
        env_module._humanoidverse_env_singleton = None
        active.clear()
        del recorder, env, states
        gc.collect()


if __name__ == "__main__":
    main()
