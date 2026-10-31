"""Run the native tracking evaluator on a saved model and write numeric results.

The released native evaluator and tracking video script use one-frame prompts;
the model's generic tracking_inference method uses seq_length instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-folder", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--episodes-per-motion", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--simulator", choices=["isaacsim", "mujoco"], default="isaacsim")
    p.add_argument("--disable-dr", action="store_true")
    p.add_argument("--disable-obs-noise", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if args.num_envs < 1 or args.episodes_per_motion < 1:
        p.error("Environment and episode counts must be positive.")

    from experiments.train import check_data, file_hash
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.agents.evaluations.humanoidverse_isaac import HumanoidVerseIsaacTrackingEvaluationConfig
    from humanoidverse.agents.utils import set_seed_everywhere

    check_data(args.data_path)
    set_seed_everywhere(args.seed)
    model_folder = args.model_folder.resolve()
    config = json.loads((model_folder / "config.json").read_text())
    env_config = config["env"]
    overrides = [v for v in env_config["hydra_overrides"] if v.split("=")[0] not in ("simulator", "env.config.headless")]
    overrides += [f"simulator={args.simulator}", "env.config.headless=True"]
    env_config.update(lafan_tail_path=str(args.data_path.resolve()), enable_cameras=False,
                      disable_domain_randomization=args.disable_dr, disable_obs_noise=args.disable_obs_noise,
                      hydra_overrides=overrides)
    env_config["device"] = "cuda:0" if args.device == "cuda" else args.device
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=args.device)
    model.eval()
    env, _ = HumanoidVerseIsaacConfig.model_validate(env_config).build(args.num_envs)
    try:
        evaluation = HumanoidVerseIsaacTrackingEvaluationConfig(n_episodes_per_motion=args.episodes_per_motion).build()
        # extract_model accepts the actual BaseModel directly.
        metrics, aggregate = evaluation.run(timestep=0, agent_or_model=model, replay_buffer=None, logger=None, env=env)
        report = {"protocol": "native_tracking", "seed": args.seed, "simulator": args.simulator,
                  "num_envs": args.num_envs, "episodes_per_motion": args.episodes_per_motion,
                  "disable_dr": args.disable_dr, "disable_obs_noise": args.disable_obs_noise,
                  "lookahead_frames": 1,
                  "data_sha256": file_hash(args.data_path), "metrics": metrics, "aggregate": aggregate}
        output = model_folder / "evaluations"
        output.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result = output / f"tracking-{args.simulator}-seed{args.seed}-{stamp}.json"
        # Native metrics contain numpy scalar values.
        result.write_text(json.dumps(report, indent=2, default=lambda value: value.item()))
        print(f"Saved: {result}\n{json.dumps(aggregate, indent=2, default=lambda value: value.item())}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
