"""Upload locally generated videos to a separate W&B evaluation run (explicit opt-in)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-folder", type=Path, required=True)
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "glen-berseth"))
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "bfm-zero"))
    p.add_argument("--group", default="evaluation-videos")
    p.add_argument("--wandb-mode", choices=["online", "offline"], default="offline")
    args = p.parse_args(argv)
    model = args.model_folder.resolve()
    videos = sorted(p for folder in ("tracking_inference", "goal_inference", "reward_inference") for p in (model / folder).rglob("*.mp4"))
    if not videos:
        p.error("No inference MP4 files found. Generate videos with the native scripts first.")
    import wandb
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_DISABLE_GIT"] = "true"
    directory = model / "_wandb_videos"
    directory.mkdir(exist_ok=True)
    with wandb.init(entity=args.entity, project=args.project, group=args.group, name=f"videos-{model.name}",
                    job_type="evaluation", mode=args.wandb_mode, dir=str(directory)):
        for video in videos:
            wandb.log({str(video.relative_to(model)): wandb.Video(str(video), format="mp4", fps=50)})
    print(f"Logged {len(videos)} videos ({args.wandb_mode}).")


if __name__ == "__main__":
    main()
