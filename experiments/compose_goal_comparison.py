"""Compose goal-reaching rollouts with a reference pose and synchronized task timeline."""

import argparse
import json
import re
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = "#101820"
TEXT = "#f0f4f8"
MUTED = "#a8bacb"
CARD = "#1b2936"
ACCENTS = ["#ad9cff", "#5cd7c6", "#ffcd79"]


def font(size, bold=False):
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def fit_text(draw, text, xy, max_width, size=24, fill=TEXT, bold=False):
    while size > 12 and draw.textbbox((0, 0), text, font=font(size, bold))[2] > max_width:
        size -= 1
    draw.text(xy, text, fill=fill, font=font(size, bold))


def describe_goal(name):
    motion, frame = name.rsplit("_", 1)
    match = re.fullmatch(r"(.+?)(\d+)_subject(\d+)", motion)
    if match:
        activity, take, performer = match.groups()
        activity = re.sub(r"([a-z])([A-Z])", r"\1 \2", activity).lower()
        activity = activity.replace("and", "&")
        activity = activity[0].upper() + activity[1:]
        return {"motion": motion, "activity": activity, "take": take,
                "performer": performer, "frame": int(frame)}
    return {"motion": motion, "activity": motion.replace("_", " "),
            "take": "?", "performer": "?", "frame": int(frame)}


def motion_groups(goals):
    groups = []
    for i, goal in enumerate(goals):
        if not groups or groups[-1]["motion"] != goal["motion"]:
            groups.append({**goal, "start": i, "end": i + 1})
        else:
            groups[-1]["end"] = i + 1
    return groups


def load_reference(reference_dir, goal_names, size):
    if reference_dir is None:
        return None
    manifest = json.loads((reference_dir / "manifest.json").read_text())
    entries = {goal["name"]: goal for goal in manifest["goals"]}
    missing = set(goal_names) - entries.keys()
    if missing:
        raise ValueError(f"Missing reference poses: {sorted(missing)}")
    return [np.array(Image.open(reference_dir / entries[name]["image"]).convert("RGB").resize(
        (size, size), Image.Resampling.LANCZOS)) for name in goal_names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollout) != len(args.label):
        parser.error("Supply one --label per --rollout")
    metadata = [json.loads((p / "rollout.json").read_text()) for p in args.rollout]
    common_keys = ["data_path", "goal_names", "steps_per_goal", "episode_len", "control_dt",
                   "video_fps", "seed", "reset_to_default_pose", "disable_dr", "disable_obs_noise",
                   "render_size", "simulator"]
    for key in common_keys:
        if any(m[key] != metadata[0][key] for m in metadata[1:]):
            raise ValueError(f"Rollout protocols differ: {key}")
    protocol = metadata[0]
    frames, input_size, fps = protocol["episode_len"], protocol["render_size"], protocol["video_fps"]
    captures = [cv2.VideoCapture(str(p / "goal.mp4")) for p in args.rollout]
    for capture in captures:
        if (not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frames
                or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != input_size
                or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != input_size
                or not np.isclose(capture.get(cv2.CAP_PROP_FPS), fps, atol=0.001)):
            raise ValueError("Video dimensions, frame count, or frame rate do not match metadata")

    # Four equal panels fill a 1920-pixel frame when a reference is supplied.
    margin, gap, size, top, strip = 24, 16, 456, 212, 82
    panel_count = len(captures) + (args.reference_dir is not None)
    width = 2 * margin + panel_count * size + (panel_count - 1) * gap
    height = 1080
    image_top, image_bottom = top + strip, top + strip + size
    references = load_reference(args.reference_dir, protocol["goal_names"], size)
    goals = [describe_goal(name) for name in protocol["goal_names"]]
    groups = motion_groups(goals)
    steps_per_goal = protocol["steps_per_goal"]
    duration = steps_per_goal * protocol["control_dt"]
    template = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(template)
    draw.text((margin, 20), "BFM-ZERO   /   STATIC GOAL REACHING", fill=MUTED, font=font(21, True))
    draw.text((margin, 778), "TIME ON CURRENT TARGET", fill=MUTED, font=font(18, True))
    draw.text((margin, 888), "MOTION SEQUENCE", fill=MUTED, font=font(18, True))
    draw.text((margin, 1020),
              f"MuJoCo  |  1x speed  |  seed {protocol['seed']}  |  same initial state and target schedule",
              fill=MUTED, font=font(18))
    notes = ["Target pose changes without a manual reset"]
    if protocol["disable_obs_noise"]:
        notes.append("observation noise off")
    if protocol["disable_dr"]:
        notes.append("domain randomization off")
    draw.text((margin, 1048), "  |  ".join(notes), fill=MUTED, font=font(17))
    panel_specs = []
    if references is not None:
        panel_specs.append(("Reference target", "Frozen pose  /  desired velocity = 0", "#84b9ef"))
    for i, (label, meta) in enumerate(zip(args.label, metadata)):
        panel_specs.append((label, f"{meta['training_steps'] / 1_000_000:g}M training steps", ACCENTS[i % len(ACCENTS)]))
    for i, (label, subtitle, color) in enumerate(panel_specs):
        x = margin + i * (size + gap)
        draw.rounded_rectangle((x, top, x + size - 1, image_bottom + 1), radius=8, fill=CARD)
        draw.rectangle((x, top, x + size - 1, top + 3), fill=color)
        fit_text(draw, label, (x + 16, top + 13), size - 32, size=28, bold=True)
        fit_text(draw, subtitle, (x + 16, top + 48), size - 32, size=20, fill=color)

    timeline_left, timeline_width = margin, width - 2 * margin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    preview_frames = {0, 99, 100, frames // 2, frames - 1}
    preview_frames.update(g["start"] * steps_per_goal for g in groups if g["start"])
    try:
        with media.VideoWriter(str(args.output), shape=(height, width), fps=fps,
                               codec="h264", crf=18) as writer:
            for frame_idx in range(frames):
                goal_idx = (frame_idx // steps_per_goal) % len(goals)
                goal = goals[goal_idx]
                step = frame_idx % steps_per_goal + 1
                group_idx = next(i for i, group in enumerate(groups) if group["start"] <= goal_idx < group["end"])
                group = groups[group_idx]
                if goal_idx not in cache:
                    image = template.copy()
                    title = ImageDraw.Draw(image)
                    fit_text(title, goal["activity"], (margin, 61), width - 400, size=44, bold=True)
                    title.text((margin, 121),
                               f"Take {goal['take']}  /  Performer {goal['performer']}  /  Reference frame {goal['frame']}",
                               fill=MUTED, font=font(23))
                    title.text((width - 300, 75), f"TARGET  {goal_idx + 1:02d} / {len(goals):02d}",
                               fill=TEXT, font=font(25, True))
                    # Each timeline block represents one motion, partitioned into its selected poses.
                    for j, item in enumerate(groups):
                        x0 = timeline_left + round(item["start"] / len(goals) * timeline_width)
                        x1 = timeline_left + round(item["end"] / len(goals) * timeline_width) - 4
                        active = j == group_idx
                        title.rounded_rectangle((x0, 934, x1, 986), radius=5,
                                                fill="#32516a" if active else CARD)
                        short = item["activity"].replace(" & ", "+")
                        fit_text(title, short, (x0 + 9, 940), x1 - x0 - 16,
                                 size=17, fill=TEXT if active else MUTED, bold=active)
                        fit_text(title, f"Take {item['take']} / P{item['performer']}", (x0 + 9, 963),
                                 x1 - x0 - 16, size=14, fill=MUTED)
                    if references is not None:
                        x = margin
                        image.paste(Image.fromarray(references[goal_idx]), (x, image_top))
                    cache[goal_idx] = image
                canvas = cache[goal_idx].copy()
                annotation = ImageDraw.Draw(canvas)
                is_new_motion = goal_idx == group["start"]
                if step <= min(25, steps_per_goal):
                    transition = "NEW MOTION" if is_new_motion else "SAME MOTION / NEXT POSE"
                    transition_color = "#84b9ef" if is_new_motion else "#5cd7c6"
                else:
                    transition, transition_color = "HOLDING TARGET POSE", MUTED
                annotation.text((margin, 169), transition, fill=transition_color, font=font(19, True))
                within_motion = goal_idx - group["start"] + 1
                annotation.text((width - 360, 169),
                                f"Pose {within_motion} of {group['end'] - group['start']} in this motion",
                                fill=MUTED, font=font(18))
                progress_label = f"Step {step:03d} / {steps_per_goal}    |    {step * protocol['control_dt']:.2f}s / {duration:.2f}s"
                label_width = annotation.textbbox((0, 0), progress_label, font=font(22, True))[2]
                annotation.text((width - margin - label_width, 771), progress_label, fill=TEXT, font=font(22, True))
                annotation.rounded_rectangle((margin, 815, width - margin, 837), radius=9, fill=CARD)
                end = margin + round((width - 2 * margin) * step / steps_per_goal)
                annotation.rounded_rectangle((margin, 815, max(margin + 10, end), 837), radius=9, fill="#84b9ef")
                annotation.text((margin, 852), "0s", fill=MUTED, font=font(15))
                annotation.text((width - margin - 48, 852), f"{duration:g}s", fill=MUTED, font=font(15))
                # A moving marker shows the current pose within the grouped motion timeline.
                center = timeline_left + (goal_idx + 0.5) / len(goals) * timeline_width
                annotation.polygon([(center, 924), (center - 7, 915), (center + 7, 915)], fill="#84b9ef")
                annotation.text((width - 330, 888), "P = performer in the dataset", fill=MUTED, font=font(16))
                array = np.array(canvas)
                offset = 1 if references is not None else 0
                for i, capture in enumerate(captures):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Cannot read rollout {i} frame {frame_idx}")
                    x = margin + (i + offset) * (size + gap)
                    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                    array[image_top:image_bottom, x:x + size] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.add_image(array)
                if frame_idx in preview_frames:
                    Image.fromarray(array).save(args.output.with_name(f"preview-{frame_idx:04d}.png"))
    finally:
        for capture in captures:
            capture.release()
    args.output.with_suffix(".json").write_text(json.dumps({
        "rollouts": [str(p.resolve()) for p in args.rollout],
        "labels": args.label,
        "reference_dir": str(args.reference_dir.resolve()) if args.reference_dir else None,
        "metadata": metadata,
        "layout": {"width": width, "height": height},
    }, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
