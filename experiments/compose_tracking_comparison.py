"""Compose native tracking videos into reference/reproduced/author panels."""

import argparse
import json
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw

from experiments.compose_goal_comparison import font, fit_text, describe_goal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduced", type=Path, required=True)
    parser.add_argument("--author", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dirs = [args.reproduced, args.author]
    metadata = [json.loads((p / "rollout.json").read_text()) for p in dirs]
    a, b = metadata
    for key in ["motions", "prompt_window", "seed", "simulator", "disable_dr",
                "disable_obs_noise", "control_dt", "render_size", "data_path"]:
        if a[key] != b[key]:
            raise ValueError(f"Tracking protocols differ: {key}")
    for motion in a["motions"]:
        name = f"reference_qpos_{motion['motion_id']}.npy"
        if not np.array_equal(np.load(dirs[0] / name), np.load(dirs[1] / name)):
            raise ValueError(f"Reference trajectories differ: {name}")
    w, h, size, gap, margin = 1448, 1080, 456, 16, 24
    top, image_y = 212, 294
    bg, text, muted = "#101820", "#f0f4f8", "#a8bacb"
    colors = ["#84b9ef", "#ad9cff", "#5cd7c6"]
    dt = a["control_dt"]
    fps = 1 / dt
    total = sum(m["control_steps"] for m in a["motions"])
    template = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(template)
    draw.text((margin, 20), "BFM-ZERO   /   MOTION TRACKING", font=font(21, True), fill=muted)
    specs = [("Reference motion", "Moving target / pose + velocity"),
             ("Reproduced", f"{a['training_steps']/1e6:g}M training steps"),
             ("Author checkpoint", f"{b['training_steps']/1e6:g}M training steps")]
    for i, (title, sub) in enumerate(specs):
        x = margin + i * (size + gap)
        draw.rectangle((x, top, x + size - 1, image_y), fill="#1b2936")
        draw.rectangle((x, top, x + size - 1, top + 3), fill=colors[i])
        draw.text((x + 16, top + 13), title, font=font(28, True), fill=text)
        fit_text(draw, sub, (x + 16, top + 48), size - 32, size=20, fill=colors[i])
    draw.text((margin, 778), "TIME IN CURRENT MOTION SEGMENT", font=font(18, True), fill=muted)
    draw.text((margin, 888), "MOTION SEQUENCE", font=font(18, True), fill=muted)
    draw.text((margin, 1020), "MuJoCo  |  1x speed  |  same reference-state start for each segment", font=font(18), fill=muted)
    draw.text((margin, 1048), "One-frame FB prompts / z updated every step  |  observation noise and domain randomization off", font=font(17), fill=muted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with media.VideoWriter(str(args.output), shape=(h, w), fps=fps, codec="h264", crf=18) as writer:
        for motion_idx, motion in enumerate(a["motions"]):
            count = motion["control_steps"]
            details = describe_goal(motion["motion_name"] + "_0")
            base = template.copy()
            d = ImageDraw.Draw(base)
            d.text((margin, 61), details["activity"], font=font(44, True), fill=text)
            d.text((margin, 121), f"Take {details['take']}  /  Performer {details['performer']}  /  continuous tracking", font=font(23), fill=muted)
            d.text((w - 310, 75), f"MOTION {motion_idx+1:02d} / {len(a['motions']):02d}", font=font(25, True), fill=text)
            for j, item in enumerate(a["motions"]):
                info = describe_goal(item["motion_name"] + "_0")
                x0 = margin + round(j / len(a["motions"]) * (w-2*margin))
                x1 = margin + round((j+1) / len(a["motions"]) * (w-2*margin)) - 4
                d.rounded_rectangle((x0, 934, x1, 986), radius=5, fill="#32516a" if j == motion_idx else "#1b2936")
                fit_text(d, info["activity"].replace(" & ", "+"), (x0+8, 940), x1-x0-16, size=17, bold=j == motion_idx)
                d.text((x0+8, 963), f"Take {info['take']} / P{info['performer']}", font=font(14), fill=muted)
            captures = [cv2.VideoCapture(str(p / motion["video"])) for p in dirs]
            try:
                for cap in captures:
                    if (not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != count + 1
                            or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != 2*a["render_size"]
                            or not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps)):
                        raise ValueError("Unexpected native tracking video format")
                    if not cap.read()[0]:
                        raise ValueError("Missing initial reference frame")
                for step in range(1, count+1):
                    imgs = []
                    for cap in captures:
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError("Tracking video ended early")
                        imgs.append(frame)
                    # Both source videos contain the same rendered reference trajectory.
                    # Independently encoded H264 sources can differ slightly in pixel values.
                    n = a["render_size"]
                    panels = [imgs[0][:, :n], imgs[0][:, n:], imgs[1][:, n:]]
                    canvas = base.copy()
                    d = ImageDraw.Draw(canvas)
                    d.text((margin, 169), "NEW MOTION / REFERENCE-STATE RESET" if step <= 25 else "CONTINUOUS TRACKING / z UPDATED EVERY STEP", font=font(18, True), fill=colors[0] if step <= 25 else muted)
                    d.text((w-350, 169), f"Reference frame {motion['start_frame']+step}", font=font(18), fill=muted)
                    label = f"Step {step:03d} / {count}  |  {step*dt:.2f}s / {count*dt:.2f}s"
                    width = d.textbbox((0,0), label, font=font(22, True))[2]
                    d.text((w-margin-width, 771), label, font=font(22, True), fill=text)
                    d.rounded_rectangle((margin, 815, w-margin, 837), radius=9, fill="#1b2936")
                    d.rounded_rectangle((margin, 815, max(margin+10, margin+round((w-2*margin)*step/count)), 837), radius=9, fill=colors[0])
                    d.text((margin, 852), "0s", font=font(15), fill=muted)
                    d.text((w-margin-48, 852), f"{count*dt:g}s", font=font(15), fill=muted)
                    x = margin + (motion_idx+step/count)/len(a["motions"])*(w-2*margin)
                    d.polygon([(x,924),(x-7,915),(x+7,915)], fill=colors[0])
                    array = np.array(canvas)
                    for i, frame in enumerate(panels):
                        x0 = margin+i*(size+gap)
                        array[image_y:image_y+size, x0:x0+size] = cv2.cvtColor(cv2.resize(frame,(size,size)),cv2.COLOR_BGR2RGB)
                    writer.add_image(array)
                    if step in {1, count} or written == total//2:
                        Image.fromarray(array).save(args.output.with_name(f"preview-{written:04d}.png"))
                    written += 1
            finally:
                for cap in captures:
                    cap.release()
    assert written == total
    args.output.with_suffix(".json").write_text(json.dumps({"source_rollouts":metadata,"frames":written,"fps":fps}, indent=2)+'\n')
    print(args.output.resolve())


if __name__ == "__main__":
    main()
