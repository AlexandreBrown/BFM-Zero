"""Render recorded native rollouts and compose the requested h=1 and horizon-grid videos."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw

from experiments.compose_goal_comparison import font, fit_text, describe_goal


HORIZONS = [1, 2, 4, 6, 8, 10, 12]
BG, FG, MUTED = "#101820", "#f0f4f8", "#a8bacb"


def render_states(source, target, renderer, fps):
    qpos = np.load(source)
    env = renderer.mujoco_env
    with media.VideoWriter(str(target), shape=(384, 384), fps=fps, codec="h264", crf=18) as out:
        for q in qpos:
            env.reset(options={"qpos": q})
            out.add_image(env.render())
    print(f"Rendered {target}", flush=True)


def validate_metadata(root):
    metadata = {family: [json.loads((root/family/f"h{h}.json").read_text()) for h in HORIZONS]
                for family in ["author", "reproduced"]}
    first = metadata["author"][0]
    reference = np.load(root/"author/reference.npy")
    assert np.array_equal(reference, np.load(root/"reproduced/reference.npy"))
    for family, records in metadata.items():
        for h, item in zip(HORIZONS, records):
            for key in ["seed", "motion_id", "motion_name", "frames", "control_dt",
                        "native_script_sha256", "disable_dr", "disable_obs_noise"]:
                assert item[key] == first[key], (family, h, key)
            assert item["horizon"] == h
    return metadata


def read_frame(cap, frame_idx):
    ok, img = cap.read()
    if not ok:
        raise ValueError(f"Video ended before frame {frame_idx}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compose(root, grid=False):
    if grid:
        metadata = validate_metadata(root)
    else:
        metadata = {f: [json.loads((root/f/"h1.json").read_text())]
                    for f in ["author", "reproduced"]}
        assert metadata['author'][0]['frames'] == metadata['reproduced'][0]['frames']
        assert np.array_equal(np.load(root/'author/reference.npy'), np.load(root/'reproduced/reference.npy'))
    first = metadata["author"][0]
    frames, fps = first["frames"], 1/first["control_dt"]
    output = root/("tracking-horizon-grid.mp4" if grid else "tracking-h1-comparison.mp4")
    ref = cv2.VideoCapture(str(root/"rendered/reference.mp4"))
    keys = [(f,h) for f in ["author", "reproduced"] for h in HORIZONS] if grid else [("reproduced",1),("author",1)]
    caps = {key: cv2.VideoCapture(str(root/f"rendered/{key[0]}-h{key[1]}.mp4")) for key in keys}
    for cap in [ref, *caps.values()]:
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
        assert np.isclose(cap.get(cv2.CAP_PROP_FPS), fps)
    width, height = (3840, 1440) if grid else (1448, 1080)
    template = Image.new("RGB", (width,height), BG)
    d = ImageDraw.Draw(template)
    if grid:
        d.text((24,20), "BFM-ZERO  /  MOTION TRACKING", font=font(38,True), fill=FG)
        if not first.get('segments'):
            d.text((24,76), "Walking  /  Take 1  /  Performer 1   |   Averaging window used to compute each task prompt z", font=font(26), fill=MUTED)
        d.text((24,174), "Reference motion", font=font(29,True), fill="#84b9ef")
        for i,h in enumerate(HORIZONS):
            x=608+i*456
            d.rounded_rectangle((x,126,x+448,169), radius=5, fill="#1b2936")
            label=f"h = {h} frame"+('s' if h>1 else '')
            d.text((x+18,132),label,font=font(26,True),fill=FG)
        for row,family in enumerate(["author","reproduced"]):
            y=184+row*550
            steps=metadata[family][0]['training_steps']/1e6
            name="Author checkpoint" if family=='author' else "Reproduced"
            d.text((608,y),f"{name}  /  {steps:g}M training steps",font=font(27,True),fill="#5cd7c6" if row==0 else '#ad9cff')
        d.rounded_rectangle((24,230,584,1228),radius=8,fill='#1b2936')
        d.text((48,260),'One shared reference',font=font(27,True),fill='#84b9ef')
        d.text((48,302),'Pose and motion velocities',font=font(23),fill=MUTED)
        d.text((24,1390),'Native tracking settings  |  same seed 4728  |  z updated every control step  |  MuJoCo / 1x speed',font=font(25),fill=MUTED)
        progress=(608,1324,3792,1346)
    else:
        d.text((24,20),'BFM-ZERO  /  MOTION TRACKING',font=font(21,True),fill=MUTED)
        if not first.get('segments'):
            d.text((24,66),'Walking',font=font(44,True),fill=FG)
            d.text((24,126),'Take 1  /  Performer 1  /  full reference motion',font=font(23),fill=MUTED)
        labels=[('Reference motion','Moving target / pose + velocity','#84b9ef'),
                ('Reproduced',f"{metadata['reproduced'][0]['training_steps']/1e6:g}M training steps",'#ad9cff'),
                ('Author checkpoint',f"{metadata['author'][0]['training_steps']/1e6:g}M training steps",'#5cd7c6')]
        for i,(name,sub,color) in enumerate(labels):
            x=24+i*472
            d.rectangle((x,212,x+455,294),fill='#1b2936')
            d.rectangle((x,212,x+455,215),fill=color)
            d.text((x+16,225),name,font=font(28,True),fill=FG)
            fit_text(d,sub,(x+16,260),424,size=20,fill=color)
        d.text((24,890),'h = 1  /  a new reference-state prompt every control step',font=font(23,True),fill=FG)
        d.text((24,1020),'Original tracking initialization and default noise/randomization settings',font=font(20),fill=MUTED)
        d.text((24,1050),'MuJoCo  |  1x speed  |  same seed 4728',font=font(18),fill=MUTED)
        progress=(24,815,1424,837)
    try:
        with media.VideoWriter(str(output),shape=(height,width),fps=fps,codec='h264',crf=18) as writer:
            for idx in range(frames):
                image=template.copy();d=ImageDraw.Draw(image)
                elapsed=idx/fps
                label=f"Frame {idx:05d} / {frames-1}   |   {elapsed:.2f}s / {(frames-1)/fps:.2f}s"
                progress_idx, progress_frames = idx, frames-1
                if first.get('segments'):
                    segment = next(s for s in first['segments'] if s['frame_start'] <= idx < s['frame_start']+s['frames'])
                    info = describe_goal(segment['motion_name']+'_0')
                    local = idx-segment['frame_start']+1
                    progress_idx, progress_frames = local, segment['frames']
                    number=first['segments'].index(segment)+1
                    sub=f"Take {info['take']} / Performer {info['performer']} / Motion {number} of {len(first['segments'])}"
                    if grid:
                        d.text((24,76),f"{info['activity']}   |   {sub}",font=font(28),fill=MUTED)
                    else:
                        d.text((24,66),info['activity'],font=font(44,True),fill=FG)
                        d.text((24,126),sub,font=font(23),fill=MUTED)
                    label=f"Step {local:03d} / {segment['frames']}  |  {local/fps:.2f}s / {segment['frames']/fps:.2f}s  |  Reference frame {segment['start_frame']+local}"
                d.text((608,1273) if grid else (24,771),label,font=font(27 if grid else 22,True),fill=FG)
                x0,y0,x1,y1=progress
                d.rounded_rectangle(progress,radius=9,fill='#1b2936')
                d.rounded_rectangle((x0,y0,max(x0+10,x0+round((x1-x0)*progress_idx/progress_frames)),y1),radius=9,fill='#84b9ef')
                array=np.array(image)
                reference=read_frame(ref,idx)
                if grid:
                    array[444:1004,24:584]=cv2.resize(reference,(560,560))
                    for row,family in enumerate(['author','reproduced']):
                        for col,h in enumerate(HORIZONS):
                            x,y=608+col*456,230+row*550
                            array[y:y+448,x:x+448]=cv2.resize(read_frame(caps[(family,h)],idx),(448,448))
                else:
                    for col,img in enumerate([reference,read_frame(caps[('reproduced',1)],idx),read_frame(caps[('author',1)],idx)]):
                        x=24+col*472
                        array[294:750,x:x+456]=cv2.resize(img,(456,456))
                writer.add_image(array)
                if idx in [0,frames//2,frames-1]:Image.fromarray(array).save(output.with_name(output.stem+f'-preview-{idx:05d}.png'))
                if idx%2000==0:print(f'Compose {output.name}: {idx}/{frames}',flush=True)
    finally:
        ref.release()
        for cap in caps.values():cap.release()
    output.with_suffix('.json').write_text(json.dumps({'metadata':metadata,'frames':frames,'fps':fps,'grid':grid},indent=2)+'\n')
    print(f'COMPLETE {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    (root/'rendered').mkdir(parents=True,exist_ok=True)
    from humanoidverse.agents.envs.humanoidverse_isaac import IsaacRendererWithMuJoco
    renderer=IsaacRendererWithMuJoco(render_size=384)
    jobs=[('reference',root/'reproduced/reference.npy',root/'reproduced/h1.json')]
    jobs += [(f'{f}-h{h}',root/f/f'qpos-h{h}.npy',root/f/f'h{h}.json')
             for h in HORIZONS for f in ['author','reproduced']]
    completed=set();one_frame_done=False
    while len(completed)<len(jobs):
        progress=False
        for name,source,meta in jobs:
            if name in completed or not meta.exists():continue
            target=root/f'rendered/{name}.mp4'
            if not target.exists():
                fps=1/json.loads(meta.read_text())['control_dt']
                render_states(source,target,renderer,fps)
            completed.add(name);progress=True
        if not one_frame_done and {'reference','author-h1','reproduced-h1'} <= completed:
            compose(root,grid=False);one_frame_done=True
        if not progress:time.sleep(5)
    compose(root,grid=True)


if __name__=='__main__':main()
