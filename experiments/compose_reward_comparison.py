"""Compose readable reward targets beside reproduced and author reward rollouts."""

import argparse
import io
import json
import os
from pathlib import Path

os.environ['MUJOCO_GL']='egl'
os.environ['MPLCONFIGDIR']='/tmp/bfm-reward-mpl'

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset']='stix'
matplotlib.rcParams['savefig.transparent']=True
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.compose_goal_comparison import font,fit_text
from experiments.extract_minimal_data_fields_for_reward_inference import TASKS
from humanoidverse.agents.envs.humanoidverse_isaac import IsaacRendererWithMuJoco
from humanoidverse.envs.g1_env_helper.robot import make_from_name
from humanoidverse.envs.g1_env_helper.rewards import (
    LocomotionReward, RotationReward, ArmsReward, MoveArmsReward, SitOnGroundReward,
)


TASK_LABELS = {
    'move-ego-0-0': ('Stand still', 'Stand'),
    'move-ego-0-0.7': ('Move forward  /  0.7 m/s', 'Forward'),
    'move-ego-90-0.3': ('Move left  /  0.3 m/s', 'Left'),
    'move-ego-180-0.3': ('Move backward  /  0.3 m/s', 'Backward'),
    'move-ego--90-0.3': ('Move right  /  0.3 m/s', 'Right'),
    'rotate-z-5-0.5': ('Turn counterclockwise', 'Turn +'),
    'rotate-z--5-0.5': ('Turn clockwise', 'Turn -'),
    'raisearms-l-l': ('Low arm pose', 'Arms low'),
    'raisearms-m-m': ('Raise both arms', 'Arms high'),
    'move-arms-0-0.7-m-m': ('Walk forward with raised arms', 'Move + arms'),
    'crouch-0.25': ('Crouch low', 'Crouch'),
    'sitonground': ('Sit on the ground', 'Sit'),
}


def task_card(task):
    return TASK_LABELS[task]


def paste_math(canvas,formula,x,y,max_width=416,size=23,color='#f0f4f8'):
    buf=io.BytesIO()
    math_to_image('$'+formula+'$',buf,prop=FontProperties(size=size),dpi=140,format='png',color=color)
    buf.seek(0);img=Image.open(buf).convert('RGBA')
    if img.width>max_width:
        img=img.resize((max_width,round(img.height*max_width/img.width)),Image.Resampling.LANCZOS)
    canvas.paste(img,(x+(max_width-img.width)//2,y),img)


def readable_reward(task):
    height=r'\mathrm{pelvis\ height}\geq0.50\,\mathrm{m}'
    upright=r'\mathrm{torso}\approx\mathrm{upright}'
    still=r'\mathrm{speed}\approx0'
    no_spin=r'\mathrm{turn\ rate}\approx0'
    if task=='move-ego-0-0':
        return ['Stand upright','and stay still'],[height,upright,still,no_spin]
    if task.startswith('move-ego-'):
        angle,speed=map(float,task[len('move-ego-'):].rsplit('-',1))
        direction={0:'forward',90:'left',180:'backward',-90:'right'}[int(angle)]
        return [f'Move {direction}',f'at {speed:g} m/s'],[
            rf'\mathrm{{speed}}\approx{speed:g}\,\mathrm{{m/s}}',
            rf'\mathrm{{direction}}=\mathrm{{{direction}}}',height,upright]
    if task.startswith('rotate-z-'):
        positive=task=='rotate-z-5-0.5'
        bounds='[5,10]' if positive else '[-10,-5]'
        return ['Stay upright and', 'turn '+('counterclockwise' if positive else 'clockwise')],[
            rf'\mathrm{{yaw\ rate}}\in{bounds}\,\mathrm{{rad/s}}',height,
            r'\mathrm{pelvis\ up}_z\geq0.9']
    if task.startswith('raisearms-'):
        high=task=='raisearms-m-m'
        rule=r'\geq1.0' if high else r'\in[0.6,0.8]'
        return (['Raise both hands','and stay still'] if high else ['Hold both hands low','and stay still']),[
            rf'\mathrm{{hand\ heights}}{rule}\,\mathrm{{m}}',height,upright,
            r'\mathrm{speed}\approx0,\quad\mathrm{turn\ rate}\approx0']
    if task.startswith('move-arms-'):
        return ['Walk forward','with hands raised'],[
            r'\mathrm{speed}\approx0.7\,\mathrm{m/s}',r'\mathrm{direction}=\mathrm{forward}',
            r'\mathrm{hand\ heights}\geq1.0\,\mathrm{m}',height]
    crouch=task=='crouch-0.25'
    return (['Crouch low','and stay still'] if crouch else ['Sit on the ground','and stay still']),[
        r'\mathrm{pelvis\ height}\in[0.25,0.35]\,\mathrm{m}' if crouch else r'\mathrm{pelvis\ height}\in[0,0.1]\,\mathrm{m}',
        r'\mathrm{knee\ heights}\in[0.2,1.0]\,\mathrm{m}' if crouch else r'\mathrm{knee\ heights}\in[0,0.1]\,\mathrm{m}',
        upright,r'\mathrm{speed}\approx0,\quad\mathrm{turn\ rate}\approx0']


def draw_reward_panel(card,task):
    d=ImageDraw.Draw(card)
    d.rectangle((24,294,479,749),fill='#1b2936')
    description,targets=readable_reward(task)
    d.text((44,308),'DESIRED BEHAVIOR',font=font(15,True),fill='#a8bacb')
    for i,line in enumerate(description):
        fit_text(d,line,(44,337+i*32),416,size=25,bold=True,fill='#84b9ef')
    d.line((44,412,459,412),fill='#354757',width=1)
    d.text((44,424),'REWARD FUNCTION',font=font(15,True),fill='#a8bacb')
    for i,target in enumerate(targets):paste_math(card,target,44,456+i*50,size=20)


def measurement_rows(reward, values, step):
    """Return fixed-column metric, measured value, and target cells."""
    speed=np.linalg.norm(values['com_velocity'][step,:2])
    pelvis=values['pelvis_height'][step]
    if isinstance(reward,(LocomotionReward,MoveArmsReward)):
        lines=[('Speed (m/s)',f'{speed:.2f}',f'{reward.move_speed:.2f}')]
        if reward.move_speed>0:
            heading=values['heading_degrees'][step]
            lines.append(('Heading (\N{DEGREE SIGN})',f'{heading:+.0f}',f'{reward.move_angle:g}'))
        else:
            spin=np.linalg.norm(values['angular_velocity'][step])
            lines.append(('Spin (rad/s)',f'{spin:.2f}','0.00'))
        if isinstance(reward,MoveArmsReward):
            left,right=values['wrist_heights'][step]
            lines.append(('Hands L/R (m)',f'{left:.2f} / {right:.2f}','\u22651.00'))
        else:
            lines.append(('Pelvis (m)',f'{pelvis:.2f}','\u22650.50'))
        return lines
    if isinstance(reward,RotationReward):
        yaw=values['angular_velocity'][step,2]
        interval='5 to 10' if reward.target_ang_velocity>0 else '-10 to -5'
        return [('Yaw (rad/s)',f'{yaw:+.2f}',interval),
                ('Pelvis (m)',f'{pelvis:.2f}','\u22650.50'),
                ('Pelvis up',f'{values["pelvis_up"][step]:.2f}','\u22650.90')]
    if isinstance(reward,ArmsReward):
        left,right=values['wrist_heights'][step]
        target='\u22651.00' if reward.left_pose=='m' else '0.6 to 0.8'
        return [('Left hand (m)',f'{left:.2f}',target),
                ('Right hand (m)',f'{right:.2f}',target),
                ('Speed (m/s)',f'{speed:.2f}','0.00')]
    if isinstance(reward,SitOnGroundReward):
        left,right=values['knee_heights'][step]
        low=reward.pelvis_height_th
        return [('Pelvis (m)',f'{pelvis:.2f}',f'{low:g} to {low+.1:g}'),
                ('Knees L/R (m)',f'{left:.2f} / {right:.2f}',
                 '0.2 to 1.0' if reward.knees_not_on_ground else '0 to 0.1'),
                ('Speed (m/s)',f'{speed:.2f}','0.00')]
    raise ValueError(type(reward))


MONO=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',17)


def draw_measurements(draw,x,rows,accent):
    """Keep columns, separators, fonts, and numeric anchors stationary."""
    draw.rounded_rectangle((x,756,x+455,878),radius=7,fill='#192632')
    for offset,label in [(14,'METRIC'),(166,'MEASURED'),(326,'TARGET')]:
        draw.text((x+offset,765),label,font=font(12,True),fill='#98aabc')
    draw.line((x+14,785,x+441,785),fill='#304151')
    draw.line((x+312,765,x+312,866),fill='#304151')
    for row,(metric,value,target) in enumerate(rows):
        y=793+26*row
        draw.text((x+14,y),metric,font=font(16),fill='#b7c5d3')
        draw.text((x+166,y),value,font=MONO,fill=accent)
        draw.text((x+326,y),target,font=font(15),fill='#e5ebf1')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    metadata={f:json.loads((root/f/'rollout.json').read_text()) for f in ['reproduced','author']}
    a,b=metadata.values()
    for key in ['tasks','control_dt','steps_per_task','samples_per_task','n_inferences','seed','initial_states']:
        assert a[key]==b[key],key
    steps=a['steps_per_task'];fps=1/a['control_dt'];total=steps*len(TASKS)
    width,height=1448,1210
    bg,fg,muted='#101820','#f0f4f8','#a8bacb'
    template=Image.new('RGB',(width,height),bg);d=ImageDraw.Draw(template)
    d.text((24,20),'BFM-ZERO  /  ZERO-SHOT REWARD INFERENCE',font=font(21,True),fill=muted)
    specs=[('Reward Inference','','#84b9ef'),
           ('Reproduced',f"{a['training_steps']/1e6:g}M training steps",'#ad9cff'),
           ('Author checkpoint',f"{b['training_steps']/1e6:g}M training steps",'#5cd7c6')]
    for i,(title,sub,color) in enumerate(specs):
        x=24+i*472
        d.rectangle((x,212,x+455,294),fill='#1b2936');d.rectangle((x,212,x+455,215),fill=color)
        d.text((x+16,225),title,font=font(28,True),fill=fg)
        d.text((x+16,260),sub,font=font(20),fill=color)
    d.text((24,908),'TIME ON CURRENT REWARD TASK',font=font(18,True),fill=muted)
    d.text((24,1018),'REWARD TASKS',font=font(18,True),fill=muted)
    d.text((24,1150),'One inferred z per task  |  held fixed throughout each rollout  |  no policy fine-tuning',font=font(19),fill=muted)
    d.text((24,1180),'MuJoCo / 1x speed  |  same standing start  |  150,000 replay samples per task',font=font(18),fill=muted)
    renderer=IsaacRendererWithMuJoco(render_size=456)
    env=renderer.mujoco_env
    out=root/'reward-inference-comparison.mp4'
    preview_indices={0,steps//2,steps+steps//2,6*steps+steps//2,10*steps+steps//2,total-1}
    written=0
    with media.VideoWriter(str(out),shape=(height,width),fps=fps,codec='h264',crf=18) as writer:
        for task_idx,task in enumerate(TASKS):
            title,short=task_card(task)
            card=template.copy();d=ImageDraw.Draw(card)
            fit_text(d,'Target: '+title,(24,66),1050,size=40,bold=True)
            d.text((width-280,76),f'TASK {task_idx+1:02d} / {len(TASKS)}',font=font(25,True),fill=fg)
            d.text((24,132),'Left: reward targets. Policy panels: behavior and measured outcomes.',font=font(22),fill=muted)
            draw_reward_panel(card,task)
            d=ImageDraw.Draw(card)
            for j,t in enumerate(TASKS):
                _,label=task_card(t);x0=24+round(j/len(TASKS)*1400);x1=24+round((j+1)/len(TASKS)*1400)-3
                d.rounded_rectangle((x0,1064,x1,1116),radius=4,fill='#32516a' if j==task_idx else '#1b2936')
                fit_text(d,label,(x0+6,1081),x1-x0-12,size=15,bold=j==task_idx,fill=fg if j==task_idx else muted)
            states={f:np.load(root/f/f'qpos-{task}.npy') for f in metadata}
            velocities={f:np.load(root/f/f'qvel-{task}.npy') for f in metadata}
            measurements={f:dict(np.load(root/f/f'measured-{task}.npz')) for f in metadata}
            reward=make_from_name(task)
            assert all(q.shape==(steps,36) and np.isfinite(q).all() for q in states.values())
            assert all(q.shape==(steps,35) and np.isfinite(q).all() for q in velocities.values())
            assert all(np.isfinite(v).all() and len(v)==steps for m in measurements.values() for v in m.values())
            for step in range(steps):
                image=card.copy();d=ImageDraw.Draw(image)
                d.text((990,905),'STEP',font=font(13,True),fill=muted)
                d.text((1046,903),f'{step+1:03d} / {steps}',font=MONO,fill=fg)
                d.line((1176,903,1176,924),fill='#354757')
                d.text((1194,905),'TIME',font=font(13,True),fill=muted)
                d.text((1246,903),f'{(step+1)/fps:04.2f} / {steps/fps:.2f}s',font=MONO,fill=fg)
                d.rounded_rectangle((24,945,1424,967),radius=9,fill='#1b2936')
                d.rounded_rectangle((24,945,max(34,24+round(1400*(step+1)/steps)),967),radius=9,fill='#84b9ef')
                d.text((24,982),'0s',font=font(15),fill=muted);d.text((1388,982),f'{steps/fps:g}s',font=font(15),fill=muted)
                for col,family in enumerate(['reproduced','author'],1):
                    x=24+col*472
                    draw_measurements(d,x,measurement_rows(reward,measurements[family],step),
                                      '#c0b3ff' if family=='reproduced' else '#79e0d1')
                array=np.array(image)
                for col,family in enumerate(['reproduced','author'],1):
                    env.reset(options={'qpos':states[family][step],'qvel':velocities[family][step]})
                    array[294:750,24+col*472:480+col*472]=env.render()
                writer.add_image(array)
                if written in preview_indices:Image.fromarray(array).save(root/f'preview-{written:04d}.png')
                written+=1
            print(f'Composed {task_idx+1}/{len(TASKS)}: {task}',flush=True)
    (root/'comparison.json').write_text(json.dumps({'rollouts':metadata,'frames':written,'fps':fps,
        'formula_note':'Panels show plain-language objectives and readable mathematical target summaries; no unexplained factor codes or substituted quadratic rewards.',
        'measurement_note':'Instantaneous physical measurements from the rerun, not target values. Horizontal speed and body-relative heading use torso subtree COM velocity, matching the reward code.'},indent=2)+'\n')
    # Release the EGL context before interpreter shutdown.
    if env.unwrapped.renderer is not None:
        env.unwrapped.renderer.close()
        env.unwrapped.renderer=None
    print(out.resolve())


if __name__=='__main__':main()
