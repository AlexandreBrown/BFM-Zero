"""Infer reward prompts from matched native replay samples and record reward rollouts."""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import h5py
import mujoco
import numpy as np
import torch

from experiments.extract_minimal_data_fields_for_reward_inference import TASKS
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.utils import set_seed_everywhere
from humanoidverse.envs.g1_env_helper.robot import make_from_name
import humanoidverse.envs.g1_env_helper.bench.reward_eval_hv as reward_module
from humanoidverse.envs.g1_env_helper.rewards import get_xpos, get_sensor_data, get_xmat, rot2eul


WORKER_MODEL = None


def worker_init(xml):
    global WORKER_MODEL
    torch.set_num_threads(1)
    WORKER_MODEL = mujoco.MjModel.from_xml_path(xml)


def score_chunk(task, arrays):
    return reward_module._relabel_worker(arrays, model=WORKER_MODEL, reward_fn=make_from_name(task))


class TaskSamples:
    def __init__(self, path):
        self.path = path
        self.task = None

    def size(self):
        return 150000

    def get_full_buffer(self):
        with h5py.File(self.path) as f:
            g = f[self.task]
            return {'action':torch.from_numpy(g['action'][:]),
                    'next':{'qpos':torch.from_numpy(g['qpos'][:]),
                            'qvel':torch.from_numpy(g['qvel'][:]),
                            'observation':{k:torch.from_numpy(g[k][:]) for k in ['state','privileged_state']}}}


def qpos(env):
    root = env.simulator.robot_root_states[:, [0,1,2,6,3,4,5]].detach().cpu().numpy()
    joints = env.simulator.dof_pos.detach().cpu().numpy()
    return np.concatenate([root,joints],axis=1)[0].copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-folder',type=Path,required=True)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=400)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    xml=str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml')
    model=load_model_from_checkpoint_dir(args.model_folder/'checkpoint',device='cpu').to('cuda')
    model.eval()
    config=json.loads((args.model_folder/'config.json').read_text())
    env_config=config['env']
    env_config['lafan_tail_path']=str(root/'humanoidverse/data/lafan_29dof.pkl')
    env_config['hydra_overrides'] += ['simulator=mujoco','env.config.headless=True','env.config.max_episode_length_s=10000']
    set_seed_everywhere(4728)
    wrapped,_=HumanoidVerseIsaacConfig(**env_config).build(1)
    sim=wrapped._env.simulator
    # The native reward XML includes the IMU and COM sensors absent from the rollout XML.
    measurement_model=mujoco.MjModel.from_xml_path(xml)
    measure=mujoco.MjData(measurement_model)
    torso_id=mujoco.mj_name2id(measurement_model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
    assert torso_id>=0 and sim.model.nv==measurement_model.nv==35
    dataset=TaskSamples(args.samples)
    metadata={'training_steps':json.loads((args.model_folder/'checkpoint/train_status.json').read_text())['time'],
              'model_folder':str(args.model_folder.resolve()),'samples':str(args.samples.resolve()),
              'samples_per_task':150000,'n_inferences':1,'inference_function':'reward_wr_inference',
              'control_dt':float(wrapped._env.dt),'steps_per_task':args.steps,'seed':4728,
              'tasks':TASKS,'env_config':env_config,'best_of_n':False,'initial_states':[]}
    import multiprocessing
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init,initargs=(xml,)) as pool:
        def persistent_relabel(mj_model, positions, velocities, actions, reward_fn, **kwargs):
            n=len(positions);chunk=(n+7)//8
            futures=[pool.submit(score_chunk,dataset.task,(positions[i:i+chunk],velocities[i:i+chunk],actions[i:i+chunk]))
                     for i in range(0,n,chunk)]
            rewards=np.concatenate([f.result() for f in futures])
            np.save(args.output/f'replay-rewards-{dataset.task}.npy',rewards)
            return rewards
        reward_module.relabel=persistent_relabel
        wrapper=reward_module.RewardWrapperHV(model=model,inference_dataset=dataset,
            num_samples_per_inference=150000,inference_function='reward_wr_inference',
            max_workers=8,process_executor=True,env_model=xml)
        for task in TASKS:
            print(f'INFER {task}',flush=True);started=time.time()
            dataset.task=task
            z_path=args.output/f'z-{task}.npy'
            if z_path.exists():
                z=torch.from_numpy(np.load(z_path)).to('cuda')
            else:
                with torch.inference_mode(): z=wrapper.reward_inference(task)
                np.save(z_path,z.cpu().numpy())
            print(f'INFERRED {task} in {time.time()-started:.1f}s',flush=True)
            states=[];velocities=[]
            measured={k:[] for k in ['com_velocity','angular_velocity','pelvis_height','wrist_heights','knee_heights','heading_degrees','pelvis_up']}
            with torch.inference_mode():
                obs,info=wrapped.reset(to_numpy=False,reset_to_default_pose=True,seed=4728)
                metadata['initial_states'].append(qpos(wrapped._env).tolist())
                for step in range(args.steps):
                    action=model.act(obs,z,mean=True)
                    obs,_,_,_,_=wrapped.step(action,to_numpy=False)
                    states.append(qpos(wrapped._env))
                    velocities.append(sim.data.qvel.copy())
                    # A separate MjData avoids changing the live simulator while measuring.
                    measure.qpos[:]=sim.data.qpos
                    measure.qvel[:]=sim.data.qvel
                    mujoco.mj_forward(measurement_model,measure)
                    mujoco.mj_subtreeVel(measurement_model,measure)
                    velocity=measure.subtree_linvel[torso_id].copy()
                    measured['com_velocity'].append(velocity)
                    measured['angular_velocity'].append(get_sensor_data(measurement_model,measure,'imu-angular-velocity').copy())
                    measured['pelvis_height'].append(get_xpos(measurement_model,measure,'pelvis')[2])
                    measured['wrist_heights'].append([get_xpos(measurement_model,measure,n)[2] for n in ['left_wrist_roll_link','right_wrist_roll_link']])
                    measured['knee_heights'].append([get_xpos(measurement_model,measure,n)[2] for n in ['left_knee_link','right_knee_link']])
                    matrix=get_xmat(measurement_model,measure,'pelvis')
                    heading=np.degrees(np.arctan2(velocity[1],velocity[0])-rot2eul(matrix)[-1])
                    measured['heading_degrees'].append((heading+180)%360-180)
                    measured['pelvis_up'].append(matrix[2,2])
            np.save(args.output/f'qpos-{task}.npy',np.stack(states))
            np.save(args.output/f'qvel-{task}.npy',np.stack(velocities))
            np.savez(args.output/f'measured-{task}.npz',**{k:np.asarray(v) for k,v in measured.items()})
            print(f'RECORDED {task}: {len(states)} frames',flush=True)
    metadata['measurements']='Live rollout qpos/qvel evaluated on separate MjData using the native reward XML and sensors; COM is torso subtree as in reward code'
    (args.output/'rollout.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
