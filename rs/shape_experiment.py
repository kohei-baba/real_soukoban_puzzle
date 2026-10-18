"""Isolated triangle/star runs; the original square outputs remain untouched."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from stable_baselines3 import SAC
from .planner import make_board,DIRECTIONS,add
from .sim import Sim,STATE_COLUMNS
from .env import PushEnv,evaluate_local,policy_action
from .train import run as train_run
from .select import select
from .experiment import roll_puzzle


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def setup(code_root,shape):
    root=code_root/'experiments'/shape
    root.mkdir(parents=True,exist_ok=True)
    source=json.loads((code_root/'config.json').read_text())
    source.update(shape=shape,shape_radius=.34 if shape=='triangle' else .35,
                  shape_angle_tolerance=.18,shape_angle_reward=2.)
    if shape=='star': source['shape_arm_width']=.13
    path=root/'config.json'
    if path.exists():
        cfg=json.loads(path.read_text())
        if cfg!=source:
            if (root/'outputs'/'models'/'last.zip').exists():
                raise ValueError('Existing trained shape config differs from current template')
            path.write_text(json.dumps(source,indent=2)+'\n')
            cfg=source
    else:
        path.write_text(json.dumps(source,indent=2)+'\n')
    return root,source


def save_rollout(root,cfg,method,seed,model=None,video=False):
    out=root/'outputs'; out.mkdir(exist_ok=True)
    tag=f'local_{method}_{seed}'
    records=[]
    board=make_board(cfg,'train')
    if video:
        # Match PushEnv.reset's first RNG draw, so the visible goal is the real
        # per-episode goal. This static marker has no collision geometry.
        choices=[]
        for r in range(1,board.rows-1):
            for c in range(1,board.cols-1):
                p=(r,c)
                for dr in DIRECTIONS:
                    if all(board.free(q) for q in (p,add(p,dr),add(p,(-dr[0],-dr[1])))):
                        choices.append((p,dr))
        p,dr=choices[int(np.random.default_rng(seed).integers(len(choices)))]
        board.goal=add(p,dr)
    with Sim(root,board,cfg,camera=video,tag=tag) as sim:
        env=PushEnv(sim,cfg)
        path=out/'videos'/f'{tag}.mp4'
        if video:
            path.parent.mkdir(exist_ok=True)
            sim.begin_video(path)
        # The camera warm-up advances physics. Reset *after* warm-up so the
        # policy begins at the same seeded state as the headless batch test.
        obs,_=env.reset(seed=seed)
        if video: assert np.allclose(env.goal,board.xy(board.goal))
        initial={'robot':sim.s.robot.tolist(),'box':sim.s.box.tolist(),'yaw':sim.s.yaw,
                 'goal':env.goal.tolist(),'direction':env.direction.tolist()}
        total=0.
        for i in range(cfg['episode_steps']):
            action=policy_action(model,obs,env,baseline=method=='baseline')
            obs,reward,done,trunc,info=env.step(action)
            total+=reward
            records.append(dict(zip(STATE_COLUMNS,sim.s.row()),action_forward=float(action[0]),
                                action_side=float(action[1]),reward=reward))
            if done or trunc: break
        if video: sim.end_video(path)
    result=dict(shape=cfg['shape'],method=method,seed=seed,steps=i+1,reward=total,
                **info,initial=initial,video=str(path) if video else '')
    write_csv(out/'traces'/f'{tag}.csv',records)
    (out/f'{tag}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('shape',choices=['triangle','star'])
    p.add_argument('task',choices=['init','smoke','baseline','train','select','evaluate','video','puzzles','complex'])
    p.add_argument('--steps',type=int,default=150000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,default=20000)
    args=p.parse_args()
    code_root=Path(__file__).resolve().parent.parent
    root,cfg=setup(code_root,args.shape)
    out=root/'outputs'; out.mkdir(exist_ok=True)
    if args.task=='init':
        print(root); return
    if args.task=='smoke':
        result=save_rollout(root,cfg,'baseline',args.seed,video=True)
        (out/'smoke_ok.json').write_text(json.dumps({'cfg':cfg,'result':result},indent=2))
        return
    if args.task=='baseline':
        with Sim(root,make_board(cfg,'train'),cfg,tag='baseline_batch') as sim:
            rows=evaluate_local(PushEnv(sim,cfg),None,range(20000,20030),baseline=True)
        write_csv(out/'baseline.csv',rows)
        print(f"baseline {sum(r['is_success'] for r in rows)}/{len(rows)}",flush=True)
        return
    if args.task=='train':
        marker=out/'smoke_ok.json'
        if not marker.exists() or json.loads(marker.read_text())['cfg']!=cfg:
            raise RuntimeError('Run shape smoke with this config first')
        train_run(root,cfg,args.steps,args.resume)
        return
    if args.task=='select':
        select(root); return
    model=SAC.load(out/'models'/'best.zip',device='cpu')
    if args.task=='evaluate':
        with Sim(root,make_board(cfg,'train'),cfg,tag='sac_batch') as sim:
            rows=evaluate_local(PushEnv(sim,cfg),model,range(20000,20030))
        write_csv(out/'sac.csv',rows)
        print(f"SAC {sum(r['is_success'] for r in rows)}/{len(rows)}",flush=True)
    elif args.task=='video':
        save_rollout(root,cfg,'baseline',args.seed,video=True)
        save_rollout(root,cfg,'SAC',args.seed,model,video=True)
    elif args.task=='puzzles':
        rows=[]
        for case in ('straight','turn','maze'):
            for method in ('baseline','SAC'):
                rows.append(roll_puzzle(root,cfg,case,method,
                                        None if method=='baseline' else out/'models'/'best.zip',
                                        seed=args.seed,video=True))
        write_csv(out/'puzzles.csv',[{k:v for k,v in row.items() if k!='initial'} for row in rows])
    elif args.task=='complex':
        roll_puzzle(root,cfg,'maze_complex','SAC',out/'models'/'best.zip',
                    seed=args.seed,video=True)


if __name__=='__main__': main()
