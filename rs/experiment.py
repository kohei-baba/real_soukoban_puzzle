"""Unsnapped physical rollouts, paired evaluations, native Gazebo camera videos."""
from pathlib import Path
import csv,json,math,time
import numpy as np
from stable_baselines3 import SAC
from .planner import make_board,push_plan,robot_path
from .sim import Sim,STATE_COLUMNS
from .env import PushEnv,evaluate_local,frame,observation,baseline_action,reached,angle_error
from .shapes import support

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def roll_puzzle(root,cfg,case,method,model_path=None,seed=20000,video=True,tag=None):
    root=Path(root); out=root/'outputs'; tag=tag or f'{case}_{method}_{seed}'
    (out/'videos').mkdir(parents=True,exist_ok=True)
    board=make_board(cfg,case); plan=push_plan(board)
    model=None if method=='baseline' else SAC.load(model_path,device='cpu')
    rng=np.random.default_rng(seed)
    initial_offset=rng.uniform(-.03,.03,2); initial_yaw=math.radians(rng.uniform(-15,15))
    records=[]; events=[]; completed=0; success=False; reason='not_started'; recording=False
    path=out/'videos'/f'{tag}.mp4'
    with Sim(root,board,cfg,camera=video,tag=tag) as sim:
        def tick(v,phase,macro):
            s=sim.step(v)
            records.append(dict(zip(STATE_COLUMNS,s.row()),phase=phase,macro=macro,
                                command_x=float(v[0]),command_y=float(v[1])))
            if np.max(np.abs(s.tilt))>.6: raise ValueError('転倒')
            return s
        try:
            if video: sim.begin_video(path); recording=True
            sim.reset(board.xy(board.robot),board.xy(board.box)+initial_offset,initial_yaw)
            initial=dict(box=sim.s.box.tolist(),yaw=sim.s.yaw,robot=sim.s.robot.tolist())
            start_time=sim.s.t
            prev_direction=None
            for macro,m in enumerate(plan):
                goal=board.xy(m['target']); nominal=board.xy(m['box'])
                d=(goal-nominal)/board.cell
                if np.linalg.norm(sim.s.box-nominal)>cfg['success_tolerance']:
                    raise ValueError('箱の格子位置への対応が崩れました')
                if prev_direction is not None:
                    # Ordinary navigation withdraws the robot; the box is untouched.
                    for _ in range(8): tick(-prev_direction*.18,'navigation',macro)
                    for _ in range(3): tick(np.zeros(2),'navigation',macro)
                before_nav=sim.s.box.copy()
                route=robot_path(board,sim.s.robot,m['behind'],sim.s.box,sim.s.yaw,
                                 cfg['box_size'],cfg['robot_radius'],cfg)
                for waypoint in route:
                    for _ in range(120):
                        error=waypoint-sim.s.robot
                        if np.linalg.norm(error)<.035: break
                        v=error*1.5
                        v*=min(1.,.26/max(np.linalg.norm(v),1e-8))
                        tick(v,'navigation',macro)
                        if np.linalg.norm(sim.s.box-before_nav)>.05:
                            raise ValueError('回り込み中に箱へ接触しました')
                    else: raise ValueError('回り込みの時間切れ')
                for _ in range(3): tick(np.zeros(2),'navigation',macro)
                # Begin the learned skill at the same near-contact distribution as training.
                extent=support(cfg,sim.s.yaw,-d)
                staging=sim.s.box-d*(extent+cfg['robot_radius']+.07)
                for _ in range(70):
                    error=staging-sim.s.robot
                    if np.linalg.norm(error)<.025: break
                    v=error*1.5; v*=min(1.,.22/max(np.linalg.norm(v),1e-8))
                    tick(v,'navigation',macro)
                else: raise ValueError('押し始め位置へ到達できません')
                for _ in range(3): tick(np.zeros(2),'navigation',macro)
                previous=np.zeros(2); stable=0
                for k in range(cfg['episode_steps']):
                    obs=observation(sim.s,goal,d,board,cfg,k,previous)
                    action=(baseline_action(sim.s,goal,d,cfg) if method=='baseline'
                            else model.predict(obs,deterministic=True)[0])
                    action=np.clip(action,-1,1); previous=action
                    tick(frame(d).T@action*cfg['max_speed'],'push',macro)
                    stable=stable+1 if reached(sim.s,goal,cfg) else 0
                    if stable>=4: break
                    if np.linalg.norm(sim.s.box-goal)>board.cell*2.3:
                        raise ValueError('箱が目標から離れすぎました')
                else: raise ValueError('押し動作の時間切れ')
                for _ in range(5): tick(np.zeros(2),'brake',macro)
                if np.linalg.norm(sim.s.box-goal)>cfg['success_tolerance']:
                    raise ValueError('制動後に目標から外れました')
                completed+=1; prev_direction=d
                events.append({'push':macro,'target':m['target'],'position':sim.s.box.tolist(),
                               'angle_deg':math.degrees(angle_error(sim.s.yaw,cfg)),'sim_time':sim.s.t})
            success=np.linalg.norm(sim.s.box-board.xy(board.goal))<cfg['success_tolerance']
            reason='success' if success else 'goal_error'
        except ValueError as e:
            # Physical failures are results. Infrastructure errors are NOT swallowed.
            reason=str(e)
        finally:
            final_distance=float(np.linalg.norm(sim.s.box-board.xy(board.goal)))
            sim_seconds=sim.s.t-start_time if 'start_time' in locals() else 0
            duration=sim.end_video(path) if recording else 0
        result=dict(case=case,method=method,seed=seed,success=bool(success),reason=reason,
            pushes_done=completed,pushes_planned=len(plan),final_distance=final_distance,
            sim_seconds=sim_seconds,video_seconds=duration,video=str(path) if video else '',
            model=str(model_path) if model_path else '',initial=initial)
        (out/f'{tag}.json').write_text(json.dumps(dict(result=result,events=events,plan=plan),indent=2))
        write_csv(out/'traces'/f'{tag}.csv',records)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return result

def smoke(root,cfg):
    root=Path(root); (root/'outputs').mkdir(exist_ok=True)
    board=make_board(cfg,'train')
    with Sim(root,board,cfg,tag='physics_check') as sim:
        # Non-contact command must not move a remote box.
        b=sim.s.box.copy(); r=sim.s.robot.copy()
        # RS_MOTION_DIAGNOSTICS_V1: keep evidence even when an assertion fails.
        motion=[dict(zip(STATE_COLUMNS,sim.s.row()))]
        for _ in range(8):
            sim.step((-.2,0))
            motion.append(dict(zip(STATE_COLUMNS,sim.s.row())))
        write_csv(root/'outputs'/'smoke_motion.csv',motion)
        print(f'physics: dt={sim.s.t-motion[0]["sim_time"]:.3f}s '
              f'robot displacement={np.linalg.norm(sim.s.robot-r):.6f}m '
              f'box displacement={np.linalg.norm(sim.s.box-b):.6f}m '
              f'robot velocity={sim.s.rv.tolist()}',flush=True)
        assert np.linalg.norm(sim.s.robot-r)>.05,'Robot does not move'
        assert np.linalg.norm(sim.s.box-b)<.02,'Box moves without contact'
        # An off-centre physical contact should translate or rotate the box.
        centre=board.xy(board.box)
        sim.reset(centre+[-cfg['box_size']/2-cfg['robot_radius']-.03,.20],centre,0)
        contact=[]
        for _ in range(25):
            sim.step((.25,0))
            contact.append(dict(zip(STATE_COLUMNS,sim.s.row())))
        write_csv(root/'outputs'/'smoke_contact.csv',contact)
        print(f'contact: box displacement={np.linalg.norm(sim.s.box-centre):.6f}m '
              f'box yaw={sim.s.yaw:.6f}rad max_yaw={max(abs(x["box_yaw"]) for x in contact):.6f}rad '
              f'robot={sim.s.robot.tolist()}',flush=True)
        assert np.linalg.norm(sim.s.box-centre)>.06,'Contact does not push the box'
        assert abs(sim.s.yaw)>.01,'Off-centre contact does not rotate the box'
        env=PushEnv(sim,cfg); obs,_=env.reset(seed=42)
        assert obs.shape==(18,) and np.isfinite(obs).all()
        t=time.monotonic()
        for _ in range(60):
            _,_,done,trunc,_=env.step(env.action_space.sample())
            if done or trunc: env.reset()
        speed=60/(time.monotonic()-t)
        print(f'計測: {speed:.1f} transitions/s（学習更新を含まない上限の目安）',flush=True)
    result=roll_puzzle(root,cfg,'straight','baseline',seed=20000,video=True,tag='smoke_camera')
    # Camera and dynamics are mandatory; baseline success is useful diagnosis, not fake pass criteria.
    assert result['video_seconds']>=1
    (root/'outputs'/'smoke_ok.json').write_text(json.dumps({'cfg':cfg,'baseline':result,'steps_per_sec':speed},indent=2))
    if not result['success']:
        print('通常制御も失敗しました。動画とtraceを確認してから学習へ進んでください。',flush=True)

def evaluate(root,cfg):
    root=Path(root); out=root/'outputs'; model=SAC.load(out/'models'/'best.zip',device='cpu')
    rows=[]
    with Sim(root,make_board(cfg,'train'),cfg,tag='heldout') as sim:
        env=PushEnv(sim,cfg)
        for method in ('baseline','SAC'):
            results=evaluate_local(env,model,range(20000,20030),baseline=method=='baseline')
            rows.extend(dict(method=method,**r) for r in results)
        env.close()
    write_csv(out/'test_local.csv',rows)
    summary=[]
    for case in ('straight','turn','maze'):
        for method in ('baseline','SAC'):
            summary.append(roll_puzzle(root,cfg,case,method,out/'models'/'best.zip',
                                      seed=20000,video=True))
    write_csv(out/'test_puzzles.csv',[{k:v for k,v in r.items() if k!='initial'} for r in summary])
    # Same initial state and same board for untrained / intermediate / best policy clips.
    checkpoints=sorted((out/'models').glob('step_*.zip'))
    selected=list(dict.fromkeys([checkpoints[0],checkpoints[len(checkpoints)//2],out/'models'/'best.zip']))
    for ckpt in selected:
        roll_puzzle(root,cfg,'straight','SAC',ckpt,seed=20000,video=True,tag=f'learning_{ckpt.stem}')
    print('固定30条件の局所テスト、3盤面×2方式、学習段階の動画を保存しました。',flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('task',choices=['smoke','train','evaluate'])
    p.add_argument('--root',required=True); p.add_argument('--steps',type=int,default=150000)
    p.add_argument('--resume',action='store_true'); p.add_argument('--backup'); args=p.parse_args()
    cfg=json.loads((Path(args.root)/'config.json').read_text())
    if args.task=='smoke': smoke(args.root,cfg)
    elif args.task=='evaluate': evaluate(args.root,cfg)
    else:
        marker=Path(args.root)/'outputs'/'smoke_ok.json'
        if not marker.exists() or json.loads(marker.read_text())['cfg']!=cfg:
            raise RuntimeError('現在の設定でsmokeセルを先に完了してください。')
        from .train import run
        run(args.root,cfg,args.steps,args.resume,args.backup)

if __name__=='__main__': main()
