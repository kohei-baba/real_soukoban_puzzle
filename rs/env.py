"""Goal-conditioned learned pushing: action is robot velocity, not box motion."""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .planner import DIRECTIONS,add

from .control import frame,angle_error,observation,baseline_action,reached
from .shapes import support

class PushEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,sim,cfg):
        self.sim=sim; self.board=sim.board; self.cfg=cfg
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(18,),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        self.difficulty=1.; self.elapsed=0; self.previous=np.zeros(2); self.stable=0
        self.goal=self.board.xy(self.board.goal); self.direction=np.array([1.,0.])

    def reset(self,seed=None,options=None):
        super().reset(seed=seed); rng=self.np_random
        choices=[]
        for r in range(1,self.board.rows-1):
            for c in range(1,self.board.cols-1):
                p=(r,c)
                for dr in DIRECTIONS:
                    if all(self.board.free(q) for q in (p,add(p,dr),add(p,(-dr[0],-dr[1])))):
                        choices.append((p,dr))
        p,dr=choices[int(rng.integers(len(choices)))]
        self.direction=(self.board.xy(add(p,dr))-self.board.xy(p))/self.board.cell
        f=frame(self.direction)
        yaw=math.radians(rng.uniform(-20,20))*self.difficulty
        box=self.board.xy(p)+rng.uniform(-.035,.035,2)*self.difficulty
        self.goal=self.board.xy(add(p,dr))
        extent=support(self.cfg,yaw,-self.direction)
        gap=rng.uniform(.035,.10)
        robot=box-self.direction*(extent+self.cfg['robot_radius']+gap)
        robot+=f[1]*rng.uniform(-.06,.06)*self.difficulty
        self.sim.reset(robot,box,yaw)
        self.elapsed=0; self.stable=0; self.previous=np.zeros(2,dtype=np.float32)
        return self.obs(),{}

    def obs(self):
        return observation(self.sim.s,self.goal,self.direction,self.board,self.cfg,self.elapsed,self.previous)

    def step(self,action):
        action=np.clip(np.asarray(action,dtype=float),-1,1)
        old=self.sim.s; olderr=np.linalg.norm(old.box-self.goal)
        olddist=np.linalg.norm(old.robot-old.box)
        if self.cfg.get('shape','square')=='square':
            oldapproach=max(0.,olddist-self.cfg['box_size']/2-self.cfg['robot_radius'])
        else:
            olddir=(old.robot-old.box)/max(olddist,1e-8)
            oldapproach=max(0.,olddist-support(self.cfg,old.yaw,olddir)-self.cfg['robot_radius'])
        s=self.sim.step(frame(self.direction).T@action*self.cfg['max_speed'])
        self.elapsed+=1
        error=np.linalg.norm(s.box-self.goal)
        newdist=np.linalg.norm(s.robot-s.box)
        if self.cfg.get('shape','square')=='square':
            approach=max(0.,newdist-self.cfg['box_size']/2-self.cfg['robot_radius'])
        else:
            newdir=(s.robot-s.box)/max(newdist,1e-8)
            approach=max(0.,newdist-support(self.cfg,s.yaw,newdir)-self.cfg['robot_radius'])
        reward=35*(olderr-error)/self.board.cell+2*(oldapproach-approach)/self.board.cell
        reward-=.035+.004*float(action@action)+.015*abs(angle_error(s.yaw,self.cfg))
        if self.cfg.get('shape','square')!='square':
            reward+=self.cfg.get('shape_angle_reward',2.)*(abs(angle_error(old.yaw,self.cfg))-
                                                             abs(angle_error(s.yaw,self.cfg)))
        reward-=.003*float(np.sum((action-self.previous)**2))
        self.stable=self.stable+1 if reached(s,self.goal,self.cfg) else 0
        success=self.stable>=4
        fail=(error>2.3*self.board.cell or np.max(np.abs(s.tilt))>.6 or
              s.bz<.03 or not self.board.free(self.board.cell_at(s.box)))
        terminated=bool(success or fail); truncated=self.elapsed>=self.cfg['episode_steps'] and not terminated
        if success: reward+=15.
        if fail: reward-=8.
        self.previous=action.astype(np.float32)
        info={'is_success':success,'distance':float(error),'angle_deg':math.degrees(abs(angle_error(s.yaw,self.cfg))),
              'reason':'success' if success else 'physical_failure' if fail else 'timeout' if truncated else 'running'}
        return self.obs(),float(reward),terminated,truncated,info

def policy_action(model,obs,env,baseline=False):
    if baseline: return baseline_action(env.sim.s,env.goal,env.direction,env.cfg)
    return model.predict(obs,deterministic=True)[0]

def evaluate_local(env,model,seeds,baseline=False):
    result=[]; env.difficulty=1.
    for seed in seeds:
        obs,_=env.reset(seed=int(seed)); total=0.
        initial={'initial_box_x':float(env.sim.s.box[0]),'initial_box_y':float(env.sim.s.box[1]),
                 'initial_yaw':float(env.sim.s.yaw),'initial_robot_x':float(env.sim.s.robot[0]),
                 'initial_robot_y':float(env.sim.s.robot[1]),'goal_x':float(env.goal[0]),'goal_y':float(env.goal[1])}
        for i in range(env.cfg['episode_steps']):
            a=policy_action(model,obs,env,baseline)
            obs,r,done,trunc,info=env.step(a); total+=r
            if done or trunc: break
        result.append(dict(seed=int(seed),steps=i+1,reward=total,**info,**initial))
    return result
