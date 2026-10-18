"""Pure numerical observation, baseline and task metrics."""
import math
import numpy as np
from .shapes import support,angle_error as shape_angle_error

def frame(direction):
    d=np.asarray(direction,dtype=float); d=d/np.linalg.norm(d)
    return np.array([d,[-d[1],d[0]]])

def angle_error(yaw,cfg=None):
    return shape_angle_error(yaw,cfg or {'shape':'square'})

def observation(s,goal,direction,board,cfg,elapsed,previous):
    f=frame(direction); cell=board.cell
    rays=[]
    for d in (f[0],f[1],-f[0],-f[1]):
        distance=3*cell
        for t in np.linspace(.1*cell,3*cell,30):
            if not board.free(board.cell_at(s.box+d*t)): distance=t; break
        rays.append(distance/(3*cell))
    return np.concatenate([f@(goal-s.box)/cell,f@(s.robot-s.box)/cell,
        f@s.rv/cfg['max_speed'],f@s.bv/cfg['max_speed'],
        [math.sin({'square':4,'triangle':3,'star':5}[cfg.get('shape','square')]*s.yaw),
         math.cos({'square':4,'triangle':3,'star':5}[cfg.get('shape','square')]*s.yaw),s.bw/3],rays,
        [elapsed/cfg['episode_steps']],previous]).astype(np.float32)

def baseline_action(s,goal,direction,cfg):
    """Feedback baseline, not an intentionally weak fixed-speed/open-loop script."""
    d=np.asarray(goal)-s.box; dist=np.linalg.norm(d)
    if dist < cfg['success_tolerance']*.8: return np.zeros(2,dtype=np.float32)
    d=d/max(dist,1e-6)
    back_extent=support(cfg,s.yaw,-d)
    rel=s.robot-s.box; along=float(rel@d); side=rel-along*d
    # Re-align behind the moving box before applying forward pressure.
    if np.linalg.norm(side)>.09 or along>-.05:
        target=s.box-d*(back_extent+cfg['robot_radius']+.05)
        world=1.5*(target-s.robot)
    else:
        forward=min(cfg['max_speed']*.8,1.3*dist)
        world=d*forward-2.8*side
    return np.clip(frame(direction)@world/cfg['max_speed'],-1,1).astype(np.float32)

def reached(s,goal,cfg):
    return bool(np.linalg.norm(s.box-goal)<cfg['success_tolerance'] and
                np.linalg.norm(s.bv)<.06 and abs(s.bw)<.15 and
                (cfg.get('shape','square')=='square' or
                 abs(angle_error(s.yaw,cfg))<cfg.get('shape_angle_tolerance',.18)))

