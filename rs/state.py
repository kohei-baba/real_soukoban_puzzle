"""ROS-independent state protocol types."""
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    t: float
    robot: np.ndarray
    rv: np.ndarray
    box: np.ndarray
    yaw: float
    bv: np.ndarray
    bw: float
    tilt: np.ndarray
    rz: float
    bz: float

    @classmethod
    def parse(cls, tokens):
        a=np.array(tokens,dtype=float)
        if len(a)!=17 or not np.isfinite(a).all(): raise ValueError('invalid state')
        return cls(a[0],a[1:3],a[4:6],a[6:8],a[9],a[10:12],a[12],a[13:17],a[3],a[8])

    def row(self):
        return [self.t,*self.robot,*self.rv,*self.box,self.yaw,*self.bv,self.bw,*self.tilt]

STATE_COLUMNS=['sim_time','robot_x','robot_y','robot_vx','robot_vy','box_x','box_y',
               'box_yaw','box_vx','box_vy','box_wz','robot_roll','robot_pitch','box_roll','box_pitch']

