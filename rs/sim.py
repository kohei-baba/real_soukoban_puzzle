"""ROS 2 command/feedback bridge; owns only its own child process groups."""
from pathlib import Path
import os, re, signal, subprocess, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from .world import write_world

from .state import State,STATE_COLUMNS

class Sim:
    def __init__(self,root,board,cfg,camera=False,tag='sim'):
        self.root=Path(root); self.cfg=cfg; self.board=board; self.camera=camera
        self.children=[]; self.handles=[]; self.node=None; self.context=None; self.executor=None
        self.seq=int(time.time()*1e6); self.responses={}; self.s=None
        logdir=self.root/'outputs'/'logs'; logdir.mkdir(parents=True,exist_ok=True)
        self.logdir=logdir; self.tag=tag
        world=logdir/f'{tag}.sdf'; write_world(world,board,cfg,camera)
        env=os.environ.copy()
        env['GZ_PARTITION']=f'real_sokoban_{os.getpid()}_{self.seq}'
        env['ROS_DOMAIN_ID']=str(20+os.getpid()%40)
        env.pop('ROS_LOCALHOST_ONLY',None)
        env['ROS_AUTOMATIC_DISCOVERY_RANGE']='LOCALHOST'
        code_root=Path(__file__).resolve().parent.parent
        env['GZ_SIM_SYSTEM_PLUGIN_PATH']=str(code_root/'build')+':'+env.get('GZ_SIM_SYSTEM_PLUGIN_PATH','')
        env['LIBGL_ALWAYS_SOFTWARE']='1'; env['GALLIUM_DRIVER']='llvmpipe'
        self.env=env
        try:
            # Start PAUSED. Every command advances exactly N physics steps.
            self._start(['gz','sim','-s','-v','3',*(['--headless-rendering'] if camera else []),str(world)],'gazebo')
            self._start(['ros2','run','ros_gz_bridge','parameter_bridge',
                         '/rs/command@std_msgs/msg/String]gz.msgs.StringMsg',
                         '/rs/state@std_msgs/msg/String[gz.msgs.StringMsg'],'bridge')
            self.context=rclpy.context.Context()
            os.environ.pop('ROS_LOCALHOST_ONLY',None)
            os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE']='LOCALHOST'
            rclpy.init(context=self.context,domain_id=int(env['ROS_DOMAIN_ID']))
            self.node=Node('real_sokoban_client',context=self.context)
            self.executor=SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.pub=self.node.create_publisher(String,'/rs/command',10)
            self.sub=self.node.create_subscription(String,'/rs/state',self._receive,10)
            deadline=time.monotonic()+45
            while self.pub.get_subscription_count()==0:
                self._alive()
                if time.monotonic()>deadline: raise TimeoutError('ROS bridge discovery')
                self.executor.spin_once(timeout_sec=.1)
            # Give Gazebo Transport discovery time before the first request.
            self.s=self.reset(board.xy(board.robot),board.xy(board.box),0)
        except BaseException:
            self.close(); raise

    def _start(self,args,name):
        f=open(self.logdir/f'{self.tag}_{name}.log','w'); self.handles.append(f)
        self.children.append(subprocess.Popen(args,env=self.env,cwd=self.root,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True))

    def _alive(self):
        if any(p.poll() is not None for p in self.children):
            raise RuntimeError(f'Gazebo/bridge exited; inspect {self.logdir}/{self.tag}_*.log')

    def _receive(self,msg):
        parts=msg.data.split()
        if not parts: return
        if parts[0]=='ERR': self.responses[int(parts[1])]=RuntimeError(msg.data)
        else:
            try: self.responses[int(parts[0])]=State.parse(parts[1:])
            except (ValueError,IndexError) as e: raise RuntimeError('State protocol error') from e

    def _request(self,command,timeout=35):
        self.seq+=1; seq=self.seq
        self.responses.clear()
        msg=String(); msg.data=f'{seq} {command}'
        deadline=time.monotonic()+timeout; sent=-1e9
        while seq not in self.responses:
            self._alive(); now=time.monotonic()
            if now>deadline: raise TimeoutError(f'{command}: inspect {self.logdir}/{self.tag}_*.log')
            if now-sent>.75: self.pub.publish(msg); sent=now  # same ID is idempotent
            self.executor.spin_once(timeout_sec=.02)
        response=self.responses.pop(seq)
        if isinstance(response,Exception): raise response
        self.s=response; return response

    def reset(self,robot,box,yaw):
        s=self._request(f'RESET {robot[0]} {robot[1]} {box[0]} {box[1]} {yaw} 80')
        if np.linalg.norm(s.box-box)>.1*self.board.cell:
            raise RuntimeError('Reset moved the box too far: overlapping geometry / unstable physics')
        return s

    def step(self,velocity):
        velocity=np.asarray(velocity,dtype=float)
        if velocity.shape!=(2,) or not np.isfinite(velocity).all(): raise ValueError(velocity)
        velocity=np.clip(velocity,-self.cfg['max_speed'],self.cfg['max_speed'])
        n=self.cfg['action_steps']; before=self.s.t
        s=self._request(f'STEP {velocity[0]} {velocity[1]} {n}')
        dt=n*self.cfg['physics_dt']
        if abs(s.t-before-dt)>self.cfg['physics_dt']*1.5:
            raise RuntimeError(f'Synchronous stepping failed: expected {dt}, got {s.t-before}')
        return s

    def settle(self,steps=5):
        for _ in range(steps): self.step((0,0))
        return self.s

    def record_request(self,req):
        result=subprocess.run(['gz','service','-s','/rs/record','--reqtype','gz.msgs.VideoRecord',
            '--reptype','gz.msgs.Boolean','--timeout','5000','--req',req],
            env=self.env,capture_output=True,text=True,timeout=10)
        if result.returncode or not re.search(r'\bdata:\s*true\b',result.stdout):
            raise RuntimeError('Camera recorder: '+result.stdout+result.stderr)

    def begin_video(self,path):
        if not self.camera: raise RuntimeError('Start Sim(camera=True)')
        self.raw=Path(path).with_suffix('.raw.mp4'); self.raw.unlink(missing_ok=True)
        self.settle(15)  # initialize the camera while the world advances
        self.record_request(f'start: true, format: "mp4", save_filename: "{self.raw}"')
        self.settle(10)

    def end_video(self,path):
        self.settle(10)
        self.record_request('stop: true')
        self.settle(5)  # recorder finalizes in a render callback
        deadline=time.monotonic()+15
        while not self.raw.exists():
            if time.monotonic()>deadline: raise RuntimeError('No camera MP4; inspect Gazebo log')
            self.settle(2)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(self.raw),
            '-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],
            check=True,timeout=90)
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
                         '-of','default=nw=1:nk=1',str(path)],text=True))
        if duration<.8: raise RuntimeError('Camera produced too few frames')
        self.raw.unlink()
        return duration

    def close(self):
        if self.executor is not None:
            if self.node is not None: self.executor.remove_node(self.node)
            self.executor.shutdown(timeout_sec=2.0)
            self.executor=None
        if self.node is not None:
            self.node.destroy_node(); self.node=None
        if self.context is not None and self.context.ok(): self.context.shutdown()
        for p in reversed(self.children):
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGINT)
                except ProcessLookupError: pass
        for p in reversed(self.children):
            try: p.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
        for f in self.handles: f.close()
        self.children=[]; self.handles=[]

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
