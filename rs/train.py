"""SAC from scratch, fixed validation seeds, periodic recoverable checkpoints."""
from pathlib import Path
import copy,csv,json,time,shutil
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from .sim import Sim
from .planner import make_board
from .env import PushEnv,evaluate_local

class Progress(BaseCallback):
    def __init__(self,total): super().__init__(); self.total=total; self.wall=time.monotonic(); self.start=0
    def _on_training_start(self): self.start=self.num_timesteps; self.wall=time.monotonic()
    def _on_step(self):
        if self.num_timesteps%2000==0:
            speed=(self.num_timesteps-self.start)/max(time.monotonic()-self.wall,1e-6)
            eta=(self.total-self.num_timesteps)/max(speed,.001)/3600
            print(f'step={self.num_timesteps}/{self.total}  {speed:.1f} steps/s  残り概算={eta:.2f} h',flush=True)
        return True

def run(root,cfg,total,resume=False,backup=None):
    root=Path(root); out=root/'outputs'; out.mkdir(exist_ok=True)
    models=out/'models'; models.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    last=models/'last.zip'; replay=models/'replay.pkl'; meta=models/'training_state.json'
    if last.exists() and not resume:
        raise FileExistsError('既存学習があります。RESUME=Trueにするか別のRUN_DIRを指定してください。')
    if resume and not all(p.exists() for p in (last,replay,meta)):
        raise FileNotFoundError('再開にはlast.zip、replay.pkl、training_state.jsonが必要です。')
    with Sim(root,make_board(cfg,'train'),cfg,tag='train') as sim:
        raw=PushEnv(sim,cfg); env=Monitor(raw,str(out/'train'),info_keywords=('is_success','distance'),
                                         override_existing=not resume)
        raw.reset(seed=cfg['seed'])
        state=json.loads(meta.read_text()) if resume else {'best':-1.,'cfg':cfg}
        if resume and state['cfg']!=cfg: raise ValueError('学習再開時の設定が以前と異なります。')
        if resume:
            model=SAC.load(last,env=env,device='cpu'); model.load_replay_buffer(replay)
            if model.num_timesteps!=state['steps'] or model.replay_buffer.pos!=state['replay_pos']:
                raise RuntimeError('チェックポイントが不整合です。保存完了済みのバックアップを復元してください。')
        else:
            model=SAC('MlpPolicy',env,learning_rate=3e-4,buffer_size=100_000,
                learning_starts=2000,batch_size=128,tau=.005,gamma=.98,
                train_freq=4,gradient_steps=4,ent_coef='auto',
                policy_kwargs={'net_arch':[128,128]},seed=cfg['seed'],device='cpu',verbose=0)
            model.save(models/'step_000000')
        best=float(state['best']); callback=Progress(total)
        def save():
            model.save(models/'last'); model.save_replay_buffer(replay)
            meta.write_text(json.dumps({'best':best,'cfg':cfg,'steps':model.num_timesteps,
                                        'replay_pos':model.replay_buffer.pos},indent=2))
            if backup:
                dest=Path(backup); (dest/'models').mkdir(parents=True,exist_ok=True)
                for p in models.iterdir():
                    if p.is_file(): shutil.copy2(p,dest/'models'/p.name)
                for p in out.glob('*.csv'): shutil.copy2(p,dest/p.name)
                shutil.copy2(root/'config.json',dest/'config.json')
        try:
            while model.num_timesteps<total:
                # Gradually widen reset errors; evaluation always uses the full range.
                raw.difficulty=min(1.,.25+model.num_timesteps/50000)
                model.set_env(env,force_reset=True)
                n=min(cfg['eval_interval'],total-model.num_timesteps)
                model.learn(total_timesteps=n,reset_num_timesteps=False,callback=callback)
                model.save(models/f'step_{model.num_timesteps:06d}')
                rng_state=copy.deepcopy(raw.np_random.bit_generator.state)
                rows=evaluate_local(raw,model,range(10000,10012))
                raw.np_random.bit_generator.state=rng_state
                score=float(np.mean([x['is_success'] for x in rows]))
                for row in rows: row['train_steps']=model.num_timesteps
                path=out/'validation.csv'; exists=path.exists()
                with path.open('a',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));
                    if not exists: writer.writeheader()
                    writer.writerows(rows)
                if score>best:
                    best=score; model.save(models/'best')
                print(f'validation={score:.1%} / best={best:.1%}（12試行）',flush=True)
                save()
        finally:
            save()  # KeyboardInterrupt keeps weights and replay when Python can unwind.
            env.close()
    print('学習保存完了。優劣は未使用のテスト条件で確認してください。',flush=True)
