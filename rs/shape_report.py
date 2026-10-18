"""Graphs and a machine-readable summary of isolated shape experiments."""
from pathlib import Path
import csv
import json
import numpy as np


def read(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def summarize(rows):
    return dict(trials=len(rows),successes=sum(r['is_success']=='True' for r in rows),
                mean_distance=float(np.mean([float(r['distance']) for r in rows])),
                median_angle_deg=float(np.median([float(r['angle_deg']) for r in rows])),
                mean_steps=float(np.mean([int(r['steps']) for r in rows])))


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    summary={}
    for ax,shape in zip(axes,('triangle','star')):
        out=root/'experiments'/shape/'outputs'
        baseline=read(out/'baseline.csv')
        validation=read(out/'validation.csv')
        steps=sorted({int(r['train_steps']) for r in validation})
        rates=[np.mean([r['is_success']=='True' for r in validation if int(r['train_steps'])==step])
               for step in steps]
        ax.plot(steps,np.array(rates)*100,'o-',label='SAC validation (12 seeds)')
        ax.set(title=shape.capitalize(),xlabel='Speed commands during training',ylim=(-4,104))
        ax.grid(alpha=.3); ax.legend(loc='lower right',fontsize=8)
        summary[shape]={'baseline':summarize(baseline),
                        'validation':[dict(step=s,successes=int(round(12*r))) for s,r in zip(steps,rates)]}
        test=out/'sac.csv'
        if test.exists(): summary[shape]['SAC']=summarize(read(test))
        selected=out/'model_selection.json'
        if selected.exists(): summary[shape]['selected']=json.loads(selected.read_text())['selected']
    axes[0].set_ylabel('Success rate (%)')
    fig.tight_layout()
    dest=root/'experiments'/'shape_learning.png'
    fig.savefig(dest,dpi=160)
    plt.close(fig)
    (root/'experiments'/'shape_summary.json').write_text(json.dumps(summary,indent=2))
    print(dest)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    report(p.parse_args().root)
