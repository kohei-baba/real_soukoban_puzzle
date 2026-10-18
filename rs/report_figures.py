"""Draw the actual top-view crate geometry used by the simulations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D

from .shapes import vertices, star_arms


def make(root):
    root = Path(root)
    output = root / 'report_assets' / 'box_shapes.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    configs = [json.loads((root / 'experiments' / shape / 'config.json').read_text())
               for shape in ('triangle', 'star')]
    names = ['Triangle', 'Five-arm star']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=160)
    for ax, cfg, name in zip(axes, configs, names):
        ax.add_patch(Rectangle((-.4, -.4), .8, .8, facecolor='#f4f6f7',
                               edgecolor='#b9c5cb', linewidth=1.5))
        shape = cfg.get('shape', 'square')
        if shape == 'triangle':
            ax.add_patch(Polygon(vertices(cfg), closed=True,
                                 facecolor='#edb64d', edgecolor='#744b0b', linewidth=2))
        else:
            for x, y, length, arm_width, yaw in star_arms(cfg):
                ax.add_patch(Rectangle((-length / 2, -arm_width / 2), length,
                                       arm_width, facecolor='#edb64d',
                                       edgecolor='#744b0b', linewidth=.8,
                                       transform=Affine2D().rotate(yaw).translate(x, y)
                                       + ax.transData))
        ax.set(xlim=(-.43, .43), ylim=(-.43, .43), title=name, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(pad=.9, w_pad=1.6)
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    make(parser.parse_args().root)
