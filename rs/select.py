"""Choose a checkpoint using validation trials only, before held-out evaluation."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import shutil


def select(root):
    root = Path(root)
    out = root / 'outputs'
    groups = defaultdict(list)
    with (out / 'validation.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            groups[int(row['train_steps'])].append(row)
    if not groups:
        raise ValueError('No validation results')
    candidates = []
    for step, rows in sorted(groups.items()):
        checkpoint = out / 'models' / f'step_{step:06d}.zip'
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        candidates.append({
            'step': step,
            'successes': sum(row['is_success'] == 'True' for row in rows),
            'trials': len(rows),
            'mean_distance': sum(float(row['distance']) for row in rows) / len(rows),
        })
    if len({item['trials'] for item in candidates}) != 1:
        raise ValueError('Validation trial counts differ between checkpoints')
    chosen = min(candidates, key=lambda item: (-item['successes'], item['mean_distance'], item['step']))
    shutil.copy2(out / 'models' / f"step_{chosen['step']:06d}.zip", out / 'models' / 'best.zip')
    result = {'selection_rule': 'maximum validation successes, then minimum mean final distance',
              'selected': chosen, 'candidates': candidates}
    (out / 'model_selection.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    select(parser.parse_args().root)
