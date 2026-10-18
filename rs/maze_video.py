"""Extract the two successful maze demonstrations from the existing montage."""
from pathlib import Path
import json
import subprocess


def make(root):
    root = Path(root)
    experiments = root / 'experiments'
    work = experiments / 'montage_parts'
    clips = []
    for shape in ('triangle', 'star'):
        result_file = experiments / shape / 'outputs' / 'maze_SAC_20000.json'
        result = json.loads(result_file.read_text())['result']
        if not (result['success'] and result['pushes_done'] == result['pushes_planned'] == 5):
            raise ValueError(f'Maze was not completed: {result_file}')
        clip = work / f'maze_SAC_20000_{shape}.mp4'
        if not clip.is_file():
            raise FileNotFoundError(clip)
        clips.append(clip)

    playlist = work / 'maze_successes_concat.txt'
    playlist.write_text(''.join(f"file '{clip}'\n" for clip in clips))
    output = experiments / 'maze_successes.mp4'
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
        '-safe', '0', '-i', str(playlist), '-c', 'copy', '-movflags', '+faststart',
        str(output),
    ], check=True)
    print(output)


def make_extended(root):
    """Show both solved maze boards for each shape in a compact labelled video."""
    root = Path(root)
    experiments = root / 'experiments'
    work = experiments / 'montage_parts'
    work.mkdir(parents=True, exist_ok=True)
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    clips = []
    for shape in ('triangle', 'star'):
        for case in ('maze', 'maze_complex'):
            tag = f'{case}_SAC_20000'
            base = experiments / shape / 'outputs'
            result = json.loads((base / f'{tag}.json').read_text())['result']
            expected = 5 if case == 'maze' else 8
            if not (result['success'] and result['pushes_done'] ==
                    result['pushes_planned'] == expected):
                raise ValueError(f'Maze did not finish: {shape}/{tag}')
            original = base / 'videos' / f'{tag}.mp4'
            if not original.is_file():
                raise FileNotFoundError(original)
            caption = work / f'extended_{shape}_{case}.txt'
            label = 'MAZE 7x7' if case == 'maze' else 'MAZE 9x9'
            caption.write_text(f'{shape.upper()} | {label} | SAC\n'
                               f'SUCCESS | pushes {expected}/{expected} | 3x SPEED')
            clip = work / f'extended_{shape}_{case}.mp4'
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-i', str(original), '-vf',
                f'setpts=PTS/3,fps=10,scale=640:480,setsar=1,'
                f'drawtext=fontfile={font}:textfile={caption}:'
                'x=14:y=18:fontsize=20:fontcolor=white:'
                'box=1:boxcolor=black@0.7:boxborderw=8',
                '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(clip),
            ], check=True)
            clips.append(clip)
    playlist = work / 'maze_extended_concat.txt'
    playlist.write_text(''.join(f"file '{clip}'\n" for clip in clips))
    output = experiments / 'maze_extended.mp4'
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
        '-safe', '0', '-i', str(playlist), '-c', 'copy', '-movflags', '+faststart',
        str(output),
    ], check=True)
    print(output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    (make_extended if args.extended else make)(args.root)
