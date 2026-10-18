#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

command="${1:-}"
if [[ -z "$command" ]]; then
  echo 'Usage: ./run.sh {build|shape|learning-curve|figures|video}' >&2
  exit 2
fi
shift

if [[ "$command" == build ]]; then
  docker build -t real-sokoban:local .
  exit
fi

case "$command" in
  shape)
    shape="${1:-}"; task="${2:-}"
    if [[ "$shape" != triangle && "$shape" != star ]]; then
      echo 'shape must be triangle or star' >&2; exit 2
    fi
    case "$task" in
      init|smoke|baseline|train|select|evaluate|video|puzzles|complex) ;;
      *) echo 'task must be init|smoke|baseline|train|select|evaluate|video|puzzles|complex' >&2; exit 2 ;;
    esac
    shift 2
    inner="python -m rs.shape_experiment $shape $task"
    if [[ "$task" == train ]]; then
      steps="${1:-150000}"
      if [[ ! "$steps" =~ ^[0-9]+$ ]]; then echo 'steps must be an integer' >&2; exit 2; fi
      inner+=" --steps $steps"
      if [[ $# -gt 0 ]]; then shift; fi
      if [[ "${1:-}" == --resume ]]; then inner+=' --resume'; shift; fi
    fi
    ;;
  learning-curve) inner='python -m rs.shape_report --root /workspace' ;;
  figures) inner='python -m rs.report_figures --root /workspace' ;;
  video) inner='python -m rs.maze_video --root /workspace --extended' ;;
  *) echo "Unknown command: $command" >&2; exit 2 ;;
esac

if [[ $# -gt 0 ]]; then echo "Unexpected arguments: $*" >&2; exit 2; fi

docker run --rm --init -v "$PWD:/workspace" real-sokoban:local \
  "set -eo pipefail; source /opt/ros/jazzy/setup.bash; source /opt/rs-venv/bin/activate; if [[ ! -f build/libRealSokoban.so ]]; then cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2; fi; $inner"
