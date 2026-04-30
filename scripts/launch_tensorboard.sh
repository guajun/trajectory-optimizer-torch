#!/usr/bin/env bash
# Launch TensorBoard against this repo's training logs.
#
# Usage:
#   scripts/launch_tensorboard.sh                          # auto-discover logs under repo root
#   scripts/launch_tensorboard.sh configs/examples         # restrict scan root
#   scripts/launch_tensorboard.sh -p 7006                  # custom port
#   scripts/launch_tensorboard.sh --bind-all               # bind to 0.0.0.0 (LAN access)
#   scripts/launch_tensorboard.sh --no-reload              # disable file watcher
#
# Recommended remote workflow (TensorBoard stays on the server):
#   On your laptop:   ssh -N -L 6006:127.0.0.1:6006 user@server
#   Then open:        http://127.0.0.1:6006
#
# Notes:
# - TensorBoard recursively scans the given logdir, so any subfolder named
#   `tensorboard/` produced by run_training will be picked up automatically.
# - Default bind is 127.0.0.1 so it is only reachable through SSH forwarding.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

logdir=""
port=6006
host="127.0.0.1"
reload_interval=5
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      port="$2"; shift 2 ;;
    -h|--host)
      host="$2"; shift 2 ;;
    --bind-all)
      host="0.0.0.0"; shift ;;
    --no-reload)
      reload_interval=0; shift ;;
    --help)
      sed -n '1,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --)
      shift; extra_args+=("$@"); break ;;
    -*)
      extra_args+=("$1"); shift ;;
    *)
      if [[ -z "$logdir" ]]; then
        logdir="$1"
      else
        extra_args+=("$1")
      fi
      shift ;;
  esac
done

if [[ -z "$logdir" ]]; then
  logdir="$repo_root"
fi
logdir="$(cd "$logdir" && pwd)"

if [[ -x "$repo_root/.venv/bin/tensorboard" ]]; then
  tb="$repo_root/.venv/bin/tensorboard"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  tb_py="$repo_root/.venv/bin/python"
  tb="$tb_py -m tensorboard.main"
else
  tb="tensorboard"
fi

echo "tensorboard logdir : $logdir"
echo "tensorboard host   : $host"
echo "tensorboard port   : $port"
if [[ "$host" == "127.0.0.1" || "$host" == "localhost" ]]; then
  echo "ssh tunnel example : ssh -N -L ${port}:127.0.0.1:${port} <user>@<server>"
fi
echo

exec $tb \
  --logdir "$logdir" \
  --host "$host" \
  --port "$port" \
  --reload_interval "$reload_interval" \
  "${extra_args[@]}"
