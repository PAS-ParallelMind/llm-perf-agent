#!/usr/bin/env bash
# Manage the parallelmind_harness webui tmux session.
#
# Usage:
#   webui/webui.sh [start | stop | restart | status]
#
# Env vars (all optional):
#   PORT     port to bind (default 9099)
#   SESSION  tmux session name (default webui)

set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-9099}"
SESSION="${SESSION:-webui}"
PYTHON="/mnt/disk2/elton7318/venv/bin/python"

is_running() { tmux has-session -t "$SESSION" 2>/dev/null; }
port_pids() {
  # PIDs bound on $PORT (excluding socket lines that don't carry pid=)
  ss -tnlp 2>/dev/null | awk -v p=":$PORT$" '$4 ~ p { print $NF }' \
    | sed -nE 's/.*pid=([0-9]+).*/\1/p' | sort -u
}

cmd="${1:-start}"

case "$cmd" in
  start)
    if is_running; then
      echo "session '$SESSION' already up. (use \`$0 restart\` to recreate)"
      exit 0
    fi
    if pids="$(port_pids)" && [[ -n "$pids" ]]; then
      echo "port $PORT already held by pid(s): $pids"
      echo "  → \`$0 stop\` first, or set PORT=<n> to pick another port."
      exit 1
    fi
    tmux new-session -d -s "$SESSION" -c "$HARNESS_ROOT" \
      "PORT=$PORT $PYTHON -m webui.backend.server"
    sleep 2
    echo "tmux session '$SESSION' started"
    echo "  → http://localhost:$PORT/"
    echo "  attach: tmux attach -t $SESSION   (Ctrl-B d to detach)"
    ;;
  stop)
    if is_running; then
      tmux kill-session -t "$SESSION"
      echo "tmux session '$SESSION' stopped"
    else
      echo "session '$SESSION' not running"
    fi
    # Reload-mode uvicorn spawns workers whose cmdline doesn't contain
    # "webui.backend.server" — kill anything still bound to the port.
    if pids="$(port_pids)" && [[ -n "$pids" ]]; then
      echo "killing orphan(s) on port $PORT: $pids"
      echo "$pids" | xargs -r kill -9
    fi
    ;;
  restart)
    "$0" stop
    sleep 1
    "$0" start
    ;;
  status)
    if is_running; then
      echo "tmux session '$SESSION': up"
    else
      echo "tmux session '$SESSION': NOT running"
    fi
    if pids="$(port_pids)" && [[ -n "$pids" ]]; then
      echo "port $PORT bound by pid(s): $pids"
    else
      echo "port $PORT: free"
    fi
    if curl -sS -o /dev/null --max-time 2 \
        "http://localhost:$PORT/api/health" 2>/dev/null; then
      echo "/api/health: 200 OK"
    else
      echo "/api/health: not reachable"
    fi
    ;;
  *)
    echo "usage: $0 {start | stop | restart | status}"
    exit 1
    ;;
esac
