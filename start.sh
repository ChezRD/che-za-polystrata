#!/bin/sh
# Start the server for one model config. The config names the engine, the model and the port.
# Extra arguments go to serve.server.
#
#   ./start.sh strata-flint-mix12.json
#   ./start.sh strata-gigachat-audio.json --lazy
set -eu
cd "$(dirname "$0")"
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "usage: ./start.sh strata-<model>.json [server arguments]" >&2
  exit 2
fi
config=$1
shift
port=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("port") or 8080)' "$config")
exec .venv/bin/python -m serve.server --engine strata --config "$config" --port "$port" "$@"
