#!/bin/bash
# usage: MODEL=model.gguf bench.sh N [extra args...]
set -u
N=$1; shift
if [ -x engine-poly/strata-poly ]; then
  BIN=${BIN:-engine-poly/strata-poly}
elif [ -x build/strata-poly ]; then
  BIN=${BIN:-build/strata-poly}
else
  BIN=${BIN:?set BIN to the strata-poly binary}
fi
MODEL=${MODEL:?set MODEL to the GGUF}
TOK=${TOK:-100}
if [ "${SEQ:-rep}" = "mix" ]; then
  IDS=$(python3 -c "print(','.join(str(100+(i*7919)%120000) for i in range($N)))")
else
  IDS=$(python3 -c "print(','.join(['$TOK']*$N))")
fi
"$BIN" --model "$MODEL" \
  --max-context 49152 --kv q8_0 --no-draft --threads 8 --prompt-batch 1024 \
  --tokens "$IDS" --max-new 32 --vram-reserve-mib 1024 "$@" 2>&1 | tail -4
