#!/bin/bash
# M19 L5 small repairs on real engines, entail on (the working copy): DEFERRED 40 (the start-up path check on NaN
# log-probabilities: vllm#33560's case on vLLM 0.16 float16, and the finite 0.30 control) and DEFERRED 41 (the KV
# extent rule on a cross-attention group: vllm#33091's Whisper case on 0.30 and 0.14).
#   bash lowlevel/l5/verify_repairs.sh      (from WSL; writes lowlevel/l5/verify/)
set -u
ROOT=<workspace>
O=$ROOT/lowlevel/l5/verify
mkdir -p $O/entail_logs
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
cd $ROOT

on() {  # name venv script args...
  local name=$1 venv=$2 script=$3
  shift 3
  rm -f $O/$name.record.jsonl
  env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$O/$name.record.jsonl ENTAIL_LOG_DIR=$O/entail_logs \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 timeout 3600 /home/<user>/venvs/$venv/bin/python $script $O/$name.json "$@" \
    > $O/$name.log 2>&1
  echo "$name rc=$? $(grep -h '^RESULT' $O/$name.log | cut -c1-200)"
  python3 - "$O/$name.record.jsonl" <<'PY'
import json, sys, collections
n = collections.Counter()
notes = []
for line in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(line)
    name, verdict = d.get("name") or d.get("fact"), d.get("verdict")
    n[(name, verdict)] += 1
    if name in ("PathAgreement", "KvExtent") and verdict != "pass":
        notes.append(f"  {name} {verdict}: {(d.get('note') or '')[:220]}")
for (name, verdict), c in sorted(n.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
    if name in ("PathAgreement", "KvExtent"):
        print(f"  {name:<14} {verdict:<8} {c}")
print("\n".join(notes[:6]))
PY
}

on r40_vl33560_0160 vllm0160 testbed/m19/replay4/cases/vl33560.py float16
on r40_vl33560_0300 vllm testbed/m19/replay4/cases/vl33560.py float16
on r41_vl33091_0300 vllm testbed/m19/replay4/cases/vl33091.py
on r41_vl33091_0140 vllm0140 testbed/m19/replay4/cases/vl33091.py
