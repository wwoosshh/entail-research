#!/bin/bash
# P6 external evaluation, item 2: where S4's +0.9 %p at batch 1 and 8 comes from. 1.3.0 (a86bd3f) and 2.0.0 (2ee6e8e)
# alternated with off and control in one session, M19 L4's harness (testbed/m19_s4.py): states on130, on200, off,
# control in a rotated order per round. Both versions exported to the Linux disk, so they are imported the same way.
#   bash testbed/p6_s4_ab.sh [ROUNDS]      (START=5 SKIP_EXPORT=1 adds rounds 5.. to the same folder; the summary
#                                          takes every round there)
set -u
ROOT=<workspace>
cd $ROOT
OUT=$ROOT/testbed/results/p6/s4_ab
mkdir -p $OUT
ROUNDS=${1:-4}
START=${START:-1}
PY=/home/<user>/venvs/vllm/bin/python
if [ -z "${SKIP_EXPORT:-}" ]; then
  rm -rf ~/ci_tmp/ab_130 ~/ci_tmp/ab_200 && mkdir -p ~/ci_tmp/ab_130 ~/ci_tmp/ab_200
  (cd $ROOT/entail && git archive a86bd3f) | tar -xf - -C ~/ci_tmp/ab_130
  (cd $ROOT/entail && git archive 2ee6e8e) | tar -xf - -C ~/ci_tmp/ab_200
fi
echo "1.3.0 a86bd3f vs 2.0.0 2ee6e8e, rounds $START..$((START + ROUNDS - 1)), $(date '+%F %T')" | tee -a $OUT/run.log
states=(on130 on200 off control)
for r in $(seq $START $((START + ROUNDS - 1))); do
  k=$(( (r - 1) % 4 ))
  for i in 0 1 2 3; do
    s=${states[$(( (k + i) % 4 ))]}
    case $s in
      on130) SHIM=$HOME/ci_tmp/ab_130:$HOME/ci_tmp/ab_130/entail/adapters/autoinstall ;;
      on200) SHIM=$HOME/ci_tmp/ab_200:$HOME/ci_tmp/ab_200/entail/adapters/autoinstall ;;
      *) SHIM="" ;;
    esac
    if [ -n "$SHIM" ]; then
      env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_LOG_DIR=$OUT/entail_logs_$s VLLM_LOGGING_LEVEL=WARNING \
        M19_S4_STATE=$s M19_S4_ROUND=$r M19_S4_OUT=$OUT $PY testbed/m19_s4.py > $OUT/${s}_$r.log 2>&1
    else
      env -u PYTHONPATH -u ENTAIL VLLM_LOGGING_LEVEL=WARNING \
        M19_S4_STATE=$s M19_S4_ROUND=$r M19_S4_OUT=$OUT $PY testbed/m19_s4.py > $OUT/${s}_$r.log 2>&1
    fi
    echo "[$(date +%T)] round $r $s exit $? $(grep -h 'B= 32' $OUT/${s}_$r.log | tail -1)" | tee -a $OUT/run.log
  done
done
source ~/venvs/gpu/bin/activate
python - $OUT <<'EOF' | tee $OUT/summary.txt
import glob, json, os, statistics, sys
out = sys.argv[1]
runs = {}
for p in glob.glob(os.path.join(out, "*_*.json")):
    d = json.load(open(p, encoding="utf-8"))
    runs.setdefault(d["round"], {})[d["state"]] = d
rounds = sorted(r for r in runs if all(s in runs[r] for s in ("on130", "on200", "off", "control")))
for b in sorted({b for r in rounds for b in runs[r]["off"]["batches"]}, key=int):
    med = lambda r, s: statistics.median(runs[r][s]["batches"][b]["seconds"])
    ratio = lambda s: statistics.median(med(r, s) / med(r, "off") for r in rounds)
    print(f"B={int(b):3d}  on130/off {ratio('on130'):.4f}  on200/off {ratio('on200'):.4f}  control/off {ratio('control'):.4f}  "
          f"on200/on130 {statistics.median(med(r, 'on200') / med(r, 'on130') for r in rounds):.4f}  (rounds {len(rounds)})")
EOF
