#!/bin/bash
# P6: S4 (request cost, vLLM 0.30, Qwen3-4B, CUDA graphs) with entail 2.0's candidate, M19 L4's procedure
# (testbed/m19_s4.sh: rounds of on, off and control processes in rotated order) - once as it is, once with
# `entail serve` running on the log folder the "on" processes write to, to see that the platform adds nothing to the
# engine's cost (it runs in its own process and only reads the records).
#   bash testbed/p6_s4.sh [ROUNDS]
set -u
ROOT=<workspace>
cd $ROOT
ROUNDS=${1:-4}
M19_S4_OUT=testbed/results/p6/s4 bash testbed/m19_s4.sh $ROUNDS
OUT=testbed/results/p6/s4_serve
mkdir -p $OUT/entail_logs
source ~/venvs/gpu/bin/activate
PYTHONPATH=$ROOT/entail python -m entail serve --dir $OUT/entail_logs --port 8771 > $OUT/serve.log 2>&1 &
SERVE=$!
# a page watching the newest run, as a user's browser would: the event stream, read continuously
python - > $OUT/watch.log 2>&1 <<'EOF' &
import http.client, json, time
for _ in range(3600):
    try:
        c = http.client.HTTPConnection("127.0.0.1", 8771, timeout=10)
        c.request("GET", "/api/runs", headers={"Host": "127.0.0.1:8771"})
        runs = json.loads(c.getresponse().read())["runs"]
        c.close()
        if runs:
            c = http.client.HTTPConnection("127.0.0.1", 8771, timeout=10)
            c.request("GET", "/api/graph?run=" + runs[0]["run"], headers={"Host": "127.0.0.1:8771"})
            c.getresponse().read()
            c.close()
        print(time.time(), len(runs), flush=True)
    except Exception as e:
        print("error", type(e).__name__, e, flush=True)
    time.sleep(2)
EOF
WATCH=$!
M19_S4_OUT=$OUT bash testbed/m19_s4.sh $ROUNDS
kill $WATCH $SERVE 2> /dev/null; wait $WATCH $SERVE 2> /dev/null
source ~/venvs/gpu/bin/activate
python testbed/m19_s4_summary.py testbed/results/p6/s4 > testbed/results/p6/s4/summary.txt 2>&1
python testbed/m19_s4_summary.py $OUT > $OUT/summary.txt 2>&1
echo "== without the platform"; cat testbed/results/p6/s4/summary.txt
echo "== with entail serve watching"; cat $OUT/summary.txt
echo "watch polls: $(grep -vc error $OUT/watch.log) errors: $(grep -c error $OUT/watch.log)"
