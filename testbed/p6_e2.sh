#!/bin/bash
# P6: the 102 normal runs (E2) with entail 2.0's candidate (the product branch) and the platform on: M19 L4's
# procedure (testbed/m15_e2_run.sh, the 38 models of list_30 and list_fresh, each engine's defaults, entail's hook
# only, the off runs of M10/M11 as the control), while `entail serve` reads the runs' records and a poller asks it for
# them (testbed/p6_platform_poll.py). Then the E2 summary (testbed/m10_e2_summarize.py).
#   bash testbed/p6_e2.sh
set -u
ROOT=<workspace>
cd $ROOT
R=testbed/results/p6/e2
mkdir -p $R/engines $R/platform_view
rm -f $R/stop
source ~/venvs/gpu/bin/activate
PYTHONPATH=$ROOT/entail python -m entail serve --dir $R/platform_view --port 8770 > $R/serve.log 2>&1 &
SERVE=$!
sleep 2
python testbed/p6_platform_poll.py $R/engines $R/platform_view 8770 $R/stop $R/platform_poll.json > $R/poll.log 2>&1 &
POLL=$!
E2_DIR=$R bash testbed/m15_e2_run.sh testbed/results/m11/e2/list_30.txt testbed/results/m11/e2/list_fresh.txt
touch $R/stop
wait $POLL
kill $SERVE 2> /dev/null; wait $SERVE 2> /dev/null
M10_E2_DIR=$ROOT/$R python testbed/m10_e2_summarize.py > $R/summarize.log 2>&1
tail -3 $R/poll.log
echo "e2 done"
