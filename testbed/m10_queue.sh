#!/bin/bash
# M10: the GPU work left, one after another - after E2's second batch, the last E3 case (sglang#33493), then E2's
# third batch once the downloads are done. Usage: bash testbed/m10_queue.sh
ROOT=<workspace>
cd $ROOT
R=$ROOT/testbed/results/m10
while pgrep -f "m10_e2_run.sh" > /dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] E3 sglang#33493"
bash testbed/m10_e3/run_case.sh sglang sg33493 testbed/m10_e3/sg33493.py /home/<user>/models/m10/openbmb__MiniCPM5-2B \
  /home/<user>/models/m10/openbmb__MiniCPM5-2B-DSpark
while pgrep -f "m10_e2_download.py" > /dev/null; do sleep 30; done
/home/<user>/venvs/gpu/bin/python testbed/m10_e2_list.py
comm -23 <(sort $R/e2/list_downloaded.txt) <(sort $R/e2/list_batch2.txt) > $R/e2/list_batch3.txt
echo "[$(date +%H:%M:%S)] E2 batch 3: $(wc -l < $R/e2/list_batch3.txt) models"
bash testbed/m10_e2_run.sh $R/e2/list_batch3.txt
echo "[$(date +%H:%M:%S)] QUEUE DONE"
