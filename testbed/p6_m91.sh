#!/bin/bash
# P6: S1 (preservation), S2 (the test problems, default and strict policy) and S3 on M9.1's healthy runs, with entail
# 2.0's candidate - M9.1's phases A, B, S and T again into testbed/results/p6/m91 (as M11.7 did for 1.0.1), then the
# two readers. Phase R (GSM8K with the RoPE override) and D/E (locating, the evaluation score) are not re-run here.
#   bash testbed/p6_m91.sh
set -u
ROOT=<workspace>
cd $ROOT
OUT=$ROOT/testbed/results/p6/m91
M91_OUT=$OUT bash testbed/m91_run.sh A B S T
source ~/venvs/gpu/bin/activate
python testbed/m91_s1.py $OUT/engines $OUT/s1_s3.json > $OUT/s1_s3.log 2>&1; echo "s1 rc=$?"
M91_DIR=$OUT python testbed/m91_problems.py > $OUT/problems.log 2>&1; echo "problems rc=$?"
echo "m91 done"
