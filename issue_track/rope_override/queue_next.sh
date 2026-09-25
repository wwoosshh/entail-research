#!/bin/bash
# Start the next batch once the A2 control run has finished.
cd <workspace>/issue_track/rope_override
until ! pgrep -f "rope_override.py A2" > /dev/null; do sleep 5; done
bash run_all.sh C_rolecheck LA LA2 LC LE LC_rolecheck
