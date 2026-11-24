#!/bin/bash
# TEMPORARY (red-team iteration): runs every red-team run in one CPU job.
# Restored to the gym's version with git at the end of the iteration.
set -e
cd /home/user
bash redteam/run_all.sh
