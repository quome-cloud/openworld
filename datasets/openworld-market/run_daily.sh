#!/bin/bash
# Unattended daily forward-test for the E50 trading world model.
# Runs the signal (logs today's pick + scores past picks) into daily.log.
cd /Users/jim/Desktop/openworld || exit 1
echo "===== $(date) =====" >> datasets/openworld-market/daily.log
/Users/jim/.pyenv/versions/3.9.18/bin/python \
  datasets/openworld-market/daily_signal.py >> datasets/openworld-market/daily.log 2>&1
