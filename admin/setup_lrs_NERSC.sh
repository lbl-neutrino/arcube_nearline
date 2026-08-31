#!/bin/bash
#
# Sets up the "lrs_nearline" screen session with 5 windows:
#   0: flow_light_data watcher
#   1: light_dqm watcher
#   2-4: rlaunch rapidfire workers
#
# Usage: ./start_lrs_nearline.sh
# Then attach with: screen -r lrs_nearline

set -euo pipefail

SESSION="lrs_nearline"
BASE_DIR="/global/common/software/dune/dunepr/arcube_nearline.2x2_lrs.run3"
LAUNCH_DIR="/pscratch/sd/d/dunepr/launchers/nearline_2x2_lrs_run3"

# Refuse to clobber an existing session with the same name
if screen -list | grep -q "\.${SESSION}[[:space:]]"; then
    echo "A screen session named '${SESSION}' already exists. Check it out. Delete it if you want to start a new one."
    exit 1
fi

# Start a detached session (this becomes window 0)
screen -dmS "$SESSION"

send() {
    # send <window_index> <command string>
    local win="$1"
    local cmd="$2"
    screen -S "$SESSION" -p "$win" -X stuff "${cmd}$(printf \\r)"
}

# --- Window 1 (default wondow on NERSC): flow_light_data watcher ---
send 1 "echo 'Starting flow_light_data watcher...'"
send 1 "cd ${BASE_DIR}"
send 1 "source admin/load.sh"
send 1 "./watcher.py --ext data --path /global/cfs/cdirs/dune/www/data/2x2/LRS_run3/warm_commission actions/flow_light_data.sh"

# --- Window 0: light_dqm watcher ---
screen -S "$SESSION" -X screen   # create window 0
send 0 "echo 'Starting light_dqm watcher...'"
send 0 "cd ${BASE_DIR}"
send 0 "source admin/load.sh"
send 0 "./watcher.py --path /global/cfs/cdirs/dune/www/data/2x2/nearline_run3/flowed_light actions/light_dqm.sh"

# --- Windows 2, 3, 4: rlaunch rapidfire workers ---
for i in 2 3 4; do
    screen -S "$SESSION" -X screen   # create new window
    send "$i" "echo 'Starting rlaunch rapidfire worker in window $i...'"
    send "$i" "cd ${BASE_DIR}"
    send "$i" "source admin/load.sh"
    send "$i" "cd ${LAUNCH_DIR}"
    send "$i" "rlaunch rapidfire --nlaunches infinite &"
done

echo "Session '${SESSION}' created with 5 windows (0-4)."
echo "Attach with: screen -r ${SESSION}"
echo "Switch windows with Ctrl+a then a window number (0-4), or Ctrl+a n / Ctrl+a p."