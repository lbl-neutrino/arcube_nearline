#!/usr/bin/env bash

stage=light_dqm

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

# light_dqm.py expects a trailing slash
inbase=/global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_light

# For example,
inpath=$1; shift
indir=$(dirname "$inpath")
reldir=$(echo "$indir" | sed "s|^$inbase/||")

# Remove the extension and the final numbers. The final numbers are either the
# run number (123 if the filename is like "mpd_run_foo_rctl_123.FLOW.hdf5") or
# the subrun number (789 if the filename is like
# "mpd_run_bar_rctl_456_p789.FLOW.hdf5"). We end up with a file_syntax of either
# "mpd_run_foo_rctl_" or "mpd_run_bar_rctl_456_p"
file_syntax=$(basename "$inpath" .FLOW.hdf5 | sed 's/[0-9]\+$//')

# Extract the same final numbers we stripped out above (welcome to regex hell)
start_run=$(basename "$inpath" .FLOW.hdf5 | sed -n 's/.*[^0-9]\([0-9]\+\)$/\1/p')

get_outpath() {
    outbase=$1
    ext=$2
    outname=$(basename "$inpath" .hdf5).light_dqm.$ext
    mkdir -p "$outbase/$reldir"
    realpath "$outbase/$reldir/$outname"
}

channel_status=$(dirname "${BASH_SOURCE[0]}")/light_dqm/channel_status.csv

plotpath1=$(get_outpath "$plot_outbase" main.pdf)
plotpath2=$(get_outpath "$plot_outbase" baseline.pdf)
plotpath3=$(get_outpath "$plot_outbase" flatline.pdf)

logpath=$(get_outpath "$log_outbase" log)

cd "$(dirname "${BASH_SOURCE[0]}")/light_dqm"

if [[ "$(stat -c %s "$inpath")" -gt 50000000000 ]]; then
    echo "File is larger than 50 GB; bailing"
    exit 1
fi

# light_dqm.py expects the trailing slash on input_path
python3 light_dqm.py --input_path "$inbase/$reldir/" \
                     --output_dir "$plot_outbase" \
                     --tmp_dir "$plot_outbase/tmp" \
                     --file_syntax "$file_syntax" \
                     --output_dir "$(dirname "$plotpath1")" \
                     --channel_status_file "$channel_status"\
                     --start_run $start_run \
                     2>&1 | tee "$logpath"


latest_outbase="$plot_outbase/latest"
latest_path1="$latest_outbase/latest_main.pdf"
latest_path2="$latest_outbase/latest_baseline.png"
latest_path3="$latest_outbase/latest_dead_chan.png"

if [ ! -d "$latest_outbase" ]; then
    mkdir -p "$latest_outbase"
fi

ln -sf $plotpath1 $latest_path1
ln -sf $plotpath2 $latest_path2
ln -sf $plotpath3 $latest_path3
