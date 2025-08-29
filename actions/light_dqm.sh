#!/usr/bin/env bash

stage=light_dqm

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=/global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_light/warm_commissioning

filesyntax=$1; shift
startrun=$1; shift
inpath=$inbase


get_outpath() {
    outbase=$1
    ext=$2
    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
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
python3 light_dqm.py --input_path "$inbase" \
                     --output_dir "$plot_outbase" \
                     --tmp_dir "$plot_outbase/tmp" \
                     --file_syntax "$file_syntax" \
                     --output_dir $get_outpath \
                     --tmp_dir 
                     --channel_status_file "$channel_status"\
                     --start_run $startrun \
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
