#!/usr/bin/env bash

stage=light_dqm

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=/global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_light

inpath=$1; shift

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath" .hdf5).light_dqm.$ext
    mkdir -p "$outbase/$reldir"
    realpath "$outbase/$reldir/$outname"
}

foas=$(dirname "${BASH_SOURCE[0]}")/light_dqm/v3_FOAS-2.json

plotpath1=$(get_outpath "$plot_outbase" main.pdf)
plotpath2=$(get_outpath "$plot_outbase" baseline.png)
plotpath3=$(get_outpath "$plot_outbase" dead_chan.png)

logpath=$(get_outpath "$log_outbase" log)

cd "$(dirname "${BASH_SOURCE[0]}")/light_dqm"

if [[ "$(stat -c %s "$inpath")" -gt 50000000000 ]]; then
    echo "File is larger than 50 GB; bailing"
    exit 1
fi

python3 light_dqm.py --input_file "$inpath" \
    --dead_json "$foas" \
    --output_file_1 "$plotpath1" \
    --output_file_2 "$plotpath2" \
    --output_file_3 "$plotpath3" \
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
