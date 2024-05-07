#!/usr/bin/env bash

set -o errexit

inbase=/global/cfs/cdirs/dune/www/data/2x2/nearline/flowed_light
plot_outbase=/global/cfs/cdirs/dune/www/data/2x2/nearline/light_dqm/plots
log_outbase=/pscratch/sd/d/dunepr/logs/nearline/light_dqm

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

plotpath1=$(get_outpath "$plot_outbase" main.pdf)
plotpath2=$(get_outpath "$plot_outbase" baseline.pdf)
plotpath3=$(get_outpath "$plot_outbase" dead_chan.pdf)

logpath=$(get_outpath "$log_outbase" log)

cd "$(dirname "${BASH_SOURCE[0]}")/light_dqm"

if [[ "$(stat -c %s "$inpath")" -gt 50000000000 ]]; then
    echo "File is larger than 50 GB; bailing"
    exit 1
fi

python3 light_dqm.py --input_file "$inpath" \
    --output_file_1 "$plotpath1" \
    --output_file_2 "$plotpath2" \
    --output_file_3 "$plotpath3" \
    2>&1 | tee "$logpath"
