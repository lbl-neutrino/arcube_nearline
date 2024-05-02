#!/usr/bin/env bash

set -o errexit

inbase=/global/cfs/cdirs/dune/www/data/2x2/LRS
data_outbase=/global/cfs/cdirs/dune/www/data/2x2/nearline/flowed_light
log_outbase=/pscratch/sd/d/dunepr/logs/nearline/flowed_light

inpath=$1; shift

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath" .data).$ext
    mkdir -p "$outbase/$reldir"
    realpath "$outbase/$reldir/$outname"
}

flowpath=$(get_outpath "$data_outbase" FLOW.hdf5)
logpath=$(get_outpath "$log_outbase" log)

if [[ "$(stat -c %s "$inpath")" -gt 50000000000 ]]; then
    echo "File is larger than 50 GB; bailing"
    exit 1
fi

cd $ROOT_OF_ARCUBE_NEARLINE/_install/ndlar_flow

workflow='yamls/proto_nd_flow/workflows/light/light_event_building_adc64.yaml'

rm -f "$flowpath"

h5flow -c "$workflow" -i "$inpath" -o "$flowpath" 2>&1 | tee "$logpath"
