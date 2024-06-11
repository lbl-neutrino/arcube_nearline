#!/usr/bin/env bash

set -o errexit

inbase=/global/cfs/cdirs/dune/www/data/2x2/nearline/packet
data_outbase=/global/cfs/cdirs/dune/www/data/2x2/nearline/flowed_charge
log_outbase=/pscratch/sd/d/dunepr/logs/nearline/flowed_charge

inpath=$1; shift

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath" .h5).$ext
    mkdir -p "$outbase/$reldir"
    realpath "$outbase/$reldir/$outname"
}

flowpath=$(get_outpath "$data_outbase" FLOW.hdf5)
logpath=$(get_outpath "$log_outbase" log)

if [[ "$(stat -c %s "$inpath")" -gt 1000000000 ]]; then
    echo "File is larger than 1 GB; bailing"
    exit 1
fi

cd $ROOT_OF_ARCUBE_NEARLINE/_install/ndlar_flow

workflow1='yamls/proto_nd_flow/workflows/charge/charge_event_building.yaml'
workflow2='yamls/proto_nd_flow/workflows/charge/charge_event_reconstruction.yaml'
workflow3='yamls/proto_nd_flow/workflows/combined/combined_reconstruction.yaml'
workflow4='yamls/proto_nd_flow/workflows/charge/prompt_calibration.yaml'
workflow5='yamls/proto_nd_flow/workflows/charge/final_calibration.yaml'

rm -f "$flowpath"

time h5flow -c "$workflow1" "$workflow2" "$workflow3" "$workflow4" "$workflow5" \
    -i "$inpath" -o "$flowpath" 2>&1 | tee "$logpath"
