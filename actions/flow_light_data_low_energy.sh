#!/usr/bin/env bash

stage=flowed_light_low_energy

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=$nearline_root/flowed_light

inpath=$1; shift

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath" .FLOW.hdf5).$ext
    mkdir -p "$outbase/$reldir"
    realpath "$outbase/$reldir/$outname"
}

flowpath=$(get_outpath "$data_outbase" FLOW_LE.hdf5)
logpath=$(get_outpath "$log_outbase" log)

if [[ "$(stat -c %s "$inpath")" -gt 50000000000 ]]; then
    echo "File is larger than 50 GB; bailing"
    exit 1
fi

cd $ROOT_OF_ARCUBE_NEARLINE/_install/ndlar_flow

#workflow='yamls/proto_nd_flow/workflows/light/light_event_building_mpd_Run2.yaml'
workflow_1='yamls/proto_nd_flow/workflows/light/light_event_reconstruction_data_LowEnergy.yaml'

rm -f "$flowpath"

set -o errexit
set -o pipefail
h5flow -c "$workflow_1" -i "$inpath" -o "$flowpath.tmp" --compression gzip 2>&1 | tee "$logpath"
mv "$flowpath.tmp" "$flowpath"
