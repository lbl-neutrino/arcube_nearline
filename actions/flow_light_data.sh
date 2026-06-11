#!/usr/bin/env bash

stage=flowed_light

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inname=${ARCUBE_NEARLINE_DATA_DIR_NAME:-LRS_run3}
inbase=$data_root/$inname

inpath=$1; shift
outstamp=$(TZ=America/Chicago date --date=@$(jq '.metadata."core.start_time"' $inpath.json) +%Y_%m_%d_%H_%S)

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    inbase_name=$(basename "$inpath")
    if [[ "$inbase_name" =~ ^mpd_(.+)_rctl_([0-9]+)_p([0-9]+)\.data$ ]]; then
        config="${BASH_REMATCH[1]}"
        run=$(printf "%06d" "${BASH_REMATCH[2]}")
        subrun=$(printf "%05d" "${BASH_REMATCH[3]}")
        outname="mpd_${config}_${outstamp}_${run}_p${subrun}.$ext"
    else
        outname=$(basename "$inpath" .data).$ext
    fi
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

workflow='yamls/proto_nd_flow/workflows/light/light_event_building_mpd_Run2.yaml'

rm -f "$flowpath"

set -o errexit
set -o pipefail
h5flow -c "$workflow" -i "$inpath" -o "$flowpath.tmp" 2>&1 | tee "$logpath"
mv "$flowpath.tmp" "$flowpath"
