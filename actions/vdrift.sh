#!/usr/bin/env bash

stage=vdrift

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=$data_root/$nearline_name/flowed_charge

inpath=$1; shift

# Detector option of the flow vdrift scripts; selects LArData.yaml for the calibrated v_drift
detector=2x2

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath").vdrift.$ext
    echo "$outbase/$reldir/$outname"
}

jsonpath="$json_outbase/json_vdrift.json"
logpath=$(get_outpath "$log_outbase" log)

globalplotpath="$plot_outbase/vdrift_time_series.png"

# The analysis and plotting live in ndlar_flow
cd $ROOT_OF_ARCUBE_NEARLINE/_install/ndlar_flow/scripts/vdrift

if [[ "$(stat -c %s "$inpath")" -gt 10000000000 ]]; then
    echo "File is larger than 10 GB; bailing"
    exit 1
fi

mkdir -p "$(dirname "$logpath")" "$(dirname "$jsonpath")" "$(dirname "$globalplotpath")"

python v_drift.py --input_file "$inpath" --output_file_json "$jsonpath" --detector "$detector" 2>&1 | tee "$logpath"
python vdrift_timeseries.py --input_file "$jsonpath" --output_file "$globalplotpath" --detector "$detector" 2>&1 | tee -a "$logpath"
