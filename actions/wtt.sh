#!/usr/bin/env bash
# Whole-tile-trigger (WTT) monitor, modelled on actions/lifetime.sh:
# tags WTTs in one flow file, adds its summary to the shared json, then redraw the WTT time series.

stage=wtt

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=$data_root/$nearline_name/flowed_charge

inpath=$1; shift

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath").wtt.$ext
    echo "$outbase/$reldir/$outname"
}

jsonpath="$json_outbase/json_wtt.json"
logpath=$(get_outpath "$log_outbase" log)

globalplotpath="$plot_outbase/wtt_time_series.png"

cd "$(dirname "${BASH_SOURCE[0]}")/wtt"

if [[ "$(stat -c %s "$inpath")" -gt 10000000000 ]]; then
    echo "File is larger than 10 GB; bailing"
    exit 1
fi

mkdir -p "$(dirname "$logpath")" "$(dirname "$jsonpath")" "$(dirname "$globalplotpath")"

python wtt.py --input_file "$inpath" --output_file_json "$jsonpath" 2>&1 | tee "$logpath"
python wtt_timeseries.py --input_file "$jsonpath" --output_file "$globalplotpath" 2>&1 | tee -a "$logpath"
