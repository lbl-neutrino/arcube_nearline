#!/usr/bin/env bash

stage=lifetime

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=$data_root/$nearline_name/flowed_charge

inpath=$1; shift

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath").lifetime.$ext
    echo "$outbase/$reldir/$outname"
}

jsonpath="$json_outbase/json_elifetime.json"
plotpath=$(get_outpath "$plot_outbase" png)
logpath=$(get_outpath "$log_outbase" log)

globalplotpath="$plot_outbase/elifetime_time_series.png"

cd "$(dirname "${BASH_SOURCE[0]}")/lifetime"

if [[ "$(stat -c %s "$inpath")" -gt 10000000000 ]]; then
    echo "File is larger than 10 GB; bailing"
    exit 1
fi

mkdir -p "$(dirname "$plotpath")" "$(dirname "$logpath")" "$(dirname "$jsonpath")" 

python lifetime.py --input_file "$inpath" --output_file_plot "$plotpath"  --output_file_json "$jsonpath" 2>&1 | tee "$logpath"
python lifetime_timeseries.py --input_file "$jsonpath" --output_file "$globalplotpath" 2>&1 |tee "$logpath"
