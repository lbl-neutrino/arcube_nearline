#!/usr/bin/env bash

stage=rate_trigger

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

plot_outbase=$data_outbase/plots

inbase=$nearline_root/packet

inpath=$1; shift

get_outpath() {
    outbase=$1
    ext=$2

    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath").rate_trigger.$ext
    echo "$outbase/$reldir/$outname"
}

plotpath=$(get_outpath "$plot_outbase" png)
logpath=$(get_outpath "$log_outbase" log)

cd "$(dirname "${BASH_SOURCE[0]}")/plot_rate_trigger"

if [[ "$(stat -c %s "$inpath")" -gt 10000000000 ]]; then
    echo "File is larger than 10 GB; bailing"
    exit 1
fi

mkdir -p "$(dirname "$plotpath")" "$(dirname "$logpath")"

./plot_rate_trigger.py --input_file "$inpath" --output_file "$plotpath" 2>&1 | tee "$logpath"
