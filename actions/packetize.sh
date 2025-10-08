#!/usr/bin/env bash

stage=packet

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=$data_root/CRS.run2

inpath=$1; shift

get_outpath() {
    inpath=$1
    outbase=$2
    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath" | sed s/binary/packet/)
    echo "$outbase/$reldir/$outname"
}

outpath=$(get_outpath "$inpath" "$data_outbase")
logpath=$(get_outpath "$inpath" "$log_outbase").log

mkdir -p "$(dirname "$outpath")" "$(dirname "$logpath")"

rm -f "$outpath"

# If files were directly recorded in packet format, just copy them
if [[ "$(basename "$inpath")" == "packet-"* ]]; then
    cp "$inpath" "$outpath"
    echo COPIED "$inpath" "$outpath" | tee "$logpath"
    exit
fi

convert_rawhdf5_to_hdf5.py \
    --direct \
    --input_filename "$inpath" \
    --output_filename "$outpath" \
    2>&1 | tee "$logpath"
