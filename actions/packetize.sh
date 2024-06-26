#!/usr/bin/env bash

stage=packet

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=$data_root/CRS

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

convert_rawhdf5_to_hdf5.py \
    --direct \
    --input_filename "$inpath" \
    --output_filename "$outpath" \
    | tee "$logpath"
