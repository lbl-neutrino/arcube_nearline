#!/usr/bin/env bash

set -o errexit

inbase=/global/cfs/cdirs/dune/www/data/2x2/CRS
data_outbase=/global/cfs/cdirs/dune/www/data/2x2/nearline/packet
log_outbase=/pscratch/sd/d/dunepr/logs/nearline/packet

get_outpath() {
    inpath=$1
    outbase=$2
    indir=$(dirname "$inpath")
    reldir=$(echo "$indir" | sed "s|^$inbase/||")
    outname=$(basename "$inpath" | sed s/binary/packet/)
    echo "$outbase/$reldir/$outname"
}

inpath=$1; shift

outpath=$(get_outpath "$inpath" "$data_outbase")
logpath=$(get_outpath "$inpath" "$log_outbase").log

mkdir -p "$(dirname "$outpath")" "$(dirname "$logpath")"

rm -f "$outpath"

convert_rawhdf5_to_hdf5.py \
    --direct \
    --input_filename "$inpath" \
    --output_filename "$outpath" \
    | tee "$logpath"
