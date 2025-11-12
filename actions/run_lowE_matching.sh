#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_txt_file>"
    exit 1
fi

stage=flowed_CLmatching_low_energy

source $(dirname $BASH_SOURCE)/../lib/init.inc.sh

inbase=$data_root/$nearline_name/lowE_matching_inputs
#echo "inbase = ${inbase}"

outdir=$data_root/$nearline_name/$stage/
mkdir -p $outdir
#echo "outdir = ${outdir}"

inpath=$1; shift

relative_path="${inpath#$inbase/}"
#echo "Relative path: $relative_path"
input_lrs_path=$data_root/$nearline_name/${relative_path%.INPUTS.txt}.FLOW_LE.hdf5
#echo "Light file path: ${input_lrs_path}"

output_rel_path=$data_root/$nearline_name/$stage/$(dirname "${relative_path}")
#echo "output_rel_path = ${output_rel_path}"
mkdir -p $output_rel_path

if [ ! -r "$inpath" ]; then
    echo "Error: Cannot read file '$inpath'"
    exit 1
fi

workflow1=yamls/proto_nd_flow/workflows/combined/low_energy_charge_light_matching_nearline.yaml

cd $ROOT_OF_ARCUBE_NEARLINE/_install/ndlar_flow

#echo "light_filename = ${light_filename}"
while IFS= read -r filepath; do
    echo "Matching $filepath to $input_lrs_path"
    if [ -f "$output_path" ]; then
        continue
    fi
    output_path=$output_rel_path/$(basename "${filepath%.FLOW.hdf5}")_$(basename "${relative_path%.INPUTS.txt}")_CLmatched.FLOW_LE.hdf5

    cp $filepath $output_path.tmp
    h5copy -i $input_lrs_path -o $output_path.tmp -s light -d light -f ref
    
    time h5flow -c "$workflow1" \
    -i "$output_path.tmp" -o "$output_path.tmp" --compression gzip 2>&1 | tee "$logpath"

    mv "$output_path.tmp" "$output_path"
    
done < "$inpath"
