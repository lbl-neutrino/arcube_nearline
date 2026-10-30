#!/usr/bin/env bash
# Convert a raw (binary) LArPix HDF5 file to packet format with larpix-control,
# as in ndproto_nearline (branch justin) justin/packetize_charge_data.jobscript.
# Usage: packetize.sh RAW_FILE PACKET_FILE
# Needs larpix-control branch direct-convert-mixed-v2-v3 (Python >= 3.10) in the environment.
# ASIC version per io_group can be overridden with the ASIC_VERSION env variable.

set -euo pipefail

if [[ "$#" != "2" ]]; then
    echo "Usage: $0 RAW_FILE PACKET_FILE"
    exit 1
fi

inpath=$1
outpath=$2
# coldbox CRP: io_groups 3,4 are LArPix-v3, the others LArPix-v2
asic_version=${ASIC_VERSION:-1:2,2:2,3:3,4:3,5:2,6:2}

if [[ ! -f "$inpath" ]]; then
    echo "Input file not found: $inpath" >&2
    exit 1
fi
if ! command -v convert_rawhdf5_to_hdf5.py > /dev/null; then
    echo "convert_rawhdf5_to_hdf5.py not found in PATH: activate the larpix-control environment first" >&2
    exit 1
fi

mkdir -p "$(dirname "$outpath")"
# the converter appends to an existing file
rm -f "$outpath"

convert_rawhdf5_to_hdf5.py \
    --input_filename "$inpath" --output_filename "$outpath" \
    --asic-version "$asic_version" \
    --format-version 3.0 --direct
