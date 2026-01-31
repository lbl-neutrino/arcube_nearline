#!/usr/bin/env bash

set -o errexit
set -o pipefail

f=$1; shift

f_meta="$f.json"

if [[ ! -f "$f_meta" ]]; then
  echo "Missing metadata for $f. Skipping."
  exit 1
fi

dest_dir=/data/DAQ

out_name=$(basename "$f" .h5).hdf5

f_out="$dest_dir/$out_name"

if [[ -e "$f_out" || -e "$f_out.copied" ]]; then
  continue
fi

mv "$f" "$f_out"

f_out_meta="$f_out.json"

cp -i "$f_meta" "$f_out_meta.tmp"

cksum=$(xrdadler32 "$f_out" | awk '{print $1}')
size=$(stat -c %s "$f_out")

# my apologies
sed -i "s/\.h5\",/.hdf5\",/" "$f_out_meta.tmp"
sed -i "s/\"adler32\": \"[^\"]*\"/\"adler32\": \"$cksum\"/" "$f_out_meta.tmp"
sed -i "s/\"size\": [^,]*/\"size\": $size/" "$f_out_meta.tmp"
mv -i "$f_out_meta.tmp" "$f_out_meta"

mv "$f_meta" "$f_meta.moved"
