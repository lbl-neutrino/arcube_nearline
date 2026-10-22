#!/usr/bin/env bash

src=$1; shift

source $(dirname "${BASH_SOURCE[0]}")/inc/nersc.inc.sh

while true; do
    load_manifest
    find "$src" -type f -print0 | while IFS= read -r -d '' file; do
        f_out=/data/DAQ/$(basename "$file")
        if [[ ! (-e "$f_out" || -e "$f_out.copied") ]]; then
            continue
        fi

        fname=$(basename "$file")
        if [[ -z ${manifest["$fname"]} ]]; then
            continue
        fi

        if [[ -e "$f_out" ]]; then
            file2=$f_out
        else
            file2=$f_out.copied
        fi

        sum1=$(md5sum "$file")
        sum2=$(md5sum "$file2")
        sum3=${manifest[fname]}

        if [[ "$sum1" == "$sum2" && "$sum1" == "$sum3" ]]; then
            rm -f "$file" "$file.json"
        fi
    sleep 60
done
