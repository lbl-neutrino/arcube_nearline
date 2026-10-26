#!/usr/bin/env bash

set -euo pipefail

src=$1; shift

copy_one() {
    srcfile=$1; shift

    if [[ ! -f "$srcfile.json" ]]; then
        echo "Missing metadata for $srcfile. Skipping."
        exit 1
    fi

    destdir=/data/DAQ

    newname=$(basename "$srcfile" | sed 's/^mpd_run_data/mpd_run_run3_data/')

    cp "$srcfile" "$destdir/$newname"
    cat "$srcfile.json" | jq ".name |= \"$newname\"" > "$destdir/$newname.json"
    # mv "$srcfile.json" "$srcfile.json.moved"
}

while true; do
    find "$src" -type f -name '*.data' -print0 | while IFS= read -r -d '' file; do
        f_out=/data/DAQ/$(basename "$file")
        if [[ -e "$f_out" || -e "$f_out.copied" ]]; then
            continue
        fi
        copy_one "$file"
    done
    sleep 60
done
