#!/usr/bin/env bash

url=$1; shift

manifest_tmpdir=$(mktemp -d)
trap EXIT rm -rf "$manifest_tmpdir"

manifest_file=$manifest_tmpdir/manifest.txt
declare -gA manifest=()

load_manifest() {
    if [[ "$(find "$manifest_file" -mmin -10)" && ${#manifest[@]} -gt 0 ]]; then
        return
    fi
    curl "$url" > "$manifest_file"
    read_manifest
}

read_manifest() {
    declare -gA manifest=()

    while read -r filename checksum; do
        manifest["$filename"]="$checksum"
    done < "$manifest_tmpdir/manifest.txt"
}
