#!/bin/bash

set -o errexit
set -o pipefail

srcfile=$1; shift

if [[ ! -f "$srcfile.json" ]]; then
  echo "Missing metadata for $srcfile. Skipping."
  exit 1
fi

destdir=../dropbox

newname=$(basename "$srcfile" | sed 's/^mpd_run_data/mpd_run_run2data/')

mv "$srcfile" "$destdir/$newname"
cat "$srcfile.json" | jq ".name |= \"$newname\"" > "$destdir/$newname.json"
# mv "$srcfile.json" "$srcfile.json.moved"
rm -f "$srcfile.json"
