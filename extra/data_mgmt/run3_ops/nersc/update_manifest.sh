#!/usr/bin/env bash
#
# manifest.sh - maintain a manifest of files under a directory.
#
# Usage: manifest.sh DIRECTORY [MANIFEST_FILE] [INTERVAL_SECONDS]
#
# Each manifest line has the form:  <filename> <md5>
# Filenames may contain spaces; the last space-separated field is always
# the checksum, so parsing strips it from the end of each line.

set -u

die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'Warning: %s\n' "$*" >&2; }

[[ $# -ge 1 ]] || die "usage: $0 DIRECTORY [MANIFEST_FILE] [INTERVAL_SECONDS]"

DIR=$1
[[ -d $DIR ]] || die "'$DIR' is not a directory"
DIR=$(cd "$DIR" && pwd) || die "cannot access '$DIR'"   # absolute, no trailing slash

MANIFEST=${2:-"$DIR/.manifest"}
INTERVAL=${3:-60}
[[ $INTERVAL =~ ^[0-9]+$ ]] || die "interval must be a whole number of seconds"

# Make the manifest path absolute so we can recognise (and skip) it during the scan.
mdir=$(cd "$(dirname -- "$MANIFEST")" && pwd) || die "manifest directory does not exist"
MANIFEST="$mdir/$(basename -- "$MANIFEST")"

declare -A known=()

# --- Load existing manifest (once, at startup) -------------------------------
if [[ -f $MANIFEST ]]; then
    while IFS= read -r line || [[ -n $line ]]; do
        [[ -z $line ]] && continue
        name=${line% *}            # strip trailing " <md5>"
        known["$name"]=1
    done < "$MANIFEST"
    printf 'Loaded %d entries from %s\n' "${#known[@]}" "$MANIFEST"
fi

# --- Scan the directory and append new files ---------------------------------
scan() {
    local path name sum added=0
    while IFS= read -r -d '' path; do
        [[ $path == "$MANIFEST" ]] && continue
        name=${path##*/}

        [[ -n ${known["$name"]+x} ]] && continue

        if [[ $name == *$'\n'* ]]; then
            warn "skipping filename containing a newline: $(printf '%q' "$name")"
            continue
        fi

        # Read via stdin so md5sum doesn't escape unusual filenames in its output.
        # If the file vanished or is unreadable, skip it; it will be retried next pass.
        sum=$(md5sum < "$path" 2>/dev/null) || { warn "cannot checksum '${path#"$DIR"/}'"; continue; }
        sum=${sum%% *}

        printf '%s %s\n' "$name" "$sum" >> "$MANIFEST" \
            || die "cannot write to manifest '$MANIFEST'"
        known["$name"]=1
        (( added++ ))
    done < <(find "$DIR" -type f -print0)

    (( added > 0 )) && printf '%s: added %d file(s)\n' "$(date '+%F %T')" "$added"
}

trap 'printf "\nExiting.\n"; exit 0' INT TERM

while true; do
    scan
    sleep "$INTERVAL" &
    wait $!        # lets the trap fire immediately instead of after sleep finishes
done
